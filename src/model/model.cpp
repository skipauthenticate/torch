#include "model/model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/error.h"
#include "core/placement.h"
#include "core/swaps.h"
#include "formats/dequant.h"
#include "kernels/attention/attention.h"
#include "kernels/deltanet/deltanet.h"
#include "kernels/gemm/gemm.h"
#include "kernels/gemv/gemv.h"
#include "kernels/hc/hc.h"
#include "kernels/indexer/indexer.h"
#include "kernels/moe/moe.h"
#include "kernels/norm/norm.h"
#include "kernels/ple/ple.h"
#include "kernels/sampling/argmax.h"
#include "kernels/sampling/sample.h"
#include "model/weights.h"
#include "runtime/direct_io.h"
#include "runtime/expert_store.h"
#include "runtime/ple_rows.h"
#include "runtime/prefill_reads.h"

namespace torch::model {
namespace {

using kernels::BlockQ8;

constexpr int kReadThreads = 16;  // 128 KiB reads reach ~5.4 GB/s at queue depth 16 (bench/ssd_reads.cpp)

using Clock = std::chrono::steady_clock;
double ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}
constexpr int kMaxContext = 262144;  // tokens the engine allows; above indexer_top_k attention is sparse
// Attention is dense up to indexer_top_k tokens of context and sparse above. TORCH_DENSE_ATTENTION=1 keeps the dense
// kernel at any context, which is exact only up to 2,051 tokens: a test of the sparse path's identity with it.
bool sparse_attention(const Config& c, int context) {
    return context > c.indexer_top_k && std::getenv("TORCH_DENSE_ATTENTION") == nullptr;
}
// A sparse prefill chunk's indexer scores, selection and cell attention run on this many columns at a time.
constexpr int kAttentionBatch = 128;
constexpr uint64_t kCudaContext = kGiB / 2;  // the CUDA context and allocator slack [est]

// Lays out device buffers in one allocation. The dry run (no base) only adds up the bytes, so the memory budget and
// the allocation come from the same list.
class DeviceLayout {
public:
    explicit DeviceLayout(uint8_t* base, int column = 0, int copies = kMaxColumns)
        : base_(base), column_(column), copies_(copies) {}

    // One allocation of `count` elements.
    template <class T>
    T* take(size_t count) {
        T* at = base_ ? reinterpret_cast<T*>(base_ + used_) : nullptr;
        used_ += (count * sizeof(T) + 255) / 256 * 256;
        return at;
    }
    // `copies` (kMaxColumns, or a prefill chunk's columns) of `count` elements, column after column; returns the
    // layout's column.
    template <class T>
    T* cols(size_t count) {
        T* at = take<T>(count * static_cast<size_t>(copies_));
        return at ? at + count * static_cast<size_t>(column_) : nullptr;
    }
    uint64_t used() const noexcept { return std::max(used_, peak_); }
    // What is laid out between mark() and rewind(mark) shares its bytes with what is laid out after the rewind: a
    // prefill chunk's blocks never run at once.
    uint64_t mark() const noexcept { return used_; }
    void rewind(uint64_t at) noexcept {
        peak_ = std::max(peak_, used_);
        used_ = at;
    }

private:
    uint8_t* base_;
    int column_, copies_;
    uint64_t used_ = 0, peak_ = 0;
};

// Everything the forward pass keeps on the GPU. The sequence state comes first, so reset() clears one prefix.
struct Buffers {
    // Sequence state.
    std::vector<uint16_t*> k_cache, v_cache;    // attention layers: [context][kv heads x head dim], F16
    std::vector<float*> conv_state, ssm_state;  // DeltaNet layers: [3][channels]; [v heads][128][128]
    float* ple_history = nullptr;               // [9][hc x n_embd]
    uint64_t state_bytes = 0;
    // A window reads the DeltaNet state from one buffer and writes the state after it to the other; which is which
    // flips with every window (Impl::parity). Each DeltaNet layer keeps its window's inputs, which accept() needs to
    // commit the convolution history and, when it keeps fewer tokens than the window ran, to replay the state.
    std::vector<float*> ssm_other;                               // the second state buffer per DeltaNet layer
    std::vector<float*> win_qkv, win_conv, win_alpha, win_beta;  // [kMaxColumns][...] per DeltaNet layer

    // Activations: kMaxColumns copies of each, one per window column (DeviceLayout::cols); a Buffers laid out for
    // column t points at column t's copies, and at the same sequence state.
    int32_t* pos = nullptr;  // this column's position, read by the kernels
    float *res, *xn, *lo, *hc_gate, *mixed, *inject, *out;
    float *embd, *ple_embd;
    float *qkv, *z, *beta, *alpha, *conv_out, *o;
    float *q, *k, *v, *attn, *attn_scratch;
    // Sparse attention (a context above indexer_top_k): per attention layer the 16-row ring of raw indexer keys and the
    // pooled keys by block, both F16 and indexed by position; per column the indexer's projections and normed queries;
    // and the window's scores, cell lists and attention scratch.
    std::vector<uint16_t*> idx_ring, idx_pool;
    float *idx_q_raw = nullptr, *idx_k_raw = nullptr, *idx_q = nullptr;
    uint16_t* idx_raw = nullptr;  // a chunk's F16 raw indexer keys, [column][kIndexDim]
    float* idx_scores = nullptr;
    kernels::IndexCells cells{};
    float *router, *weights, *h, *y, *shared_g, *shared_u, *shared_h, *shared_y, *shared_gate;
    int32_t* ids;
    float *key, *key_n, *value, *query, *ple_gate, *gated, *ple_x;
    float* logits;
    kernels::ArgmaxResult *argmax, *argmax_scratch;
    float* sample_scratch = nullptr;  // the sampler's distributions, kMaxColumns of n_vocab (windows' buffers only)
    // Q8 copies of the vectors quantized products read, one buffer per vector so that no column's copy is overwritten
    // by another vector's before it is read: the block input, the attention or DeltaNet output (6,144 wide), the shared
    // expert's hidden vector, the routed experts' hidden vectors, and any other input (up to hc x n_embd wide).
    BlockQ8 *q8, *q8_x, *q8_sh, *q8_h, *q8_wide;
    // A chunk's quantized product input re-laid for the tensor-core kernels (kernels/gemm), the widest input's size;
    // and its routed experts: the routes' buckets, each slot's input re-laid, the gate and up products (the gate's
    // becomes h), h quantized and re-laid, and the device table of the experts' matrices ([3][n_expert]).
    uint8_t* q8_columns = nullptr;
    uint8_t *route_buckets = nullptr, *q8_routes = nullptr, *q8_h_routes = nullptr;
    float *gate_routes = nullptr, *up_routes = nullptr;
    const uint8_t** expert_table = nullptr;

    // d's column picks the activations' copy. chunk > 0 lays out a prefill chunk of that many columns instead: no
    // sequence state (Impl::prefill points this at the windows'), and one set of DeltaNet inputs for every layer, since
    // a chunk commits each layer's convolution history right after the layer.
    void lay_out(DeviceLayout& d, const Config& c, int context, int chunk = 0) {
        if (chunk > 0) {
            lay_out_chunk(d, c, context, chunk);
        } else {
            lay_out_state(d, c, context);
            lay_out_activations(d, c, context, 0, sparse_attention(c, context));
        }
    }

    // A chunk's buffers: those read across blocks, then one region that each block's own buffers share in turn (the
    // mixer's temporaries, DeltaNet's, attention's, the experts' with their grouped workspace, the PLE block's), since
    // a chunk runs one block at a time. The windows' buffers (lay_out_activations) are separate.
    void lay_out_chunk(DeviceLayout& d, const Config& c, int context, int chunk) {
        const size_t E = static_cast<size_t>(c.n_embd), HCD = static_cast<size_t>(c.hc) * E;
        const size_t KV = static_cast<size_t>(c.n_head_kv * c.head_dim),
                     HD = static_cast<size_t>(c.n_head * c.head_dim);
        const size_t CC = static_cast<size_t>(c.conv_channels());
        const size_t HV = static_cast<size_t>(c.ssm_v_heads), DS = static_cast<size_t>(c.ssm_head_dim);
        const size_t V = HV * DS, K = static_cast<size_t>(c.n_expert_used), FF = static_cast<size_t>(c.n_ff_expert);
        const size_t SH = static_cast<size_t>(c.n_ff_shared), N = static_cast<size_t>(chunk);
        pos = d.cols<int32_t>(1);
        res = d.cols<float>(HCD), mixed = d.cols<float>(E), inject = d.cols<float>(static_cast<size_t>(c.hc));
        out = d.cols<float>(E), embd = d.cols<float>(E), ple_embd = d.cols<float>(E), ple_x = d.cols<float>(HCD);
        q8 = d.cols<BlockQ8>(E / 32), q8_x = d.cols<BlockQ8>(std::max(V, HD) / 32), q8_wide = d.cols<BlockQ8>(HCD / 32);
        q8_columns = d.take<uint8_t>(kernels::q8_columns_bytes(static_cast<int>(HCD / 32), chunk));
        logits = d.take<float>(static_cast<size_t>(c.n_vocab));  // the head runs for the chunk's last column only
        argmax = d.take<kernels::ArgmaxResult>(1),
        argmax_scratch = d.take<kernels::ArgmaxResult>(kernels::kArgmaxBlocks);
        expert_table = d.take<const uint8_t*>(3 * static_cast<size_t>(c.n_expert));
        qkv = conv_out = beta = alpha = nullptr;  // a window's; a chunk's DeltaNet inputs are win_* below
        const uint64_t region = d.mark();
        xn = d.cols<float>(HCD), lo = d.cols<float>(static_cast<size_t>(c.hc_rank)), hc_gate = d.cols<float>(HCD);
        d.rewind(region);  // DeltaNet: one set of inputs for every layer (each layer commits its history at once)
        float *qkv_in = d.take<float>(N * CC), *conv_in = d.take<float>(N * CC);
        float *alpha_in = d.take<float>(N * HV), *beta_in = d.take<float>(N * HV);
        win_qkv.assign(static_cast<size_t>(c.n_layer), qkv_in), win_conv.assign(win_qkv.size(), conv_in);
        win_alpha.assign(win_qkv.size(), alpha_in), win_beta.assign(win_qkv.size(), beta_in);
        z = d.cols<float>(V), o = d.cols<float>(V);
        d.rewind(region);  // attention
        q = d.cols<float>(2 * HD), k = d.cols<float>(KV), v = d.cols<float>(KV), attn = d.cols<float>(HD);
        if (sparse_attention(c, context)) {
            idx_q_raw = d.cols<float>(kernels::kIndexHeads * kernels::kIndexDim);
            idx_k_raw = d.cols<float>(kernels::kIndexDim);
            idx_q = d.cols<float>(kernels::kIndexHeads * kernels::kIndexDim);
            // scored, selected and attended in batches of kAttentionBatch columns (Impl::attention)
            const int batch = std::min(chunk, kAttentionBatch);
            idx_raw = d.take<uint16_t>(N * kernels::kIndexDim);
            idx_scores = d.take<float>(static_cast<size_t>(kernels::index_score_floats(context, batch)));
            cells.blocks = d.take<int32_t>(static_cast<size_t>(batch) * kernels::kIndexTopBlocks);
            cells.info = d.take<int32_t>(static_cast<size_t>(batch) * 4);
            attn_scratch = d.take<float>(static_cast<size_t>(kernels::attention_cells_scratch_floats(c.n_head, batch)));
        }
        d.rewind(region);  // the experts, the routed ones grouped (Impl::routed_experts)
        router = d.cols<float>(static_cast<size_t>(c.n_expert)), weights = d.cols<float>(K);
        shared_gate = d.cols<float>(1), ids = d.cols<int32_t>(K), y = d.cols<float>(K * E);
        shared_g = d.cols<float>(SH), shared_u = d.cols<float>(SH), shared_h = d.cols<float>(SH);
        shared_y = d.cols<float>(E), q8_sh = d.cols<BlockQ8>(SH / 32);
        h = nullptr, q8_h = nullptr;  // a window's
        const int routes = chunk * c.n_expert_used, slots = kernels::expert_slots(c.n_expert, routes);
        const size_t S = static_cast<size_t>(slots);
        route_buckets = d.take<uint8_t>(kernels::expert_buckets_bytes(c.n_expert, routes));
        q8_routes = d.take<uint8_t>(kernels::q8_columns_bytes(static_cast<int>(E / 32), slots + 64));
        gate_routes = d.take<float>(S * FF), up_routes = d.take<float>(S * FF);
        q8_h_routes = d.take<uint8_t>(kernels::q8_columns_bytes(static_cast<int>(FF / 32), slots + 64));
        d.rewind(region);  // the PLE block (its ple_x stays: the ring takes it at the chunk's end)
        key = d.cols<float>(HCD), key_n = d.cols<float>(HCD), value = d.cols<float>(E), query = d.cols<float>(HCD);
        ple_gate = d.cols<float>(static_cast<size_t>(c.hc)), gated = d.cols<float>(HCD);
        d.rewind(region);
    }

    void lay_out_state(DeviceLayout& d, const Config& c, int context) {
        const size_t CC = static_cast<size_t>(c.conv_channels()), KV = static_cast<size_t>(c.n_head_kv * c.head_dim);
        const size_t E = static_cast<size_t>(c.n_embd), HCD = static_cast<size_t>(c.hc) * E;
        const size_t HV = static_cast<size_t>(c.ssm_v_heads), DS = static_cast<size_t>(c.ssm_head_dim);
        k_cache.assign(static_cast<size_t>(c.n_layer), nullptr);
        v_cache = k_cache;
        conv_state.assign(static_cast<size_t>(c.n_layer), nullptr);
        ssm_state = conv_state;
        for (int il = 0; il < c.n_layer; ++il)
            if (c.recurrent(il)) {
                conv_state[static_cast<size_t>(il)] = d.take<float>(3 * CC);
                ssm_state[static_cast<size_t>(il)] = d.take<float>(HV * DS * DS);
            }
        ple_history = d.take<float>(9 * HCD);
        state_bytes = d.used();
        // K/V rows are written before they are read, so the reset leaves them (and the experts lent their tail) alone
        for (int il = 0; il < c.n_layer; ++il)
            if (!c.recurrent(il)) {
                k_cache[static_cast<size_t>(il)] = d.take<uint16_t>(static_cast<size_t>(context) * KV);
                v_cache[static_cast<size_t>(il)] = d.take<uint16_t>(static_cast<size_t>(context) * KV);
            }
        // position-indexed and written before read: not sequence state, no reset needed
        const bool sparse = sparse_attention(c, context);
        idx_ring.assign(static_cast<size_t>(c.n_layer), nullptr);
        idx_pool = idx_ring;
        for (int il = 0; il < c.n_layer && sparse; ++il)
            if (!c.recurrent(il)) {
                idx_ring[static_cast<size_t>(il)] = d.take<uint16_t>(kernels::kIndexRing * kernels::kIndexDim);
                idx_pool[static_cast<size_t>(il)] =
                    d.take<uint16_t>(static_cast<size_t>(context / kernels::kIndexPool) * kernels::kIndexDim);
            }
        ssm_other.assign(static_cast<size_t>(c.n_layer), nullptr);
        win_qkv = win_conv = win_alpha = win_beta = ssm_other;
        for (int il = 0; il < c.n_layer; ++il) {
            const size_t l = static_cast<size_t>(il);
            if (c.recurrent(il)) {
                ssm_other[l] = d.take<float>(HV * DS * DS);
                win_qkv[l] = d.take<float>(kMaxColumns * CC), win_conv[l] = d.take<float>(kMaxColumns * CC);
                win_alpha[l] = d.take<float>(kMaxColumns * HV), win_beta[l] = d.take<float>(kMaxColumns * HV);
            }
        }
    }

    void lay_out_activations(DeviceLayout& d, const Config& c, int context, int, bool sparse) {
        const size_t E = static_cast<size_t>(c.n_embd), HCD = static_cast<size_t>(c.hc) * E;
        const size_t KV = static_cast<size_t>(c.n_head_kv * c.head_dim);
        const size_t CC = static_cast<size_t>(c.conv_channels());
        const size_t HV = static_cast<size_t>(c.ssm_v_heads), DS = static_cast<size_t>(c.ssm_head_dim);
        const size_t V = HV * DS, K = static_cast<size_t>(c.n_expert_used), FF = static_cast<size_t>(c.n_ff_expert);
        pos = d.cols<int32_t>(1);
        res = d.cols<float>(HCD), xn = d.cols<float>(HCD), lo = d.cols<float>(static_cast<size_t>(c.hc_rank));
        hc_gate = d.cols<float>(HCD), mixed = d.cols<float>(E), inject = d.cols<float>(static_cast<size_t>(c.hc));
        out = d.cols<float>(E), embd = d.cols<float>(E), ple_embd = d.cols<float>(E);
        qkv = d.cols<float>(CC), z = d.cols<float>(V), beta = d.cols<float>(HV), alpha = d.cols<float>(HV);
        conv_out = d.cols<float>(CC), o = d.cols<float>(V);
        q = d.cols<float>(2 * static_cast<size_t>(c.n_head * c.head_dim)), k = d.cols<float>(KV);
        v = d.cols<float>(KV), attn = d.cols<float>(static_cast<size_t>(c.n_head * c.head_dim));
        // one scratch for the whole window: attention_decode indexes it by column (a chunk runs these kernels on groups
        // of up to kMaxColumns columns, with the windows' scratch: Impl::share_state)
        if (sparse) {
            idx_q_raw = d.cols<float>(kernels::kIndexHeads * kernels::kIndexDim);
            idx_k_raw = d.cols<float>(kernels::kIndexDim);
            idx_q = d.cols<float>(kernels::kIndexHeads * kernels::kIndexDim);
        }
        if (sparse) {
            idx_scores = d.take<float>(static_cast<size_t>(kernels::index_score_floats(context, kMaxColumns)));
            cells.blocks = d.take<int32_t>(kMaxColumns * kernels::kIndexTopBlocks);
            cells.info = d.take<int32_t>(kMaxColumns * 4);
            attn_scratch =
                d.take<float>(static_cast<size_t>(kernels::attention_cells_scratch_floats(c.n_head, kMaxColumns)));
        } else {
            attn_scratch =
                d.take<float>(static_cast<size_t>(kernels::attention_scratch_floats(c.n_head, context, kMaxColumns)));
        }
        router = d.cols<float>(static_cast<size_t>(c.n_expert)), weights = d.cols<float>(K);
        ids = d.cols<int32_t>(K), h = d.cols<float>(K * FF), y = d.cols<float>(K * E);
        const size_t SH = static_cast<size_t>(c.n_ff_shared);
        shared_g = d.cols<float>(SH), shared_u = d.cols<float>(SH), shared_h = d.cols<float>(SH);
        shared_y = d.cols<float>(E), shared_gate = d.cols<float>(1);
        key = d.cols<float>(HCD), key_n = d.cols<float>(HCD), value = d.cols<float>(E), query = d.cols<float>(HCD);
        ple_gate = d.cols<float>(static_cast<size_t>(c.hc)), gated = d.cols<float>(HCD), ple_x = d.cols<float>(HCD);
        logits = d.cols<float>(static_cast<size_t>(c.n_vocab));
        argmax = d.cols<kernels::ArgmaxResult>(1),
        argmax_scratch = d.cols<kernels::ArgmaxResult>(kernels::kArgmaxBlocks);
        q8 = d.cols<BlockQ8>(E / 32),
        q8_x = d.cols<BlockQ8>(std::max(V, static_cast<size_t>(c.n_head * c.head_dim)) / 32);
        q8_sh = d.cols<BlockQ8>(SH / 32), q8_h = d.cols<BlockQ8>(K * FF / 32), q8_wide = d.cols<BlockQ8>(HCD / 32);
        sample_scratch = d.take<float>(static_cast<size_t>(kernels::sampling_scratch_floats(c.n_vocab, kMaxColumns)));
    }
};

// Pinned host memory the GPU copies to and from, kMaxColumns of each, column after column.
struct HostBuffers {
    float* logits;                  // [columns][n_vocab]
    float *embd, *ple;              // [columns][n_embd]: each token's embedding row and PLE rows
    kernels::ArgmaxResult* argmax;  // [columns]
    int32_t* pos;                   // [columns]

    static uint64_t bytes(const Config& c) {
        return kMaxColumns * ((2 * static_cast<uint64_t>(c.n_embd) + static_cast<uint64_t>(c.n_vocab)) * sizeof(float) +
                              sizeof(kernels::ArgmaxResult) + sizeof(int32_t)) +
               sampling_bytes(c);
    }
    // The sampler's mapped memory (allocated on its own): the history counts, then its small inputs and outputs.
    static constexpr int kSamplingInts = 4 * kMaxColumns;
    static uint64_t sampling_bytes(const Config& c) {
        return (static_cast<uint64_t>(c.n_vocab) + kSamplingInts) * sizeof(int32_t);
    }
    void lay_out(uint8_t* host, const Config& c) {
        logits = reinterpret_cast<float*>(host);
        embd = logits + static_cast<size_t>(kMaxColumns) * static_cast<size_t>(c.n_vocab);
        ple = embd + static_cast<size_t>(kMaxColumns) * static_cast<size_t>(c.n_embd);
        argmax = reinterpret_cast<kernels::ArgmaxResult*>(ple + static_cast<size_t>(kMaxColumns) *
                                                                    static_cast<size_t>(c.n_embd));
        pos = reinterpret_cast<int32_t*>(argmax + kMaxColumns);
    }
};

}  // namespace

struct Model::Impl {
    gguf::File file;             // every shard's tensors (gguf::File::open_model)
    runtime::DirectFile direct;  // reads them where file's offsets say
    Config cfg;
    int context;
    runtime::ReadPool pool{kReadThreads};
    Budget budget;
    RouteCounts ranking;  // the order experts are made resident in, without --resident
    std::unique_ptr<Weights> weights;
    std::unique_ptr<runtime::ExpertStore> store;
    std::unique_ptr<runtime::PleRows> ple;
    cudaStream_t stream = nullptr;
    // --lend-kv: per attention layer its K rows, V rows and pooled keys, bytes per position; the experts with a matrix
    // below position lend_mark of a region are no longer resident there (Options::lend_from).
    struct LendRegion {
        uint8_t* base;
        uint64_t per_position;
    };
    std::vector<LendRegion> lend_regions;
    int lend_mark = 0, lend_step = 0, tenants_now = 0;
    std::atomic<bool> cancel_requested{false};  // Model::cancel(), from any thread
    bool cancelled = false;
    uint8_t* device = nullptr;
    uint8_t* host = nullptr;
    std::array<Buffers, kMaxColumns> cols;  // each window column's view of the buffers
    Buffers b;                              // the column being enqueued (each_column)
    int col_ = 0;
    HostBuffers hb{};
    // Prefill (Options::prefill_columns): one chunk's activations, chunk_cap columns of each, in a device allocation of
    // their own, and its pinned token rows; the ops read `ops`, the windows' buffers (cols[0]) or the chunk's.
    int chunk_cap = 0;
    Buffers chunk, chunk_last;  // the chunk from its column 0, and from its last column (the head's)
    uint8_t* chunk_device = nullptr;
    uint64_t chunk_device_bytes = 0;
    uint8_t* chunk_host = nullptr;
    // Options::lend_chunk: experts lent the chunk's device buffer until the next prefill, ranked by prompt_routes
    bool lend_chunk_on = false;
    bool fast_prefill = false;            // Options::fast_prefill: kernels::gemm_bf16_tc for a chunk's BF16 products
    std::vector<uint64_t> prompt_routes;  // the prefill chunks' route counts since the last lend
    int chunk_tenants = 0;
    bool lent_early = false;                            // lend_chunk(true) has lent the part after the residuals
    float *chunk_embd = nullptr, *chunk_ple = nullptr;  // [chunk_cap][n_embd]
    std::future<double> chunk_ple_read;  // the chunk's PLE rows into chunk_ple (ms), joined by ple_rows_in
    int32_t* chunk_pos = nullptr;        // [chunk_cap]
    // The routed experts of a chunk's layer: their route counts, and each pass's experts and matrix table on the host,
    // a ring of kTableRing (a pass's copies to the GPU read them after the host has moved on).
    static constexpr int kTableRing = 4;
    int32_t* chunk_counts = nullptr;                // [n_expert]
    int32_t* chunk_lists[kTableRing] = {};          // [n_expert + 1]: the pass's experts, their count last
    const uint8_t** chunk_tables[kTableRing] = {};  // [3][n_expert]
    // A chunk's route counts of each layer that ran routed_experts ([n_layer][n_expert], pinned, copied without a
    // wait), tallied as hits and misses once the chunk is in; and the experts its passes read.
    int32_t* layer_counts = nullptr;
    std::vector<char> layer_grouped;
    uint64_t prefill_hits = 0, prefill_misses = 0, prefill_reads_n = 0;
    // Sessions, option B (milestone M6): the sequence state after the last prefill chunk, in an allocation of its own
    // (never lent): the bytes reset() clears, each DeltaNet state in ssm_state's place, then the indexer's raw-key
    // rings. rewind() restores it; snapshot_at is its position, or -1.
    uint8_t* snapshot = nullptr;
    int snapshot_at = -1;
    std::array<cudaEvent_t, kTableRing> table_copied{};  // after a ring entry's copies
    int table_next = 0;
    // Wide chunks read their layers' non-resident experts ahead (runtime/prefill_reads.h): per layer, its passes'
    // tickets and experts, planned at the chunk's start; empty: the chunk waits for each layer's route counts instead.
    static constexpr int kReadAheadColumns = 256;  // (1 - 10/512)^256 < 1%: nearly every expert is routed
    std::unique_ptr<runtime::PrefillReads> prefill_reads;
    std::vector<std::vector<std::pair<int, std::vector<int32_t>>>> chunk_passes;
    const Buffers* ops = nullptr;
    bool prefilling = false;
    static constexpr int kPleBatch = 256;  // a chunk's PLE reads go this many tokens' worth at a time
    // TORCH_PREFILL_STAGES=1: the GPU time of each stage of the prefill chunks (CUDA events), printed at the end.
    enum Stage { kDense, kBf16, kExperts, kAttention, kDeltaNet, kTotal, kStages };
    const bool stage_times = std::getenv("TORCH_PREFILL_STAGES") != nullptr;
    std::vector<cudaEvent_t> stage_events;  // a pool, two per mark
    std::vector<int> stage_marks;           // the stage of each recorded pair, in order
    std::array<double, kStages> stage_ms{};
    int stage_chunks = 0;
    // Runs f, timed as stage s when profiling a chunk.
    template <class F>
    void stage(Stage s, F&& f) {
        if (!prefilling || !stage_times) {
            f();
            return;
        }
        const size_t at = 2 * stage_marks.size();  // the mark's slot first: stages nest (kTotal holds the others)
        stage_marks.push_back(s);
        while (stage_events.size() < at + 2) {
            cudaEvent_t e = nullptr;
            TORCH_CUDA_CHECK(cudaEventCreate(&e));
            stage_events.push_back(e);
        }
        TORCH_CUDA_CHECK(cudaEventRecord(stage_events[at], stream));
        f();
        TORCH_CUDA_CHECK(cudaEventRecord(stage_events[at + 1], stream));
    }
    void collect_stages() {  // after the chunk's work has finished
        for (size_t i = 0; i < stage_marks.size(); ++i) {
            float elapsed = 0;
            TORCH_CUDA_CHECK(cudaEventElapsedTime(&elapsed, stage_events[2 * i], stage_events[2 * i + 1]));
            stage_ms[static_cast<size_t>(stage_marks[i])] += elapsed;
        }
        stage_marks.clear();
        ++stage_chunks;
    }
    // TORCH_DECODE_STAGES=1: the GPU time between consecutive marks of a window, by the stage each mark ends, recorded
    // as event nodes in each width's graph (cudaEventRecordExternal), so the windows still replay graphs. Printed at
    // each new prompt (the windows since the last one) and at the end. Graph mode only.
    enum DStage {
        dCopies,
        dPle,
        dMixers,
        dDense,
        dAttention,
        dDeltaNet,
        dRouter,
        dGate,
        dGateUp,
        dQuantize,
        dDown,
        dShared,
        dCombine,
        dHead,
        dOther,
        kDStages
    };
    static constexpr const char* kDStageNames[kDStages] = {"copies",   "ple",    "mixers",    "dense",   "attention",
                                                           "deltanet", "router", "gate+wait", "gate+up", "quantize",
                                                           "down",     "shared", "combine",   "head",    "other"};
    const bool dstage_times = std::getenv("TORCH_DECODE_STAGES") != nullptr;  // used only when !eager
    struct Timeline {
        std::vector<cudaEvent_t> events;
        std::vector<int> stages;
    };
    std::array<std::array<Timeline, kMaxColumns + 1>, 2> timelines;
    Timeline* recording = nullptr;  // the timeline a graph capture is filling
    std::array<std::array<double, kDStages>, kMaxColumns + 1> dstage_ms{};
    std::array<int, kMaxColumns + 1> dstage_windows{};
    int dstage_report = 0;
    void mark(DStage s) {
        if (!recording || prefilling)
            return;
        cudaEvent_t e = nullptr;
        TORCH_CUDA_CHECK(cudaEventCreate(&e));
        TORCH_CUDA_CHECK(cudaEventRecordWithFlags(e, stream, cudaEventRecordExternal));
        recording->events.push_back(e);
        recording->stages.push_back(s);
    }
    void collect_window(const Timeline& t, int columns) {  // after the window's work has finished
        for (size_t i = 1; i < t.events.size(); ++i) {
            float elapsed = 0;
            TORCH_CUDA_CHECK(cudaEventElapsedTime(&elapsed, t.events[i - 1], t.events[i]));
            dstage_ms[static_cast<size_t>(columns)][static_cast<size_t>(t.stages[i])] += elapsed;
        }
        ++dstage_windows[static_cast<size_t>(columns)];
    }
    void report_windows() {
        int windows = 0;
        std::array<double, kDStages> sum{};
        for (int w = 1; w <= kMaxColumns; ++w) {
            const auto& row = dstage_ms[static_cast<size_t>(w)];
            const int n = dstage_windows[static_cast<size_t>(w)];
            if (n == 0)
                continue;
            windows += n;
            double total = 0;
            std::string line;
            for (int k = 0; k < kDStages; ++k) {
                sum[static_cast<size_t>(k)] += row[static_cast<size_t>(k)];
                total += row[static_cast<size_t>(k)];
                line += std::string(" ") + kDStageNames[k] + " " + std::to_string(row[static_cast<size_t>(k)] / n);
            }
            std::fprintf(stderr, "DECODE %d width %d: %d windows, GPU ms per window %.3f:%s\n", dstage_report, w, n,
                         total / n, line.c_str());
        }
        if (windows) {
            double total = 0;
            std::string line;
            for (int k = 0; k < kDStages; ++k) {
                total += sum[static_cast<size_t>(k)];
                line +=
                    std::string(" ") + kDStageNames[k] + " " + std::to_string(sum[static_cast<size_t>(k)] / windows);
            }
            std::fprintf(stderr, "DECODE %d all: %d windows, GPU ms per window %.3f:%s\n", dstage_report, windows,
                         total / windows, line.c_str());
        }
        dstage_ms = {}, dstage_windows = {};
        ++dstage_report;
    }
    size_t window_start = 0;  // the last window's first position, and its width
    int window = 0;
    std::vector<int32_t> history;
    std::string error;
    DecodeTimes times;
    // TORCH_DUMP=1: after every stage, one line on stderr with the stage's sum, mean |x| and count of non-finite
    // values, for bisecting numerics. TORCH_DUMP_DIR=dir also writes each stage's values to
    // dir/<position>-<layer>-<stage>.f32. TORCH_DUMP_STAGES=a,b limits both to those stages, TORCH_DUMP_FROM=p to
    // positions p and later. Slow: every dumped stage waits for the GPU.
    const char* dump_dir = std::getenv("TORCH_DUMP_DIR");
    const std::string dump_stages =
        std::getenv("TORCH_DUMP_STAGES") ? std::string(",") + std::getenv("TORCH_DUMP_STAGES") + "," : "";
    const bool dumping = std::getenv("TORCH_DUMP") != nullptr || dump_dir != nullptr;
    const size_t dump_from =
        std::getenv("TORCH_DUMP_FROM") ? std::strtoull(std::getenv("TORCH_DUMP_FROM"), nullptr, 10) : 0;
    // A token runs as one replayed CUDA graph; TORCH_EAGER=1 (and the stage dumps, from TORCH_DUMP_FROM on) launch
    // kernel by kernel instead.
    const bool eager = dumping || std::getenv("TORCH_EAGER") != nullptr;
    // One graph per DeltaNet state parity and window width, recorded at its first use.
    std::array<std::array<cudaGraphExec_t, kMaxColumns + 1>, 2> graphs{};
    int parity = 0;            // which DeltaNet state buffer holds the state before the next window
    bool window_open = false;  // the last window is not committed yet
    bool want_logits = true;
    // Sampling (sample()): mapped memory the host writes and the kernels read, its device address, and the last
    // window's logits on the device with the position of the token its column 0 draws.
    int32_t *sampling_host = nullptr, *sampling_device = nullptr;
    const float* last_logits = nullptr;
    size_t first_draw = 0;
    int layer_ = -1;              // the layer being computed, for dump lines
    uint64_t reserved_later = 0;  // Options::reserve, allocated by the caller after the model
    // Swaps (Options::swaps_per_window): which experts are resident; their recency is the store's stamps.
    int swaps_per_window = 0;
    Residency residency;
    std::vector<uint32_t> stamps_cache;  // the store's stamps of the layers a swap decision reads
    std::mutex swap_mutex;               // residency, stamps_cache and swapped_in_window: swaps on either thread
    int swapped_in_window = 0;
    std::atomic<uint32_t> window_stamp{0};  // the gate's stamp for the running window (window_start + 1)

    explicit Impl(const Options& o)
        : file(gguf::File::open_model(o.model)), direct(file), cfg(Config::read(file)), context(o.context) {
        want_logits = o.logits;
        if (context < 1 || context > kMaxContext)
            throw Error("The context must be between 1 and " + std::to_string(kMaxContext) + " tokens.");

        // MemAvailable is read before the CUDA context exists; the budget reserves room for it.
        DeviceLayout dry(nullptr);
        b.lay_out(dry, cfg, context);
        const uint64_t KV = static_cast<uint64_t>(cfg.n_head_kv * cfg.head_dim), N = static_cast<uint64_t>(context);
        const uint64_t pooled = sparse_attention(cfg, context) ? kernels::kIndexRing + N / kernels::kIndexPool : 0;
        uint64_t kv_bytes = 0;  // per attention layer: F16 K and V rows and indexer keys, all allocated up front
        for (int il = 0; il < cfg.n_layer; ++il)
            kv_bytes += cfg.recurrent(il) ? 0 : 2 * (2 * N * KV + pooled * kernels::kIndexDim);
        std::vector<Reservation> reservations = {
            {"weights other than routed experts", Weights::base_bytes(cfg, file)},
            {"KV cache and indexer keys (" + std::to_string(context) + " tokens)", kv_bytes},
            {"DeltaNet state and its window copy, activations", dry.used() + HostBuffers::bytes(cfg) - kv_bytes},
            {"expert staging (" + std::to_string(runtime::kGateSlots) + " slots), map and tables",
             runtime::ExpertStore::fixed_bytes(cfg, file)},
            {"CUDA context and allocator slack", kCudaContext},
        };
        chunk_cap = o.prefill_columns;
        if (chunk_cap < 0 || chunk_cap > kMaxPrefillColumns)
            throw Error("A prefill chunk holds 1 to " + std::to_string(kMaxPrefillColumns) + " tokens.");
        if (chunk_cap > 0) {
            DeviceLayout chunk_dry(nullptr, 0, chunk_cap);
            chunk.lay_out(chunk_dry, cfg, context, chunk_cap);
            reservations.push_back({"prefill chunk of " + std::to_string(chunk_cap) + " tokens: activations, rows",
                                    chunk_dry.used() + chunk_host_bytes()});
            if (chunk_cap >= kReadAheadColumns && o.prefill_staging > 0)
                reservations.push_back({"prefill expert reads ahead (2 x " + std::to_string(o.prefill_staging) + ")",
                                        runtime::PrefillReads::bytes(cfg, file, o.prefill_staging)});
            reservations.push_back({"session state after a prefill (sessions)", snapshot_bytes()});
        }
        for (const Reservation& r : o.reserve) {
            reservations.push_back(r);
            reserved_later += r.bytes;
        }
        budget = plan_budget(mem_available(), kFreeFloor, std::move(reservations));
        const Placement placement = place(o);

        TORCH_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        kernels::upload_iq_tables();
        void* d = nullptr;
        TORCH_CUDA_CHECK(cudaMalloc(&d, dry.used()));
        device = static_cast<uint8_t*>(d);
        // Every page now, so the top-up's MemAvailable reading does not depend on when the driver backs them.
        TORCH_CUDA_CHECK(cudaMemset(d, 0, dry.used()));
        TORCH_CUDA_CHECK(cudaDeviceSynchronize());  // before anything on the model's non-blocking stream writes them
        for (int t = 0; t < kMaxColumns; ++t) {
            DeviceLayout layout(device, t);
            cols[static_cast<size_t>(t)].lay_out(layout, cfg, context);
        }
        b = cols[0];
        ops = &cols[0];
        void* h = nullptr;
        TORCH_CUDA_CHECK(
            cudaHostAlloc(&h, HostBuffers::bytes(cfg) - HostBuffers::sampling_bytes(cfg), cudaHostAllocDefault));
        host = static_cast<uint8_t*>(h);
        hb.lay_out(host, cfg);
        TORCH_CUDA_CHECK(cudaHostAlloc(&h, HostBuffers::sampling_bytes(cfg), cudaHostAllocMapped));
        sampling_host = static_cast<int32_t*>(h);
        std::fill_n(sampling_host, cfg.n_vocab + HostBuffers::kSamplingInts, 0);
        TORCH_CUDA_CHECK(cudaHostGetDevicePointer(&h, sampling_host, 0));
        sampling_device = static_cast<int32_t*>(h);
        if (chunk_cap > 0) {
            DeviceLayout chunk_dry(nullptr, 0, chunk_cap);
            chunk.lay_out(chunk_dry, cfg, context, chunk_cap);
            TORCH_CUDA_CHECK(cudaMalloc(&d, chunk_dry.used()));
            chunk_device = static_cast<uint8_t*>(d);
            chunk_device_bytes = chunk_dry.used();
            DeviceLayout layout(chunk_device, 0, chunk_cap);
            chunk.lay_out(layout, cfg, context, chunk_cap);
            share_state(chunk);
            TORCH_CUDA_CHECK(cudaHostAlloc(&h, chunk_host_bytes(), cudaHostAllocDefault));
            chunk_host = static_cast<uint8_t*>(h);
            chunk_embd = reinterpret_cast<float*>(chunk_host);
            chunk_ple = chunk_embd + static_cast<size_t>(chunk_cap) * static_cast<size_t>(cfg.n_embd);
            chunk_pos = reinterpret_cast<int32_t*>(chunk_ple +
                                                   static_cast<size_t>(chunk_cap) * static_cast<size_t>(cfg.n_embd));
            uint8_t* at =
                reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(chunk_pos + chunk_cap) + 15) & ~uintptr_t{15});
            const size_t NE = static_cast<size_t>(cfg.n_expert);
            for (int k = 0; k < kTableRing; ++k) {
                chunk_tables[k] = reinterpret_cast<const uint8_t**>(at);
                chunk_lists[k] = reinterpret_cast<int32_t*>(at + 3 * NE * sizeof(void*));
                at += (3 * NE * sizeof(void*) + (NE + 1) * sizeof(int32_t) + 15) / 16 * 16;
                TORCH_CUDA_CHECK(
                    cudaEventCreateWithFlags(&table_copied[static_cast<size_t>(k)], cudaEventDisableTiming));
            }
            chunk_counts = reinterpret_cast<int32_t*>(at);
            kernels::upload_gemm_tables();
            TORCH_CUDA_CHECK(cudaMalloc(&d, snapshot_bytes()));
            snapshot = static_cast<uint8_t*>(d);
            TORCH_CUDA_CHECK(cudaHostAlloc(&h, sizeof(int32_t) * static_cast<size_t>(cfg.n_layer * cfg.n_expert),
                                           cudaHostAllocDefault));
            layer_counts = static_cast<int32_t*>(h);
            layer_grouped.assign(static_cast<size_t>(cfg.n_layer), 0);
        }

        weights = std::make_unique<Weights>(cfg, file, direct, pool);
        store = std::make_unique<runtime::ExpertStore>(cfg, weights->layers, direct, placement.resident);
        ple = std::make_unique<runtime::PleRows>(cfg, file, direct,
                                                 std::max(kMaxColumns, std::min(chunk_cap, kPleBatch)));
        if (chunk_cap >= kReadAheadColumns && o.prefill_staging > 0)
            prefill_reads = std::make_unique<runtime::PrefillReads>(weights->layers, direct, o.prefill_staging);
        if (!o.resident)
            top_up(placement, o.resident_total);
        if (o.lend_from > 0)
            lend(o.lend_from);
        swaps_per_window = o.swaps_per_window;
        // the CPU copies those swaps (lent tenants are device memory); TORCH_SWAP_WHEN_FULL=0 turns them off where no
        // option reaches (the C API, torch_server)
        const char* fast = std::getenv("TORCH_FAST_PREFILL");
        fast_prefill = o.fast_prefill || (fast && std::string(fast) == "1");
        const char* lend = std::getenv("TORCH_LEND_CHUNK");
        lend_chunk_on = o.lend_chunk && chunk_device != nullptr && !(lend && std::string(lend) == "0");
        const char* full = std::getenv("TORCH_SWAP_WHEN_FULL");
        if (o.swap_when_full && o.lend_from == 0 && !(full && std::string(full) == "0"))
            store->on_full([this](std::span<const runtime::ExpertStore::Staged> s) { swap_staged(s, true); });
        residency.n_layer = cfg.n_layer, residency.n_expert = cfg.n_expert;
        const size_t pairs = static_cast<size_t>(cfg.n_layer) * static_cast<size_t>(cfg.n_expert);
        residency.resident.assign(pairs, 0);
        for (int l = 0; l < cfg.n_layer; ++l)
            for (int e = 0; e < cfg.n_expert; ++e)
                residency.resident[residency.at(l, e)] = store->resident(l, e) ? 1 : 0;
        stamps_cache.assign(pairs, 0);
        residency.last_use = stamps_cache;
        residency.calibration = ranking.counts.size() == pairs ? ranking.counts : std::vector<uint64_t>(pairs, 0);
        reset();
    }

    // Measures MemAvailable now that everything is allocated and makes the next-ranked experts resident until it is
    // down to the floor plus kTopUpMargin (budget.h), or until `total` experts are resident. Output does not change,
    // only the hit rate.
    void top_up(const Placement& placement, std::optional<int> total) {
        budget.available_after_load = mem_available();
        const uint64_t keep = budget.floor + kTopUpMargin + reserved_later;
        const int want = total ? *total - placement.count() : INT_MAX;
        if (budget.available_after_load > keep && want > 0) {
            const Placement more =
                extend_ranked(ranking, runtime::ExpertStore::layer_expert_bytes(cfg, file), placement,
                              budget.available_after_load - keep, static_cast<size_t>(want));
            store->add(more.resident);
            budget.added = more.count();
            budget.added_bytes = more.bytes;
        }
        if (total && placement.count() + budget.added < *total)
            throw Error("Only " + std::to_string(placement.count() + budget.added) +
                        " experts fit in free memory, not " + std::to_string(*total) +
                        "; ask for fewer with --resident-total.");
    }

    // Lends the bytes of positions [from, context) of every attention layer's K rows, V rows and pooled keys to the
    // next-ranked experts after the residency: each expert's three slots together, page-aligned, region by region.
    void lend(int from) {
        if (from < 64 || from % 64 != 0)
            throw Error("--lend-from takes a multiple of 64 positions.");
        lend_mark = lend_step = from;
        if (from >= context)
            return;
        const Buffers& c = cols[0];
        const uint64_t row = static_cast<uint64_t>(cfg.n_head_kv * cfg.head_dim) * sizeof(uint16_t);
        for (int il = 0; il < cfg.n_layer; ++il) {
            const size_t l = static_cast<size_t>(il);
            if (cfg.recurrent(il))
                continue;
            lend_regions.push_back({reinterpret_cast<uint8_t*>(c.k_cache[l]), row});
            lend_regions.push_back({reinterpret_cast<uint8_t*>(c.v_cache[l]), row});
            if (c.idx_pool[l])  // one pooled row of 128 F16 per 4 positions
                lend_regions.push_back({reinterpret_cast<uint8_t*>(c.idx_pool[l]),
                                        kernels::kIndexDim * sizeof(uint16_t) / kernels::kIndexPool});
        }
        uint64_t lent = 0;
        for (const LendRegion& r : lend_regions)
            lent += static_cast<uint64_t>(context - from) * r.per_position;
        budget.lend_from = from;
        budget.tenants = tenants_now = lend_range(from, context);
        budget.lent_bytes = lent;
    }

    // Places the next-ranked experts that are not resident in positions [from, to) of every lend region and returns
    // how many.
    int lend_range(int from, int to) {
        Placement have;
        have.resident.resize(static_cast<size_t>(cfg.n_layer));
        for (int l = 0; l < cfg.n_layer; ++l)
            for (int e = 0; e < cfg.n_expert; ++e)
                if (store->resident(l, e))
                    have.resident[static_cast<size_t>(l)].push_back(e);
        uint64_t room = 0;
        for (const LendRegion& r : lend_regions)
            room += static_cast<uint64_t>(to - from) * r.per_position;
        const std::vector<uint64_t> bytes = runtime::ExpertStore::layer_expert_bytes(cfg, file);
        const RouteCounts order = ranking.counts.empty() ? uniform_ranking(cfg.n_layer, cfg.n_expert) : ranking;
        const Placement next = extend_ranked(order, bytes, have, room);
        std::vector<runtime::ExpertStore::Tenant> tenants;
        size_t r = 0;
        uint64_t at = 0, end = 0;
        const auto open = [&] {
            const uint64_t base = reinterpret_cast<uint64_t>(lend_regions[r].base);
            at = runtime::page_ceil(base + static_cast<uint64_t>(from) * lend_regions[r].per_position);
            end = base + static_cast<uint64_t>(to) * lend_regions[r].per_position;
        };
        open();
        for (int l = 0; l < cfg.n_layer && r < lend_regions.size(); ++l)
            for (const int32_t e : next.resident[static_cast<size_t>(l)]) {
                const uint64_t need = bytes[static_cast<size_t>(l)];
                while (at + need > end && ++r < lend_regions.size())
                    open();
                if (r == lend_regions.size())
                    break;  // the fragments left over hold no more experts
                tenants.push_back({l, e, reinterpret_cast<uint8_t*>(at)});
                at += need;
            }
        store->add_tenants(tenants);
        if (!residency.resident.empty())  // a re-lend after a reset: swaps may pick the new tenants
            for (const runtime::ExpertStore::Tenant& t : tenants)
                residency.resident[residency.at(t.layer, t.expert)] = 1;
        return static_cast<int>(tenants.size());
    }

    // Moves the lend mark one step: the experts with a matrix below it stop being resident (later routes are misses).
    void unlend() {
        lend_mark = std::min(context, lend_mark + lend_step);
        for (const LendRegion& r : lend_regions) {
            const uint8_t* below = r.base + static_cast<uint64_t>(lend_mark) * r.per_position;
            for (const auto& [l, e] : store->evict(r.base, below)) {
                residency.resident[residency.at(l, e)] = 0;
                --tenants_now;
            }
        }
    }

    // Model::lend_chunk(): the experts that are not resident, the prompt's most routed first and then the placement's
    // ranking, each expert's three slots together and page-aligned, in the chunk's device buffer. Residency (the swaps'
    // view) keeps them out, so no swap evicts them or writes into the buffer; the store routes to them as to any
    // resident expert.
    // early: only the part after the chunk's residuals, which the draft's KV-only passes still read (stage_residuals),
    // from another thread while they run; the prefill's GPU work is done. Then a later lend_chunk() adds the residuals'
    // part. Not early: the whole buffer, or the residuals' part after an early lend.
    int lend_chunk(bool early) {
        if (!lend_chunk_on || chunk_device == nullptr || (chunk_tenants > 0 && !(lent_early && !early)))
            return 0;
        const uint64_t base = reinterpret_cast<uint64_t>(chunk_device);
        const uint64_t residuals_end =
            reinterpret_cast<uint64_t>(chunk.res) +
            static_cast<uint64_t>(chunk_cap) * static_cast<uint64_t>(cfg.hc) * static_cast<uint64_t>(cfg.n_embd) * 4;
        uint64_t begin = runtime::page_ceil(base), end = base + chunk_device_bytes;
        if (early)
            begin = runtime::page_ceil(residuals_end);
        else if (lent_early)
            end = runtime::page_ceil(residuals_end);  // where the early lend began
        if (!early)
            TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));  // nothing still reads the chunk's buffers
        store->before_window();                               // swaps still queued land first
        Placement have;
        have.resident.resize(static_cast<size_t>(cfg.n_layer));
        for (int l = 0; l < cfg.n_layer; ++l)
            for (int e = 0; e < cfg.n_expert; ++e)
                if (store->resident(l, e))
                    have.resident[static_cast<size_t>(l)].push_back(e);
        const std::vector<uint64_t> bytes = runtime::ExpertStore::layer_expert_bytes(cfg, file);
        RouteCounts order = ranking.counts.empty() ? uniform_ranking(cfg.n_layer, cfg.n_expert) : ranking;
        if (prompt_routes.size() == order.counts.size())
            for (size_t i = 0; i < order.counts.size(); ++i)  // calibration counts stay far below 2^32
                order.counts[i] += prompt_routes[i] << 32;
        if (!early)
            std::fill(prompt_routes.begin(), prompt_routes.end(), 0);
        const Placement next = extend_ranked(order, bytes, have, end > begin ? end - begin : 0);
        std::vector<runtime::ExpertStore::Tenant> tenants;
        uint64_t at = begin;
        for (int l = 0; l < cfg.n_layer; ++l)
            for (const int32_t e : next.resident[static_cast<size_t>(l)]) {
                const uint64_t need = bytes[static_cast<size_t>(l)];
                if (at + need > end)
                    break;
                tenants.push_back({l, e, reinterpret_cast<uint8_t*>(at)});
                at += need;
            }
        store->add_tenants(tenants);
        chunk_tenants += static_cast<int>(tenants.size());
        lent_early = early;
        return static_cast<int>(tenants.size());
    }
    // Before a prefill writes the chunk's buffers: the lent experts are not resident any more.
    void reclaim_chunk() {
        if (chunk_tenants == 0)
            return;
        const auto out = store->evict(chunk_device, chunk_device + chunk_device_bytes);
        chunk_tenants = 0, lent_early = false;
        if (out.empty())
            throw Error("The prefill chunk's lent experts were not resident; this is a bug in the engine.");
    }

    // Which experts are resident: exactly --resident per layer, the most-routed in Options::placement, or as many of
    // every layer's first experts as fit. Fills in the budget's placement lines and the ranking top_up() continues.
    Placement place(const Options& o) {
        const std::vector<uint64_t> bytes = runtime::ExpertStore::layer_expert_bytes(cfg, file);
        Placement p;
        if (o.resident_total && (o.resident || o.placement.empty() || *o.resident_total < 0))
            throw Error(
                "--resident-total takes a count of experts and keeps a ranked placement: give --placement, "
                "not --resident.");
        if (o.resident) {
            if (*o.resident < 0 || *o.resident > cfg.n_expert)
                throw Error("--resident must be between 0 and " + std::to_string(cfg.n_expert) + ".");
            p = place_uniform(*o.resident, cfg.n_expert, bytes);
            if (p.bytes > budget.room())
                throw Error("Only " + std::to_string(uniform_capacity(budget.room(), cfg.n_expert, bytes)) +
                            " experts per layer fit in free memory, not " + std::to_string(*o.resident) +
                            "; ask for fewer with --resident.");
            budget.placement = *o.resident == 0
                                   ? "no experts resident (--resident 0)"
                                   : "experts 0 to " + std::to_string(*o.resident - 1) + " of every layer (--resident)";
        } else if (!o.placement.empty()) {
            const RouteCounts counts = RouteCounts::read(o.placement);
            if (counts.n_layer != cfg.n_layer || counts.n_expert != cfg.n_expert)
                throw Error("The route counts in " + o.placement.string() + " were recorded with another model.");
            p = place_ranked(counts, bytes, budget.placement_room(),
                             o.resident_total ? static_cast<size_t>(*o.resident_total) : SIZE_MAX);
            budget.placement = "the most-routed experts in " + o.placement.string() +
                               (o.resident_total ? ", " + std::to_string(*o.resident_total) + " in total" : "");
            ranking = counts;
        } else {
            const int k = uniform_capacity(budget.placement_room(), cfg.n_expert, bytes);
            p = place_uniform(k, cfg.n_expert, bytes);
            budget.placement = "experts 0 to " + std::to_string(k - 1) + " of every layer (no route counts given)";
            ranking = uniform_ranking(cfg.n_layer, cfg.n_expert);
        }
        budget.resident = p.count();
        budget.experts = cfg.n_layer * cfg.n_expert;
        budget.resident_bytes = p.bytes;
        return p;
    }

    ~Impl() {
        if (stream)
            cudaStreamSynchronize(stream);
        if (stage_times && stage_chunks) {
            const double sum =
                stage_ms[kDense] + stage_ms[kBf16] + stage_ms[kExperts] + stage_ms[kAttention] + stage_ms[kDeltaNet];
            std::fprintf(
                stderr,
                "PREFILL GPU ms over %d chunks: dense %.1f bf16 %.1f experts %.1f attention %.1f deltanet %.1f "
                "other %.1f total %.1f\n",
                stage_chunks, stage_ms[kDense], stage_ms[kBf16], stage_ms[kExperts], stage_ms[kAttention],
                stage_ms[kDeltaNet], stage_ms[kTotal] - sum, stage_ms[kTotal]);
        }
        for (cudaEvent_t e : stage_events)
            cudaEventDestroy(e);
        if (dstage_times)
            report_windows();
        for (const auto& row : timelines)
            for (const Timeline& t : row)
                for (cudaEvent_t e : t.events)
                    cudaEventDestroy(e);
        if (std::getenv("TORCH_HOST_STAGES") && hs_windows)
            std::fprintf(
                stderr,
                "HOST per window: commit %.3f embed %.3f before_window %.3f launch %.3f swaps %.3f ms (%d windows)\n",
                hs[0] / hs_windows, hs[1] / hs_windows, hs[2] / hs_windows, hs[3] / hs_windows, hs[4] / hs_windows,
                hs_windows);
        for (const auto& row : graphs)
            for (cudaGraphExec_t g : row)
                if (g)
                    cudaGraphExecDestroy(g);
        for (const auto& a : commit_graphs)
            for (const auto& b2 : a)
                for (cudaGraphExec_t g : b2)
                    if (g)
                        cudaGraphExecDestroy(g);
        if (device)
            cudaFree(device);
        if (host)
            cudaFreeHost(host);
        if (sampling_host)
            cudaFreeHost(sampling_host);
        prefill_reads.reset();  // its reads finish first
        for (cudaEvent_t e : table_copied)
            if (e)
                cudaEventDestroy(e);
        if (chunk_device)
            cudaFree(chunk_device);
        if (chunk_host)
            cudaFreeHost(chunk_host);
        if (snapshot)
            cudaFree(snapshot);
        if (layer_counts)
            cudaFreeHost(layer_counts);
        if (stream)
            cudaStreamDestroy(stream);
    }

    void dump(const char* stage, int layer, const float* d, int n) {
        if (!dumping || window_start + static_cast<size_t>(col_) < dump_from ||
            (!dump_stages.empty() && dump_stages.find(std::string(",") + stage + ",") == std::string::npos))
            return;
        std::vector<float> h(static_cast<size_t>(n));
        TORCH_CUDA_CHECK(cudaMemcpyAsync(h.data(), d, sizeof(float) * h.size(), cudaMemcpyDeviceToHost, stream));
        TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (dump_dir) {
            const std::string path = std::string(dump_dir) + "/" +
                                     std::to_string(window_start + static_cast<size_t>(col_)) + "-" +
                                     std::to_string(layer) + "-" + stage + ".f32";
            if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(h.data(), sizeof(float), h.size(), f);
                std::fclose(f);
            }
        }
        double sum = 0, abs = 0;
        int bad = 0;
        for (float v : h) {
            if (!std::isfinite(v)) {
                ++bad;
                continue;
            }
            sum += v;
            abs += std::fabs(v);
        }
        std::fprintf(stderr, "dump pos %zu layer %2d %-16s n %6d sum %14.6g mean|x| %12.6g non-finite %d [%g %g %g]\n",
                     window_start + static_cast<size_t>(col_), layer, stage, n, sum, abs / n, bad, h[0],
                     n > 1 ? h[1] : 0.0f, n > 2 ? h[2] : 0.0f);
    }

    void reset() {
        history.clear();
        snapshot_at = -1;
        parity = 0;  // the state before the first window is in ssm_state, which the memset below clears
        window_open = false;
        TORCH_CUDA_CHECK(cudaMemsetAsync(device, 0, b.state_bytes, stream));
        TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
        store->before_window();  // swaps still queued land first
        store->clear_stamps();
        if (!lend_regions.empty() && lend_mark > lend_step) {  // the new sequence starts below the lent tail again
            const int moved_to = lend_mark;
            lend_mark = lend_step;
            tenants_now += lend_range(lend_step, moved_to);
        }
    }

    // y = m . x: BF16 weights take x as F32, quantized weights take it quantized to Q8 (gemv.h).
    void matvec(const Matrix& m, const float* x, float* y, bool quantized = false, int columns = 1) {
        products(x, {{&m, y}}, quantized, columns);
    }

    // y_i = m_i . x for matrices that read the same x (at most 4, one width): x is quantized once into its Q8 buffer
    // for the quantized ones, which share one launch when gemv_q8_group takes their formats; two BF16 ones share one
    // too. `quantized`: the kernel that wrote x also wrote its Q8 copy (hc_collapse for b.mixed, rms_norm_q8 for b.o).
    struct Product {
        const Matrix* m;
        float* y;
    };
    // With `columns`, x and every y are column 0's pointers of a window, laid out column after column (the matrices'
    // multi-column launches read each weight row once for all columns).
    void products(const float* x, std::initializer_list<Product> list, bool quantized = false, int columns = 1) {
        Product bf16[4], q8[4];
        kernels::Q8Part parts[4];
        int nb = 0, nq = 0, np = 0;
        for (const Product& p : list)
            (p.m->type == Type::BF16 ? bf16[nb++] : q8[nq++]) = p;
        BlockQ8* xq = nq > 0 ? q8_of(x) : nullptr;
        if (nq > 0 && !quantized)
            kernels::quantize_q8(x, int64_t{q8[0].m->cols} * columns, xq, stream);
        stage(kDense, [&] {
            // A chunk multiplies the formats kernels/gemm has on the tensor cores, from its input re-laid once.
            const bool wide = prefilling && columns > kMaxColumns;
            kernels::Q8Columns xc{};
            for (int i = 0; i < nq; ++i) {
                const Matrix& m = *q8[i].m;
                if (wide && kernels::gemm_q8_supports(m.type) && kernels::gemm_q8_fits(m.cols)) {
                    if (xc.q == nullptr) {
                        xc = kernels::q8_columns_at(ops->q8_columns, m.cols / 32, columns);
                        kernels::q8_to_columns(xq, xc, stream);
                    }
                    kernels::gemm_q8(m.type, m.data, m.rows, m.cols, xc, q8[i].y, stream);
                } else if (nq > 1 && kernels::gemv_q8_groups(m.type))
                    parts[np++] = {m.type, m.data, m.rows, q8[i].y};
                else
                    groups(columns, [&](int g, int n) {
                        kernels::gemv_q8(m.type, m.data, m.rows, m.cols, xq + int64_t{g} * (m.cols / 32),
                                         q8[i].y + int64_t{g} * m.rows, stream, n);
                    });
            }
            const int64_t units = nq > 0 ? q8[0].m->cols / 32 : 0;
            if (np == 1)
                groups(columns, [&](int g, int n) {
                    kernels::gemv_q8(parts[0].type, parts[0].w, parts[0].rows, q8[0].m->cols, xq + g * units,
                                     parts[0].y + int64_t{g} * parts[0].rows, stream, n);
                });
            else if (np > 1)
                groups(columns, [&](int g, int n) {
                    kernels::Q8Part at[4];
                    for (int i = 0; i < np; ++i)
                        at[i] = {parts[i].type, parts[i].w, parts[i].rows, parts[i].y + int64_t{g} * parts[i].rows};
                    kernels::gemv_q8_group(at, np, q8[0].m->cols, xq + g * units, false, stream, n);
                });
        });
        stage(kBf16, [&] {
            if (prefilling && nb > 0 && kernels::gemm_bf16_wins(bf16[0].m->cols)) {  // a chunk: the window order on all
                                                                                     // its columns at once, pairs too
                for (int i = 0; i < nb; i += 2) {
                    const Matrix& m = *bf16[i].m;
                    const Matrix* m2 = i + 1 < nb ? bf16[i + 1].m : nullptr;
                    bf16_chunk(m.data, m.rows, m.cols, x, bf16[i].y, columns, 0.0f, m2 ? m2->data : nullptr,
                               m2 ? m2->rows : 0, m2 ? bf16[i + 1].y : nullptr);
                }
                return;
            }
            for (int i = 0; i < nb; i += 2) {
                const Matrix& m = *bf16[i].m;
                groups(columns, [&](int g, int n) {
                    if (i + 1 < nb)
                        kernels::gemv_bf16_pair(m.data, m.rows, bf16[i + 1].m->data, bf16[i + 1].m->rows, m.cols,
                                                x + int64_t{g} * m.cols, bf16[i].y + int64_t{g} * m.rows,
                                                bf16[i + 1].y + int64_t{g} * bf16[i + 1].m->rows, stream, 0.0f, n);
                    else
                        kernels::gemv_bf16(m.data, m.rows, m.cols, x + int64_t{g} * m.cols,
                                           bf16[i].y + int64_t{g} * m.rows, stream, n);
                });
            }
        });
    }

    // The window kernels take 1 to kMaxColumns columns: a prefill chunk runs them on its columns in groups of that many
    // (or `width`), in order, f(first column, count); a window is one group.
    template <class F>
    void groups(int columns, F&& f, int width = kMaxColumns) {
        for (int g = 0; g < columns; g += width)
            f(g, std::min(width, columns - g));
    }

    // A chunk's BF16 product: the window order (gemm_bf16, bitwise the windows), or with Options::fast_prefill the
    // tensor cores (gemm_bf16_tc), which take K in multiples of 64.
    void bf16_chunk(const uint8_t* w, int rows, int width, const float* x, float* y, int columns, float silu_scale,
                    const uint8_t* w2, int rows2, float* y2) {
        if (!fast_prefill || !kernels::gemm_bf16_tc(w, rows, width, x, y, columns, silu_scale, stream, w2, rows2, y2))
            kernels::gemm_bf16(w, rows, width, x, y, columns, silu_scale, stream, w2, rows2, y2, ops->q8_columns,
                               kernels::q8_columns_bytes(cfg.hc * cfg.n_embd / 32, chunk_cap));
    }

    // A hyper-connection mixer over the window's columns: the hc streams of each column's residual, normalized, mixed
    // into one block input (b.mixed, and its Q8), and, when `inject` is given, the block output's per-stream weights
    // (b.inject). With `combine`, the previous block's output (b.out, weighted by b.inject) is first added to the
    // residual, in the same launch as the norm.
    void hc_mix(const float* norm, const Matrix& down, const Matrix& up, const Matrix* inject, bool combine,
                int columns) {
        const int E = cfg.n_embd;
        const Buffers& c = *ops;
        if (combine)
            kernels::hc_combine_norm(c.res, c.out, c.inject, norm, c.xn, E, cfg.hc, cfg.eps, stream, columns);
        else
            kernels::rms_norm(c.res, E, norm, E, nullptr, 0, c.xn, E, E, cfg.hc, cfg.eps, stream, columns);
        // A BF16 down projection applies the low-rank SiLU as it stores; BF16 inject rows read the same input, so
        // they share its launch.
        const float scale = 1.0f / static_cast<float>(cfg.hc);
        const bool q8_down = !prefilling && down.type == Type::Q8_0;
        const bool paired =
            inject && (down.type == Type::BF16 || q8_down) && inject->type == Type::BF16 && inject->cols == down.cols;
        if (down.type == Type::BF16 && prefilling && kernels::gemm_bf16_wins(down.cols)) {  // a chunk: all columns
            stage(kBf16, [&] {
                bf16_chunk(down.data, down.rows, down.cols, c.xn, c.lo, columns, scale, paired ? inject->data : nullptr,
                           paired ? inject->rows : 0, c.inject);
            });
        } else if (down.type == Type::BF16) {
            stage(kBf16, [&] {
                groups(columns, [&](int g, int n) {
                    kernels::gemv_bf16_pair(down.data, down.rows, paired ? inject->data : nullptr,
                                            paired ? inject->rows : 0, down.cols, c.xn + int64_t{g} * down.cols,
                                            c.lo + int64_t{g} * down.rows, c.inject + int64_t{g} * cfg.hc, stream,
                                            scale, n);
                });
            });
        } else if (q8_down) {
            BlockQ8* xq = q8_of(c.xn);
            kernels::quantize_q8(c.xn, int64_t{columns} * down.cols, xq, stream);
            kernels::mixer_down_q8(down.data, paired ? inject->data : nullptr, down.rows, paired ? inject->rows : 0,
                                   down.cols, xq, c.xn, c.lo, c.inject, scale, columns, stream);
        } else {
            matvec(down, c.xn, c.lo, false, columns);
            kernels::silu_scale(c.lo, cfg.hc_rank, scale, stream, columns);
        }
        if (inject && !paired)
            matvec(*inject, c.xn, c.inject, false, columns);
        // A window's up projection and collapse share one launch, bitwise the two (gemv_bf16_collapse).
        if (!prefilling && up.type == Type::Q8_0 && up.cols == 320 && cfg.hc == 4 && E % 32 == 0)
            kernels::mixer_up_q8(up.data, E, c.lo, c.xn, c.mixed, c.q8, columns, stream);
        else if (prefilling || up.type != Type::BF16 ||
                 !kernels::gemv_bf16_collapse(up.data, E, cfg.hc, up.cols, c.lo, c.xn, c.mixed, c.q8, stream,
                                              columns)) {
            matvec(up, c.lo, c.hc_gate, false, columns);
            kernels::hc_collapse(c.xn, c.hc_gate, c.mixed, c.q8, E, cfg.hc, stream, columns);
        }
        each_column(columns, [&] {
            dump("hc_norm", layer_, b.xn, cfg.hc * E);
            dump("hc_lo", layer_, b.lo, cfg.hc_rank);
            dump("hc_mixed", layer_, b.mixed, E);
            if (inject)
                dump("hc_inject", layer_, b.inject, cfg.hc);
        });
    }

    // The PLE block over the window's columns. Its convolution reads the 9-row ring for the positions before the
    // window and the earlier columns for the rest, and leaves the ring to commit() (kernels/ple/ple.h).
    void ple_block(const Layer& L, int columns) {
        const int E = cfg.n_embd;
        const Buffers& c = *ops;
        matvec(L.ple_key, c.ple_embd, c.key, false, columns);
        matvec(L.ple_value, c.ple_embd, c.value, false, columns);
        kernels::rms_norm(c.key, E, L.ple_norm_key, E, nullptr, 0, c.key_n, E, E, cfg.hc, cfg.eps, stream, columns);
        kernels::rms_norm(c.res, E, L.ple_norm_query, E, nullptr, 0, c.query, E, E, cfg.hc, cfg.eps, stream, columns);
        const int64_t HCD = int64_t{cfg.hc} * E;
        for (int t = 0; t < columns; ++t) {
            kernels::ple_gate(c.key_n + t * HCD, c.query + t * HCD, c.ple_gate + t * cfg.hc, E, cfg.hc, stream);
            kernels::ple_gated_value(c.value + int64_t{t} * E, c.ple_gate + t * cfg.hc, c.gated + t * HCD, E, cfg.hc,
                                     stream);
        }
        kernels::rms_norm(c.gated, E, L.ple_norm_conv, E, nullptr, 0, c.ple_x, E, E, cfg.hc, cfg.eps, stream, columns);
        kernels::ple_conv_window(c.res, c.gated, c.ple_x, c.ple_history, L.ple_conv, cfg.hc * E, c.pos, columns,
                                 stream);
        each_column(columns, [&] {
            dump("ple_gate", layer_, b.ple_gate, cfg.hc);
            dump("ple_out", layer_, b.res, cfg.hc * E);
        });
    }

    // The DeltaNet state buffers of layer l for the next window: the state before it, and where the state after it
    // goes.
    float* state_in(size_t l) const { return parity ? cols[0].ssm_other[l] : cols[0].ssm_state[l]; }
    float* state_out(size_t l) const { return parity ? cols[0].ssm_state[l] : cols[0].ssm_other[l]; }

    // A DeltaNet block over the window's columns. Its inputs stay in the layer's window buffers for commit(); the
    // convolution reads the history without writing it, and the recurrence reads state_in and writes state_out.
    void deltanet(const Layer& L, int il, int columns) {
        const int DS = cfg.ssm_head_dim, QK = cfg.ssm_k_heads * DS, HV = cfg.ssm_v_heads, CC = cfg.conv_channels();
        const size_t l = static_cast<size_t>(il);
        const Buffers& c = *ops;
        float *qkv = c.win_qkv[l], *conv = c.win_conv[l], *alpha = c.win_alpha[l], *beta = c.win_beta[l];
        products(c.mixed, {{&L.qkv, qkv}, {&L.z, c.z}, {&L.beta, beta}, {&L.alpha, alpha}}, true, columns);
        mark(dDense);
        // q and k are L2-normalized in the convolution's launch; the gates are computed in the recurrence's
        stage(kDeltaNet, [&] {
            kernels::conv_silu_l2_window(c.conv_state[l], qkv, L.conv, conv, CC, 2 * QK, cfg.eps, columns, stream);
            kernels::gdn_window(state_in(l), state_out(l), conv, conv + QK, conv + 2 * QK, CC, alpha, beta, L.dt_bias,
                                L.a, nullptr, c.o, cfg.ssm_k_heads, HV, 1.0f / std::sqrt(static_cast<float>(DS)),
                                columns, stream);
        });
        if (prefilling)  // the chunk is kept whole, and the next layer reuses its input buffers
            kernels::conv_commit(c.conv_state[l], qkv, CC, columns, stream);
        mark(dDeltaNet);
        kernels::rms_norm_q8(c.o, DS, L.ssm_norm, 0, c.z, DS, c.o, DS, c.q8_x, DS, HV, cfg.eps, stream, columns);
        matvec(L.ssm_out, c.o, c.out, true, columns);
        mark(dDense);
        each_column(columns, [&] {
            const size_t t = static_cast<size_t>(col_);
            dump("qkv", layer_, qkv + t * static_cast<size_t>(CC), CC);
            dump("conv_l2", layer_, conv + t * static_cast<size_t>(CC), CC);
            dump("gdn_norm", layer_, b.o, HV * DS);
            dump("linear_attn_out", layer_, b.out, cfg.n_embd);
        });
    }

    // A gated attention block over the window's columns: every column's K/V rows go into the cache first, then
    // column t attends to rows 0 .. pos + t (kernels/attention/attention.h).
    void attention(const Layer& L, int il, int columns) {
        const int D = cfg.head_dim, H = cfg.n_head, HKV = cfg.n_head_kv;
        const size_t l = static_cast<size_t>(il);
        const Buffers& c = *ops;
        products(c.mixed, {{&L.wq, c.q}, {&L.wk, c.k}, {&L.wv, c.v}}, true, columns);  // q: per head [q | gate]
        const bool sparse = sparse_attention(cfg, context);
        if (sparse)
            products(c.mixed, {{&L.idx_q, c.idx_q_raw}, {&L.idx_k, c.idx_k_raw}}, true, columns);
        const int64_t QS = int64_t{H} * 2 * D, KVS = int64_t{HKV} * D, IQ = kernels::kIndexHeads * kernels::kIndexDim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        // A dense chunk's groups go through the whole block in order, as windows would: the raw-key ring holds 16
        // positions. A sparse chunk stores every K/V row, raw key and pooled key first (later rows are outside a
        // column's cells), then scores, selects and attends in batches: bitwise windows of 8 (indexer_test).
        const bool batched = prefilling && sparse;
        mark(dDense);
        stage(kAttention, [&] {
            if (batched) {
                kernels::qk_norm_rope_store(c.q, 2 * D, L.q_norm, c.k, L.k_norm, c.v, c.k_cache[l], c.v_cache[l], H,
                                            HKV, cfg.n_rot, c.pos, cfg.rope_base, cfg.eps, columns, stream);
                kernels::index_prepare_chunk(c.idx_q_raw, c.idx_k_raw, L.idx_q_norm, c.idx_q, c.idx_raw, c.pos,
                                             cfg.n_rot, cfg.rope_base, cfg.eps, columns, stream);
                kernels::index_pool_chunk(c.idx_ring[l], c.idx_raw, L.idx_k_norm, c.idx_pool[l], c.pos, cfg.n_rot,
                                          cfg.rope_base, cfg.eps, columns, stream);
            }
            groups(
                columns,
                [&](int g, int n) {
                    float* q = c.q + g * QS;
                    const int32_t* pos = c.pos + g;
                    if (!batched)
                        kernels::qk_norm_rope_store(q, 2 * D, L.q_norm, c.k + g * KVS, L.k_norm, c.v + g * KVS,
                                                    c.k_cache[l], c.v_cache[l], H, HKV, cfg.n_rot, pos, cfg.rope_base,
                                                    cfg.eps, n, stream);
                    if (sparse) {  // the indexer picks each column's cells
                        float* idx_q = c.idx_q + g * IQ;
                        if (!batched) {
                            kernels::index_prepare(c.idx_q_raw + g * IQ, c.idx_k_raw + int64_t{g} * kernels::kIndexDim,
                                                   L.idx_q_norm, idx_q, c.idx_ring[l], pos, cfg.n_rot, cfg.rope_base,
                                                   cfg.eps, n, stream);
                            kernels::index_pool(c.idx_ring[l], L.idx_k_norm, c.idx_pool[l], pos, cfg.n_rot,
                                                cfg.rope_base, cfg.eps, n, stream);
                        }
                        kernels::index_score(idx_q, c.idx_pool[l], c.idx_scores, pos, context / kernels::kIndexPool, n,
                                             stream);
                        kernels::index_select(c.idx_scores, c.cells, pos, context / kernels::kIndexPool, n, stream);
                        each_column(n, [&] {  // the scores of a column that drops blocks (tests/model/parity.py --gaps)
                            const int visible = static_cast<int>(window_start) + col_ + 1;
                            const int blocks = visible / kernels::kIndexPool;
                            if (blocks > kernels::kIndexTopBlocks)
                                dump("index_scores", layer_,
                                     c.idx_scores + int64_t{col_} * (context / kernels::kIndexPool), blocks);
                        });
                        kernels::attention_decode_cells(q, q + D, 2 * D, c.k_cache[l], c.v_cache[l],
                                                        c.attn + g * (QS / 2), c.attn_scratch, pos, H, HKV, scale,
                                                        c.cells, n, stream);
                    } else {
                        kernels::attention_decode(q, q + D, 2 * D, c.k_cache[l], c.v_cache[l], c.attn + g * (QS / 2),
                                                  c.attn_scratch, pos, H, HKV, scale, context, n, stream);
                    }
                },
                batched ? kAttentionBatch : kMaxColumns);
        });
        mark(dAttention);
        matvec(L.wo, c.attn, c.out, false, columns);
        mark(dDense);
        each_column(columns, [&] {
            dump("attn", layer_, b.attn, H * D);
            dump("attn_out", layer_, b.out, cfg.n_embd);
        });
    }

    // The MoE block over the window's columns: routers, then each column's experts, then the combine.
    void experts(const Layer& L, int il, int columns) {
        const int E = cfg.n_embd, K = cfg.n_expert_used;
        const Buffers& c = *ops;
        products(c.mixed, {{&L.router, c.router}, {&L.shared_gate_inp, c.shared_gate}}, true, columns);
        each_column(columns, [&] { dump("router", layer_, b.router, cfg.n_expert); });
        kernels::router_top_k(c.router, cfg.n_expert, K, c.ids, c.weights, stream, columns);
        mark(dRouter);
        if (prefilling && kernels::gemm_q8_experts_supports(L.gate_exps.type) &&  // a chunk of any width
            kernels::gemm_q8_experts_supports(L.up_exps.type) && kernels::gemm_q8_experts_supports(L.down_exps.type)) {
            stage(kExperts, [&] { routed_experts(L, il, columns); });
            shared_expert(L, columns);
            kernels::moe_combine(c.y, c.weights, K, c.shared_y, c.shared_gate, c.out, E, stream, columns);
            return;
        }
        if (prefilling)  // a chunk has no window expert buffers (Buffers::lay_out_chunk)
            throw Error("Prefill has no grouped kernel for layer " + std::to_string(il) + "'s expert formats.");
        // One gate launch places the whole window's experts: each missing one is read once, all of a layer's misses
        // in one request (a chunk: per group of columns, in order).
        const runtime::GateTables gt = store->gate_tables(il);
        const int FF = cfg.n_ff_expert;
        stage(kExperts, [&] {
            groups(columns, [&](int g, int n) {
                const int64_t routes = int64_t{g} * K;
                float* h = c.h + routes * FF;
                BlockQ8* q8_h = c.q8_h + routes * FF / 32;
                runtime::expert_gate(c.ids + routes, K, n, il, store->map(), cfg.n_expert, gt, store->mailbox(),
                                     store->seq(), store->counters(), store->route_counts(), store->zeros(), c.pos + g,
                                     store->last_use(), stream);
                mark(dGate);
                // Each unique expert's gate, up and down rows are read once for all the columns that chose it.
                kernels::gemv_q8_experts_window(L.gate_exps.type, gt.table, gt.table + runtime::kGateSlots, gt.columns,
                                                gt.rank, gt.count, std::min(n * K, runtime::kGateSlots), K, FF, E,
                                                c.q8 + int64_t{g} * (E / 32), h, n, stream);  // b.mixed's Q8
                mark(dGateUp);
                // h is quantized by its own launch: writing it as Q8 from gate and up measured slower (+5-9 us a routed
                // launch, +0.15-0.3 ms a token against 0.14 ms of quantize_q8; branch m3-gemv-q8-epilogues).
                kernels::quantize_q8(h, int64_t{n} * K * FF, q8_h, stream);
                mark(dQuantize);
                // Q2_0's down measured faster per (column, rank) route (105 against 112 us at T = 3); both bitwise
                // equal.
                if (n == 1 || L.down_exps.type == Type::Q2_0)
                    kernels::gemv_q8_experts(L.down_exps.type, gt.down_routes, nullptr, n * K, E, FF, q8_h, FF / 32,
                                             c.y + routes * E, stream);
                else
                    kernels::gemv_q8_experts_down_window(
                        L.down_exps.type, gt.table + 2 * runtime::kGateSlots, gt.columns, gt.rank, gt.count,
                        std::min(n * K, runtime::kGateSlots), K, E, FF, q8_h, c.y + routes * E, n, stream);
                mark(dDown);
            });
        });
        shared_expert(L, columns);
        mark(dShared);
        kernels::moe_combine(c.y, c.weights, K, c.shared_y, c.shared_gate, c.out, E, stream, columns);
        mark(dCombine);
        each_column(columns, [&] {
            dump("experts_h", layer_, b.h, K * cfg.n_ff_expert);
            dump("experts_y", layer_, b.y, K * E);
            dump("shared_y", layer_, b.shared_y, E);
            dump("ffn_out", layer_, b.out, E);
        });
    }

    // A chunk's routed experts, grouped by expert (kernels/gemm): the routes' buckets; each slot's input; gate and up
    // over each expert's slots, SwiGLU, h quantized and re-laid, down into y in route order, so moe_combine reads y as
    // a window's (each route bitwise the window kernels' product). Non-resident experts are read into the staging
    // buffers kGateSlots at a time: each pass runs gate, up and down for its experts (the first pass also for the
    // resident ones), waiting for the GPU before the next pass's reads reuse the buffers.
    void routed_experts(const Layer& L, int il, int columns) {
        const int E = cfg.n_embd, K = cfg.n_expert_used, FF = cfg.n_ff_expert, NE = cfg.n_expert;
        const Buffers& c = *ops;
        const int routes = columns * K;
        const kernels::ExpertBuckets bk = kernels::expert_buckets_at(c.route_buckets, NE, routes);
        kernels::expert_buckets(c.ids, routes, K, NE, bk, il, c.pos, store->route_counts(), store->last_use(), stream);
        TORCH_CUDA_CHECK(cudaMemcpyAsync(layer_counts + int64_t{il} * NE, bk.counts,
                                         sizeof(int32_t) * static_cast<size_t>(NE), cudaMemcpyDeviceToHost, stream));
        layer_grouped[static_cast<size_t>(il)] = 1;
        kernels::Q8Columns xg = kernels::q8_columns_at(c.q8_routes, E / 32, bk.slots + 64);  // gemm.h: 64 more
        xg.columns = bk.slots;
        kernels::q8_gather_columns(c.q8, E / 32, bk, K, xg, stream);  // b.mixed's Q8, from hc_collapse
        kernels::Q8Columns hc = kernels::q8_columns_at(c.q8_h_routes, FF / 32, bk.slots + 64);
        hc.columns = bk.slots;
        // The layer's passes: (ticket, -1 or kResidentPass, experts read for it). Reading ahead, the resident experts
        // run as their own pass after the passes the read-ahead's two pools hold, so the reads of the passes after it
        // (into the pools those free) run beside its GPU work; else the first pass also runs them.
        constexpr int kResidentPass = -2;
        std::vector<int32_t> present;
        std::vector<std::pair<int, std::vector<int32_t>>> passes;
        const bool ahead = !chunk_passes.empty();
        if (ahead) {  // read ahead: every resident expert (an unrouted one has an empty bucket) and the planned passes
            for (int e = 0; e < NE; ++e)
                if (store->resident(il, e))
                    present.push_back(e);
            passes = chunk_passes[static_cast<size_t>(il)];
        } else {  // wait for the route counts, then read the misses kGateSlots at a time
            TORCH_CUDA_CHECK(cudaMemcpyAsync(chunk_counts, bk.counts, sizeof(int32_t) * static_cast<size_t>(NE),
                                             cudaMemcpyDeviceToHost, stream));
            TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<int32_t> missing;
            for (int e = 0; e < NE; ++e)
                if (chunk_counts[e] > 0)
                    (store->resident(il, e) ? present : missing).push_back(e);
            for (size_t at = 0; at < missing.size(); at += static_cast<size_t>(runtime::kGateSlots))
                passes.emplace_back(
                    -1, std::vector<int32_t>(
                            missing.begin() + static_cast<std::ptrdiff_t>(at),
                            missing.begin() + static_cast<std::ptrdiff_t>(std::min(
                                                  missing.size(), at + static_cast<size_t>(runtime::kGateSlots)))));
        }
        size_t resident_pass = 0;
        if (ahead) {
            resident_pass = std::min<size_t>(2, passes.size());
            passes.insert(passes.begin() + static_cast<std::ptrdiff_t>(resident_pass),
                          {kResidentPass, std::vector<int32_t>{}});
        } else if (passes.empty()) {
            passes.emplace_back(-1, std::vector<int32_t>{});
        }
        for (size_t p = 0; p < passes.size(); ++p) {
            const auto& [ticket, staged_ids] = passes[p];
            const std::vector<const uint8_t*> staged = ticket >= 0 ? prefill_reads->wait(ticket)
                                                       : ticket == kResidentPass
                                                           ? std::vector<const uint8_t*>{}
                                                           : store->stage_for_prefill(il, staged_ids);
            prefill_reads_n += staged_ids.size();
            const int slot = table_next++ % kTableRing;  // the ring entry's last copies have run
            TORCH_CUDA_CHECK(cudaEventSynchronize(table_copied[static_cast<size_t>(slot)]));
            const uint8_t** table = chunk_tables[slot];
            int32_t* list = chunk_lists[slot];
            std::vector<int32_t> pass = p == resident_pass ? present : std::vector<int32_t>{};
            for (int e : pass) {
                const std::array<const uint8_t*, 3> m = store->resident_matrices(il, e);
                for (int kind = 0; kind < 3; ++kind)
                    table[kind * NE + e] = m[static_cast<size_t>(kind)];
            }
            for (size_t i = 0; i < staged_ids.size(); ++i) {
                const int e = staged_ids[i];
                pass.push_back(e);
                for (int kind = 0; kind < 3; ++kind)
                    table[kind * NE + e] = staged[3 * i + static_cast<size_t>(kind)];
            }
            std::sort(pass.begin(), pass.end());
            std::copy(pass.begin(), pass.end(), list);
            list[NE] = static_cast<int32_t>(pass.size());
            TORCH_CUDA_CHECK(cudaMemcpyAsync(c.expert_table, table, sizeof(void*) * 3 * static_cast<size_t>(NE),
                                             cudaMemcpyHostToDevice, stream));
            TORCH_CUDA_CHECK(
                cudaMemcpyAsync(bk.list, list, sizeof(int32_t) * pass.size(), cudaMemcpyHostToDevice, stream));
            TORCH_CUDA_CHECK(cudaMemcpyAsync(bk.n_list, list + NE, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            TORCH_CUDA_CHECK(cudaEventRecord(table_copied[static_cast<size_t>(slot)], stream));
            kernels::gemm_q8_experts(L.gate_exps.type, c.expert_table, bk, NE, FF, E, xg, nullptr, c.gate_routes,
                                     stream);
            kernels::gemm_q8_experts(L.up_exps.type, c.expert_table + NE, bk, NE, FF, E, xg, nullptr, c.up_routes,
                                     stream);
            kernels::expert_h_columns(c.gate_routes, c.up_routes, bk, NE, hc, stream);  // the pass's slots only
            kernels::gemm_q8_experts(L.down_exps.type, c.expert_table + 2 * NE, bk, NE, E, FF, hc, bk.slot_route, c.y,
                                     stream);
            if (ticket >= 0)
                prefill_reads->release(ticket, stream);  // its bytes are free once these kernels have run
            else if (ticket != kResidentPass)
                TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));  // the store's staging is reused by the next pass
        }
    }

    // The shared expert of the window's columns: gate and up with their SwiGLU in one launch, then down.
    void shared_expert(const Layer& L, int columns) {
        const int E = cfg.n_embd;
        const Buffers& c = *ops;
        const Matrix &sg = L.shared_gate, &su = L.shared_up;
        const int64_t SH = cfg.n_ff_shared;
        if (prefilling && columns > kMaxColumns && kernels::gemm_q8_supports(sg.type) &&
            kernels::gemm_q8_supports(su.type) && kernels::gemm_q8_fits(E)) {  // a chunk: tensor cores, SwiGLU at once
            stage(kDense, [&] {
                const kernels::Q8Columns xc = kernels::q8_columns_at(c.q8_columns, E / 32, columns);
                kernels::q8_to_columns(c.q8, xc, stream);  // b.mixed's Q8
                kernels::gemm_q8(sg.type, sg.data, sg.rows, E, xc, c.shared_g, stream);
                kernels::gemm_q8(su.type, su.data, su.rows, E, xc, c.shared_u, stream);
            });
            kernels::swiglu(c.shared_g, c.shared_u, c.shared_h, static_cast<int>(SH * columns), stream);
        } else if (kernels::gemv_q8_groups(sg.type) && kernels::gemv_q8_groups(su.type) && sg.rows == su.rows) {
            stage(kDense, [&] {
                groups(columns, [&](int g, int n) {
                    const kernels::Q8Part parts[2] = {{sg.type, sg.data, sg.rows, c.shared_h + g * SH},
                                                      {su.type, su.data, su.rows, nullptr}};
                    kernels::gemv_q8_group(parts, 2, E, c.q8 + int64_t{g} * (E / 32), true, stream, n);  // b.mixed's Q8
                });
            });
        } else {
            groups(columns, [&](int g, int n) {
                kernels::gemv_q8(sg.type, sg.data, sg.rows, E, c.q8 + int64_t{g} * (E / 32), c.shared_g + g * SH,
                                 stream, n);
                kernels::gemv_q8(su.type, su.data, su.rows, E, c.q8 + int64_t{g} * (E / 32), c.shared_u + g * SH,
                                 stream, n);
            });
            for (int t = 0; t < columns; ++t)
                kernels::swiglu(c.shared_g + t * SH, c.shared_u + t * SH, c.shared_h + t * SH, cfg.n_ff_shared, stream);
        }
        matvec(L.shared_down, c.shared_h, c.shared_y, false, columns);
    }

    // Runs `f` once per window column with `b` pointing at that column's buffers, columns in order. Each op of the
    // forward runs over all columns before the next op starts (op-major), so a multi-column kernel can replace the
    // loop with one launch; ops on sequence state (attention, DeltaNet, the PLE ring) see the columns in order.
    template <class F>
    void each_column(int columns, F&& f) {
        if (prefilling)
            return;  // stage dumps are per window column
        for (int t = 0; t < columns; ++t) {
            b = cols[static_cast<size_t>(t)];
            col_ = t;
            f();
        }
        b = cols[0];
        col_ = 0;
    }

    // The current column's Q8 buffer for input vector x (see Buffers::q8).
    BlockQ8* q8_of(const float* x) const {
        if (x == b.mixed)
            return b.q8;
        if (x == b.o || x == b.attn)
            return b.q8_x;
        if (x == b.shared_h)
            return b.q8_sh;
        return b.q8_wide;
    }

    // Everything the GPU does for a window of `columns` tokens, on `stream`: run directly, or recorded once per width
    // into that width's graph.
    void enqueue(int columns) {
        const int E = cfg.n_embd;
        const Buffers& c0 = cols[0];
        const size_t n = static_cast<size_t>(columns);
        mark(dOther);  // the timeline's start
        TORCH_CUDA_CHECK(cudaMemcpyAsync(c0.pos, hb.pos, sizeof(int32_t) * n, cudaMemcpyHostToDevice, stream));
        TORCH_CUDA_CHECK(cudaMemcpyAsync(c0.embd, hb.embd, sizeof(float) * n * static_cast<size_t>(E),
                                         cudaMemcpyHostToDevice, stream));
        TORCH_CUDA_CHECK(cudaMemcpyAsync(c0.ple_embd, hb.ple, sizeof(float) * n * static_cast<size_t>(E),
                                         cudaMemcpyHostToDevice, stream));
        mark(dCopies);

        layer_ = -1;
        kernels::hc_broadcast(c0.embd, c0.res, E, cfg.hc, stream, columns);
        each_column(columns, [&] {
            dump("embd", layer_, b.embd, E);
            dump("ple_rows", layer_, b.ple_embd, E);
        });
        // A block's output is added to the residual in the next mixer's norm launch (hc_combine_norm), except before a
        // PLE block, which reads the residual first, and when dumping stages, which shows the residual per layer.
        bool pending = false;  // b.out and b.inject still to be added to b.res
        const auto settle = [&] {
            if (pending)
                kernels::hc_combine(c0.res, c0.out, c0.inject, E, cfg.hc, stream, columns);
            pending = false;
        };
        for (int il = 0; il < cfg.n_layer; ++il) {
            const Layer& L = weights->layers[static_cast<size_t>(il)];
            layer_ = il;
            if (L.ple) {
                settle();
                ple_block(L, columns);
                mark(dPle);
            }
            const bool combine = pending;
            hc_mix(L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, &L.hc_attn_inject, combine, columns);
            mark(dMixers);
            if (L.recurrent)
                deltanet(L, il, columns);
            else
                attention(L, il, columns);
            hc_mix(L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, &L.hc_ffn_inject, true, columns);
            mark(dMixers);
            experts(L, il, columns);
            pending = true;
            if (dumping) {
                settle();
                each_column(columns, [&] { dump("residual", layer_, b.res, cfg.hc * E); });
            }
        }
        layer_ = -1;
        // The head's mixer is the output norm: there is no separate one.
        const bool combine = pending;
        hc_mix(weights->head_hc_norm, weights->head_hc_down, weights->head_hc_up, nullptr, combine, columns);
        mark(dMixers);
        matvec(weights->output, c0.mixed, c0.logits, true, columns);
        mark(dHead);
        each_column(columns, [&] { dump("logits", layer_, b.logits, cfg.n_vocab); });
        kernels::argmax(c0.logits, cfg.n_vocab, c0.argmax_scratch, c0.argmax, stream, columns);
        TORCH_CUDA_CHECK(
            cudaMemcpyAsync(hb.argmax, c0.argmax, sizeof(kernels::ArgmaxResult) * n, cudaMemcpyDeviceToHost, stream));
        if (want_logits)
            TORCH_CUDA_CHECK(cudaMemcpyAsync(hb.logits, c0.logits, sizeof(float) * n * static_cast<size_t>(cfg.n_vocab),
                                             cudaMemcpyDeviceToHost, stream));
        mark(dOther);
    }

    // Draws columns first .. first + n - 1 of the last window (sample_columns), the later columns conditional on the
    // window's inputs after column `first` as drafts, then keeps the longest prefix of those drafts equal to the drawn
    // tokens and the first drawn token that differs (accept_drafts). The inputs, position and tokens go through the
    // mapped memory after the history counts; the scratch is the windows' (the draws run after the window).
    std::vector<int32_t> sample(kernels::SamplingParams params, int first, int n) {
        if (last_logits == nullptr || first < 0 || n < 1 || first + n > window)
            throw Error("Sample columns of the last window, before accept().");
        int32_t* in = sampling_host + cfg.n_vocab;  // drafts [kMaxColumns], position, sampled, accepted + tokens
        int32_t *drafts = in, *position = in + kMaxColumns, *sampled = position + 1, *accepted = sampled + kMaxColumns;
        const size_t off = static_cast<size_t>(cfg.n_vocab);
        for (int j = 0; j + 1 < n; ++j)
            drafts[j] = history[window_start + static_cast<size_t>(first + 1 + j)];
        *position = static_cast<int32_t>(first_draw) + first;
        const bool penalties =
            params.repetition_penalty != 1.0f || params.presence_penalty != 0.0f || params.frequency_penalty != 0.0f;
        int32_t* dev = sampling_device + off;  // the same ints at their device address
        kernels::sample_columns(last_logits + static_cast<int64_t>(first) * cfg.n_vocab, cfg.n_vocab, n,
                                penalties ? sampling_device : nullptr, n > 1 ? dev : nullptr, params, dev + kMaxColumns,
                                cols[0].sample_scratch, dev + kMaxColumns + 1, stream);
        if (n > 1)
            kernels::accept_drafts(dev + kMaxColumns + 1, dev, n, dev + 2 * kMaxColumns + 1, dev + 2 * kMaxColumns + 2,
                                   stream);
        TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
        TORCH_CUDA_CHECK(cudaGetLastError());
        if (n == 1)
            return {sampled[0]};
        return std::vector<int32_t>(accepted + 1, accepted + 1 + *accepted);
    }

    // Settles the last window, keeping its first n tokens: the convolution histories and the PLE ring take the kept
    // columns, and when n is short of the window the DeltaNet state is recomputed from the state before the window
    // over the kept columns (the same kernel and order as the window, so bitwise the same state). The KV rows of the
    // dropped columns stay; they are rewritten before they are read.
    // One graph per DeltaNet state parity, kept columns and whether the state is replayed (n short of the window): the
    // commit's ~70 small launches cost ~0.3 ms of host time launched one by one.
    std::array<std::array<std::array<cudaGraphExec_t, 2>, kMaxColumns + 1>, 2> commit_graphs{};

    void enqueue_commit(int n) {
        const int DS = cfg.ssm_head_dim, QK = cfg.ssm_k_heads * DS, HV = cfg.ssm_v_heads, CC = cfg.conv_channels();
        const Buffers& c = cols[0];
        for (int il = 0; il < cfg.n_layer; ++il) {
            const size_t l = static_cast<size_t>(il);
            if (!cfg.recurrent(il))
                continue;
            if (n < window) {
                const Layer& L = weights->layers[l];
                kernels::gdn_window(state_in(l), state_out(l), c.win_conv[l], c.win_conv[l] + QK,
                                    c.win_conv[l] + 2 * QK, CC, c.win_alpha[l], c.win_beta[l], L.dt_bias, L.a, nullptr,
                                    c.o, cfg.ssm_k_heads, HV, 1.0f / std::sqrt(static_cast<float>(DS)), n, stream);
            }
            kernels::conv_commit(c.conv_state[l], c.win_qkv[l], CC, n, stream);
        }
        kernels::ple_commit(c.ple_history, c.ple_x, cfg.hc * cfg.n_embd, c.pos, n, stream);
    }

    void commit(int n) {
        cudaGraphExec_t& graph = commit_graphs[static_cast<size_t>(parity)][static_cast<size_t>(n)][n < window ? 1 : 0];
        if (eager) {
            enqueue_commit(n);
        } else {
            if (graph == nullptr) {
                cudaGraph_t g = nullptr;
                TORCH_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                try {
                    enqueue_commit(n);
                } catch (...) {
                    cudaStreamEndCapture(stream, &g);
                    if (g)
                        cudaGraphDestroy(g);
                    throw;
                }
                TORCH_CUDA_CHECK(cudaStreamEndCapture(stream, &g));
                const cudaError_t made = cudaGraphInstantiate(&graph, g, 0);
                cudaGraphDestroy(g);
                TORCH_CUDA_CHECK(made);
            }
            TORCH_CUDA_CHECK(cudaGraphLaunch(graph, stream));
        }
        parity ^= 1;
        history.resize(window_start + static_cast<size_t>(n));
        window_open = false;
    }

    std::array<double, 5> hs{};  // measurement: host ms in commit, embedding, before_window, launch, swaps
    int hs_windows = 0;
    void verify(std::span<const int32_t> tokens) {
        const auto start = Clock::now();
        const int columns = static_cast<int>(tokens.size());
        if (columns < 1 || columns > kMaxColumns)
            throw Error("A window holds 1 to " + std::to_string(kMaxColumns) + " tokens, not " +
                        std::to_string(columns) + ".");
        for (const int32_t token : tokens)
            if (token < 0 || token >= cfg.n_vocab)
                throw Error("Token id " + std::to_string(token) + " is outside the vocabulary of " +
                            std::to_string(cfg.n_vocab) + ".");
        if (history.size() + tokens.size() > static_cast<size_t>(context))
            throw Error("The context of " + std::to_string(context) + " tokens is full; start a new sequence.");
        const auto t_commit = Clock::now();
        if (window_open)
            commit(window);  // the last window was not cut: keep all of it
        hs[0] += ms(Clock::now() - t_commit);
        const auto t_embed = Clock::now();
        const int E = cfg.n_embd;
        window_start = history.size();
        window_stamp.store(static_cast<uint32_t>(window_start) + 1, std::memory_order_release);
        window = columns;
        last_logits = cols[0].logits, first_draw = window_start + 1;

        // The token embedding and the PLE rows are single rows: the CPU dequantizes them to F32 into pinned memory,
        // which the window's first copies read.
        const Matrix& te = weights->token_embd;
        for (int t = 0; t < columns; ++t) {
            const int32_t token = tokens[static_cast<size_t>(t)];
            history.push_back(token);
            const size_t at = static_cast<size_t>(t) * static_cast<size_t>(E);
            dequant_row(te.type, te.data + static_cast<uint64_t>(token) * te.row_bytes(), E, hb.embd + at);
            hb.pos[t] = static_cast<int32_t>(window_start) + t;
        }
        hs[1] += ms(Clock::now() - t_embed);
        // All of the window's PLE rows in one batch of reads (rows are E floats a column, hb.ple's layout).
        const auto ple_start = Clock::now();
        ple->read_window(history, window_start, columns, hb.ple);
        const double ple_ms = ms(Clock::now() - ple_start);
        times.ple += ple_ms;
        times.ple_max = std::max(times.ple_max, ple_ms);
        times.ple_slow += ple_ms > 5.0 ? 1 : 0;

        const auto t_bw = Clock::now();
        store->before_window();  // the last window's swaps have landed and its staging buffers are free
        {
            const std::lock_guard lock(swap_mutex);
            swapped_in_window = 0;
        }
        while (!lend_regions.empty() && lend_mark < context &&
               window_start + static_cast<size_t>(columns) > static_cast<size_t>(lend_mark))
            unlend();  // the window writes K/V rows at or past the lend mark
        hs[2] += ms(Clock::now() - t_bw);
        const auto t_launch = Clock::now();
        cudaGraphExec_t& graph = graphs[static_cast<size_t>(parity)][static_cast<size_t>(columns)];
        if (eager && window_start + static_cast<size_t>(columns) > dump_from) {  // earlier windows replay graphs
            enqueue(columns);
        } else {
            if (graph == nullptr) {  // recorded at the first window of this width and parity; its pointers stay fixed
                cudaGraph_t g = nullptr;
                TORCH_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                if (dstage_times)
                    recording = &timelines[static_cast<size_t>(parity)][static_cast<size_t>(columns)];
                try {
                    enqueue(columns);
                    recording = nullptr;
                } catch (...) {
                    recording = nullptr;
                    cudaStreamEndCapture(stream, &g);  // leave the stream usable; the half-recorded graph is dropped
                    if (g)
                        cudaGraphDestroy(g);
                    throw;
                }
                TORCH_CUDA_CHECK(cudaStreamEndCapture(stream, &g));
                const cudaError_t made = cudaGraphInstantiate(&graph, g, 0);
                cudaGraphDestroy(g);
                TORCH_CUDA_CHECK(made);
            }
            TORCH_CUDA_CHECK(cudaGraphLaunch(graph, stream));
        }
        hs[3] += ms(Clock::now() - t_launch);
        const auto gpu_start = Clock::now();
        TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
        TORCH_CUDA_CHECK(cudaGetLastError());
        if (dstage_times && !eager)
            collect_window(timelines[static_cast<size_t>(parity)][static_cast<size_t>(columns)], columns);
        store->check();
        window_open = true;
        times.gpu += ms(Clock::now() - gpu_start);
        const auto t_sw = Clock::now();
        if (swaps_per_window > 0)
            swap_missed();
        hs[4] += ms(Clock::now() - t_sw);
        ++hs_windows;
        times.total += ms(Clock::now() - start);
    }

    uint64_t chunk_host_bytes() const {
        const uint64_t E = static_cast<uint64_t>(cfg.n_expert);
        return static_cast<uint64_t>(chunk_cap) *
                   (2 * static_cast<uint64_t>(cfg.n_embd) * sizeof(float) + sizeof(int32_t)) +
               E * sizeof(int32_t) + kTableRing * ((E + 1) * sizeof(int32_t) + 3 * E * sizeof(void*) + 16) + 16;
    }

    // A chunk's buffers use the windows' sequence state and scratch (the chunk's own layout has none).
    void share_state(Buffers& x) const {
        const Buffers& w = cols[0];
        x.k_cache = w.k_cache, x.v_cache = w.v_cache, x.conv_state = w.conv_state, x.ssm_state = w.ssm_state;
        x.ple_history = w.ple_history, x.state_bytes = w.state_bytes, x.ssm_other = w.ssm_other;
        x.idx_ring = w.idx_ring, x.idx_pool = w.idx_pool;
        if (!sparse_attention(cfg, context))  // a sparse chunk has its own (Buffers::lay_out_chunk)
            x.attn_scratch = w.attn_scratch;
    }

    // A prefill chunk (Options::prefill_columns): the tokens run through the model op by op over all their columns. The
    // kernels capped at kMaxColumns (matrix products, the expert gate and expert kernels, attention and the indexer)
    // run on groups of that many columns in order, so every column is bitwise what feeding the tokens in windows of
    // kMaxColumns gives. The chunk is kept whole: each DeltaNet layer commits its convolution history right after it,
    // the PLE ring at the end. Only the last column runs the output head; argmax(0) and logits(0) are its.
    void prefill(std::span<const int32_t> tokens) {
        const auto start = Clock::now();
        if (dstage_times && std::any_of(dstage_windows.begin(), dstage_windows.end(), [](int w) { return w > 0; }))
            report_windows();  // the windows of the previous request
        const int n = static_cast<int>(tokens.size());
        if (chunk_cap == 0)
            throw Error("Prefill needs a chunk size (Options::prefill_columns, torch run --prefill N).");
        if (n < 1 || n > chunk_cap)
            throw Error("A prefill chunk holds 1 to " + std::to_string(chunk_cap) + " tokens, not " +
                        std::to_string(n) + ".");
        for (const int32_t token : tokens)
            if (token < 0 || token >= cfg.n_vocab)
                throw Error("Token id " + std::to_string(token) + " is outside the vocabulary of " +
                            std::to_string(cfg.n_vocab) + ".");
        if (history.size() + tokens.size() > static_cast<size_t>(context))
            throw Error("The context of " + std::to_string(context) + " tokens is full; start a new sequence.");
        if (window_open)
            commit(window);
        snapshot_at = -1;  // until this chunk is in: a failed chunk leaves the sequence to reset()
        std::fill(layer_grouped.begin(), layer_grouped.end(), 0);
        const int E = cfg.n_embd;
        window_start = history.size();
        const Matrix& te = weights->token_embd;
        for (int t = 0; t < n; ++t) {
            const int32_t token = tokens[static_cast<size_t>(t)];
            history.push_back(token);
            dequant_row(te.type, te.data + static_cast<uint64_t>(token) * te.row_bytes(), E,
                        chunk_embd + static_cast<size_t>(t) * static_cast<size_t>(E));
            chunk_pos[t] = static_cast<int32_t>(window_start) + t;
        }
        // The PLE rows (each distinct row once) are read while the GPU runs the layers before the PLE layer;
        // ple_rows_in() waits for them there.
        chunk_ple_read = std::async(std::launch::async, [this, first = window_start, n] {
            const auto ple_start = Clock::now();
            ple->read_tokens(history, first, n, chunk_ple);
            return ms(Clock::now() - ple_start);
        });
        store->before_window();
        reclaim_chunk();  // the chunk's buffers are written below
        while (!lend_regions.empty() && lend_mark < context &&
               window_start + static_cast<size_t>(n) > static_cast<size_t>(lend_mark))
            unlend();  // the chunk writes K/V rows at or past the lend mark
        plan_reads(n);
        prefilling = true;
        try {
            enqueue_chunk(n);
        } catch (...) {
            prefilling = false, ops = &cols[0], b = cols[0];
            if (chunk_ple_read.valid())
                chunk_ple_read.wait();  // its thread writes chunk_ple
            throw;
        }
        prefilling = false, ops = &cols[0], b = cols[0];
        const auto gpu_start = Clock::now();
        TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
        TORCH_CUDA_CHECK(cudaGetLastError());
        store->check();
        times.gpu += ms(Clock::now() - gpu_start);
        if (stage_times)
            collect_stages();
        for (int il = 0; il < cfg.n_layer; ++il)  // residency did not change during the chunk
            for (int e = 0; e < cfg.n_expert && layer_grouped[static_cast<size_t>(il)]; ++e)
                (store->resident(il, e) ? prefill_hits : prefill_misses) +=
                    static_cast<uint64_t>(layer_counts[il * cfg.n_expert + e]);
        if (lend_chunk_on) {
            prompt_routes.resize(static_cast<size_t>(cfg.n_layer) * static_cast<size_t>(cfg.n_expert), 0);
            for (int il = 0; il < cfg.n_layer; ++il)
                for (int e = 0; e < cfg.n_expert && layer_grouped[static_cast<size_t>(il)]; ++e)
                    prompt_routes[static_cast<size_t>(il * cfg.n_expert + e)] +=
                        static_cast<uint64_t>(layer_counts[il * cfg.n_expert + e]);
        }
        parity ^= 1;  // every DeltaNet layer wrote its state after the chunk to state_out
        window = 1;   // logits(0) and argmax(0): the chunk's last token's
        last_logits = chunk_last.logits, first_draw = history.size();
        window_open = false;
        take_snapshot();
        times.total += ms(Clock::now() - start);
    }

    uint64_t snapshot_bytes() const {
        const uint64_t ring = sparse_attention(cfg, context) ? kernels::kIndexRing * kernels::kIndexDim * 2 : 0;
        uint64_t bytes = b.state_bytes;
        for (int il = 0; il < cfg.n_layer; ++il)
            bytes += cfg.recurrent(il) ? 0 : ring;
        return bytes;
    }

    // Copies (to==true) the sequence state to the snapshot, or back; every DeltaNet state goes through ssm_state's
    // place, so a restored state is the one before the next window at parity 0.
    void copy_snapshot(bool to) {
        const Buffers& w = cols[0];
        const auto copy = [&](void* snap, void* live, size_t n) {
            TORCH_CUDA_CHECK(cudaMemcpyAsync(to ? snap : live, to ? live : snap, n, cudaMemcpyDeviceToDevice, stream));
        };
        copy(snapshot, device, w.state_bytes);
        const size_t state = sizeof(float) * static_cast<size_t>(cfg.ssm_v_heads) *
                             static_cast<size_t>(cfg.ssm_head_dim) * static_cast<size_t>(cfg.ssm_head_dim);
        uint8_t* ring = snapshot + w.state_bytes;
        for (size_t l = 0; l < w.ssm_state.size(); ++l) {
            if (to && parity && w.ssm_state[l])  // the state before the next window is in ssm_other
                copy(snapshot + (reinterpret_cast<uint8_t*>(w.ssm_state[l]) - device), w.ssm_other[l], state);
            if (!w.idx_ring.empty() && w.idx_ring[l]) {
                copy(ring, w.idx_ring[l], kernels::kIndexRing * kernels::kIndexDim * sizeof(uint16_t));
                ring += kernels::kIndexRing * kernels::kIndexDim * sizeof(uint16_t);
            }
        }
    }

    void take_snapshot() {
        copy_snapshot(true);
        snapshot_at = static_cast<int>(history.size());
    }

    // Back to the end of the last prefill chunk (sessions, option B): K/V rows and pooled keys are indexed by position
    // and written before they are read, so only the state the snapshot holds goes back.
    void rewind() {
        if (snapshot_at < 0)
            throw Error("There is no prefill to go back to; start a new sequence.");
        window_open = false;  // the last window's columns are dropped, not committed
        copy_snapshot(false);
        parity = 0;
        history.resize(static_cast<size_t>(snapshot_at));
        window = 0;  // no results until the next window
    }

    // A wide chunk routes nearly every expert, so each layer's non-resident experts are read ahead, in passes of the
    // reader's slots, queued now in layer order (routed_experts waits for each pass).
    void plan_reads(int n) {
        chunk_passes.clear();
        if (!prefill_reads || n < kReadAheadColumns)
            return;
        prefill_reads->reset();  // the GPU is idle between chunks
        chunk_passes.resize(static_cast<size_t>(cfg.n_layer));
        for (int il = 0; il < cfg.n_layer; ++il) {
            std::vector<int32_t> missing;
            for (int e = 0; e < cfg.n_expert; ++e)
                if (!store->resident(il, e))
                    missing.push_back(e);
            for (size_t at = 0; at < missing.size(); at += static_cast<size_t>(prefill_reads->slots())) {
                std::vector<int32_t> pass(
                    missing.begin() + static_cast<std::ptrdiff_t>(at),
                    missing.begin() + static_cast<std::ptrdiff_t>(
                                          std::min(missing.size(), at + static_cast<size_t>(prefill_reads->slots()))));
                const int ticket = prefill_reads->request(il, pass);
                chunk_passes[static_cast<size_t>(il)].emplace_back(ticket, std::move(pass));
            }
        }
    }

    void enqueue_chunk(int n) {
        DeviceLayout last(chunk_device, n - 1, chunk_cap);
        chunk_last.lay_out(last, cfg, context, chunk_cap);
        share_state(chunk_last);
        ops = &chunk, b = chunk;
        stage(kTotal, [&] { enqueue_chunk_work(n); });
    }

    // The chunk's PLE rows, once their reads are done, into the chunk's device rows (before the PLE layer reads them).
    void ple_rows_in(int n) {
        if (!chunk_ple_read.valid())
            return;
        const double ple_ms = chunk_ple_read.get();  // rethrows a failed read
        times.ple += ple_ms;
        times.ple_max = std::max(times.ple_max, ple_ms);
        TORCH_CUDA_CHECK(cudaMemcpyAsync(chunk.ple_embd, chunk_ple,
                                         sizeof(float) * static_cast<size_t>(n) * static_cast<size_t>(cfg.n_embd),
                                         cudaMemcpyHostToDevice, stream));
    }

    void enqueue_chunk_work(int n) {
        const int E = cfg.n_embd;
        const size_t N = static_cast<size_t>(n);
        TORCH_CUDA_CHECK(cudaMemcpyAsync(chunk.pos, chunk_pos, sizeof(int32_t) * N, cudaMemcpyHostToDevice, stream));
        TORCH_CUDA_CHECK(cudaMemcpyAsync(chunk.embd, chunk_embd, sizeof(float) * N * static_cast<size_t>(E),
                                         cudaMemcpyHostToDevice, stream));
        layer_ = -1;
        kernels::hc_broadcast(chunk.embd, chunk.res, E, cfg.hc, stream, n);
        bool pending = false;  // as enqueue()
        for (int il = 0; il < cfg.n_layer; ++il) {
            const Layer& L = weights->layers[static_cast<size_t>(il)];
            layer_ = il;
            if (L.ple) {
                if (pending)
                    kernels::hc_combine(chunk.res, chunk.out, chunk.inject, E, cfg.hc, stream, n);
                pending = false;
                ple_rows_in(n);  // the layers before ran while its rows were read
                ple_block(L, n);
            }
            hc_mix(L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, &L.hc_attn_inject, pending, n);
            if (L.recurrent)
                deltanet(L, il, n);
            else
                attention(L, il, n);
            hc_mix(L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, &L.hc_ffn_inject, true, n);
            experts(L, il, n);
            pending = true;
        }
        ple_rows_in(n);  // (joined above when a layer is the PLE layer)
        layer_ = -1;
        // The draft reads every residual. Only the last column feeds the output head.
        if (fast_prefill) {
            hc_mix(weights->head_hc_norm, weights->head_hc_down, weights->head_hc_up, nullptr, pending, n);
            ops = &chunk_last, b = chunk_last;
        } else {
            if (pending && n > 1)
                kernels::hc_combine(chunk.res, chunk.out, chunk.inject, E, cfg.hc, stream, n - 1);
            ops = &chunk_last, b = chunk_last;
            hc_mix(weights->head_hc_norm, weights->head_hc_down, weights->head_hc_up, nullptr, pending, 1);
        }
        matvec(weights->output, chunk_last.mixed, chunk_last.logits, true, 1);
        kernels::argmax(chunk_last.logits, cfg.n_vocab, chunk_last.argmax_scratch, chunk_last.argmax, stream, 1);
        TORCH_CUDA_CHECK(cudaMemcpyAsync(hb.argmax, chunk_last.argmax, sizeof(kernels::ArgmaxResult),
                                         cudaMemcpyDeviceToHost, stream));
        if (want_logits)
            TORCH_CUDA_CHECK(cudaMemcpyAsync(hb.logits, chunk_last.logits,
                                             sizeof(float) * static_cast<size_t>(cfg.n_vocab), cudaMemcpyDeviceToHost,
                                             stream));
        kernels::ple_commit(chunk.ple_history, chunk.ple_x, cfg.hc * E, chunk.pos, n, stream);
    }

    // Copies the 4-stream residual of columns first .. first + n - 1 of the last chunk to the windows' columns 0 .. n -
    // 1 (Model::residual), for the draft layer's passes over the chunk.
    void stage_residuals(int first, int n) {
        if (chunk_cap == 0 || first < 0 || n < 1 || n > kMaxColumns || first + n > chunk_cap)
            throw Error("Stage 1 to " + std::to_string(kMaxColumns) + " columns of the last prefill chunk.");
        const size_t HCD = static_cast<size_t>(cfg.hc) * static_cast<size_t>(cfg.n_embd);
        TORCH_CUDA_CHECK(cudaMemcpyAsync(cols[0].res, chunk.res + static_cast<size_t>(first) * HCD,
                                         sizeof(float) * HCD * static_cast<size_t>(n), cudaMemcpyDeviceToDevice,
                                         stream));
    }

    // Keeps the last window's first n tokens.
    void accept(int n) {
        if (!window_open || n < 1 || n > window)
            throw Error("accept() keeps 1 to " + std::to_string(window) + " tokens of the last window, once, not " +
                        std::to_string(n) + ".");
        commit(n);
    }

    void decode(int32_t token) {
        verify(std::span(&token, 1));
        commit(1);
    }

    // After a window: the experts it read from the SSD, still in the staging buffers, are swapped in (swap_staged).
    void swap_missed() { swap_staged(store->staged(), false); }

    // Up to the window's swap budget (swaps_per_window, less the swaps it has made) of `staged` take the slots of their
    // layers' least recently routed experts. After the window on the main thread, or, with Options::swap_when_full,
    // on the I/O thread when the staging fills (their layers are done for this window).
    void swap_staged(std::span<const runtime::ExpertStore::Staged> staged, bool full) {
        const auto start = Clock::now();
        const std::lock_guard lock(swap_mutex);
        const int cap = swaps_per_window - swapped_in_window;
        if (staged.empty() || cap <= 0)
            return;
        std::vector<std::pair<int, int>> missed;
        for (const runtime::ExpertStore::Staged& s : staged)
            missed.push_back({s.layer, s.expert});
        // the stamps of the layers that missed, in host memory (the CPU reads the mapped array slowly)
        const std::span<const uint32_t> stamps = store->stamps();
        const size_t row = static_cast<size_t>(cfg.n_expert);
        for (const auto& [layer, expert] : missed)
            std::copy_n(stamps.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(layer) * row), row,
                        stamps_cache.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(layer) * row));
        const uint32_t now = window_stamp.load(std::memory_order_acquire);  // the gate's stamp for this window
        for (const Swap& w : online_swaps(residency, missed, now, cap)) {
            const auto it = std::find_if(staged.begin(), staged.end(),
                                         [&](const auto& s) { return s.layer == w.layer && s.expert == w.in; });
            store->swap_in(*it, w.out, full);
            ++times.swaps;
            times.swaps_when_full += full ? 1 : 0;
            ++swapped_in_window;
        }
        if (!full)
            store->end_swaps();
        times.swap_ms += ms(Clock::now() - start);
    }
};

Model::Model(const Options& options) : impl_(std::make_unique<Impl>(options)) {}
Model::~Model() = default;

const Config& Model::config() const noexcept {
    return impl_->cfg;
}

const Budget& Model::budget() const noexcept {
    return impl_->budget;
}

DecodeTimes Model::decode_times() const noexcept {
    DecodeTimes t = impl_->times;
    t.expert_reads = impl_->store->read_ms();
    t.ple_hedged = impl_->ple->hedged_reads();
    return t;
}

ExpertStats Model::expert_stats() const {
    const auto [hits, misses] = impl_->store->hits_and_misses();
    return {impl_->budget.resident + impl_->budget.added + impl_->tenants_now,
            impl_->budget.experts,
            hits + impl_->prefill_hits,
            misses + impl_->prefill_misses,
            impl_->store->hedged_reads(),
            impl_->tenants_now,
            impl_->prefill_reads_n};
}

RouteCounts Model::route_counts() const {
    RouteCounts c;
    c.n_layer = impl_->cfg.n_layer;
    c.n_expert = impl_->cfg.n_expert;
    c.counts = impl_->store->read_route_counts();
    return c;
}

int Model::position() const noexcept {
    return static_cast<int>(impl_->history.size());
}

bool Model::decode(int32_t token) noexcept {
    if ((impl_->cancelled = impl_->cancel_requested.exchange(false))) {
        impl_->error = "The request was cancelled between windows.";
        return false;
    }
    try {
        impl_->error.clear();
        impl_->decode(token);
        return true;
    } catch (const std::exception& e) {
        impl_->error = e.what();
        return false;
    }
}

bool Model::verify(std::span<const int32_t> tokens) noexcept {
    if ((impl_->cancelled = impl_->cancel_requested.exchange(false))) {
        impl_->error = "The request was cancelled between windows.";
        return false;
    }
    try {
        impl_->error.clear();
        impl_->verify(tokens);
        return true;
    } catch (const std::exception& e) {
        impl_->error = e.what();
        return false;
    }
}

bool Model::prefill(std::span<const int32_t> tokens) noexcept {
    if ((impl_->cancelled = impl_->cancel_requested.exchange(false))) {
        impl_->error = "The request was cancelled between windows.";
        return false;
    }
    try {
        impl_->error.clear();
        impl_->prefill(tokens);
        return true;
    } catch (const std::exception& e) {
        impl_->error = e.what();
        return false;
    }
}

void Model::stage_residuals(int first, int n) {
    impl_->stage_residuals(first, n);
}

bool Model::accept(int n) noexcept {
    try {
        impl_->error.clear();
        impl_->accept(n);
        return true;
    } catch (const std::exception& e) {
        impl_->error = e.what();
        return false;
    }
}

std::span<const float> Model::logits(int column) const noexcept {
    if (!impl_->want_logits || column < 0 || column >= impl_->window)
        return {};
    const size_t n = static_cast<size_t>(impl_->cfg.n_vocab);
    return {impl_->hb.logits + static_cast<size_t>(column) * n, n};
}

int32_t Model::argmax(int column) const noexcept {
    return column >= 0 && column < impl_->window ? impl_->hb.argmax[column].index : -1;
}

std::vector<int32_t> Model::sample(const kernels::SamplingParams& params, int first, int n) {
    return impl_->sample(params, first, n);
}

std::span<int32_t> Model::history_counts() noexcept {
    return {impl_->sampling_host, static_cast<size_t>(impl_->cfg.n_vocab)};
}

const std::string& Model::error() const noexcept {
    return impl_->error;
}

void Model::set_swaps_per_window(int n) noexcept {
    impl_->swaps_per_window = std::max(n, 0);
}

int Model::lend_chunk(bool early) {
    return impl_->lend_chunk(early);
}

const Weights& Model::weights() const noexcept {
    return *impl_->weights;
}

CUstream_st* Model::stream() const noexcept {
    return impl_->stream;
}

const float* Model::residual(int column) const noexcept {
    return impl_->cols[static_cast<size_t>(column)].res;
}

void Model::reset() {
    impl_->reset();
}

int Model::prefill_end() const noexcept {
    return impl_->snapshot_at;
}

void Model::rewind() {
    impl_->rewind();
}

int Model::checkpoint_limit() const noexcept {
    return impl_->lend_step > 0 ? impl_->lend_step : impl_->context;
}

std::span<uint8_t> Model::checkpoint_state() const noexcept {
    return {impl_->snapshot, impl_->snapshot_bytes()};
}

std::span<uint8_t> Model::checkpoint_residual() const noexcept {
    return {reinterpret_cast<uint8_t*>(impl_->cols[0].res), sizeof(float) * impl_->cfg.hc * impl_->cfg.n_embd};
}

void Model::checkpoint_rows(int first, int end, std::vector<std::span<uint8_t>>& out) const {
    if (first < 0 || end < first || end > checkpoint_limit())
        throw Error("The checkpoint exceeds the active cache rows.");
    const auto& b = impl_->cols[0];
    const size_t row = static_cast<size_t>(impl_->cfg.n_head_kv) * impl_->cfg.head_dim;
    const auto add = [&](uint16_t* data, int a, int z, size_t width) {
        if (data && z > a)
            out.emplace_back(reinterpret_cast<uint8_t*>(data + a * width), (z - a) * width * sizeof(uint16_t));
    };
    for (size_t l = 0; l < b.k_cache.size(); ++l) {
        add(b.k_cache[l], first, end, row);
        add(b.v_cache[l], first, end, row);
        if (!b.idx_pool.empty())
            add(b.idx_pool[l], first / kernels::kIndexPool, end / kernels::kIndexPool, kernels::kIndexDim);
    }
}

void Model::restore_checkpoint(std::span<const int32_t> ids) {
    if (ids.empty() || ids.size() > static_cast<size_t>(checkpoint_limit()))
        throw Error("The checkpoint position exceeds the active cache rows.");
    impl_->history.assign(ids.begin(), ids.end());
    impl_->snapshot_at = static_cast<int>(ids.size());
    impl_->rewind();
}

void Model::cancel() noexcept {
    impl_->cancel_requested = true;
}

bool Model::cancelled() const noexcept {
    return impl_->cancelled;
}

}  // namespace torch::model
