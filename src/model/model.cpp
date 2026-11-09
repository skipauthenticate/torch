#include "model/model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/error.h"
#include "core/placement.h"
#include "core/swaps.h"
#include "formats/dequant.h"
#include "kernels/attention/attention.h"
#include "kernels/deltanet/deltanet.h"
#include "kernels/gemv/gemv.h"
#include "kernels/hc/hc.h"
#include "kernels/moe/moe.h"
#include "kernels/norm/norm.h"
#include "kernels/ple/ple.h"
#include "kernels/sampling/argmax.h"
#include "model/weights.h"
#include "runtime/direct_io.h"
#include "runtime/expert_store.h"
#include "runtime/ple_rows.h"

namespace torch::model {
namespace {

using kernels::BlockQ8;

constexpr int kReadThreads = 16;  // 128 KiB reads reach ~5.4 GB/s at queue depth 16 (bench/ssd_reads.cpp)

using Clock = std::chrono::steady_clock;
double ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}
constexpr uint64_t kCudaContext = kGiB / 2;  // the CUDA context and allocator slack [est]

// Lays out device buffers in one allocation. The dry run (no base) only adds up the bytes, so the memory budget and
// the allocation come from the same list.
class DeviceLayout {
public:
    explicit DeviceLayout(uint8_t* base, int column = 0) : base_(base), column_(column) {}

    // One allocation of `count` elements.
    template <class T>
    T* take(size_t count) {
        T* at = base_ ? reinterpret_cast<T*>(base_ + used_) : nullptr;
        used_ += (count * sizeof(T) + 255) / 256 * 256;
        return at;
    }
    // kMaxColumns copies of `count` elements, column after column; returns the layout's column.
    template <class T>
    T* cols(size_t count) {
        T* at = take<T>(count * kMaxColumns);
        return at ? at + count * static_cast<size_t>(column_) : nullptr;
    }
    uint64_t used() const noexcept { return used_; }

private:
    uint8_t* base_;
    int column_;
    uint64_t used_ = 0;
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
    float *router, *weights, *h, *y, *shared_g, *shared_u, *shared_h, *shared_y, *shared_gate;
    int32_t* ids;
    float *key, *key_n, *value, *query, *ple_gate, *gated, *ple_x;
    float* logits;
    kernels::ArgmaxResult *argmax, *argmax_scratch;
    // Q8 copies of the vectors quantized products read, one buffer per vector so that no column's copy is overwritten
    // by another vector's before it is read: the block input, the attention or DeltaNet output (6,144 wide), the shared
    // expert's hidden vector, the routed experts' hidden vectors, and any other input (up to hc x n_embd wide).
    BlockQ8 *q8, *q8_x, *q8_sh, *q8_h, *q8_wide;

    void lay_out(DeviceLayout& d, const Config& c, int context) {  // d's column picks the activations' copy
        const size_t E = static_cast<size_t>(c.n_embd), HCD = static_cast<size_t>(c.hc) * E;
        const size_t KV = static_cast<size_t>(c.n_head_kv * c.head_dim);
        const size_t CC = static_cast<size_t>(c.conv_channels());
        const size_t HV = static_cast<size_t>(c.ssm_v_heads), DS = static_cast<size_t>(c.ssm_head_dim);
        const size_t V = HV * DS, K = static_cast<size_t>(c.n_expert_used), FF = static_cast<size_t>(c.n_ff_expert);
        k_cache.assign(static_cast<size_t>(c.n_layer), nullptr);
        v_cache = k_cache;
        conv_state.assign(static_cast<size_t>(c.n_layer), nullptr);
        ssm_state = conv_state;
        for (int il = 0; il < c.n_layer; ++il) {
            const size_t l = static_cast<size_t>(il);
            if (c.recurrent(il)) {
                conv_state[l] = d.take<float>(3 * CC);
                ssm_state[l] = d.take<float>(HV * DS * DS);
            } else {
                k_cache[l] = d.take<uint16_t>(static_cast<size_t>(context) * KV);
                v_cache[l] = d.take<uint16_t>(static_cast<size_t>(context) * KV);
            }
        }
        ple_history = d.take<float>(9 * HCD);
        state_bytes = d.used();
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

        pos = d.cols<int32_t>(1);
        res = d.cols<float>(HCD), xn = d.cols<float>(HCD), lo = d.cols<float>(static_cast<size_t>(c.hc_rank));
        hc_gate = d.cols<float>(HCD), mixed = d.cols<float>(E), inject = d.cols<float>(static_cast<size_t>(c.hc));
        out = d.cols<float>(E), embd = d.cols<float>(E), ple_embd = d.cols<float>(E);
        qkv = d.cols<float>(CC), z = d.cols<float>(V), beta = d.cols<float>(HV), alpha = d.cols<float>(HV);
        conv_out = d.cols<float>(CC), o = d.cols<float>(V);
        q = d.cols<float>(2 * static_cast<size_t>(c.n_head * c.head_dim)), k = d.cols<float>(KV);
        v = d.cols<float>(KV), attn = d.cols<float>(static_cast<size_t>(c.n_head * c.head_dim));
        // one scratch for the whole window: attention_decode indexes it by column
        attn_scratch =
            d.take<float>(static_cast<size_t>(kernels::attention_scratch_floats(c.n_head, context, kMaxColumns)));
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
                              sizeof(kernels::ArgmaxResult) + sizeof(int32_t));
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
    gguf::File shard1, shard2;
    runtime::DirectFile file1, file2;
    Config cfg;
    int context;
    runtime::ReadPool pool{kReadThreads};
    Budget budget;
    RouteCounts ranking;  // the order experts are made resident in, without --resident
    std::unique_ptr<Weights> weights;
    std::unique_ptr<runtime::ExpertStore> store;
    std::unique_ptr<runtime::PleRows> ple;
    cudaStream_t stream = nullptr;
    uint8_t* device = nullptr;
    uint8_t* host = nullptr;
    std::array<Buffers, kMaxColumns> cols;  // each window column's view of the buffers
    Buffers b;                              // the column being enqueued (each_column)
    int col_ = 0;
    HostBuffers hb{};
    size_t window_start = 0;  // the last window's first position, and its width
    int window = 0;
    std::vector<int32_t> history;
    std::string error;
    DecodeTimes times;
    // TORCH_DUMP=1: after every stage, one line on stderr with the stage's sum, mean |x| and count of non-finite
    // values, for bisecting numerics. TORCH_DUMP_DIR=dir also writes each stage's values to
    // dir/<position>-<layer>-<stage>.f32. TORCH_DUMP_STAGES=a,b limits both to those stages. Slow: every dumped stage
    // waits for the GPU.
    const char* dump_dir = std::getenv("TORCH_DUMP_DIR");
    const std::string dump_stages =
        std::getenv("TORCH_DUMP_STAGES") ? std::string(",") + std::getenv("TORCH_DUMP_STAGES") + "," : "";
    const bool dumping = std::getenv("TORCH_DUMP") != nullptr || dump_dir != nullptr;
    // A token runs as one replayed CUDA graph; TORCH_EAGER=1 (and the stage dumps) launch kernel by kernel instead.
    const bool eager = dumping || std::getenv("TORCH_EAGER") != nullptr;
    // One graph per DeltaNet state parity and window width, recorded at its first use.
    std::array<std::array<cudaGraphExec_t, kMaxColumns + 1>, 2> graphs{};
    int parity = 0;            // which DeltaNet state buffer holds the state before the next window
    bool window_open = false;  // the last window is not committed yet
    bool want_logits = true;
    int layer_ = -1;              // the layer being computed, for dump lines
    uint64_t reserved_later = 0;  // Options::reserve, allocated by the caller after the model
    // Swaps (Options::swaps_per_window): which experts are resident; their recency is the store's stamps.
    int swaps_per_window = 0;
    Residency residency;
    std::vector<uint32_t> stamps_cache;  // the store's stamps of the layers a swap decision reads

    explicit Impl(const Options& o)
        : shard1(gguf::File::open(o.shard1)),
          shard2(gguf::File::open(o.shard2)),
          file1(o.shard1),
          file2(o.shard2),
          cfg(Config::read(shard1)),
          context(o.context) {
        want_logits = o.logits;
        if (context < 1 || context > cfg.indexer_top_k)
            throw Error("The context must be between 1 and " + std::to_string(cfg.indexer_top_k) +
                        " tokens in this version, where attention is computed densely.");

        // MemAvailable is read before the CUDA context exists; the budget reserves room for it.
        DeviceLayout dry(nullptr);
        b.lay_out(dry, cfg, context);
        std::vector<Reservation> reservations = {
            {"weights other than routed experts", Weights::base_bytes(cfg, shard1)},
            {"KV cache, DeltaNet state and its window copy, activations (" + std::to_string(context) + " tokens)",
             dry.used() + HostBuffers::bytes(cfg)},
            {"expert staging (" + std::to_string(runtime::kGateSlots) + " slots), map and tables",
             runtime::ExpertStore::fixed_bytes(cfg, shard1)},
            {"CUDA context and allocator slack", kCudaContext},
        };
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
        for (int t = 0; t < kMaxColumns; ++t) {
            DeviceLayout layout(device, t);
            cols[static_cast<size_t>(t)].lay_out(layout, cfg, context);
        }
        b = cols[0];
        void* h = nullptr;
        TORCH_CUDA_CHECK(cudaHostAlloc(&h, HostBuffers::bytes(cfg), cudaHostAllocDefault));
        host = static_cast<uint8_t*>(h);
        hb.lay_out(host, cfg);

        weights = std::make_unique<Weights>(cfg, shard1, file1, pool);
        store = std::make_unique<runtime::ExpertStore>(cfg, weights->layers, file1, placement.resident);
        ple = std::make_unique<runtime::PleRows>(cfg, shard2, file2, kMaxColumns);
        if (!o.resident)
            top_up(placement);
        swaps_per_window = o.swaps_per_window;
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
    // down to the floor plus kTopUpMargin (budget.h). Output does not change, only the hit rate.
    void top_up(const Placement& placement) {
        budget.available_after_load = mem_available();
        const uint64_t keep = budget.floor + kTopUpMargin + reserved_later;
        if (budget.available_after_load <= keep)
            return;
        const Placement more = extend_ranked(ranking, runtime::ExpertStore::layer_expert_bytes(cfg, shard1), placement,
                                             budget.available_after_load - keep);
        store->add(more.resident);
        budget.added = more.count();
        budget.added_bytes = more.bytes;
    }

    // Which experts are resident: exactly --resident per layer, the most-routed in Options::placement, or as many of
    // every layer's first experts as fit. Fills in the budget's placement lines and the ranking top_up() continues.
    Placement place(const Options& o) {
        const std::vector<uint64_t> bytes = runtime::ExpertStore::layer_expert_bytes(cfg, shard1);
        Placement p;
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
            p = place_ranked(counts, bytes, budget.room());
            budget.placement = "the most-routed experts in " + o.placement.string();
            ranking = counts;
        } else {
            const int k = uniform_capacity(budget.room(), cfg.n_expert, bytes);
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
        if (stream)
            cudaStreamDestroy(stream);
    }

    void dump(const char* stage, int layer, const float* d, int n) {
        if (!dumping || (!dump_stages.empty() && dump_stages.find(std::string(",") + stage + ",") == std::string::npos))
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
        parity = 0;  // the state before the first window is in ssm_state, which the memset below clears
        window_open = false;
        TORCH_CUDA_CHECK(cudaMemsetAsync(device, 0, b.state_bytes, stream));
        TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
        store->before_window();  // swaps still queued land first
        store->clear_stamps();
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
        for (int i = 0; i < nq; ++i) {
            const Matrix& m = *q8[i].m;
            if (nq > 1 && kernels::gemv_q8_groups(m.type))
                parts[np++] = {m.type, m.data, m.rows, q8[i].y};
            else
                kernels::gemv_q8(m.type, m.data, m.rows, m.cols, xq, q8[i].y, stream, columns);
        }
        if (np == 1)
            kernels::gemv_q8(parts[0].type, parts[0].w, parts[0].rows, q8[0].m->cols, xq, parts[0].y, stream, columns);
        else if (np > 1)
            kernels::gemv_q8_group(parts, np, q8[0].m->cols, xq, false, stream, columns);
        for (int i = 0; i < nb; i += 2) {
            const Matrix& m = *bf16[i].m;
            if (i + 1 < nb)
                kernels::gemv_bf16_pair(m.data, m.rows, bf16[i + 1].m->data, bf16[i + 1].m->rows, m.cols, x, bf16[i].y,
                                        bf16[i + 1].y, stream, 0.0f, columns);
            else
                kernels::gemv_bf16(m.data, m.rows, m.cols, x, bf16[i].y, stream, columns);
        }
    }

    // A hyper-connection mixer over the window's columns: the hc streams of each column's residual, normalized, mixed
    // into one block input (b.mixed, and its Q8), and, when `inject` is given, the block output's per-stream weights
    // (b.inject). With `combine`, the previous block's output (b.out, weighted by b.inject) is first added to the
    // residual, in the same launch as the norm.
    void hc_mix(const float* norm, const Matrix& down, const Matrix& up, const Matrix* inject, bool combine,
                int columns) {
        const int E = cfg.n_embd;
        const Buffers& c = cols[0];
        if (combine)
            kernels::hc_combine_norm(c.res, c.out, c.inject, norm, c.xn, E, cfg.hc, cfg.eps, stream, columns);
        else
            kernels::rms_norm(c.res, E, norm, E, nullptr, 0, c.xn, E, E, cfg.hc, cfg.eps, stream, columns);
        // A BF16 down projection applies the low-rank SiLU as it stores; BF16 inject rows read the same input, so
        // they share its launch.
        const float scale = 1.0f / static_cast<float>(cfg.hc);
        const bool paired =
            inject && down.type == Type::BF16 && inject->type == Type::BF16 && inject->cols == down.cols;
        if (down.type == Type::BF16) {
            kernels::gemv_bf16_pair(down.data, down.rows, paired ? inject->data : nullptr, paired ? inject->rows : 0,
                                    down.cols, c.xn, c.lo, c.inject, stream, scale, columns);
        } else {
            matvec(down, c.xn, c.lo, false, columns);
            kernels::silu_scale(c.lo, cfg.hc_rank, scale, stream, columns);
        }
        matvec(up, c.lo, c.hc_gate, false, columns);
        if (inject && !paired)
            matvec(*inject, c.xn, c.inject, false, columns);
        kernels::hc_collapse(c.xn, c.hc_gate, c.mixed, c.q8, E, cfg.hc, stream, columns);
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
        const Buffers& c = cols[0];
        matvec(L.ple_key, c.ple_embd, c.key, false, columns);
        matvec(L.ple_value, c.ple_embd, c.value, false, columns);
        kernels::rms_norm(c.key, E, L.ple_norm_key, E, nullptr, 0, c.key_n, E, E, cfg.hc, cfg.eps, stream, columns);
        kernels::rms_norm(c.res, E, L.ple_norm_query, E, nullptr, 0, c.query, E, E, cfg.hc, cfg.eps, stream, columns);
        each_column(columns, [&] {
            kernels::ple_gate(b.key_n, b.query, b.ple_gate, E, cfg.hc, stream);
            kernels::ple_gated_value(b.value, b.ple_gate, b.gated, E, cfg.hc, stream);
        });
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
        const Buffers& c = cols[0];
        float *qkv = c.win_qkv[l], *conv = c.win_conv[l], *alpha = c.win_alpha[l], *beta = c.win_beta[l];
        products(c.mixed, {{&L.qkv, qkv}, {&L.z, c.z}, {&L.beta, beta}, {&L.alpha, alpha}}, true, columns);
        // q and k are L2-normalized in the convolution's launch; the gates are computed in the recurrence's
        kernels::conv_silu_l2_window(c.conv_state[l], qkv, L.conv, conv, CC, 2 * QK, cfg.eps, columns, stream);
        kernels::gdn_window(state_in(l), state_out(l), conv, conv + QK, conv + 2 * QK, CC, alpha, beta, L.dt_bias, L.a,
                            nullptr, c.o, cfg.ssm_k_heads, HV, 1.0f / std::sqrt(static_cast<float>(DS)), columns,
                            stream);
        kernels::rms_norm_q8(c.o, DS, L.ssm_norm, 0, c.z, DS, c.o, DS, c.q8_x, DS, HV, cfg.eps, stream, columns);
        matvec(L.ssm_out, c.o, c.out, true, columns);
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
        const Buffers& c = cols[0];
        products(c.mixed, {{&L.wq, c.q}, {&L.wk, c.k}, {&L.wv, c.v}}, true, columns);  // q: per head [q | gate]
        kernels::qk_norm_rope_store(c.q, 2 * D, L.q_norm, c.k, L.k_norm, c.v, c.k_cache[l], c.v_cache[l], H, HKV,
                                    cfg.n_rot, c.pos, cfg.rope_base, cfg.eps, columns, stream);
        kernels::attention_decode(c.q, c.q + D, 2 * D, c.k_cache[l], c.v_cache[l], c.attn, c.attn_scratch, c.pos, H,
                                  HKV, 1.0f / std::sqrt(static_cast<float>(D)), context, columns, stream);
        matvec(L.wo, c.attn, c.out, false, columns);
        each_column(columns, [&] {
            dump("attn", layer_, b.attn, H * D);
            dump("attn_out", layer_, b.out, cfg.n_embd);
        });
    }

    // The MoE block over the window's columns: routers, then each column's experts, then the combine.
    void experts(const Layer& L, int il, int columns) {
        const int E = cfg.n_embd, K = cfg.n_expert_used;
        const Buffers& c = cols[0];
        products(c.mixed, {{&L.router, c.router}, {&L.shared_gate_inp, c.shared_gate}}, true, columns);
        each_column(columns, [&] { dump("router", layer_, b.router, cfg.n_expert); });
        kernels::router_top_k(c.router, cfg.n_expert, K, c.ids, c.weights, stream, columns);
        // One gate launch places the whole window's experts: each missing one is read once, all of a layer's misses
        // in one request.
        const runtime::GateTables gt = store->gate_tables(il);
        runtime::expert_gate(c.ids, K, columns, il, store->map(), cfg.n_expert, gt, store->mailbox(), store->seq(),
                             store->counters(), store->route_counts(), store->zeros(), c.pos, store->last_use(),
                             stream);
        // Each unique expert's gate, up and down rows are read once for all the columns that chose it.
        const int FF = cfg.n_ff_expert;
        kernels::gemv_q8_experts_window(L.gate_exps.type, gt.table, gt.table + runtime::kGateSlots, gt.columns, gt.rank,
                                        gt.count, std::min(columns * K, runtime::kGateSlots), K, FF, E, c.q8, c.h,
                                        columns, stream);  // c.q8: b.mixed's Q8, from hc_collapse
        // h is quantized by its own launch: writing it as Q8 from gate and up measured slower (+5-9 us a routed
        // launch, +0.15-0.3 ms a token against 0.14 ms of quantize_q8; branch m3-gemv-q8-epilogues).
        kernels::quantize_q8(c.h, int64_t{columns} * K * FF, c.q8_h, stream);
        // Q2_0's down measured faster per (column, rank) route (105 against 112 us at T = 3); both are bitwise equal.
        if (columns == 1 || L.down_exps.type == Type::Q2_0)
            kernels::gemv_q8_experts(L.down_exps.type, gt.down_routes, nullptr, columns * K, E, FF, c.q8_h, FF / 32,
                                     c.y, stream);
        else
            kernels::gemv_q8_experts_down_window(L.down_exps.type, gt.table + 2 * runtime::kGateSlots, gt.columns,
                                                 gt.rank, gt.count, std::min(columns * K, runtime::kGateSlots), K, E,
                                                 FF, c.q8_h, c.y, columns, stream);
        shared_expert(L, columns);
        kernels::moe_combine(c.y, c.weights, K, c.shared_y, c.shared_gate, c.out, E, stream, columns);
        each_column(columns, [&] {
            dump("experts_h", layer_, b.h, K * cfg.n_ff_expert);
            dump("experts_y", layer_, b.y, K * E);
            dump("shared_y", layer_, b.shared_y, E);
            dump("ffn_out", layer_, b.out, E);
        });
    }

    // The shared expert of the window's columns: gate and up with their SwiGLU in one launch, then down.
    void shared_expert(const Layer& L, int columns) {
        const int E = cfg.n_embd;
        const Buffers& c = cols[0];
        const Matrix &sg = L.shared_gate, &su = L.shared_up;
        if (kernels::gemv_q8_groups(sg.type) && kernels::gemv_q8_groups(su.type) && sg.rows == su.rows) {
            const kernels::Q8Part parts[2] = {{sg.type, sg.data, sg.rows, c.shared_h},
                                              {su.type, su.data, su.rows, nullptr}};
            kernels::gemv_q8_group(parts, 2, E, c.q8, true, stream, columns);  // b.mixed's Q8, from hc_collapse
        } else {
            kernels::gemv_q8(sg.type, sg.data, sg.rows, E, c.q8, c.shared_g, stream, columns);
            kernels::gemv_q8(su.type, su.data, su.rows, E, c.q8, c.shared_u, stream, columns);
            each_column(columns, [&] { kernels::swiglu(b.shared_g, b.shared_u, b.shared_h, cfg.n_ff_shared, stream); });
        }
        matvec(L.shared_down, c.shared_h, c.shared_y, false, columns);
    }

    // Runs `f` once per window column with `b` pointing at that column's buffers, columns in order. Each op of the
    // forward runs over all columns before the next op starts (op-major), so a multi-column kernel can replace the
    // loop with one launch; ops on sequence state (attention, DeltaNet, the PLE ring) see the columns in order.
    template <class F>
    void each_column(int columns, F&& f) {
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
        TORCH_CUDA_CHECK(cudaMemcpyAsync(c0.pos, hb.pos, sizeof(int32_t) * n, cudaMemcpyHostToDevice, stream));
        TORCH_CUDA_CHECK(cudaMemcpyAsync(c0.embd, hb.embd, sizeof(float) * n * static_cast<size_t>(E),
                                         cudaMemcpyHostToDevice, stream));
        TORCH_CUDA_CHECK(cudaMemcpyAsync(c0.ple_embd, hb.ple, sizeof(float) * n * static_cast<size_t>(E),
                                         cudaMemcpyHostToDevice, stream));

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
            }
            const bool combine = pending;
            hc_mix(L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, &L.hc_attn_inject, combine, columns);
            if (L.recurrent)
                deltanet(L, il, columns);
            else
                attention(L, il, columns);
            hc_mix(L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, &L.hc_ffn_inject, true, columns);
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
        matvec(weights->output, c0.mixed, c0.logits, true, columns);
        each_column(columns, [&] { dump("logits", layer_, b.logits, cfg.n_vocab); });
        kernels::argmax(c0.logits, cfg.n_vocab, c0.argmax_scratch, c0.argmax, stream, columns);
        TORCH_CUDA_CHECK(
            cudaMemcpyAsync(hb.argmax, c0.argmax, sizeof(kernels::ArgmaxResult) * n, cudaMemcpyDeviceToHost, stream));
        if (want_logits)
            TORCH_CUDA_CHECK(cudaMemcpyAsync(hb.logits, c0.logits, sizeof(float) * n * static_cast<size_t>(cfg.n_vocab),
                                             cudaMemcpyDeviceToHost, stream));
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
        window = columns;

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
        hs[2] += ms(Clock::now() - t_bw);
        const auto t_launch = Clock::now();
        cudaGraphExec_t& graph = graphs[static_cast<size_t>(parity)][static_cast<size_t>(columns)];
        if (eager) {
            enqueue(columns);
        } else {
            if (graph == nullptr) {  // recorded at the first window of this width and parity; its pointers stay fixed
                cudaGraph_t g = nullptr;
                TORCH_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                try {
                    enqueue(columns);
                } catch (...) {
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

    // After a window: up to swaps_per_window of the experts it read from the SSD, still in the staging buffers, take
    // the slots of their layers' least recently routed experts. The copies overlap what runs before the next window.
    void swap_missed() {
        const auto start = Clock::now();
        const std::vector<runtime::ExpertStore::Staged> staged = store->staged();
        if (staged.empty())
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
        const uint32_t now = static_cast<uint32_t>(window_start) + 1;  // the gate's stamp for this window
        for (const Swap& w : online_swaps(residency, missed, now, swaps_per_window)) {
            const auto it = std::find_if(staged.begin(), staged.end(),
                                         [&](const auto& s) { return s.layer == w.layer && s.expert == w.in; });
            store->swap_in(*it, w.out);
            ++times.swaps;
        }
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
    return {impl_->budget.resident + impl_->budget.added, impl_->budget.experts, hits, misses,
            impl_->store->hedged_reads()};
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
    try {
        impl_->error.clear();
        impl_->verify(tokens);
        return true;
    } catch (const std::exception& e) {
        impl_->error = e.what();
        return false;
    }
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

const std::string& Model::error() const noexcept {
    return impl_->error;
}

void Model::set_swaps_per_window(int n) noexcept {
    impl_->swaps_per_window = std::max(n, 0);
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

}  // namespace torch::model
