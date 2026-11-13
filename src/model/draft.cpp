#include "model/draft.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>

#include "core/cuda_check.h"
#include "core/error.h"
#include "formats/dequant.h"
#include "gguf/gguf.h"
#include "kernels/attention/attention.h"
#include "kernels/gemv/gemv.h"
#include "kernels/hc/hc.h"
#include "kernels/moe/moe.h"
#include "kernels/norm/norm.h"
#include "kernels/sampling/argmax.h"
#include "model/config.h"
#include "model/model.h"
#include "model/weights.h"
#include "runtime/direct_io.h"

namespace torch::model {
namespace {

using kernels::BlockQ8;

constexpr int kMaxDepth = 64;
constexpr int kReadThreads = 16;

// Lays out device buffers in one allocation; the dry run (no base) only adds up the bytes.
class Layout {
public:
    explicit Layout(uint8_t* base) : base_(base) {}
    template <class T>
    T* take(size_t count) {
        T* at = base_ ? reinterpret_cast<T*>(base_ + used_) : nullptr;
        used_ += (count * sizeof(T) + 255) / 256 * 256;
        return at;
    }
    uint64_t used() const noexcept { return used_; }

private:
    uint8_t* base_;
    uint64_t used_ = 0;
};

// The rows of a draft K/V cache: a ring (row = position % rows) of the last kDraftWindow positions at most.
int ring_rows(int context) {
    return std::min(context, kDraftWindow);
}

// The draft's device memory: a K/V cache per variant first (zeroed with `zeros` at start), the inputs of an extend's
// columns (embedding and positions per column), then one column of activations.
struct Buffers {
    std::vector<uint16_t*> k_cache, v_cache;  // per variant: [ring_rows][kv heads x head dim], F16
    float* zeros;                             // hc zeros: the inject that adds e to every stream with weight 1
    uint64_t zeroed = 0;
    int32_t* pos;  // [3][kMaxColumns]: positions, their ring rows, and the last ring row attention reads
    float* embd;   // [kMaxColumns][n_embd]
    float *res, *xn, *lo, *hc_gate, *mixed, *inject, *out, *en, *e, *n;
    float *q, *k, *v, *attn, *attn_scratch;
    float *router, *weights, *hx, *y, *shared_g, *shared_u, *shared_h, *shared_y, *shared_gate;
    int32_t* ids;
    const uint8_t** table;  // the routed experts' matrices: k gate, k up, k down
    float *logits, *prob;
    kernels::ArgmaxResult *best, *scratch;
    BlockQ8 *q8, *q8_h;

    void lay_out(Layout& d, const Config& c, int context, size_t variants) {
        const size_t E = static_cast<size_t>(c.n_embd), HCD = static_cast<size_t>(c.hc) * E;
        const size_t KV = static_cast<size_t>(c.n_head_kv * c.head_dim),
                     HD = static_cast<size_t>(c.n_head * c.head_dim);
        const size_t K = static_cast<size_t>(c.n_expert_used), FF = static_cast<size_t>(c.n_ff_expert);
        const size_t SH = static_cast<size_t>(c.n_ff_shared);
        k_cache.assign(variants, nullptr);
        v_cache = k_cache;
        const size_t rows = static_cast<size_t>(ring_rows(context));
        for (size_t i = 0; i < variants; ++i) {
            k_cache[i] = d.take<uint16_t>(rows * KV);
            v_cache[i] = d.take<uint16_t>(rows * KV);
        }
        zeros = d.take<float>(static_cast<size_t>(c.hc));
        zeroed = d.used();
        pos = d.take<int32_t>(3 * kMaxColumns), embd = d.take<float>(kMaxColumns * E);
        res = d.take<float>(HCD), xn = d.take<float>(HCD);
        lo = d.take<float>(static_cast<size_t>(c.hc_rank)), hc_gate = d.take<float>(HCD), mixed = d.take<float>(E);
        inject = d.take<float>(static_cast<size_t>(c.hc)), out = d.take<float>(E);
        en = d.take<float>(E), e = d.take<float>(E), n = d.take<float>(HCD);
        q = d.take<float>(2 * HD), k = d.take<float>(KV), v = d.take<float>(KV), attn = d.take<float>(HD);
        const int64_t scratch_floats = kernels::attention_scratch_floats(c.n_head, static_cast<int>(rows), 1);
        attn_scratch = d.take<float>(static_cast<size_t>(scratch_floats));
        router = d.take<float>(static_cast<size_t>(c.n_expert)), weights = d.take<float>(K);
        hx = d.take<float>(K * FF), y = d.take<float>(K * E);
        shared_g = d.take<float>(SH), shared_u = d.take<float>(SH), shared_h = d.take<float>(SH);
        shared_y = d.take<float>(E), shared_gate = d.take<float>(1);
        ids = d.take<int32_t>(K), table = d.take<const uint8_t*>(3 * K);
        logits = d.take<float>(static_cast<size_t>(c.n_vocab)), prob = d.take<float>(1);
        best = d.take<kernels::ArgmaxResult>(1), scratch = d.take<kernels::ArgmaxResult>(kernels::kArgmaxBlocks);
        q8 = d.take<BlockQ8>(std::max(HCD, HD) / 32), q8_h = d.take<BlockQ8>(K * FF / 32);
    }
};

// Pinned host memory: the columns' embeddings the CPU dequantizes and their positions, and the argmax with its
// probability.
struct Host {
    float* embd;   // [kMaxColumns][n_embd]
    int32_t* pos;  // [3][kMaxColumns], as Buffers::pos
    kernels::ArgmaxResult* best;
    float* prob;

    static uint64_t bytes(const Config& c) {
        return kMaxColumns * (static_cast<uint64_t>(c.n_embd) * sizeof(float) + 3 * sizeof(int32_t)) +
               sizeof(kernels::ArgmaxResult) + sizeof(float);
    }
    void lay_out(uint8_t* at, const Config& c) {
        embd = reinterpret_cast<float*>(at);
        pos = reinterpret_cast<int32_t*>(embd + static_cast<size_t>(kMaxColumns) * static_cast<size_t>(c.n_embd));
        best = reinterpret_cast<kernels::ArgmaxResult*>(pos + 3 * kMaxColumns);
        prob = reinterpret_cast<float*>(best + 1);
    }
};

int max_head_rows(const std::vector<DraftVariant>& variants) {
    int rows = 0;
    for (const DraftVariant& v : variants)
        rows = std::max(rows, v.head_rows);
    return rows;
}

uint64_t device_bytes(const Config& c, int context, size_t variants) {
    Layout dry(nullptr);
    Buffers b;
    b.lay_out(dry, c, context, variants);
    return dry.used();
}

}  // namespace

struct Draft::Impl {
    const Model& trunk;
    const Config& cfg;
    const Weights& tw;  // the trunk's: the token embedding and the output head
    cudaStream_t stream;
    int context;
    gguf::File file;
    runtime::DirectFile direct;
    std::unique_ptr<DraftWeights> w;
    std::vector<DraftVariant> variants;
    std::vector<int32_t> vocab;  // head row -> token id, for variants with head_rows > 0
    Matrix head;                 // those rows of the output head, gathered at load
    Buffers b;
    Host hb{};
    uint8_t* device = nullptr;
    uint8_t* host = nullptr;
    uint8_t* head_data = nullptr;
    // The passes run as CUDA graphs recorded at first use: per variant, the extends of 1 to kMaxColumns positions
    // (KV-only, and KV-only but a full last pass) and the depth step. TORCH_EAGER=1 launches kernel by kernel instead
    // (the same result, bit for bit).
    const bool eager = std::getenv("TORCH_EAGER") != nullptr;
    std::vector<cudaGraphExec_t> graphs;

    Impl(const Model& m, const std::filesystem::path& path, std::vector<DraftVariant> vs, std::vector<int32_t> ids,
         int ctx)
        : trunk(m),
          cfg(m.config()),
          tw(m.weights()),
          stream(m.stream()),
          context(ctx),
          file(gguf::File::open(path)),
          direct(path),
          variants(std::move(vs)),
          vocab(std::move(ids)) {
        if (variants.empty())
            throw Error("Give at least one draft variant.");
        const int rows = max_head_rows(variants);
        if (rows > static_cast<int>(vocab.size()) || rows > cfg.n_vocab)
            throw Error("A draft head of " + std::to_string(rows) +
                        " rows needs that many token ids in the draft vocabulary; it has " +
                        std::to_string(vocab.size()) + ".");
        std::vector<bool> seen(static_cast<size_t>(cfg.n_vocab));
        for (int i = 0; i < rows; ++i) {
            const int32_t id = vocab[static_cast<size_t>(i)];
            if (id < 0 || id >= cfg.n_vocab || seen[static_cast<size_t>(id)])
                throw Error("The draft vocabulary must list distinct token ids below " + std::to_string(cfg.n_vocab) +
                            "; " + std::to_string(id) + " is out of range or repeated.");
            seen[static_cast<size_t>(id)] = true;
        }
        runtime::ReadPool pool(kReadThreads);
        w = std::make_unique<DraftWeights>(cfg, file, direct, pool);

        void* d = nullptr;
        const uint64_t bytes = device_bytes(cfg, context, variants.size());
        TORCH_CUDA_CHECK(cudaMalloc(&d, bytes));
        device = static_cast<uint8_t*>(d);
        Layout layout(device);
        b.lay_out(layout, cfg, context, variants.size());
        TORCH_CUDA_CHECK(cudaMemsetAsync(device, 0, b.zeroed, stream));
        void* h = nullptr;
        TORCH_CUDA_CHECK(cudaHostAlloc(&h, Host::bytes(cfg), cudaHostAllocDefault));
        host = static_cast<uint8_t*>(h);
        hb.lay_out(host, cfg);
        if (rows > 0) {  // the output head's rows of the listed ids, in that order
            const Matrix& out = tw.output;
            const uint64_t rb = out.row_bytes();
            void* hd = nullptr;
            TORCH_CUDA_CHECK(cudaHostAlloc(&hd, rb * static_cast<uint64_t>(rows), cudaHostAllocMapped));
            head_data = static_cast<uint8_t*>(hd);
            for (int i = 0; i < rows; ++i)
                std::memcpy(head_data + rb * static_cast<uint64_t>(i),
                            out.data + rb * static_cast<uint64_t>(vocab[static_cast<size_t>(i)]), rb);
            head = {out.type, rows, out.cols, head_data};
        }
        TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    ~Impl() {
        cudaStreamSynchronize(stream);
        for (cudaGraphExec_t g : graphs)
            if (g)
                cudaGraphExecDestroy(g);
        if (device)
            cudaFree(device);
        if (host)
            cudaFreeHost(host);
        if (head_data)
            cudaFreeHost(head_data);
    }

    // y = m . x for `columns` vectors x (back to back, as are their results), the weights read once: BF16 weights take
    // x as F32, quantized ones x quantized into b.q8, unless `quantized` says the kernel that wrote x already put it
    // there (hc_collapse for b.mixed).
    void matvec(const Matrix& m, const float* x, float* y, bool quantized = false, int columns = 1) {
        if (m.type == Type::BF16) {
            kernels::gemv_bf16(m.data, m.rows, m.cols, x, y, stream, columns);
            return;
        }
        if (!quantized)
            kernels::quantize_q8(x, int64_t{m.cols} * columns, b.q8, stream);
        kernels::gemv_q8(m.type, m.data, m.rows, m.cols, b.q8, y, stream, columns);
    }

    // A hyper-connection mixer on b.res, as the trunk's: with `combine`, b.out weighted by the previous mixer's
    // b.inject is added first; then this mixer's inject (when given), and b.mixed (with b.q8) for the block.
    void mix(const float* norm, const Matrix& down, const Matrix& up, const Matrix* inject, bool combine) {
        const int E = cfg.n_embd;
        if (combine)
            kernels::hc_combine_norm(b.res, b.out, b.inject, norm, b.xn, E, cfg.hc, cfg.eps, stream);
        else
            kernels::rms_norm(b.res, E, norm, E, nullptr, 0, b.xn, E, E, cfg.hc, cfg.eps, stream);
        if (inject)
            matvec(*inject, b.xn, b.inject);
        matvec(down, b.xn, b.lo);
        kernels::silu_scale(b.lo, cfg.hc_rank, 1.0f / static_cast<float>(cfg.hc), stream);
        matvec(up, b.lo, b.hc_gate);
        kernels::hc_collapse(b.xn, b.hc_gate, b.mixed, b.q8, E, cfg.hc, stream);
    }

    void attention(const Layer& L, uint16_t* k_cache, uint16_t* v_cache, const int32_t* pos) {
        const int D = cfg.head_dim, H = cfg.n_head, HKV = cfg.n_head_kv;
        matvec(L.wq, b.mixed, b.q, true);  // per head [q | gate]
        matvec(L.wk, b.mixed, b.k, true);
        matvec(L.wv, b.mixed, b.v, true);
        kernels::qk_norm_rope_store(b.q, 2 * D, L.q_norm, b.k, L.k_norm, b.v, k_cache, v_cache, H, HKV, cfg.n_rot, pos,
                                    cfg.rope_base, cfg.eps, 1, stream, pos + kMaxColumns);
        // Rows 0 .. min(position, rows - 1) of the ring: every earlier position below kDraftWindow, the last
        // kDraftWindow positions from there on.
        kernels::attention_decode(b.q, b.q + D, 2 * D, k_cache, v_cache, b.attn, b.attn_scratch, pos + 2 * kMaxColumns,
                                  H, HKV, 1.0f / std::sqrt(static_cast<float>(D)), ring_rows(context), 1, stream);
        matvec(L.wo, b.attn, b.out);
    }

    void experts(const Layer& L) {
        const int E = cfg.n_embd, K = cfg.n_expert_used, FF = cfg.n_ff_expert;
        matvec(L.router, b.mixed, b.router, true);
        matvec(L.shared_gate_inp, b.mixed, b.shared_gate, true);
        kernels::router_top_k(b.router, cfg.n_expert, K, b.ids, b.weights, stream);
        kernels::expert_table(
            b.ids, K, {w->gate_exps, w->up_exps, w->down_exps, L.gate_exps.slice_bytes, L.down_exps.slice_bytes},
            b.table, stream);
        // the routed and the shared experts' gate and up read b.mixed as hc_collapse quantized it
        kernels::gemv_q8_experts(L.gate_exps.type, b.table, b.table + K, K, FF, E, b.q8, 0, b.hx, stream);
        matvec(L.shared_gate, b.mixed, b.shared_g, true);
        matvec(L.shared_up, b.mixed, b.shared_u, true);
        kernels::swiglu(b.shared_g, b.shared_u, b.shared_h, cfg.n_ff_shared, stream);
        kernels::quantize_q8(b.hx, int64_t{K} * FF, b.q8_h, stream);
        kernels::gemv_q8_experts(L.down_exps.type, b.table + 2 * K, nullptr, K, E, FF, b.q8_h, FF / 32, b.y, stream);
        matvec(L.shared_down, b.shared_h, b.shared_y);
        kernels::moe_combine(b.y, b.weights, K, b.shared_y, b.shared_gate, b.out, E, stream);
    }

    // S_s = fc_hidden(hnorm(x)_s) + fc_embedding(enorm(embedding of slot c)), into b.res.
    void input(const DraftVariant& dv, const float* x, int c) {
        const int E = cfg.n_embd, HCD = cfg.hc * E;
        kernels::rms_norm(b.embd + static_cast<size_t>(c) * E, E, w->enorm, 0, nullptr, 0, b.en, E, E, 1, cfg.eps,
                          stream);
        matvec(w->fc_embedding, b.en, b.e);
        if (dv.pooled_hnorm)
            kernels::rms_norm(x, HCD, w->hnorm, 0, nullptr, 0, b.n, HCD, HCD, 1, cfg.eps, stream);
        else
            kernels::rms_norm(x, E, w->hnorm, E, nullptr, 0, b.n, E, E, cfg.hc, cfg.eps, stream);
        matvec(w->fc_hidden, b.n, b.res, false, cfg.hc);  // the hc streams as columns: fc_hidden is read once
        kernels::hc_combine(b.res, b.e, b.zeros, E, cfg.hc, stream);  // weight 2 sigmoid(0) = 1 on every stream
    }

    // The KV-only pass of slot c, from H = the trunk's column c: the attention mixer, the K and V projections, k-norm,
    // RoPE and the cache row at slot c's position.
    void kv_only(int variant, int c) {
        const int D = cfg.head_dim, HKV = cfg.n_head_kv;
        const Layer& L = w->layer;
        input(variants[static_cast<size_t>(variant)], trunk.residual(c), c);
        mix(L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, nullptr, false);
        matvec(L.wk, b.mixed, b.k, true);
        matvec(L.wv, b.mixed, b.v, true);
        kernels::rms_norm(b.k, D, L.k_norm, 0, nullptr, 0, b.k, D, D, HKV, cfg.eps, stream);
        kernels::rope_neox(b.k, D, HKV, cfg.n_rot, b.pos + c, cfg.rope_base, stream);
        kernels::store_kv(b.k, b.v, b.k_cache[static_cast<size_t>(variant)], b.v_cache[static_cast<size_t>(variant)],
                          HKV * D, b.pos + kMaxColumns + c, stream);
    }

    // The full pass of slot c from x (H, or the block's own state S' in b.res): the block, the head's mixer (b.res
    // becomes S'), the head rows, their argmax and its probability, copied to the host.
    void full(int variant, const float* x, int c) {
        const DraftVariant& dv = variants[static_cast<size_t>(variant)];
        const Layer& L = w->layer;
        input(dv, x, c);
        mix(L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, &L.hc_attn_inject, false);
        attention(L, b.k_cache[static_cast<size_t>(variant)], b.v_cache[static_cast<size_t>(variant)], b.pos + c);
        mix(L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, &L.hc_ffn_inject, true);
        experts(L);
        mix(w->head_hc_norm, w->head_hc_down, w->head_hc_up, nullptr, true);
        const Matrix rows = dv.head_rows > 0 ? Matrix{head.type, dv.head_rows, head.cols, head.data} : tw.output;
        matvec(rows, b.mixed, b.logits, true);
        kernels::argmax(b.logits, rows.rows, b.scratch, b.best, stream);
        kernels::argmax_prob(b.logits, rows.rows, b.best, b.prob, stream);
        TORCH_CUDA_CHECK(
            cudaMemcpyAsync(hb.best, b.best, sizeof(kernels::ArgmaxResult), cudaMemcpyDeviceToHost, stream));
        TORCH_CUDA_CHECK(cudaMemcpyAsync(hb.prob, b.prob, sizeof(float), cudaMemcpyDeviceToHost, stream));
    }

    // The first n slots' embeddings and positions, from pinned host memory.
    void upload(int n) {
        const size_t E = static_cast<size_t>(cfg.n_embd), cols = static_cast<size_t>(n);
        TORCH_CUDA_CHECK(cudaMemcpyAsync(b.embd, hb.embd, sizeof(float) * cols * E, cudaMemcpyHostToDevice, stream));
        TORCH_CUDA_CHECK(
            cudaMemcpyAsync(b.pos, hb.pos, sizeof(int32_t) * 3 * kMaxColumns, cudaMemcpyHostToDevice, stream));
    }

    // Slot c's input: the embedding of `token` and position `position`, written on the host (the GPU is idle).
    void set_slot(int c, int32_t token, int position) {
        const Matrix& te = tw.token_embd;
        dequant_row(te.type, te.data + static_cast<uint64_t>(token) * te.row_bytes(), cfg.n_embd,
                    hb.embd + static_cast<size_t>(c) * static_cast<size_t>(cfg.n_embd));
        const int rows = ring_rows(context);
        hb.pos[c] = position, hb.pos[kMaxColumns + c] = position % rows;
        hb.pos[2 * kMaxColumns + c] = std::min(position, rows - 1);
    }

    // The drafted token of the last full pass, after the GPU has finished.
    Drafted result(const DraftVariant& dv) const {
        const int32_t row = hb.best->index;
        return {dv.head_rows > 0 ? vocab[static_cast<size_t>(row)] : row, *hb.prob};
    }

    // Runs what `enqueue` puts on the stream: directly when eager, else graph `index`, recorded at its first use (every
    // pointer it bakes in stays fixed; the inputs come from the pinned host slots set before the launch).
    template <class F>
    void launch(size_t index, F&& enqueue) {
        if (eager) {
            enqueue();
            return;
        }
        if (graphs.empty())
            graphs.assign(variants.size() * (2 * (kMaxColumns + 1) + 1), nullptr);
        cudaGraphExec_t& graph = graphs[index];
        if (graph == nullptr) {
            cudaGraph_t g = nullptr;
            TORCH_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            try {
                enqueue();
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
    size_t extend_graph(int variant, bool full_last, int n) const {
        return (static_cast<size_t>(variant) * 2 + (full_last ? 1 : 0)) * (kMaxColumns + 1) + static_cast<size_t>(n);
    }
    size_t step_graph(int variant) const {
        return variants.size() * 2 * (kMaxColumns + 1) + static_cast<size_t>(variant);
    }

    void run(int variant, int t, std::span<const int32_t> next, int depth, std::vector<Drafted>& out,
             const std::function<bool(int, float)>& more) {
        const int n = static_cast<int>(next.size());
        if (variant < 0 || variant >= static_cast<int>(variants.size()) || depth < 0 || depth > kMaxDepth || n < 1 ||
            n > kMaxColumns)
            throw Error("Ask for an existing draft variant, 1 to " + std::to_string(kMaxColumns) +
                        " positions and a depth between 0 and " + std::to_string(kMaxDepth) + ".");
        for (const int32_t token : next)
            if (token < 0 || token >= cfg.n_vocab)
                throw Error("Token id " + std::to_string(token) + " is outside the vocabulary.");
        if (t < 0 || t + n > trunk.position())
            throw Error("Draft from positions the trunk has computed in its last window.");
        const int last = t + n - 1 + std::max(depth - 1, 0);
        if (last >= context)
            throw Error("Drafting to position " + std::to_string(last) + " passes the context of " +
                        std::to_string(context) + " tokens.");
        const DraftVariant& dv = variants[static_cast<size_t>(variant)];
        out.clear();
        for (int i = 0; i < n; ++i)
            set_slot(i, next[static_cast<size_t>(i)], t + i);
        launch(extend_graph(variant, depth > 0, n), [&] {
            upload(n);
            for (int i = 0; i < (depth > 0 ? n - 1 : n); ++i)
                kv_only(variant, i);
            if (depth > 0)
                full(variant, trunk.residual(n - 1), n - 1);
        });
        if (depth > 0) {
            TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
            out.push_back(result(dv));
        }
        for (int j = 1; j < depth; ++j) {  // from S' and the previous draft, one position further each
            if (more && !more(j, out.back().prob))
                break;
            set_slot(0, out.back().token, t + n - 1 + j);
            launch(step_graph(variant), [&] {
                upload(1);
                full(variant, b.res, 0);
            });
            TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
            out.push_back(result(dv));
        }
        TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
        TORCH_CUDA_CHECK(cudaGetLastError());
    }

    int first_kv_difference(int a, int b2, int rows) const {
        const size_t row = static_cast<size_t>(cfg.n_head_kv * cfg.head_dim), n = row * static_cast<size_t>(rows);
        std::vector<uint16_t> ka(n), kb(n), va(n), vb(n);
        TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
        for (auto [dst, src] : {std::pair{&ka, b.k_cache[static_cast<size_t>(a)]},
                                {&kb, b.k_cache[static_cast<size_t>(b2)]},
                                {&va, b.v_cache[static_cast<size_t>(a)]},
                                {&vb, b.v_cache[static_cast<size_t>(b2)]}})
            TORCH_CUDA_CHECK(cudaMemcpy(dst->data(), src, sizeof(uint16_t) * n, cudaMemcpyDeviceToHost));
        for (int r = 0; r < rows; ++r) {
            const size_t at = static_cast<size_t>(r) * row;
            if (!std::equal(ka.begin() + at, ka.begin() + at + row, kb.begin() + at) ||
                !std::equal(va.begin() + at, va.begin() + at + row, vb.begin() + at))
                return r;
        }
        return -1;
    }
};

Reservation Draft::reservation(const std::filesystem::path& shard1, const std::filesystem::path& file,
                               const std::vector<DraftVariant>& variants, int context) {
    const gguf::File main = gguf::File::open(shard1, true), draft = gguf::File::open(file);
    const Config c = Config::read(main);
    const gguf::TensorInfo* out = main.tensor("output.weight");
    const uint64_t row = out ? torch::row_bytes(*type_info(out->type), static_cast<uint64_t>(c.n_embd)) : 0;
    const int rows = max_head_rows(variants);
    return {"draft layer, all experts resident, " + std::to_string(rows) + " head rows and " +
                std::to_string(variants.size()) + " K/V rings of " + std::to_string(ring_rows(context)) + " rows",
            DraftWeights::bytes(c, draft) + static_cast<uint64_t>(rows) * row +
                device_bytes(c, context, variants.size()) + Host::bytes(c)};
}

Draft::Draft(const Model& trunk, const std::filesystem::path& file, std::vector<DraftVariant> variants,
             std::vector<int32_t> vocab, int context)
    : impl_(std::make_unique<Impl>(trunk, file, std::move(variants), std::move(vocab), context)) {}

Draft::~Draft() = default;

void Draft::run(int variant, int position, std::span<const int32_t> next, int depth, std::vector<Drafted>& out,
                const std::function<bool(int, float)>& more) {
    impl_->run(variant, position, next, depth, out, more);
}

int Draft::first_kv_difference(int a, int b, int rows) const {
    if (a < 0 || b < 0 || a >= static_cast<int>(impl_->variants.size()) ||
        b >= static_cast<int>(impl_->variants.size()) || rows < 0 || rows > ring_rows(impl_->context))
        throw Error("Compare existing draft variants over rows within the draft's K/V ring.");
    return impl_->first_kv_difference(a, b, rows);
}

}  // namespace torch::model
