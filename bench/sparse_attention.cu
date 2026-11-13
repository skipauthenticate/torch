// Cost of one trunk attention layer's attention per verify window, dense against sparse (M6 gate K4: a sparse window at
// 4,096 tokens may cost at most the dense window at 2,048 plus 1 ms). Times, with CUDA events over CUDA graphs:
//  - attention_decode at position 2,040 (context 2,048) and at --position (context 4,096 by default);
//  - the sparse path at --position stage by stage: the BF16 indexer projections, index_prepare, index_pool,
//    index_score, index_select, attention_decode_cells; and the whole path in one graph.
// Each layer has its own KV cache, pooled keys and indexer weights in the engine, so the graph rotates through 12
// copies of them (more bytes than the 4 MB L2) and reports microseconds per layer and milliseconds per window (x 12).
// Inputs are random (finite F16 / BF16 around +-0.2-1), so the selection is a random half of the blocks per column;
// --similar makes every column's input column 0's plus 5% noise, so the columns select mostly the same blocks.
//
// With --chunk N (and --batch B, default 128) it times instead one attention layer over a prefill chunk of N tokens at
// positions P .. P + N - 1, stage by stage and whole, with each stage's per-chunk seconds for 12 layers: q/k norm, RoPE
// and K/V rows; the indexer's prepare and pool; and, batch by batch, scoring, selection and attention over the cells.
//
// usage: torch_bench_sparse MODEL-00001-of-00002.gguf [--position P] [--context N] [--similar]
//                           [--chunk N [--batch B]]
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "core/cuda_check.h"
#include "gguf/gguf.h"
#include "kernels/attention/attention.h"
#include "kernels/columns.h"
#include "kernels/gemv/gemv.h"
#include "kernels/indexer/indexer.h"
#include "model/config.h"

namespace {

using namespace torch;
constexpr int kLayers = 12;  // trunk attention layers, one copy of every per-layer buffer each
constexpr int kReplays = 20;

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint32_t next() {
    g_rng ^= g_rng << 13, g_rng ^= g_rng >> 7, g_rng ^= g_rng << 17;
    return static_cast<uint32_t>(g_rng >> 16);
}
float uniform(float lo, float hi) {
    return lo + (hi - lo) * static_cast<float>(next() & 0xffffff) / 16777216.0f;
}

template <class T>
T* device(size_t n) {
    T* p = nullptr;
    TORCH_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}
template <class T>
T* upload(const std::vector<T>& h) {
    T* p = device<T>(h.size());
    TORCH_CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
float* floats(size_t n, float lo, float hi) {
    std::vector<float> h(n);
    for (float& x : h)
        x = uniform(lo, hi);
    return upload(h);
}
uint16_t* halves(size_t n) {  // finite F16 around +-0.2, as bench/kernels.cu fills the KV cache
    std::vector<uint16_t> h(n);
    for (uint16_t& x : h)
        x = static_cast<uint16_t>((next() & 0x83FF) | 0x3000);
    return upload(h);
}
uint8_t* bf16(size_t n) {  // BF16 of uniform(-0.05, 0.05)
    std::vector<uint16_t> h(n);
    for (uint16_t& x : h) {
        const float f = uniform(-0.05f, 0.05f);
        uint32_t bits;
        std::memcpy(&bits, &f, 4);
        x = static_cast<uint16_t>(bits >> 16);
    }
    return reinterpret_cast<uint8_t*>(upload(h));
}

// Microseconds per launch(layer) over kReplays replays of one graph of kLayers launches, after one warm-up replay.
template <class Launch>
double time_layers(cudaStream_t stream, Launch&& launch) {
    launch(0);
    TORCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph;
    TORCH_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int l = 0; l < kLayers; ++l)
        launch(l);
    TORCH_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec;
    TORCH_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    cudaEvent_t start, stop;
    TORCH_CUDA_CHECK(cudaEventCreate(&start));
    TORCH_CUDA_CHECK(cudaEventCreate(&stop));
    TORCH_CUDA_CHECK(cudaGraphLaunch(exec, stream));
    TORCH_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int r = 0; r < kReplays; ++r)
        TORCH_CUDA_CHECK(cudaGraphLaunch(exec, stream));
    TORCH_CUDA_CHECK(cudaEventRecord(stop, stream));
    TORCH_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    TORCH_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start), cudaEventDestroy(stop);
    cudaGraphExecDestroy(exec), cudaGraphDestroy(graph);
    return ms * 1000.0 / (kReplays * kLayers);
}

// One attention layer over a prefill chunk (see the header), us per layer and s per chunk of 12 layers.
void run_chunk(const model::Config& cfg, int position, int context, int n, int b, cudaStream_t stream) {
    const int H = cfg.n_head, HKV = cfg.n_head_kv, D = cfg.head_dim, KV = HKV * D, ID = kernels::kIndexDim;
    const int max_blocks = context / 4;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<uint16_t*> kc(kLayers), vc(kLayers), ring(kLayers), pooled(kLayers);
    for (int l = 0; l < kLayers; ++l) {
        kc[l] = halves(static_cast<size_t>(context) * KV), vc[l] = halves(static_cast<size_t>(context) * KV);
        ring[l] = halves(static_cast<size_t>(kernels::kIndexRing) * ID);
        pooled[l] = halves(static_cast<size_t>(max_blocks) * ID);
    }
    const size_t N = static_cast<size_t>(n);
    float* q = floats(N * 2 * H * D, -1.0f, 1.0f);
    float* k = floats(N * KV, -1.0f, 1.0f);
    float* v = floats(N * KV, -1.0f, 1.0f);
    float* iq = floats(N * kernels::kIndexHeads * ID, -1.0f, 1.0f);
    float* ik = floats(N * ID, -1.0f, 1.0f);
    float *qw = floats(D, 0.5f, 1.5f), *kw = floats(D, 0.5f, 1.5f), *qn = floats(ID, 0.5f, 1.5f),
          *kn = floats(ID, 0.5f, 1.5f);
    float* qi = device<float>(N * kernels::kIndexHeads * ID);
    uint16_t* raw = device<uint16_t>(N * ID);
    float* out = device<float>(N * H * D);
    std::vector<int32_t> positions(N);
    for (size_t t = 0; t < N; ++t)
        positions[t] = position + static_cast<int>(t);
    int32_t* pos = upload(positions);
    float* scores = device<float>(static_cast<size_t>(kernels::index_score_floats(context, b)));
    kernels::IndexCells cells{device<int32_t>(static_cast<size_t>(b) * kernels::kIndexTopBlocks),
                              device<int32_t>(static_cast<size_t>(b) * 4)};
    float* scratch = device<float>(static_cast<size_t>(kernels::attention_cells_scratch_floats(H, b)));
    auto rows = [&](int l) {
        kernels::qk_norm_rope_store(q, 2 * D, qw, k, kw, v, kc[l], vc[l], H, HKV, cfg.n_rot, pos, cfg.rope_base,
                                    cfg.eps, n, stream);
    };
    auto index = [&](int l) {
        kernels::index_prepare_chunk(iq, ik, qn, qi, raw, pos, cfg.n_rot, cfg.rope_base, cfg.eps, n, stream);
        kernels::index_pool_chunk(ring[l], raw, kn, pooled[l], pos, cfg.n_rot, cfg.rope_base, cfg.eps, n, stream);
    };
    const auto batches = [&](int l, bool score, bool select, bool attend) {
        for (int c0 = 0; c0 < n; c0 += b) {
            const int w = std::min(b, n - c0);
            if (score)
                kernels::index_score(qi + static_cast<size_t>(c0) * kernels::kIndexHeads * ID, pooled[l], scores,
                                     pos + c0, max_blocks, w, stream);
            if (select)
                kernels::index_select(scores, cells, pos + c0, max_blocks, w, stream);
            if (attend) {
                float* qc = q + static_cast<size_t>(c0) * 2 * H * D;
                kernels::attention_decode_cells(qc, qc + D, 2 * D, kc[l], vc[l], out + static_cast<size_t>(c0) * H * D,
                                                scratch, pos + c0, H, HKV, scale, cells, w, stream);
            }
        }
    };
    const double t_rows = time_layers(stream, rows), t_index = time_layers(stream, index);
    const double t_score = time_layers(stream, [&](int l) { batches(l, true, false, false); });
    const double t_select = time_layers(stream, [&](int l) { batches(l, false, true, false); });
    const double t_attend = time_layers(stream, [&](int l) { batches(l, false, false, true); });
    const double t_all = time_layers(stream, [&](int l) { rows(l), index(l), batches(l, true, true, true); });
    const double chunk = 4096.0 / n * kLayers / 1e6;  // us per layer -> s per 4,096-token chunk of 12 layers
    std::printf(
        "chunk of %d at %d (context %d), batches of %d: us per layer (s per 4,096 chunk, 12 layers): rows %.0f "
        "(%.3f) + indexer prepare/pool %.0f (%.3f) + score %.0f (%.3f) + select %.0f (%.3f) + attention %.0f "
        "(%.3f); all %.0f (%.3f)\n",
        n, position, context, b, t_rows, t_rows * chunk, t_index, t_index * chunk, t_score, t_score * chunk, t_select,
        t_select * chunk, t_attend, t_attend * chunk, t_all, t_all * chunk);
}

}  // namespace

int main(int argc, char** argv) {
    std::string path;
    int position = 4000, context = 4096, chunk = 0, batch = 128;
    bool similar = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--position" && i + 1 < argc)
            position = std::atoi(argv[++i]);
        else if (a == "--context" && i + 1 < argc)
            context = std::atoi(argv[++i]);
        else if (a == "--similar")
            similar = true;
        else if (a == "--chunk" && i + 1 < argc)
            chunk = std::atoi(argv[++i]);
        else if (a == "--batch" && i + 1 < argc)
            batch = std::atoi(argv[++i]);
        else
            path = a;
    }
    if (path.empty() || position + std::max(chunk, kernels::kMaxColumns) > context || context % 4 != 0 || batch < 1) {
        std::fprintf(stderr,
                     "usage: torch_bench_sparse MODEL-00001-of-00002.gguf [--position P] [--context N] [--similar] "
                     "[--chunk N [--batch B]]\n");
        return 2;
    }
    const model::Config cfg = model::Config::read(gguf::File::open(path, /*header_only=*/true));
    if (chunk > 0) {
        cudaStream_t s;
        TORCH_CUDA_CHECK(cudaStreamCreate(&s));
        run_chunk(cfg, position, context, chunk, batch, s);
        return 0;
    }
    const int H = cfg.n_head, HKV = cfg.n_head_kv, D = cfg.head_dim, E = cfg.n_embd, KV = HKV * D;
    const int T = kernels::kMaxColumns, IH = kernels::kIndexHeads, ID = kernels::kIndexDim, max_blocks = context / 4;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    cudaStream_t stream;
    TORCH_CUDA_CHECK(cudaStreamCreate(&stream));

    // Per layer copies.
    std::vector<uint16_t*> kc(kLayers), vc(kLayers), ring(kLayers), pooled(kLayers);
    std::vector<uint8_t*> wq(kLayers), wk(kLayers);
    for (int l = 0; l < kLayers; ++l) {
        kc[l] = halves(static_cast<size_t>(context) * KV), vc[l] = halves(static_cast<size_t>(context) * KV);
        ring[l] = halves(static_cast<size_t>(kernels::kIndexRing) * ID);
        pooled[l] = halves(static_cast<size_t>(max_blocks) * ID);
        wq[l] = bf16(static_cast<size_t>(IH) * ID * E), wk[l] = bf16(static_cast<size_t>(ID) * E);
    }
    // Activations, shared (warm in L2 as in the engine).
    float* q = floats(static_cast<size_t>(T) * 2 * H * D, -1.0f, 1.0f);
    std::vector<float> xh(static_cast<size_t>(T) * E);
    for (size_t i = 0; i < xh.size(); ++i)  // --similar: every column is column 0 plus 5% noise
        xh[i] = similar && i >= static_cast<size_t>(E) ? xh[i % E] + uniform(-0.05f, 0.05f) : uniform(-1.0f, 1.0f);
    float* x = upload(xh);
    float* qn = floats(ID, 0.5f, 1.5f);
    float* kn = floats(ID, 0.5f, 1.5f);
    float* q_raw = device<float>(static_cast<size_t>(T) * IH * ID);
    float* k_raw = device<float>(static_cast<size_t>(T) * ID);
    float* qi = device<float>(static_cast<size_t>(T) * IH * ID);
    float* scores = device<float>(static_cast<size_t>(kernels::index_score_floats(context, T)));
    kernels::IndexCells cells{device<int32_t>(static_cast<size_t>(T) * kernels::kIndexTopBlocks),
                              device<int32_t>(static_cast<size_t>(T) * 4)};
    float* out = device<float>(static_cast<size_t>(T) * H * D);
    float* cell_scratch = device<float>(static_cast<size_t>(kernels::attention_cells_scratch_floats(H, T)));
    float* dense_scratch = device<float>(static_cast<size_t>(kernels::attention_scratch_floats(H, context, T)));
    int32_t* pos_sparse = upload(std::vector<int32_t>{position});
    int32_t* pos_dense = upload(std::vector<int32_t>{2040});

    std::printf("model n_embd %d, heads %d/%d x %d; sparse at position %d, context %d; us per layer, ms per window\n",
                E, H, HKV, D, position, context);
    for (int t : {8, 1, 4, 8}) {  // the first pass warms the clocks up and is not printed
        auto dense = [&](const int32_t* pos, int ctx) {
            return [&, pos, ctx](int l) {
                kernels::attention_decode(q, q + D, 2 * D, kc[l], vc[l], out, dense_scratch, pos, H, HKV, scale, ctx, t,
                                          stream);
            };
        };
        auto proj = [&](int l) {
            kernels::gemv_bf16_pair(wq[l], IH * ID, wk[l], ID, E, x, q_raw, k_raw, stream, 0.0f, t);
        };
        auto prepare = [&](int l) {
            kernels::index_prepare(q_raw, k_raw, qn, qi, ring[l], pos_sparse, cfg.n_rot, cfg.rope_base, cfg.eps, t,
                                   stream);
        };
        auto pool = [&](int l) {
            kernels::index_pool(ring[l], kn, pooled[l], pos_sparse, cfg.n_rot, cfg.rope_base, cfg.eps, t, stream);
        };
        auto score = [&](int l) { kernels::index_score(qi, pooled[l], scores, pos_sparse, max_blocks, t, stream); };
        auto select = [&](int) { kernels::index_select(scores, cells, pos_sparse, max_blocks, t, stream); };
        auto attend = [&](int l) {
            kernels::attention_decode_cells(q, q + D, 2 * D, kc[l], vc[l], out, cell_scratch, pos_sparse, H, HKV, scale,
                                            cells, t, stream);
        };
        auto sparse = [&](int l) { proj(l), prepare(l), pool(l), score(l), select(l), attend(l); };
        const double d2k = time_layers(stream, dense(pos_dense, 2048));
        const double dpk = time_layers(stream, dense(pos_sparse, context));
        const double s_proj = time_layers(stream, proj), s_prep = time_layers(stream, prepare);
        const double s_pool = time_layers(stream, pool), s_score = time_layers(stream, score);
        const double s_sel = time_layers(stream, select), s_att = time_layers(stream, attend);
        const double s_all = time_layers(stream, sparse);
        static bool warm = false;
        if (!std::exchange(warm, true))
            continue;
        std::printf(
            "W=%d dense@2040 %.1f (%.2f ms) | dense@%d %.1f (%.2f) | sparse@%d %.1f (%.2f) = proj %.1f + "
            "prepare %.1f + pool %.1f + score %.1f + select %.1f + cells %.1f | sparse - dense@2040 %.2f ms\n",
            t, d2k, d2k * kLayers / 1000, position, dpk, dpk * kLayers / 1000, position, s_all, s_all * kLayers / 1000,
            s_proj, s_prep, s_pool, s_score, s_sel, s_att, (s_all - d2k) * kLayers / 1000);
    }
    return 0;
}
