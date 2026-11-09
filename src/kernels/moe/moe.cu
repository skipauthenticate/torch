#include "kernels/common.cuh"
#include "kernels/moe/moe.h"

namespace torch::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxExperts = 1024;
constexpr int kMaxTopK = 32;

// True when (v, i) beats (best_v, best_i): larger value, or the same value at a lower index.
__device__ __forceinline__ bool better(float v, int i, float best_v, int best_i) {
    return v > best_v || (v == best_v && i < best_i);
}

// One block, one thread per expert (the launcher rounds n up to whole warps). The softmax uses the block's
// reductions. The top k is found by counting, not by k rounds of argmax: an expert's rank is the number of experts that
// beat it (larger p, or equal p at a lower index), a strict total order. Each expert first counts within its warp; the
// k with rank < k there are the warp's candidates, and every expert of the overall top k is among them. Each candidate
// then counts among all candidates, which gives its exact overall rank, so the choice, the order and the sum of the
// chosen p (taken in rank order) are those of k rounds of a block-wide argmax.
__global__ void router_kernel(const float* logits, int n, int k, int32_t* ids, float* weights) {
    logits += int64_t{blockIdx.x} * n, ids += blockIdx.x * k, weights += blockIdx.x * k;  // the window column
    __shared__ float p_s[kMaxExperts];
    __shared__ float cand_v[kMaxExperts / kWarp * kMaxTopK];
    __shared__ int cand_i[kMaxExperts / kWarp * kMaxTopK];
    __shared__ float sel_v[kMaxTopK];
    __shared__ int sel_i[kMaxTopK];
    const int t = threadIdx.x, warp = t / kWarp, warps = blockDim.x / kWarp;
    const float logit = t < n ? logits[t] : -INFINITY;
    const float peak = block_max(logit);
    const float e = t < n ? expf(logit - peak) : 0.0f;
    const float sum = block_sum(e);
    const float p = t < n ? e / sum : -INFINITY;
    p_s[t] = p;
    __syncwarp();

    int rank = 0;  // within the warp
    for (int j = warp * kWarp; j < warp * kWarp + kWarp; ++j)
        rank += better(p_s[j], j, p, t) ? 1 : 0;
    if (rank < k) {
        cand_v[warp * k + rank] = p;
        cand_i[warp * k + rank] = t;
    }
    __syncthreads();
    if (rank < k) {
        rank = 0;  // overall
        for (int c = 0; c < warps * k; ++c)
            rank += better(cand_v[c], cand_i[c], p, t) ? 1 : 0;
        if (rank < k) {
            sel_v[rank] = p;
            sel_i[rank] = t;
        }
    }
    __syncthreads();
    if (t == 0) {
        float chosen = 0.0f;
        for (int r = 0; r < k; ++r)
            chosen += sel_v[r];
        const float norm = fmaxf(chosen, 6.103515625e-5f);
        for (int r = 0; r < k; ++r) {
            ids[r] = sel_i[r];
            weights[r] = sel_v[r] / norm;
        }
    }
}

__global__ void swiglu_kernel(const float* g, const float* u, float* h, int n) {
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i < n)
        h[i] = silu(g[i]) * u[i];
}

__global__ void moe_combine_kernel(const float* y, const float* w, int k, const float* ys, const float* shared_gate,
                                   float* out, int n) {
    const int64_t t = blockIdx.y;  // the window column
    y += t * k * n, w += t * k, ys += t * n, shared_gate += t, out += t * n;
    const int e = blockIdx.x * kThreads + threadIdx.x;
    if (e >= n)
        return;
    float acc = 0.0f;
    for (int r = 0; r < k; ++r)
        acc += w[r] * y[int64_t{r} * n + e];
    out[e] = acc + sigmoid(shared_gate[0]) * ys[e];
}

__global__ void expert_table_kernel(const int32_t* ids, int k, ResidentExperts e, const uint8_t** table) {
    const int s = threadIdx.x;
    if (s >= k)
        return;
    const uint64_t id = static_cast<uint64_t>(ids[s]);
    table[s] = e.gate + id * e.gate_up_slice;
    table[k + s] = e.up + id * e.gate_up_slice;
    table[2 * k + s] = e.down + id * e.down_slice;
}

}  // namespace

void router_top_k(const float* logits, int n, int k, int32_t* ids, float* weights, cudaStream_t stream, int columns) {
    router_kernel<<<columns, (n + kWarp - 1) / kWarp * kWarp, 0, stream>>>(logits, n, k, ids, weights);
}

void expert_table(const int32_t* ids, int k, ResidentExperts experts, const uint8_t** table, cudaStream_t stream) {
    expert_table_kernel<<<1, kWarp, 0, stream>>>(ids, k, experts, table);
}

void swiglu(const float* g, const float* u, float* h, int n, cudaStream_t stream) {
    swiglu_kernel<<<blocks_for(n, kThreads), kThreads, 0, stream>>>(g, u, h, n);
}

void moe_combine(const float* y, const float* w, int k, const float* ys, const float* shared_gate, float* out, int n,
                 cudaStream_t stream, int columns) {
    moe_combine_kernel<<<dim3(blocks_for(n, kThreads), columns), kThreads, 0, stream>>>(y, w, k, ys, shared_gate, out,
                                                                                        n);
}

}  // namespace torch::kernels
