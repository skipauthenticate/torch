#include "kernels/common.cuh"
#include "kernels/moe/moe.h"

namespace torch::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxExperts = 1024;

// True when (v, i) beats (best_v, best_i): larger value, or the same value at a lower index.
__device__ __forceinline__ bool better(float v, int i, float best_v, int best_i) {
    return v > best_v || (v == best_v && i < best_i);
}

// One block, one thread per expert (the launcher rounds the block up to a multiple of 32).
__global__ void router_kernel(const float* logits, int n, int k, int32_t* ids, float* weights) {
    __shared__ float p[kMaxExperts];
    __shared__ float warp_v[kWarp];
    __shared__ int warp_i[kWarp];
    const int t = threadIdx.x;
    const float logit = t < n ? logits[t] : -INFINITY;
    const float peak = block_max(logit);
    const float e = t < n ? expf(logit - peak) : 0.0f;
    const float sum = block_sum(e);
    p[t] = t < n ? e / sum : -INFINITY;

    float chosen = 0.0f;
    for (int r = 0; r < k; ++r) {
        __syncthreads();
        float v = p[t];
        int i = t;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, offset);
            const int oi = __shfl_xor_sync(0xffffffffu, i, offset);
            if (better(ov, oi, v, i)) {
                v = ov;
                i = oi;
            }
        }
        if (t % kWarp == 0) {
            warp_v[t / kWarp] = v;
            warp_i[t / kWarp] = i;
        }
        __syncthreads();
        if (t == 0) {
            float bv = warp_v[0];
            int bi = warp_i[0];
            for (int w = 1; w < blockDim.x / kWarp; ++w) {
                if (better(warp_v[w], warp_i[w], bv, bi)) {
                    bv = warp_v[w];
                    bi = warp_i[w];
                }
            }
            ids[r] = bi;
            weights[r] = bv;
            p[bi] = -INFINITY;
            chosen += bv;
        }
    }
    if (t == 0) {
        const float norm = fmaxf(chosen, 6.103515625e-5f);
        for (int r = 0; r < k; ++r)
            weights[r] /= norm;
    }
}

__global__ void swiglu_kernel(const float* g, const float* u, float* h, int n) {
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i < n)
        h[i] = silu(g[i]) * u[i];
}

__global__ void moe_combine_kernel(const float* y, const float* w, int k, const float* ys, const float* shared_gate,
                                   float* out, int n) {
    const int e = blockIdx.x * kThreads + threadIdx.x;
    if (e >= n)
        return;
    float acc = 0.0f;
    for (int r = 0; r < k; ++r)
        acc += w[r] * y[int64_t{r} * n + e];
    out[e] = acc + sigmoid(shared_gate[0]) * ys[e];
}

}  // namespace

void router_top_k(const float* logits, int n, int k, int32_t* ids, float* weights, cudaStream_t stream) {
    router_kernel<<<1, (n + kWarp - 1) / kWarp * kWarp, 0, stream>>>(logits, n, k, ids, weights);
}

void swiglu(const float* g, const float* u, float* h, int n, cudaStream_t stream) {
    swiglu_kernel<<<blocks_for(n, kThreads), kThreads, 0, stream>>>(g, u, h, n);
}

void moe_combine(const float* y, const float* w, int k, const float* ys, const float* shared_gate, float* out, int n,
                 cudaStream_t stream) {
    moe_combine_kernel<<<blocks_for(n, kThreads), kThreads, 0, stream>>>(y, w, k, ys, shared_gate, out, n);
}

}  // namespace torch::kernels
