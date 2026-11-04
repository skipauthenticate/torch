#include "kernels/common.cuh"
#include "kernels/hc/hc.h"

namespace torch::kernels {
namespace {

constexpr int kThreads = 256;

__global__ void silu_scale_kernel(float* x, int n, float scale) {
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i < n)
        x[i] = silu(x[i] * scale);
}

__global__ void hc_collapse_kernel(const float* xn, const float* gate, float* mixed, int n, int hc) {
    const int e = blockIdx.x * kThreads + threadIdx.x;
    if (e >= n)
        return;
    float acc = 0.0f;
    for (int s = 0; s < hc; ++s)
        acc += xn[int64_t{s} * n + e] * sigmoid(gate[int64_t{s} * n + e]);
    mixed[e] = acc * (1.0f / static_cast<float>(hc));
}

__global__ void hc_combine_kernel(float* res, const float* out, const float* inject, int n, int hc) {
    const int e = blockIdx.x * kThreads + threadIdx.x;
    if (e >= n)
        return;
    const float o = out[e];
    for (int s = 0; s < hc; ++s)
        res[int64_t{s} * n + e] += o * (2.0f * sigmoid(inject[s] * (1.0f / static_cast<float>(hc))));
}

__global__ void hc_broadcast_kernel(const float* x, float* res, int n, int hc) {
    const int e = blockIdx.x * kThreads + threadIdx.x;
    if (e >= n)
        return;
    for (int s = 0; s < hc; ++s)
        res[int64_t{s} * n + e] = x[e];
}

}  // namespace

void silu_scale(float* x, int n, float scale, cudaStream_t stream) {
    silu_scale_kernel<<<blocks_for(n, kThreads), kThreads, 0, stream>>>(x, n, scale);
}

void hc_collapse(const float* xn, const float* gate, float* mixed, int n, int hc, cudaStream_t stream) {
    hc_collapse_kernel<<<blocks_for(n, kThreads), kThreads, 0, stream>>>(xn, gate, mixed, n, hc);
}

void hc_combine(float* res, const float* out, const float* inject, int n, int hc, cudaStream_t stream) {
    hc_combine_kernel<<<blocks_for(n, kThreads), kThreads, 0, stream>>>(res, out, inject, n, hc);
}

void hc_broadcast(const float* x, float* res, int n, int hc, cudaStream_t stream) {
    hc_broadcast_kernel<<<blocks_for(n, kThreads), kThreads, 0, stream>>>(x, res, n, hc);
}

}  // namespace torch::kernels
