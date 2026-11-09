#include "kernels/common.cuh"
#include "kernels/gemv/q8.cuh"
#include "kernels/hc/hc.h"

namespace torch::kernels {
namespace {

constexpr int kThreads = 256;

// Every kernel here takes a window's columns in blockIdx.y; a column's pointers are offset by its column-major size
// (common layout of kernels/hc/hc.h), and its threads do exactly what a one-column launch does.

__global__ void silu_scale_kernel(float* x, int n, float scale) {
    x += int64_t{blockIdx.y} * n;
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i < n)
        x[i] = silu(x[i] * scale);
}

// A warp covers 32 consecutive values (n % 32 == 0), which it quantizes as one Q8 block.
__global__ void hc_collapse_kernel(const float* xn, const float* gate, float* mixed, BlockQ8* q8, int n, int hc) {
    const int64_t t = blockIdx.y;
    xn += t * hc * n, gate += t * hc * n, mixed += t * n, q8 += t * (n / 32);
    const int e = blockIdx.x * kThreads + threadIdx.x;
    if (e >= n)
        return;  // whole warps
    float acc = 0.0f;
    for (int s = 0; s < hc; ++s)
        acc += xn[int64_t{s} * n + e] * sigmoid(gate[int64_t{s} * n + e]);
    const float m = acc * (1.0f / static_cast<float>(hc));
    mixed[e] = m;
    quantize_q8_block(m, q8 + e / 32);
}

__global__ void hc_combine_kernel(float* res, const float* out, const float* inject, int n, int hc) {
    const int64_t t = blockIdx.y;
    res += t * hc * n, out += t * n, inject += t * hc;
    const int e = blockIdx.x * kThreads + threadIdx.x;
    if (e >= n)
        return;
    const float o = out[e];
    for (int s = 0; s < hc; ++s)
        res[int64_t{s} * n + e] += o * (2.0f * sigmoid(inject[s] * (1.0f / static_cast<float>(hc))));
}

// rms_norm's block size (norm.cu): the same thread owns the same elements and the sum of squares is reduced in the
// same order, so the results are bitwise those of hc_combine followed by rms_norm.
constexpr int kNormThreads = 256;

constexpr int kMaxPerThread = 16;  // n <= 4,096

// One block per stream (blockIdx.x) and column (blockIdx.y). A thread loads all its elements before it sums them, in
// the same order, so its loads are in flight together instead of one after another.
__global__ void __launch_bounds__(kNormThreads)
    hc_combine_norm_kernel(float* res, const float* out, const float* inject, const float* w, float* xn, int n, int hc,
                           float eps) {
    const int64_t t = blockIdx.y;
    res += t * hc * n, xn += t * hc * n, out += t * n, inject += t * hc;
    const int64_t base = int64_t{blockIdx.x} * n;  // the stream, in the column and in the weight
    const float g = 2.0f * sigmoid(inject[blockIdx.x] * (1.0f / static_cast<float>(hc)));
    float v[kMaxPerThread];
#pragma unroll
    for (int j = 0; j < kMaxPerThread; ++j) {
        const int i = threadIdx.x + j * kNormThreads;
        if (i < n)
            v[j] = res[base + i] + out[i] * g;
    }
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kMaxPerThread; ++j) {
        const int i = threadIdx.x + j * kNormThreads;
        if (i < n) {
            res[base + i] = v[j];
            sum += v[j] * v[j];
        }
    }
    sum = block_sum(sum);
    const float scale = rsqrtf(sum / static_cast<float>(n) + eps);
#pragma unroll
    for (int j = 0; j < kMaxPerThread; ++j) {
        const int i = threadIdx.x + j * kNormThreads;
        if (i < n)
            xn[base + i] = v[j] * scale * w[base + i];
    }
}

__global__ void hc_broadcast_kernel(const float* x, float* res, int n, int hc) {
    const int64_t t = blockIdx.y;
    x += t * n, res += t * hc * n;
    const int e = blockIdx.x * kThreads + threadIdx.x;
    if (e >= n)
        return;
    for (int s = 0; s < hc; ++s)
        res[int64_t{s} * n + e] = x[e];
}

}  // namespace

void silu_scale(float* x, int n, float scale, cudaStream_t stream, int columns) {
    silu_scale_kernel<<<dim3(blocks_for(n, kThreads), columns), kThreads, 0, stream>>>(x, n, scale);
}

void hc_collapse(const float* xn, const float* gate, float* mixed, BlockQ8* q8, int n, int hc, cudaStream_t stream,
                 int columns) {
    hc_collapse_kernel<<<dim3(blocks_for(n, kThreads), columns), kThreads, 0, stream>>>(xn, gate, mixed, q8, n, hc);
}

void hc_combine_norm(float* res, const float* out, const float* inject, const float* w, float* xn, int n, int hc,
                     float eps, cudaStream_t stream, int columns) {
    hc_combine_norm_kernel<<<dim3(hc, columns), kNormThreads, 0, stream>>>(res, out, inject, w, xn, n, hc, eps);
}

void hc_combine(float* res, const float* out, const float* inject, int n, int hc, cudaStream_t stream, int columns) {
    hc_combine_kernel<<<dim3(blocks_for(n, kThreads), columns), kThreads, 0, stream>>>(res, out, inject, n, hc);
}

void hc_broadcast(const float* x, float* res, int n, int hc, cudaStream_t stream, int columns) {
    hc_broadcast_kernel<<<dim3(blocks_for(n, kThreads), columns), kThreads, 0, stream>>>(x, res, n, hc);
}

}  // namespace torch::kernels
