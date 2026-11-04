#include "kernels/common.cuh"
#include "kernels/norm/norm.h"

namespace torch::kernels {
namespace {

constexpr int kThreads = 256;

// One block per group. kL2 selects the L2 variant, which has no weight or gate.
template <bool kL2>
__global__ void __launch_bounds__(kThreads)
    norm_kernel(const float* x, int64_t x_stride, const float* w, int64_t w_stride, const float* gate,
                int64_t gate_stride, float* out, int64_t out_stride, int n, float eps) {
    const int g = blockIdx.x;
    x += g * x_stride;
    out += g * out_stride;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += kThreads)
        sum += x[i] * x[i];
    sum = block_sum(sum);
    const float scale = kL2 ? rsqrtf(sum / static_cast<float>(n) + eps / static_cast<float>(n))
                            : rsqrtf(sum / static_cast<float>(n) + eps);
    if constexpr (kL2) {
        const float unit = 1.0f / sqrtf(static_cast<float>(n));
        for (int i = threadIdx.x; i < n; i += kThreads)
            out[i] = (x[i] * scale) * unit;
    } else {
        if (w != nullptr)
            w += g * w_stride;
        if (gate != nullptr)
            gate += g * gate_stride;
        // each thread reads and writes only its own elements, so out == x is safe
        for (int i = threadIdx.x; i < n; i += kThreads) {
            float v = x[i] * scale;
            if (w != nullptr)
                v = v * w[i];
            if (gate != nullptr)
                v = v * sigmoid(gate[i]);
            out[i] = v;
        }
    }
}

}  // namespace

void rms_norm(const float* x, int64_t x_stride, const float* w, int64_t w_stride, const float* gate,
              int64_t gate_stride, float* out, int64_t out_stride, int n, int groups, float eps, cudaStream_t stream) {
    norm_kernel<false>
        <<<groups, kThreads, 0, stream>>>(x, x_stride, w, w_stride, gate, gate_stride, out, out_stride, n, eps);
}

void l2_norm(const float* x, int64_t x_stride, float* out, int64_t out_stride, int n, int groups, float eps,
             cudaStream_t stream) {
    norm_kernel<true><<<groups, kThreads, 0, stream>>>(x, x_stride, nullptr, 0, nullptr, 0, out, out_stride, n, eps);
}

}  // namespace torch::kernels
