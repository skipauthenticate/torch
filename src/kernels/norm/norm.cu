#include "kernels/common.cuh"
#include "kernels/gemv/q8.cuh"
#include "kernels/norm/norm.h"

namespace torch::kernels {
namespace {

constexpr int kThreads = 256;

// One block per group. kL2 selects the L2 variant, which has no weight or gate. With q8 (n <= kThreads, n % 32 == 0)
// thread t holds element t, and each warp also writes its 32 outputs as a Q8 block.
template <bool kL2>
__global__ void __launch_bounds__(kThreads)
    norm_kernel(const float* x, int64_t x_stride, const float* w, int64_t w_stride, const float* gate,
                int64_t gate_stride, float* out, int64_t out_stride, int n, float eps, BlockQ8* q8 = nullptr) {
    const int g = blockIdx.x;
    // Window column blockIdx.y: its groups follow the previous column's (each column's tensor is groups x stride).
    const int64_t column_groups = int64_t{blockIdx.y} * gridDim.x;
    x += (column_groups + g) * x_stride;
    out += (column_groups + g) * out_stride;
    if (gate != nullptr)
        gate += column_groups * gate_stride;
    if (q8 != nullptr)
        q8 += column_groups * (n / 32);
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
        float last = 0.0f;
        for (int i = threadIdx.x; i < n; i += kThreads) {
            float v = x[i] * scale;
            if (w != nullptr)
                v = v * w[i];
            if (gate != nullptr)
                v = v * sigmoid(gate[i]);
            out[i] = v;
            last = v;
        }
        if (q8 != nullptr && static_cast<int>(threadIdx.x) < n)  // whole warps
            quantize_q8_block(last, q8 + int64_t{g} * (n / 32) + threadIdx.x / kWarp);
    }
}

}  // namespace

void rms_norm(const float* x, int64_t x_stride, const float* w, int64_t w_stride, const float* gate,
              int64_t gate_stride, float* out, int64_t out_stride, int n, int groups, float eps, cudaStream_t stream,
              int columns) {
    norm_kernel<false><<<dim3(groups, columns), kThreads, 0, stream>>>(x, x_stride, w, w_stride, gate, gate_stride, out,
                                                                       out_stride, n, eps);
}

void rms_norm_q8(const float* x, int64_t x_stride, const float* w, int64_t w_stride, const float* gate,
                 int64_t gate_stride, float* out, int64_t out_stride, BlockQ8* q8, int n, int groups, float eps,
                 cudaStream_t stream, int columns) {
    norm_kernel<false><<<dim3(groups, columns), kThreads, 0, stream>>>(x, x_stride, w, w_stride, gate, gate_stride, out,
                                                                       out_stride, n, eps, q8);
}

void l2_norm(const float* x, int64_t x_stride, float* out, int64_t out_stride, int n, int groups, float eps,
             cudaStream_t stream, int columns) {
    norm_kernel<true>
        <<<dim3(groups, columns), kThreads, 0, stream>>>(x, x_stride, nullptr, 0, nullptr, 0, out, out_stride, n, eps);
}

}  // namespace torch::kernels
