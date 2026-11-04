#include "core/cuda_check.h"
#include "kernels/attention/attention.h"
#include "kernels/common.cuh"

namespace torch::kernels {
namespace {

constexpr int kHeadDim = 256;  // one thread per dimension in attention_decode
constexpr int kThreads = 256;
constexpr int kDefaultSharedBytes = 48 * 1024;

// Block h, thread i < n_rot / 2.
__global__ void rope_kernel(float* x, int64_t head_stride, int n_rot, const int32_t* pos, float base) {
    const int i = threadIdx.x;
    float* xh = x + blockIdx.x * head_stride;
    const float theta_scale = powf(base, -2.0f / static_cast<float>(n_rot));
    const float theta = static_cast<float>(*pos) * powf(theta_scale, static_cast<float>(i));
    const float c = cosf(theta), s = sinf(theta);
    const int half_rot = n_rot / 2;
    const float x0 = xh[i], x1 = xh[i + half_rot];
    xh[i] = x0 * c - x1 * s;
    xh[i + half_rot] = x0 * s + x1 * c;
}

__global__ void store_kv_kernel(const float* k, const float* v, uint16_t* k_cache, uint16_t* v_cache, int n,
                                const int32_t* pos) {
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n)
        return;
    const int64_t at = int64_t{*pos} * n + i;
    k_cache[at] = __half_as_ushort(__float2half_rn(k[i]));
    v_cache[at] = __half_as_ushort(__float2half_rn(v[i]));
}

// One block per query head, thread d owns output dimension d.
__global__ void __launch_bounds__(kThreads)
    attention_kernel(const float* q, const float* gate, int64_t q_stride, const uint16_t* k_cache,
                     const uint16_t* v_cache, float* out, const int32_t* pos, int heads, int kv_heads, float scale) {
    extern __shared__ float scores[];
    __shared__ float query[kHeadDim];
    const int h = blockIdx.x, d = threadIdx.x;
    const int lane = d % kWarp, warp = d / kWarp;
    const int rows = *pos + 1;
    const int64_t row_len = int64_t{kv_heads} * kHeadDim;
    const int64_t kv_offset = int64_t{h / (heads / kv_heads)} * kHeadDim;

    query[d] = q[h * q_stride + d];
    __syncthreads();
    // one warp per cache row at a time; lane l sums dimensions l, l + 32, ... and the lanes combine in a butterfly
    for (int p = warp; p < rows; p += kThreads / kWarp) {
        const uint16_t* key = k_cache + p * row_len + kv_offset;
        float dot = 0.0f;
        for (int i = lane; i < kHeadDim; i += kWarp)
            dot += query[i] * f16_to_f32(key[i]);
        dot = warp_sum(dot);
        if (lane == 0)
            scores[p] = dot * scale;
    }
    __syncthreads();
    float peak = -INFINITY;
    for (int p = d; p < rows; p += kThreads)
        peak = fmaxf(peak, scores[p]);
    peak = block_max(peak);
    float sum = 0.0f;
    for (int p = d; p < rows; p += kThreads) {
        scores[p] = expf(scores[p] - peak);
        sum += scores[p];
    }
    sum = block_sum(sum);  // its leading __syncthreads also publishes the scores
    float acc = 0.0f;
    for (int p = 0; p < rows; ++p)
        acc += scores[p] * f16_to_f32(v_cache[p * row_len + kv_offset + d]);
    out[h * kHeadDim + d] = acc / sum * sigmoid(gate[h * q_stride + d]);
}

}  // namespace

void rope_neox(float* x, int64_t head_stride, int heads, int n_rot, const int32_t* pos, float base,
               cudaStream_t stream) {
    rope_kernel<<<heads, n_rot / 2, 0, stream>>>(x, head_stride, n_rot, pos, base);
}

void store_kv(const float* k, const float* v, uint16_t* k_cache, uint16_t* v_cache, int n, const int32_t* pos,
              cudaStream_t stream) {
    store_kv_kernel<<<blocks_for(n, kThreads), kThreads, 0, stream>>>(k, v, k_cache, v_cache, n, pos);
}

void attention_decode(const float* q, const float* gate, int64_t q_stride, const uint16_t* k_cache,
                      const uint16_t* v_cache, float* out, const int32_t* pos, int heads, int kv_heads, float scale,
                      int max_ctx, cudaStream_t stream) {
    const size_t shared = size_t{static_cast<size_t>(max_ctx)} * sizeof(float);
    if (shared > kDefaultSharedBytes) {
        TORCH_CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(shared)));
    }
    attention_kernel<<<heads, kThreads, shared, stream>>>(q, gate, q_stride, k_cache, v_cache, out, pos, heads,
                                                          kv_heads, scale);
}

}  // namespace torch::kernels
