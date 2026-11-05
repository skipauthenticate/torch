#include "kernels/common.cuh"
#include "kernels/deltanet/deltanet.h"

namespace torch::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kHeadDim = 128;

// One channel's convolution and SiLU, shifting its state.
__device__ __forceinline__ float conv_channel(float* state, const float* x, const float* w, int channels, int c) {
    const float s0 = state[c], s1 = state[channels + c], s2 = state[2 * channels + c], x3 = x[c];
    const float* wc = w + 4 * c;
    state[c] = s1;
    state[channels + c] = s2;
    state[2 * channels + c] = x3;
    return silu(wc[0] * s0 + wc[1] * s1 + wc[2] * s2 + wc[3] * x3);
}

// The first normed / 128 blocks each take one q or k head: threads 0-127 convolve its channels, then the block takes
// their L2 norm as l2_norm does (a block of kThreads with one element per thread, the same block_sum), so the result is
// bitwise that of conv_silu followed by l2_norm. The other blocks convolve the value channels, one per thread.
__global__ void __launch_bounds__(kThreads)
    conv_silu_l2_kernel(float* state, const float* x, const float* w, float* out, int channels, int normed, float eps) {
    const int heads = normed / kHeadDim;
    if (static_cast<int>(blockIdx.x) < heads) {
        const int c = blockIdx.x * kHeadDim + threadIdx.x;
        const float v = threadIdx.x < kHeadDim ? conv_channel(state, x, w, channels, c) : 0.0f;
        float sum = 0.0f;
        sum += v * v;
        sum = block_sum(sum);
        const float n = static_cast<float>(kHeadDim);
        const float scale = rsqrtf(sum / n + eps / n);
        if (threadIdx.x < kHeadDim)
            out[c] = (v * scale) * (1.0f / sqrtf(n));
        return;
    }
    const int c = normed + (blockIdx.x - heads) * kThreads + threadIdx.x;
    if (c < channels)
        out[c] = conv_channel(state, x, w, channels, c);
}

// One block per value head; thread j owns column j of the state, so the column stays in registers between the two
// passes over i and no reduction crosses threads.
__global__ void __launch_bounds__(kHeadDim)
    gdn_step_kernel(float* state, const float* q, const float* k, const float* v, float* alpha, float* beta,
                    const float* dt_bias, const float* a, float* out, int k_heads, float scale) {
    const int h = blockIdx.x, j = threadIdx.x;
    const int kh = h % k_heads;
    __shared__ float ks[kHeadDim], qs[kHeadDim];
    ks[j] = k[kh * kHeadDim + j];
    qs[j] = q[kh * kHeadDim + j];
    // The gates: every thread computes them from the raw values, read before the barrier; thread 0 stores them after.
    const float b = sigmoid(beta[h]);
    const float g = softplus(alpha[h] + dt_bias[h]) * a[h];
    __syncthreads();
    if (j == 0) {
        beta[h] = b;
        alpha[h] = g;
    }

    float* s_head = state + int64_t{h} * kHeadDim * kHeadDim;
    float s[kHeadDim];
#pragma unroll
    for (int i = 0; i < kHeadDim; ++i)
        s[i] = s_head[i * kHeadDim + j];
    const float eg = expf(g);
    float kv = 0.0f;
#pragma unroll
    for (int i = 0; i < kHeadDim; ++i)
        kv += s[i] * ks[i];
    const float d = (v[h * kHeadDim + j] - eg * kv) * b;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < kHeadDim; ++i) {
        s[i] = eg * s[i] + ks[i] * d;
        acc += s[i] * qs[i];
        s_head[i * kHeadDim + j] = s[i];
    }
    out[h * kHeadDim + j] = acc * scale;
}

}  // namespace

void conv_silu_l2(float* state, const float* x, const float* w, float* out, int channels, int normed, float eps,
                  cudaStream_t stream) {
    const int blocks = normed / kHeadDim + blocks_for(channels - normed, kThreads);
    conv_silu_l2_kernel<<<blocks, kThreads, 0, stream>>>(state, x, w, out, channels, normed, eps);
}

void gdn_step(float* state, const float* q, const float* k, const float* v, float* alpha, float* beta,
              const float* dt_bias, const float* a, float* out, int k_heads, int v_heads, float scale,
              cudaStream_t stream) {
    gdn_step_kernel<<<v_heads, kHeadDim, 0, stream>>>(state, q, k, v, alpha, beta, dt_bias, a, out, k_heads, scale);
}

}  // namespace torch::kernels
