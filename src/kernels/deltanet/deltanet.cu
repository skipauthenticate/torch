#include "kernels/common.cuh"
#include "kernels/deltanet/deltanet.h"

namespace torch::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kHeadDim = 128;

__global__ void conv_silu_kernel(float* state, const float* x, const float* w, float* out, int channels) {
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= channels)
        return;
    const float s0 = state[c], s1 = state[channels + c], s2 = state[2 * channels + c], x3 = x[c];
    const float* wc = w + 4 * c;
    out[c] = silu(wc[0] * s0 + wc[1] * s1 + wc[2] * s2 + wc[3] * x3);
    state[c] = s1;
    state[channels + c] = s2;
    state[2 * channels + c] = x3;
}

__global__ void gdn_gates_kernel(float* beta, float* alpha, const float* dt_bias, const float* a, int heads) {
    const int h = blockIdx.x * kThreads + threadIdx.x;
    if (h >= heads)
        return;
    beta[h] = sigmoid(beta[h]);
    alpha[h] = softplus(alpha[h] + dt_bias[h]) * a[h];
}

// One block per value head; thread j owns column j of the state, so the column stays in registers between the two
// passes over i and no reduction crosses threads.
__global__ void __launch_bounds__(kHeadDim)
    gdn_step_kernel(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
                    float* out, int k_heads, float scale) {
    const int h = blockIdx.x, j = threadIdx.x;
    const int kh = h % k_heads;
    __shared__ float ks[kHeadDim], qs[kHeadDim];
    ks[j] = k[kh * kHeadDim + j];
    qs[j] = q[kh * kHeadDim + j];
    __syncthreads();

    float* s_head = state + int64_t{h} * kHeadDim * kHeadDim;
    float s[kHeadDim];
#pragma unroll
    for (int i = 0; i < kHeadDim; ++i)
        s[i] = s_head[i * kHeadDim + j];
    const float eg = expf(g[h]);
    float kv = 0.0f;
#pragma unroll
    for (int i = 0; i < kHeadDim; ++i)
        kv += s[i] * ks[i];
    const float d = (v[h * kHeadDim + j] - eg * kv) * beta[h];
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

void conv_silu(float* state, const float* x, const float* w, float* out, int channels, cudaStream_t stream) {
    conv_silu_kernel<<<blocks_for(channels, kThreads), kThreads, 0, stream>>>(state, x, w, out, channels);
}

void gdn_gates(float* beta, float* alpha, const float* dt_bias, const float* a, int heads, cudaStream_t stream) {
    gdn_gates_kernel<<<blocks_for(heads, kThreads), kThreads, 0, stream>>>(beta, alpha, dt_bias, a, heads);
}

void gdn_step(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
              float* out, int k_heads, int v_heads, float scale, cudaStream_t stream) {
    gdn_step_kernel<<<v_heads, kHeadDim, 0, stream>>>(state, q, k, v, g, beta, out, k_heads, scale);
}

}  // namespace torch::kernels
