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

// The window form of conv_silu_l2: grid (blocks, columns). Column t convolves the inputs at window positions t - 3 ..
// t, where negative positions are the three history rows (oldest first) and the others are columns of x; the history is
// only read. Same expressions and block layout as conv_silu_l2_kernel, so column t is bitwise the one-token result.
__device__ __forceinline__ float conv_channel_window(const float* state, const float* x, const float* w, int channels,
                                                     int c, int t) {
    const auto tap = [&](int u) {
        return u < 0 ? state[(3 + u) * int64_t{channels} + c] : x[u * int64_t{channels} + c];
    };
    const float s0 = tap(t - 3), s1 = tap(t - 2), s2 = tap(t - 1), x3 = tap(t);
    const float* wc = w + 4 * c;
    return silu(wc[0] * s0 + wc[1] * s1 + wc[2] * s2 + wc[3] * x3);
}

__global__ void __launch_bounds__(kThreads)
    conv_silu_l2_window_kernel(const float* state, const float* x, const float* w, float* out, int channels, int normed,
                               float eps) {
    const int t = blockIdx.y;
    float* const out_t = out + t * int64_t{channels};
    const int heads = normed / kHeadDim;
    if (static_cast<int>(blockIdx.x) < heads) {
        const int c = blockIdx.x * kHeadDim + threadIdx.x;
        const float v = threadIdx.x < kHeadDim ? conv_channel_window(state, x, w, channels, c, t) : 0.0f;
        float sum = 0.0f;
        sum += v * v;
        sum = block_sum(sum);
        const float n = static_cast<float>(kHeadDim);
        const float scale = rsqrtf(sum / n + eps / n);
        if (threadIdx.x < kHeadDim)
            out_t[c] = (v * scale) * (1.0f / sqrtf(n));
        return;
    }
    const int c = normed + (blockIdx.x - heads) * kThreads + threadIdx.x;
    if (c < channels)
        out_t[c] = conv_channel_window(state, x, w, channels, c, t);
}

// New history = the last three of (old rows ++ the first `accepted` columns of x); one thread per channel reads its old
// rows before writing, so it updates in place.
__global__ void __launch_bounds__(kThreads)
    conv_commit_kernel(float* state, const float* x, int channels, int accepted, const int32_t* accepted_ptr) {
    if (accepted_ptr != nullptr)
        accepted = *accepted_ptr;
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= channels)
        return;
    float next[3];
    for (int r = 0; r < 3; ++r) {
        const int u = accepted - 3 + r;
        next[r] = u < 0 ? state[(3 + u) * int64_t{channels} + c] : x[u * int64_t{channels} + c];
    }
    for (int r = 0; r < 3; ++r)
        state[r * int64_t{channels} + c] = next[r];
}

// The window form of gdn_step: one block per value head walks the columns in order with the state column in registers,
// reading state_in once and writing state_out once. Column strides: q, k and v advance by x_stride floats per column
// (the layout of conv_silu_l2_window's output), alpha and beta by v_heads, out by v_heads * 128. The gates go to
// gates[(t * v_heads + h) * 2 + {0, 1}] = {beta, g}; the inputs are left untouched. Each column's arithmetic is
// gdn_step_kernel's, so it is bitwise the one-token result. A long run of columns (a prefill chunk) loads column t +
// 1's inputs into registers while column t is computed, and reads k and q from shared memory four values at a time.
__global__ void __launch_bounds__(kHeadDim)
    gdn_window_kernel(const float* state_in, float* state_out, const float* q, const float* k, const float* v,
                      int64_t x_stride, const float* alpha, const float* beta, const float* dt_bias, const float* a,
                      float* gates, float* out, int k_heads, int v_heads, float scale, int columns,
                      const int32_t* columns_ptr, int window) {
    if (columns_ptr != nullptr)
        columns = *columns_ptr;
    if (columns == window)
        return;  // a fully accepted window: state_out already holds the state after all its columns
    const int h = blockIdx.x, j = threadIdx.x;
    const int kh = h % k_heads;
    __shared__ __align__(16) float ks[2][kHeadDim], qs[2][kHeadDim];
    const float* s_in = state_in + int64_t{h} * kHeadDim * kHeadDim;
    float s[kHeadDim];
#pragma unroll
    for (int i = 0; i < kHeadDim; ++i)
        s[i] = s_in[i * kHeadDim + j];
    // column t's inputs, loaded one column ahead
    float k_next = 0.0f, q_next = 0.0f, v_next = 0.0f, alpha_next = 0.0f, beta_next = 0.0f;
    const auto load = [&](int t) {
        k_next = k[t * x_stride + kh * kHeadDim + j];
        q_next = q[t * x_stride + kh * kHeadDim + j];
        v_next = v[t * x_stride + h * kHeadDim + j];
        alpha_next = alpha[t * int64_t{v_heads} + h];
        beta_next = beta[t * int64_t{v_heads} + h];
    };
    if (columns > 0)
        load(0);
    for (int t = 0; t < columns; ++t) {
        const int buf = t & 1;
        ks[buf][j] = k_next, qs[buf][j] = q_next;  // this buffer's last readers finished column t - 2
        const float v_t = v_next, alpha_t = alpha_next, beta_t = beta_next;
        __syncthreads();  // column t's k and q are in shared memory
        if (t + 1 < columns)
            load(t + 1);  // in flight while column t is computed
        const float b = sigmoid(beta_t);
        const float g = softplus(alpha_t + dt_bias[h]) * a[h];
        if (j == 0 && gates != nullptr) {
            gates[(t * int64_t{v_heads} + h) * 2] = b;
            gates[(t * int64_t{v_heads} + h) * 2 + 1] = g;
        }
        const float eg = expf(g);
        const float* ksb = ks[buf];
        const float* qsb = qs[buf];
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < kHeadDim; ++i)
            kv += s[i] * ksb[i];
        const float d = (v_t - eg * kv) * b;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < kHeadDim; ++i) {
            s[i] = eg * s[i] + ksb[i] * d;
            acc += s[i] * qsb[i];
        }
        out[(t * int64_t{v_heads} + h) * kHeadDim + j] = acc * scale;
    }
    float* s_out = state_out + int64_t{h} * kHeadDim * kHeadDim;
#pragma unroll
    for (int i = 0; i < kHeadDim; ++i)
        s_out[i * kHeadDim + j] = s[i];
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

void conv_silu_l2_window(const float* state, const float* x, const float* w, float* out, int channels, int normed,
                         float eps, int columns, cudaStream_t stream) {
    const int blocks = normed / kHeadDim + blocks_for(channels - normed, kThreads);
    conv_silu_l2_window_kernel<<<dim3(blocks, columns), kThreads, 0, stream>>>(state, x, w, out, channels, normed, eps);
}

void conv_commit(float* state, const float* x, int channels, int accepted, cudaStream_t stream) {
    conv_commit_kernel<<<blocks_for(channels, kThreads), kThreads, 0, stream>>>(state, x, channels, accepted, nullptr);
}

void conv_commit(float* state, const float* x, int channels, const int32_t* accepted, cudaStream_t stream) {
    conv_commit_kernel<<<blocks_for(channels, kThreads), kThreads, 0, stream>>>(state, x, channels, 0, accepted);
}

void gdn_window(const float* state_in, float* state_out, const float* q, const float* k, const float* v,
                int64_t x_stride, const float* alpha, const float* beta, const float* dt_bias, const float* a,
                float* gates, float* out, int k_heads, int v_heads, float scale, int columns, cudaStream_t stream) {
    gdn_window_kernel<<<v_heads, kHeadDim, 0, stream>>>(state_in, state_out, q, k, v, x_stride, alpha, beta, dt_bias, a,
                                                        gates, out, k_heads, v_heads, scale, columns, nullptr, -1);
}

void gdn_window(const float* state_in, float* state_out, const float* q, const float* k, const float* v,
                int64_t x_stride, const float* alpha, const float* beta, const float* dt_bias, const float* a,
                float* gates, float* out, int k_heads, int v_heads, float scale, const int32_t* columns, int window,
                cudaStream_t stream) {
    gdn_window_kernel<<<v_heads, kHeadDim, 0, stream>>>(state_in, state_out, q, k, v, x_stride, alpha, beta, dt_bias, a,
                                                        gates, out, k_heads, v_heads, scale, 0, columns, window);
}

}  // namespace torch::kernels
