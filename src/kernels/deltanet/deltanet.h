// Gated DeltaNet pieces for one token: the causal convolution over the q|k|v channels, the gates, and the
// recurrence on the 128 x 128 state of each value head.
//
// Parity test: tests/parity/deltanet_test.cpp (several consecutive tokens, so state carries over).
// Bytes moved: conv_silu and gdn_gates are a few KB. gdn_step reads and writes the state once, v_heads * 64 KB
// (3 MB at 48 heads); it is the one memory-bound kernel here.
#pragma once

#include <cuda_runtime.h>

namespace torch::kernels {

// Depthwise causal convolution with 4 taps, then SiLU. state is [3][channels] (the 3 previous inputs, oldest row
// first), w is [channels][4] (w[4 * c + j]), x is this token's input. out[c] = silu(w0 * s0 + w1 * s1 + w2 * s2 + w3 *
// x[c]) summed in that order; then the state shifts by one row and x becomes the newest row. out may not alias x.
void conv_silu(float* state, const float* x, const float* w, float* out, int channels, cudaStream_t stream);

// In place, per head h: beta = sigmoid(beta); alpha = softplus(alpha + dt_bias[h]) * a[h] (a is -exp(A_log)). The
// result in alpha is the log of the state decay, the `g` that gdn_step takes.
void gdn_gates(float* beta, float* alpha, const float* dt_bias, const float* a, int heads, cudaStream_t stream);

// One step of the gated delta rule. state is [v_heads][128][128] with S[h][i][j], i the key dimension and j the value
// dimension; q and k are [k_heads][128], v and out [v_heads][128], g and beta [v_heads]. Value head h uses key head
// h % k_heads (the key heads are tiled across the value heads, not interleaved). Per head, with eg = expf(g[h]):
//   kv_j = sum_i S[i][j] * k[i];  d_j = (v_j - eg * kv_j) * beta[h];  S[i][j] = eg * S[i][j] + k[i] * d_j;
//   out_j = (sum_i S[i][j] * q[i]) * scale.
// The sums run over i in increasing order.
void gdn_step(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
              float* out, int k_heads, int v_heads, float scale, cudaStream_t stream);

}  // namespace torch::kernels
