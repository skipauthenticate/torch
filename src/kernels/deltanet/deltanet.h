// Gated DeltaNet pieces for one token: the causal convolution over the q|k|v channels, the gates, and the
// recurrence on the 128 x 128 state of each value head.
//
// Parity test: tests/parity/deltanet_test.cpp (several consecutive tokens, so state carries over).
// Bytes moved: conv_silu_l2 is ~0.2 MB (state, weights and channels). gdn_step reads and writes the state once,
// v_heads * 64 KB (3 MB at 48 heads); it is the one memory-bound kernel here.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace torch::kernels {

// Depthwise causal convolution with 4 taps, then SiLU. state is [3][channels] (the 3 previous inputs, oldest row
// first), w is [channels][4] (w[4 * c + j]), x is this token's input. out[c] = silu(w0 * s0 + w1 * s1 + w2 * s2 + w3 *
// x[c]) summed in that order; then the state shifts by one row and x becomes the newest row. The first `normed`
// channels (the q and k heads, a multiple of 128) are then L2-normalized in groups of 128 as l2_norm does
// (norm.h). out may not alias x.
void conv_silu_l2(float* state, const float* x, const float* w, float* out, int channels, int normed, float eps,
                  cudaStream_t stream);

// One step of the gated delta rule. state is [v_heads][128][128] with S[h][i][j], i the key dimension and j the value
// dimension; q and k are [k_heads][128], v and out [v_heads][128], alpha and beta [v_heads]. Value head h uses key
// head h % k_heads (the key heads are tiled across the value heads, not interleaved). First the gates, stored back in
// place: beta[h] = sigmoid(beta[h]) and g = alpha[h] = softplus(alpha[h] + dt_bias[h]) * a[h] (a is -exp(A_log), so g
// is the log of the state decay). Then per head, with eg = expf(g):
//   kv_j = sum_i S[i][j] * k[i];  d_j = (v_j - eg * kv_j) * beta[h];  S[i][j] = eg * S[i][j] + k[i] * d_j;
//   out_j = (sum_i S[i][j] * q[i]) * scale.
// The sums run over i in increasing order.
void gdn_step(float* state, const float* q, const float* k, const float* v, float* alpha, float* beta,
              const float* dt_bias, const float* a, float* out, int k_heads, int v_heads, float scale,
              cudaStream_t stream);

// ---- Windows of 1 to 8 columns (a verify window). Column t is the token at position p + t. Each column's result is
// bitwise that of the one-token kernel on that column (tests/parity/deltanet_test.cpp); nothing here writes the
// sequence state except gdn_window's output state and conv_commit.

// conv_silu_l2 for `columns` columns: x and out are [columns][channels]. Column t convolves the three history rows of
// `state` (read only) and the earlier columns of x, so the history stays as it was before the window.
void conv_silu_l2_window(const float* state, const float* x, const float* w, float* out, int channels, int normed,
                         float eps, int columns, cudaStream_t stream);

// Makes `state` the history after `accepted` (1 to 8) columns: the last three of its old rows followed by the first
// `accepted` columns of x ([columns][channels], the window's inputs). Updates in place.
void conv_commit(float* state, const float* x, int channels, int accepted, cudaStream_t stream);

// gdn_step for `columns` columns in one launch, reading the sequence state state_in (before the window) and writing
// state_out (after the last column; the two must differ). q, k and v are column 0's pointers into conv_silu_l2_window's
// output and advance by x_stride floats per column; alpha and beta are the raw gate inputs [columns][v_heads]. Unlike
// gdn_step the inputs are not overwritten: the gates (beta, g) per column and head go to gates[(t * v_heads + h) * 2 +
// {0, 1}] (nullptr to skip). out is [columns][v_heads][128]. Replaying after a rejection is this launch over the first
// k + 1 columns from the same state_in.
void gdn_window(const float* state_in, float* state_out, const float* q, const float* k, const float* v,
                int64_t x_stride, const float* alpha, const float* beta, const float* dt_bias, const float* a,
                float* gates, float* out, int k_heads, int v_heads, float scale, int columns, cudaStream_t stream);

// The same two with the count in device memory (an int32 holding 1 to 8 that an earlier kernel wrote, e.g. the accept
// step), read once on the device, so a replay and a commit can be captured in a graph. The replay also takes `window`,
// the width of the window being replayed, and returns at once when *columns == window: a fully accepted window needs
// no replay because state_out already holds the state after all its columns (from the window's own gdn_window).
void conv_commit(float* state, const float* x, int channels, const int32_t* accepted, cudaStream_t stream);
void gdn_window(const float* state_in, float* state_out, const float* q, const float* k, const float* v,
                int64_t x_stride, const float* alpha, const float* beta, const float* dt_bias, const float* a,
                float* gates, float* out, int k_heads, int v_heads, float scale, const int32_t* columns, int window,
                cudaStream_t stream);

}  // namespace torch::kernels
