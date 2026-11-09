// Hyper-connection helpers for one token. The residual is hc streams of n floats (res[s * n + e], hc = 4 and
// n = 2560 in the model); the layer sees one vector mixed from them and adds its output back to every stream.
//
// All but hc_combine_norm are element-wise over e with a fixed loop over the streams, so there is no cross-thread
// reduction.
//
// Every function takes `columns` (1 to 8) tokens of a verify window, column after column: a column's stream tensor is
// hc * n floats, its vectors n floats, its inject hc floats and its Q8 output n / 32 blocks; each column's result is
// bitwise that of a one-column call on that column.
//
// Parity test: tests/parity/hc_test.cpp. Bytes moved: each operand once (hc * n floats for a stream tensor).
#pragma once

#include <cuda_runtime.h>

#include "kernels/gemv/gemv.h"

namespace torch::kernels {

// x = silu(x * scale), in place over n values.
void silu_scale(float* x, int n, float scale, cudaStream_t stream, int columns = 1);

// mixed[e] = (sum over s = 0..hc-1 in order of xn[s * n + e] * sigmoid(gate[s * n + e])) * (1.0f / hc), and mixed as
// n / 32 Q8 blocks in q8, bitwise those quantize_q8 writes (n % 32 == 0): the block's quantized products read them.
void hc_collapse(const float* xn, const float* gate, float* mixed, BlockQ8* q8, int n, int hc, cudaStream_t stream,
                 int columns = 1);

// res[s * n + e] += out[e] * (2.0f * sigmoid(inject[s] * (1.0f / hc))); inject is the raw W_inject . xn, hc values.
void hc_combine(float* res, const float* out, const float* inject, int n, int hc, cudaStream_t stream, int columns = 1);

// hc_combine, then xn = rms_norm(res) per stream with weight w (hc * n floats): what the next mixer starts with, in one
// launch, bitwise the same as hc_combine followed by rms_norm(res, n, w, n, ..., xn, n, n, hc, eps). n <= 4,096.
void hc_combine_norm(float* res, const float* out, const float* inject, const float* w, float* xn, int n, int hc,
                     float eps, cudaStream_t stream, int columns = 1);

// res[s * n + e] = x[e]: the first layer's input starts every stream equal.
void hc_broadcast(const float* x, float* res, int n, int hc, cudaStream_t stream, int columns = 1);

}  // namespace torch::kernels
