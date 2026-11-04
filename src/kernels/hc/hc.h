// Hyper-connection helpers for one token. The residual is hc streams of n floats (res[s * n + e], hc = 4 and
// n = 2560 in the model); the layer sees one vector mixed from them and adds its output back to every stream.
//
// All four are element-wise over e with a fixed loop over the streams, so there is no cross-thread reduction.
//
// Parity test: tests/parity/hc_test.cpp. Bytes moved: each operand once (hc * n floats for a stream tensor).
#pragma once

#include <cuda_runtime.h>

namespace torch::kernels {

// x = silu(x * scale), in place over n values.
void silu_scale(float* x, int n, float scale, cudaStream_t stream);

// mixed[e] = (sum over s = 0..hc-1 in order of xn[s * n + e] * sigmoid(gate[s * n + e])) * (1.0f / hc).
void hc_collapse(const float* xn, const float* gate, float* mixed, int n, int hc, cudaStream_t stream);

// res[s * n + e] += out[e] * (2.0f * sigmoid(inject[s] * (1.0f / hc))); inject is the raw W_inject . xn, hc values.
void hc_combine(float* res, const float* out, const float* inject, int n, int hc, cudaStream_t stream);

// res[s * n + e] = x[e]: the first layer's input starts every stream equal.
void hc_broadcast(const float* x, float* res, int n, int hc, cudaStream_t stream);

}  // namespace torch::kernels
