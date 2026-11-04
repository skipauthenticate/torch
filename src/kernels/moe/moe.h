// Mixture-of-experts glue for one token: the router (softmax and top-k), the SwiGLU activation and the weighted
// combination of the chosen experts with the shared expert. The expert matrix products are gemv_q8_experts.
//
// Parity test: tests/parity/moe_test.cpp. Bytes moved: a few KB each; the router is one block, the others
// element-wise.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace torch::kernels {

// Softmax over n router logits (n <= 1,024; max-subtracted, expf), then k rounds of argmax in which ties go to the
// lowest index. ids[r] is the r-th choice, weights[r] = p[ids[r]] / fmaxf(sum of the k chosen p, 6.103515625e-5f),
// the probabilities renormalized over the chosen experts. Requires k <= n.
void router_top_k(const float* logits, int n, int k, int32_t* ids, float* weights, cudaStream_t stream);

// h = silu(g) * u over n values.
void swiglu(const float* g, const float* u, float* h, int n, cudaStream_t stream);

// out[e] = (sum over r = 0..k-1 in order of w[r] * y[r * n + e]) + sigmoid(shared_gate[0]) * ys[e].
void moe_combine(const float* y, const float* w, int k, const float* ys, const float* shared_gate, float* out, int n,
                 cudaStream_t stream);

}  // namespace torch::kernels
