// `columns` runs a verify window's tokens in one launch, column after column (router logits [T][n], ids and weights
// [T][k], y [T][k][n], ys and out [T][n], shared_gate [T]); each column's result is bitwise a one-column call's.
//
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
// the probabilities renormalized over the chosen experts. Requires k <= n and k <= 32.
void router_top_k(const float* logits, int n, int k, int32_t* ids, float* weights, cudaStream_t stream,
                  int columns = 1);

// The pointer table gemv_q8_experts reads (k slots of gate, then up, then down), for experts that are all resident
// back to back, as the draft layer's are: table[s] = gate + ids[s] * gate_up_slice, table[k + s] = up + ...,
// table[2k + s] = down + ids[s] * down_slice, for s < k <= 32.
struct ResidentExperts {
    const uint8_t* gate;
    const uint8_t* up;
    const uint8_t* down;
    uint64_t gate_up_slice, down_slice;  // bytes of one expert
};
void expert_table(const int32_t* ids, int k, ResidentExperts experts, const uint8_t** table, cudaStream_t stream);

// h = silu(g) * u over n values.
void swiglu(const float* g, const float* u, float* h, int n, cudaStream_t stream);

// out[e] = (sum over r = 0..k-1 in order of w[r] * y[r * n + e]) + sigmoid(shared_gate[0]) * ys[e].
void moe_combine(const float* y, const float* w, int k, const float* ys, const float* shared_gate, float* out, int n,
                 cudaStream_t stream, int columns = 1);

}  // namespace torch::kernels
