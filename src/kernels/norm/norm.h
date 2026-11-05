// Normalizations over groups of n floats, for one token. Group g reads x + g * x_stride and writes out + g *
// out_stride.
//
// rms_norm: mean = sum(x^2) / n, scale = rsqrtf(mean + eps), out = x * scale, then the multiplications by the weight
// and by sigmoid(gate) as separate roundings, left to right. The sum of squares is a fixed-order block reduction (a
// thread sums elements t, t + 256, ... in that order), so the result never depends on timing. out may equal x.
//
// Parity test: tests/parity/norm_test.cpp. Bytes moved: x read twice (the second time from L2), plus weight, gate and
// the output once.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/gemv/gemv.h"

namespace torch::kernels {

// out = (x * rsqrtf(sum(x^2) / n + eps)) * w_g[i] * sigmoid(gate_g[i]), with w_g = w + g * w_stride (w_stride 0 shares
// one weight vector; w == nullptr means no weight) and gate_g = gate + g * gate_stride (gate == nullptr means none).
void rms_norm(const float* x, int64_t x_stride, const float* w, int64_t w_stride, const float* gate,
              int64_t gate_stride, float* out, int64_t out_stride, int n, int groups, float eps, cudaStream_t stream);

// rms_norm that also writes each group's output as n / 32 Q8 blocks, group after group, bitwise those quantize_q8
// writes: for groups that feed a quantized product. Requires n <= 256 and n % 32 == 0.
void rms_norm_q8(const float* x, int64_t x_stride, const float* w, int64_t w_stride, const float* gate,
                 int64_t gate_stride, float* out, int64_t out_stride, BlockQ8* q8, int n, int groups, float eps,
                 cudaStream_t stream);

// out = (x * rsqrtf(sum(x^2) / n + eps / n)) * (1.0f / sqrtf(n)): the unit-length vector with eps inside the root,
// computed as an RMS norm with eps / n followed by a scale of 1 / sqrt(n), in that order of roundings.
void l2_norm(const float* x, int64_t x_stride, float* out, int64_t out_stride, int n, int groups, float eps,
             cudaStream_t stream);

}  // namespace torch::kernels
