// Full attention for one decode token: 24 query heads and 2 KV heads of 256 dimensions, a rotary embedding on the
// first 64 dimensions of each head, and an F16 KV cache. Positions come from device memory (`pos`) so a captured CUDA
// graph can replay them.
//
// The KV cache is [max_ctx][kv_heads * 256] F16 for keys and again for values; row p holds position p.
//
// Parity test: tests/parity/attention_test.cpp. Bytes moved by attention_decode: the K and V rows 0..*pos once per
// KV head group (2 * (pos + 1) * kv_heads * 256 * 2 bytes, read by each of the heads / kv_heads query heads, mostly
// from L2).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace torch::kernels {

// NeoX-style rotary embedding. Head h is x + h * head_stride; pairs (i, i + n_rot / 2) for
// i < n_rot / 2 rotate by theta = (float)*pos * powf(powf(base, -2.0f / n_rot), (float)i). Dimensions >= n_rot are
// untouched.
void rope_neox(float* x, int64_t head_stride, int heads, int n_rot, const int32_t* pos, float base,
               cudaStream_t stream);

// Writes row *pos of both caches from k and v (n = kv_heads * 256 floats each), rounded to F16 (round to nearest
// even).
void store_kv(const float* k, const float* v, uint16_t* k_cache, uint16_t* v_cache, int n, const int32_t* pos,
              cudaStream_t stream);

// Dense causal attention of the query at position *pos over cache rows 0..*pos. Query head h is q + h * q_stride and
// its gate gate + h * q_stride (the projection interleaves [q | gate] per head, so q_stride is 512 and gate = q +
// 256); it reads KV head h / (heads / kv_heads). out + h * 256 = softmax(q . k * scale) v * sigmoid(gate), in F32
// from F16 cache values. The scores live in dynamic shared memory sized for max_ctx (the launcher opts in above
// 48 KB); *pos must be < max_ctx. The softmax subtracts the maximum; its sum and the weighted sum of V run in a
// fixed order.
void attention_decode(const float* q, const float* gate, int64_t q_stride, const uint16_t* k_cache,
                      const uint16_t* v_cache, float* out, const int32_t* pos, int heads, int kv_heads, float scale,
                      int max_ctx, cudaStream_t stream);

}  // namespace torch::kernels
