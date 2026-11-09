// Full attention for one decode token: 24 query heads and 2 KV heads of 256 dimensions, a rotary embedding on the
// first 64 dimensions of each head, and an F16 KV cache. Positions come from device memory (`pos`) so a captured CUDA
// graph can replay them.
//
// The KV cache is [max_ctx][kv_heads * 256] F16 for keys and again for values; row p holds position p.
//
// Parity test: tests/parity/attention_test.cpp. Bytes moved by attention_decode: the K and V rows 0..*pos once
// (2 * (pos + 1) * kv_heads * 256 * 2 bytes; the 12 query heads of a KV head share one read), plus the scratch the
// two passes write and read (heads * chunks * 258 floats at most).
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

// The five launches that precede attention_decode in one, for a window of `columns` tokens (column t is the token at
// position *pos + t): RMS norm (weight, no gate, eps) of the q heads (head h of column t is q + (t * heads + h) *
// q_stride) and of the k heads (k + (t * kv_heads + h) * 256), rope_neox on both with n_rot at position *pos + t, and
// store_kv of rows *pos + t from the normed, rotated k and from v (columns * kv_heads * 256 floats). Heads have 256
// dimensions. q and k are updated in place. Column t is bitwise equal to rms_norm, rope_neox and store_kv run in
// sequence on that column (tests/parity/attention_test.cpp).
void qk_norm_rope_store(float* q, int64_t q_stride, const float* q_weight, float* k, const float* k_weight,
                        const float* v, uint16_t* k_cache, uint16_t* v_cache, int heads, int kv_heads, int n_rot,
                        const int32_t* pos, float base, float eps, int columns, cudaStream_t stream);

// Floats of scratch attention_decode needs for a cache of max_ctx rows and a window of `columns` tokens.
int64_t attention_scratch_floats(int heads, int max_ctx, int columns);

// Dense causal attention of the window's queries: column t (1 <= columns <= kMaxColumns) attends to cache rows
// 0..*pos + t. Query head h of column t is q + (t * heads + h) * q_stride and its gate gate + (t * heads + h) *
// q_stride (the projection interleaves [q | gate] per head, so q_stride is 512 and gate = q + 256); it reads KV head
// h / (heads / kv_heads), and heads / kv_heads must be 12. out + (t * heads + h) * 256 = softmax(q . k * scale) v *
// sigmoid(gate), in F32 from F16 cache values; *pos + columns - 1 must be < max_ctx. The rows of every column must
// already be in the cache (qk_norm_rope_store writes them first).
//
// Split-K in two launches. Pass 1 gives each (KV head, 128-row chunk) one block that reads its K rows once for all
// columns and all 12 query heads, then its V rows for each column, and writes the chunk's maximum score, sum of
// exp(score - maximum) and unnormalized weighted sum of V per (column, head) to scratch (attention_scratch_floats
// floats of device memory); blocks past the last column's row exit. Pass 2 merges the used chunks of each (column,
// head) in ascending order and applies the gate. Scores, softmax and sums are F32 with IEEE expf; every reduction has
// a fixed order and column t follows exactly the order of a one-token decode at *pos + t, so its result is bitwise
// that decode's.
void attention_decode(const float* q, const float* gate, int64_t q_stride, const uint16_t* k_cache,
                      const uint16_t* v_cache, float* out, float* scratch, const int32_t* pos, int heads, int kv_heads,
                      float scale, int max_ctx, int columns, cudaStream_t stream);

}  // namespace torch::kernels
