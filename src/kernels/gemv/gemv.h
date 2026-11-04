// Matrix-vector products for one-token decode, on weights exactly as the file stores them.
//
// Quantized weights take activations quantized to 8 bits per 32 values (BlockQ8) and multiply them with integer dot
// products (dp4a); BF16 weights take F32 activations. Rows
// are row_bytes(type, cols) apart, matrices one after the other as in the GGUF file.
//
// The reduction order, the same for every launch shape: a row is split into 32-element units; each unit's integer
// dot product is exact, is multiplied by its integer sub-scales in int32, then by the weight and activation scales
// in F32 (dot_q8.cuh shows the formula per format). Lane l of the row's warp sums units l, l + 32, l + 64, ... in
// that order, and the 32 lane sums are added in a fixed butterfly. So a row's result depends only on its bytes and
// its activations, never on where the bytes live (a resident expert or a staging slot) or which block ran it.
//
// Parity test: tests/parity/gemv_test.cpp (every format against the CPU reference in src/formats/dequant.h).
// Bytes moved: the weight bytes once, plus the activations (L2-resident).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "formats/types.h"

namespace torch::kernels {

// 32 activations as int8: x ~ q * d, with d = max|x| / 127 rounded to F16 and q = round(x / d) computed with the
// unrounded d. `sum` is the exact sum of q, which formats with an offset or a minimum (Q2_0, Q4_K, Q5_K) need.
struct BlockQ8 {
    uint16_t d;  // F16 bits
    int16_t sum;
    int8_t q[32];
};
static_assert(sizeof(BlockQ8) == 36, "BlockQ8 is 36 bytes: two 16-bit fields and 32 int8");

// Quantizes n floats (n % 32 == 0) into n / 32 blocks. Several vectors back to back quantize in one call.
void quantize_q8(const float* x, int64_t n, BlockQ8* out, cudaStream_t stream);

// Copies the IQ2_S, IQ3_XXS and IQ3_S grids to the GPU. Call once before the first gemv; throws torch::Error.
void upload_iq_tables();

// True for the formats gemv_q8 multiplies: every quantized format of the model (Q2_0, Q4_K, Q5_K, Q6_K, Q8_0,
// IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS).
bool gemv_q8_supports(Type type);

// y[r] = W[r] . x for r < rows, with x as cols / 32 blocks.
void gemv_q8(Type type, const uint8_t* w, int rows, int cols, const BlockQ8* x, float* y, cudaStream_t stream);

// y[r] = W[r] . x for BF16 weights and F32 activations, accumulated in F32 (cols % 8 == 0, rows 16-byte aligned).
void gemv_bf16(const uint8_t* w, int rows, int cols, const float* x, float* y, cudaStream_t stream);

// Routed experts. For each slot s < slots, W_s = table[s] (a device-readable pointer table, so a resident expert and
// one read into a staging slot run the same code) and x_s = x + s * x_stride blocks:
//   up_table == nullptr: y[s * rows + r] = W_s[r] . x_s                      (the down projection)
//   up_table != nullptr: y[s * rows + r] = silu(W_s[r] . x_s) * (U_s[r] . x_s), U_s = up_table[s]  (gate and up)
void gemv_q8_experts(Type type, const uint8_t* const* table, const uint8_t* const* up_table, int slots, int rows,
                     int cols, const BlockQ8* x, int64_t x_stride, float* y, cudaStream_t stream);

}  // namespace torch::kernels
