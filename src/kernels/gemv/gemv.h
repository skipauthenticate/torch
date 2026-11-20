// Matrix-vector products for one-token decode, on weights exactly as the file stores them.
//
// Quantized weights take activations quantized to 8 bits per 32 values (BlockQ8) and multiply them with integer dot
// products (dp4a); BF16 weights take F32 activations. Rows are row_bytes(type, cols) apart, matrices one after the
// other as in the GGUF file.
//
// The reduction order, the same for every launch shape: a row is split into 32-element units; each unit's integer
// dot product is exact, is multiplied by its integer sub-scales in int32, then by the weight and activation scales
// in F32 (dot_q8.cuh shows the formula per format). Lane l of the row's warp sums units l, l + 32, l + 64, ... in
// that order, and the 32 lane sums are added in a fixed butterfly. BF16 rows do the same with 8-element chunks as
// units, each one multiply-add per element in order. A kernel may place a row's units on its lanes differently (a
// row on fewer lanes, or the units of several rows as one run): each canonical lane's units are still summed in
// order into a sum of their own, and the 32 sums are combined in the butterfly's order. So a row's result depends
// only on its bytes and its activations, never on where the bytes live (a resident expert or a staging slot), which
// kernel or block ran it, or which rows shared its launch. gemv_q8 and gemv_q8_experts run the same kernels, so a
// dense matrix and an expert give the same bits.
//
// Parity test: tests/parity/gemv_test.cpp (every format against the CPU reference in src/formats/dequant.h; BF16 bit
// for bit against a CPU twin of the order; experts bit for bit against gemv_q8).
// Bytes moved: the weight bytes once, plus the activations (L2-resident; staged per block in shared memory for most
// formats).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "formats/types.h"
#include "kernels/columns.h"

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

// Columns (verify windows): the dense entry points take `columns` (1 to kMaxColumns = 8) inputs and outputs, column
// after column (x[t * cols / 32 + u] or x[t * cols + i], y[t * rows + r]), and each column's output is bitwise what
// the one-column call computes on that column. A row's weights are read once for all its columns.

// y[r] = W[r] . x for r < rows, with x as cols / 32 blocks.
void gemv_q8(Type type, const uint8_t* w, int rows, int cols, const BlockQ8* x, float* y, cudaStream_t stream,
             int columns = 1);

// One dense matrix of a group launch.
struct Q8Part {
    Type type;
    const uint8_t* w;
    int rows;
    float* y;
};

// True for the formats a group launch takes: Q4_K, Q5_K, Q6_K and IQ4_XS, the projections that share an input.
bool gemv_q8_groups(Type type);

// Up to 3 dense matrices of one width that read the same activations, in one launch, each row bitwise as gemv_q8
// computes it: parts[i].y = W_i x. With `swiglu` (2 parts of equal rows, gate then up): parts[0].y = silu(W_0 x) *
// (W_1 x), bitwise as swiglu (kernels/moe) of the two products. Every part's format must pass gemv_q8_groups.
void gemv_q8_group(const Q8Part* parts, int count, int cols, const BlockQ8* x, bool swiglu, cudaStream_t stream,
                   int columns = 1);

// y[r] = W[r] . x for BF16 weights and F32 activations, accumulated in F32 (cols % 8 == 0, rows 16-byte aligned).
void gemv_bf16(const uint8_t* w, int rows, int cols, const float* x, float* y, cudaStream_t stream, int columns = 1);

// Two BF16 matrices of the same width and input in one launch: y = W x (rows rows) and y2 = W2 x (rows2 rows), each
// row bitwise as gemv_bf16 computes it. A non-zero silu_scale makes y = silu(silu_scale * W x), bitwise as
// silu_scale (kernels/hc) applied to it: a mixer's low-rank activation, with its inject rows as W2.
void gemv_bf16_pair(const uint8_t* w, int rows, const uint8_t* w2, int rows2, int cols, const float* x, float* y,
                    float* y2, cudaStream_t stream, float silu_scale = 0.0f, int columns = 1);

// A hyper-connection mixer's up projection and collapse in one launch, over `columns` (1 to kMaxColumns) columns:
// gate = W x for W of hc * n BF16 rows of `cols` values, then mixed[e] and its n / 32 Q8 blocks exactly as
// hc_collapse(xn, gate, mixed, q8, n, hc) computes them (kernels/hc/hc.h), bitwise; gate itself is not stored.
// Supports hc == 4, n % 32 == 0 and cols <= 320 (gemv_bf16's short rows); returns false, launching nothing, otherwise.
bool gemv_bf16_collapse(const uint8_t* w, int n, int hc, int cols, const float* x, const float* xn, float* mixed,
                        BlockQ8* q8, cudaStream_t stream, int columns = 1);

// The Q8_0 mixer down projection with its scaled SiLU and optional BF16 inject rows on the same F32 input.
void mixer_down_q8(const uint8_t* w, const uint8_t* inject, int rows, int inject_rows, int cols, const BlockQ8* xq,
                   const float* x, float* low, float* skip, float scale, int columns, cudaStream_t stream);

// A Q8_0 mixer with 320 input values and four streams of n values (n divisible by 32). Quantizes the input and
// collapses its projected streams into mixed and its Q8 blocks. Supports one to eight columns.
void mixer_up_q8(const uint8_t* w, int n, const float* low, const float* xn, float* mixed, BlockQ8* q8, int columns,
                 cudaStream_t stream);

// Routed experts. For each slot s < slots, W_s = table[s] (a device-readable pointer table, so a resident expert and
// one read into a staging slot run the same code) and x_s = x + s * x_stride blocks:
//   up_table == nullptr: y[s * rows + r] = W_s[r] . x_s                      (the down projection)
//   up_table != nullptr: y[s * rows + r] = silu(W_s[r] . x_s) * (U_s[r] . x_s), U_s = up_table[s]  (gate and up)
void gemv_q8_experts(Type type, const uint8_t* const* table, const uint8_t* const* up_table, int slots, int rows,
                     int cols, const BlockQ8* x, int64_t x_stride, float* y, cudaStream_t stream);

// The routed gate and up projections of a verify window of `window` columns (1 to kMaxColumns), each unique expert's
// rows read once for all the columns that routed it. Unique expert u < *count (device memory; at most `slots`) has gate
// matrix gate[u] and up matrix up[u]; bit t of columns[u] is set when column t routed it, at rank rank[u * kMaxColumns
// + t] (the window gate's GateTables: gate = table, up = table + kGateSlots). For each such (u, t) and row r < rows:
//   h[(t * k + rank) * rows + r] = silu(G_u[r] . x_t) * (U_u[r] . x_t),   x_t = x + t * cols / 32 blocks,
// bitwise what gemv_q8_experts computes for column t's routes (its h, K slots of rows), so h is [window][k][rows].
// The down projection then runs on h quantized in one launch (gemv_q8_experts_down_window). One column runs
// gemv_q8_experts itself (unique index = rank).
void gemv_q8_experts_window(Type type, const uint8_t* const* gate, const uint8_t* const* up, const uint8_t* columns,
                            const int8_t* rank, const int32_t* count, int slots, int k, int rows, int cols,
                            const BlockQ8* x, float* h, int window, cudaStream_t stream);

// The routed down projections of a verify window, each unique expert's down matrix read once for all the columns
// that routed it (rows of fewer than 1,024 values, the model's 640). Unique expert u < *count has down matrix down[u]
// (GateTables: table + 2 * kGateSlots); columns and rank as for gemv_q8_experts_window. For each routed (u, t) at rank
// r and row i < rows, with slot s = t * k + r:
//   y[s * rows + i] = D_u[i] . x_s,   x_s = x + s * cols / 32 blocks (h of slot s, quantized),
// bitwise what gemv_q8_experts(type, down_routes, nullptr, window * k, rows, cols, x, cols / 32, y) computes, so y is
// [window * k][rows] and moe_combine reads it unchanged. One column runs gemv_q8_experts itself.
void gemv_q8_experts_down_window(Type type, const uint8_t* const* down, const uint8_t* columns, const int8_t* rank,
                                 const int32_t* count, int slots, int k, int rows, int cols, const BlockQ8* x, float* y,
                                 int window, cudaStream_t stream);

}  // namespace torch::kernels
