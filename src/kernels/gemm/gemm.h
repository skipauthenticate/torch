// Matrix products over the many columns of a prefill chunk on the int8 tensor cores (milestone M6), each column bitwise
// what the one-column gemv_q8 computes on it.
//
// The activations are re-laid from BlockQ8 columns into Q8Columns, a unit-major layout whose 32 int8 of a (unit,
// column) load as tensor-core fragments. Each column's result keeps gemv_q8's canonical order (gemv.h): a unit's exact
// integer sum (mma.sync s8 x s8 into s32, one m16n8k16 per 16-value half where the format scales halves), the
// format's per-unit F32 formula, canonical lane l summing units l, l + 32, ... in order, and the lanes' butterfly.
//
// Kernel: one resident block per SM loops over work items of 16 rows x a column range whose activations stay in L2.
// An item first decodes its 16 rows once into shared memory in fragment order; then 64-column passes in which warp w
// takes lane set w % 8 (sub-block w % 8 of every 256-value block: canonical lanes w % 8 + 8j) and column half w / 8,
// streaming its own activations through a private cp.async ring (no block-wide barrier in the K loop), and ends with
// the canonical tree. Measured (Orin, 1,300 MHz, 4,096 columns, Q6_K): 2560x6144 8.70 TOPS, 10240x2560 6.94 TOPS,
// against 1.6 for gemv_q8 in windows of 8 columns (lab k5q, 2026-11-10). K <= 640 in IQ4_NL or Q2_0 (the experts'
// down projections, the shared expert's), whole 128-row items: every canonical lane holds at most one unit, so each
// warp runs its outputs' whole tree in registers, with no tree through shared memory (gemm_short_kernel).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "formats/types.h"
#include "kernels/gemv/gemv.h"

namespace torch::kernels {

// Columns are stored in multiples of kGemmColumnPad (the kernel's pass width); the padding is read, never stored.
inline constexpr int kGemmColumnPad = 64;

inline int gemm_padded_columns(int columns) {
    return (columns + kGemmColumnPad - 1) / kGemmColumnPad * kGemmColumnPad;
}

// N columns of Q8 activations, `stride` (gemm_padded_columns(N)) columns apart per unit:
//   q[(u * stride + c) * 32 ..]: the 32 int8 of unit u of column c, words in the order 0, 4, 1, 5, 2, 6, 3, 7
//   d[u * stride + c]: its scale (BlockQ8::d, F16) as F32; sum[u * stride + c]: BlockQ8::sum
struct Q8Columns {
    uint8_t* q;
    float* d;
    int16_t* sum;
    int units, columns, stride;
};

// Bytes of a Q8Columns of `units` units and `columns` columns. A routed experts' input (slots) passes columns + 64:
// its buckets start at multiples of 8, so a pass may read up to 63 columns past the last slot.
size_t q8_columns_bytes(int units, int columns);
// Lays out a Q8Columns in `memory` (q8_columns_bytes of it, 16-byte aligned).
Q8Columns q8_columns_at(void* memory, int units, int columns);

// Re-lays `columns` columns of BlockQ8 (column c's units at x + c * units) into `out`.
void q8_to_columns(const BlockQ8* x, const Q8Columns& out, cudaStream_t stream);

// True for the formats gemm_q8 multiplies: Q6_K, Q5_K, Q4_K, IQ4_XS, IQ4_NL and Q8_0.
bool gemm_q8_supports(Type type);
// True when rows of `cols` values fit gemm_q8's shared memory (640 bytes a 32-value unit, plus the rings): up to about
// 8,000 values on the Orin. Wider rows (a Q8_0 hyper-connection mixer's 10,240) take gemv_q8 a window at a time.
bool gemm_q8_fits(int cols);

// y[c * rows + r] = W[r] . x_c for every column c < x.columns, bitwise gemv_q8 on that column. rows % 16 == 0, cols a
// whole number of the format's blocks that gemm_q8_fits, and x.units == cols / 32.
void gemm_q8(Type type, const uint8_t* w, int rows, int cols, const Q8Columns& x, float* y, cudaStream_t stream);

// ---- Routed experts over a chunk --------------------------------------------------------------------------------
//
// A chunk's routes (column t's rank-k expert is route t * k_used + k) grouped by expert: expert e's routes take slots
// first[e] .. first[e] + counts[e] - 1 in route order, buckets start at multiples of 8 slots (the slots after a
// bucket's routes, up to the next multiple of 8, are padding: slot_route -1), and list[0 .. *n_list) are the routed
// experts in increasing id. Deterministic: integer counts, route order within a bucket.
struct ExpertBuckets {
    int32_t* counts;      // [n_expert]
    int32_t* first;       // [n_expert]
    int32_t* list;        // [n_expert]
    int32_t* n_list;      // [1] (device)
    int32_t* slot_route;  // [slots]: the route in each slot, -1 for padding
    int slots;            // capacity: expert_slots()
};
int expert_slots(int n_expert, int routes);
size_t expert_buckets_bytes(int n_expert, int routes);
ExpertBuckets expert_buckets_at(void* memory, int n_expert, int routes);  // 16-byte aligned memory

// Fills `b` from ids[0 .. routes) (k routes per column). Also does what the window's expert gate does for the routing
// it sees: route_counts[layer][e] += the expert's routes, and last_use[layer][e] = the window stamp feeding the chunk
// in windows of 8 from its start would leave (*pos + the first column of the last 8-column group routing e, plus 1).
void expert_buckets(const int32_t* ids, int routes, int k, int n_expert, const ExpertBuckets& b, int layer,
                    const int32_t* pos, unsigned long long* route_counts, uint32_t* last_use, cudaStream_t stream);

// out (columns = b.slots) = the Q8 activations of each slot's column (x: BlockQ8 columns of `units` units; column of
// route r = r / k); padding slots are zero.
void q8_gather_columns(const BlockQ8* x, int units, const ExpertBuckets& b, int k, const Q8Columns& out,
                       cudaStream_t stream);

// out (units = the experts' FF / 32) = the Q8 of h = silu(g) * u for the real slots of each expert in b.list[0 ..
// *b.n_list), g and u [slot][FF]: bitwise swiglu, quantize_q8 and q8_to_columns over those slots. Other slots are left
// as they are, so a read-ahead pass touches only its own experts' slots.
void expert_h_columns(const float* g, const float* u, const ExpertBuckets& b, int n_expert, const Q8Columns& out,
                      cudaStream_t stream);

// True for the formats gemm_q8_experts multiplies: IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL and Q2_0.
bool gemm_q8_experts_supports(Type type);

// For each listed expert e (W_e = table[e], rows x cols of `type`) and each of its slots s: y[o * rows + r] = W_e[r] .
// x_s with o = index[s] (index null: o = s; padding slots are not stored), bitwise gemv_q8 on that column. table is a
// device array of n_expert pointers (a resident expert's matrix or a staged copy: the same bytes).
void gemm_q8_experts(Type type, const uint8_t* const* table, const ExpertBuckets& b, int n_expert, int rows, int cols,
                     const Q8Columns& x, const int32_t* index, float* y, cudaStream_t stream);

// Copies the IQ grids to the GPU for the expert formats. Call once before the first gemm_q8_experts.
void upload_gemm_tables();

// ---- BF16 weights x F32 activations ----------------------------------------------------------------------------
//
// y[c * rows + r] = W[r] . x_c for c < columns (x_c = x + c * cols), and with silu_scale non-zero silu(silu_scale *
// that), each bitwise gemv_bf16 / gemv_bf16_pair on that column: canonical lane l's sum runs over 8-value chunks
// l, l + 32, ... element by element as one FMA chain, and the 32 lane sums meet in warp_sum's butterfly. The kernel
// runs the 32 lanes as 32 passes of an ordinary tiled product over each lane's chunks (64 rows x 64 columns a block,
// 4 x 4 a thread, one accumulator per output), in the butterfly's depth-first order (lanes 0, 16, 8, 24, ...), folding
// each finished pair into the tree at once: at most five partial values per output are live. The passes' chunks are
// staged as one sequence, so short rows' passes share stages. rows % 64 == 0 is not needed; cols % 8 == 0, 256 <= cols
// <= 262,144 (every lane has a chunk) and rows 16-byte aligned (as gemv_bf16). With w2, rows2 more rows from it go to
// y2 (no SiLU) in the same launch, as gemv_bf16_pair pairs a mixer's down projection with its inject rows. Up to 16
// rows past the last whole 64-row tile (the inject rows, the router's shared gate) run as gemv_bf16 runs a row, a warp
// a column, instead of a tile of their own. Shorter rows stay on gemv_bf16 (gemm_bf16_wins).
inline bool gemm_bf16_wins(int cols) {
    return cols >= 256;
}
// For long prompts an optional 16-byte-aligned scratch buffer holds the reordered weights and inputs that fit.
void gemm_bf16(const uint8_t* w, int rows, int cols, const float* x, float* y, int columns, float silu_scale,
               cudaStream_t stream, const uint8_t* w2 = nullptr, int rows2 = 0, float* y2 = nullptr,
               void* workspace = nullptr, size_t workspace_bytes = 0);

// gemm_bf16's products on the tensor cores, for --fast-prefill (gemm_bf16_tc.cu): the same outputs and epilogues, but
// the activations are rounded to BF16 before the products, which accumulate in F32 in a tensor-core order. Not bitwise
// gemm_bf16 or the windows; deterministic and independent of `columns`. For K <= 16,384 (the model's widest is 10,240)
// each output is within 2^-8 x sum_k |w_k x_k| of the exact sum before the SiLU: rounding x to BF16 costs at most 2^-9
// of it, the F32 accumulation at most K x 2^-24 (gemm_test checks the bound). Returns false, launching nothing, when
// cols is not a multiple of 64.
bool gemm_bf16_tc(const uint8_t* w, int rows, int cols, const float* x, float* y, int columns, float silu_scale,
                  cudaStream_t stream, const uint8_t* w2 = nullptr, int rows2 = 0, float* y2 = nullptr);

}  // namespace torch::kernels
