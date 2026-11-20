#include <algorithm>

#include "core/cuda_check.h"
#include "core/error.h"
#include "kernels/common.cuh"
#include "kernels/gemv/dot_q8.cuh"
#include "kernels/gemv/gemv.h"
#include "kernels/gemv/q8.cuh"

namespace torch::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kRowsPerBlock = kThreads / kWarp;  // one warp per row
constexpr int kQ8ShortBlocksPerSm = 6;           // gemv_q8_short_kernel blocks resident per SM

// One warp per 32 values.
__global__ void quantize_q8_kernel(const float* x, int64_t blocks, BlockQ8* out) {
    const int64_t block = blockIdx.x * int64_t{kRowsPerBlock} + threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    if (block >= blocks)
        return;
    quantize_q8_block(x[block * 32 + lane], out + block);
}

// The warp_sum butterfly over the 32 lane sums of the canonical layout, for a row that runs on kLanes lanes: lane j
// holds the sums of canonical lanes j + kLanes * k in acc[k]. Offsets 16 down to kLanes pair sums within a lane,
// the smaller ones pair lanes, so the result is bitwise warp_sum's. Every lane of the row's group gets it.
template <int kLanes>
__device__ __forceinline__ float row_sum(float (&acc)[kWarp / kLanes]) {
#pragma unroll
    for (int m = kWarp / kLanes / 2; m > 0; m /= 2)
#pragma unroll
        for (int k = 0; k < m; ++k)
            acc[k] += acc[k + m];
    float v = acc[0];
#pragma unroll
    for (int offset = kLanes / 2; offset > 0; offset /= 2)
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

// What one gemv_q8_kernel launch multiplies. Dense matrices pass `w`, experts the pointer tables (one entry per slot).
struct Q8Args {
    const uint8_t* w;
    const uint8_t* const* table;
    const uint8_t* const* up_table;  // non-null: SwiGLU of gate (table) and up
    int slots, rows, units;          // units of 32 values per row
    int64_t row_bytes;
    const BlockQ8* x;
    int64_t x_stride;  // blocks between the slots' activations; 0 when every slot reads the same
    float* y;
};

// Shared memory of one block: the format's IQ grid, the activations as Act (one set per slot unless shared), and
// each slot's matrices (gate, then up): the copies are read once per block instead of once per row group.
// Launches stay within the default 48 KB: gemv_q8_experts splits its slots to fit (about 1,000 activation blocks a
// launch; one token's down projection stages 200, a window of 8 columns 1,600).
template <Type T>
size_t q8_shared_bytes(const Q8Args& a) {
    return kGridBytes<T> + sizeof(Act) * static_cast<size_t>(a.x_stride == 0 ? 1 : a.slots) * a.units +
           2 * sizeof(const uint8_t*) * static_cast<size_t>(a.slots);
}

struct Q8Shared {
    const uint32_t* grid;
    const Act* act;
    const uint8_t* const* matrices;  // slot s: gate matrices[s], up matrices[slots + s]
};

// Fills the block's shared memory (q8_shared_bytes) and returns where each part is.
template <Type T>
__device__ Q8Shared stage_q8(const Q8Args& a, uint4* shared) {
    Act* act = reinterpret_cast<Act*>(shared + kGridBytes<T> / sizeof(uint4));
    if constexpr (kGridBytes<T> > 0) {
        const auto* source = static_cast<const uint4*>(grid_source<T>());
        for (int i = threadIdx.x; i < kGridBytes<T> / static_cast<int>(sizeof(uint4)); i += kThreads)
            shared[i] = source[i];
    }
    const int acts = (a.x_stride == 0 ? 1 : a.slots) * a.units;
    for (int i = threadIdx.x; i < acts; i += kThreads)
        act[i] = act_of<T>(a.x[i / a.units * a.x_stride + i % a.units]);
    auto* matrices = reinterpret_cast<const uint8_t**>(act + acts);
    for (int i = threadIdx.x; i < 2 * a.slots; i += kThreads) {
        const int slot = i % a.slots;
        matrices[i] = i < a.slots ? (a.table != nullptr ? a.table[slot] : a.w)
                                  : (a.up_table != nullptr ? a.up_table[slot] : nullptr);
    }
    __syncthreads();
    return {reinterpret_cast<const uint32_t*>(shared), act, matrices};
}

// gemv_q8_kernel blocks resident per SM: 4 (64 registers a thread), and 6 for IQ4_XS's plain products, which decode
// in fewer registers and gain from more warps in flight.
template <Type T, bool kSwiglu>
inline constexpr int kQ8BlocksPerSm = !kSwiglu && T == Type::IQ4_XS ? 6 : 4;

// A row runs on kLanes lanes (a warp takes 32 / kLanes rows at a time), picked so the row's units fill them: lane j
// owns the units of canonical lanes j, j + kLanes, ... (units u with u % kLanes == j) and keeps each canonical lane's
// sum apart, so row_sum gives the one-row-per-warp result bit for bit. Every block first copies the grid and the
// activations into shared memory; then warp w of W takes row groups w, w + W, ... over all slots (launch_q8 says how
// many blocks).
template <Type T, int kLanes, bool kSwiglu>
__global__ void __launch_bounds__(kThreads, kQ8BlocksPerSm<T, kSwiglu>) gemv_q8_kernel(Q8Args a) {
    constexpr int kSums = kWarp / kLanes, kGroups = kWarp / kLanes;
    extern __shared__ uint4 shared[];
    const Q8Shared sh = stage_q8<T>(a, shared);

    const int lane = threadIdx.x % kWarp, j = lane % kLanes;
    const int groups = (a.rows + kGroups - 1) / kGroups;
    const int warps = gridDim.x * (kThreads / kWarp);
    for (int item = blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp; item < groups * a.slots; item += warps) {
        const int slot = item / groups;
        const int row = item % groups * kGroups + lane / kLanes;
        const int64_t offset = min(row, a.rows - 1) * a.row_bytes;  // a group past the last row repeats it
        const uint8_t* gate = sh.matrices[slot] + offset;
        const uint8_t* up = kSwiglu ? sh.matrices[a.slots + slot] + offset : nullptr;
        const Act* xs = sh.act + (a.x_stride == 0 ? 0 : slot * a.units);
        float acc[kSums] = {}, acc_up[kSums] = {};
        for (int base = j; base < a.units; base += kWarp) {
#pragma unroll
            for (int k = 0; k < kSums; ++k) {
                const int u = base + kLanes * k;
                if (u < a.units) {
                    const Act x = xs[u];
                    acc[k] += dot_unit<T>(gate, u, x, sh.grid);
                    if constexpr (kSwiglu)
                        acc_up[k] += dot_unit<T>(up, u, x, sh.grid);
                }
            }
        }
        const float sum = row_sum<kLanes>(acc);
        const float sum_up = kSwiglu ? row_sum<kLanes>(acc_up) : 0.0f;
        if (j == 0 && row < a.rows)
            a.y[int64_t{slot} * a.rows + row] = kSwiglu ? silu(sum) * sum_up : sum;
    }
}

// Formats whose decode reads the activations as stored, with no regrouping or extra sums (Q4_K, Q5_K, Q8_0), need
// no staging: each lane reads its activation blocks from global memory (L1-resident after the block's first warp),
// one warp per row, which measures faster for them than the staged kernels.
template <Type T, bool kSwiglu>
__global__ void __launch_bounds__(kThreads) gemv_q8_direct_kernel(Q8Args a) {
    const int item = blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    if (item >= a.rows * a.slots)
        return;  // the whole warp
    const int slot = item / a.rows, row = item % a.rows;
    const int64_t offset = row * a.row_bytes;
    const uint8_t* gate = (a.table != nullptr ? a.table[slot] : a.w) + offset;
    const uint8_t* up = kSwiglu ? a.up_table[slot] + offset : nullptr;
    const BlockQ8* xs = a.x + slot * a.x_stride;
    float acc = 0.0f, acc_up = 0.0f;
    for (int u = lane; u < a.units; u += kWarp) {
        const Act x = act_of<T>(xs[u]);
        acc += dot_unit<T>(gate, u, x, nullptr);
        if constexpr (kSwiglu)
            acc_up += dot_unit<T>(up, u, x, nullptr);
    }
    const float sum = warp_sum(acc);
    const float sum_up = kSwiglu ? warp_sum(acc_up) : 0.0f;
    if (lane == 0)
        a.y[int64_t{slot} * a.rows + row] = kSwiglu ? silu(sum) * sum_up : sum;
}

// Rows of fewer than 32 units (640 columns are 20): a warp takes 8 rows at a time as one run of consecutive units,
// lane l computing units l, l + 32, ... of the run, so each load instruction reads consecutive bytes and no lane
// idles. Such a row has at most one unit per canonical lane, so each unit is a canonical lane's whole sum; the sums go
// through shared memory to the 4 lanes of their row, lane q holding canonical lanes q, q + 4, ..., q + 28 (zero past
// the row's units) for row_sum<4>.
constexpr int kQ8ShortRows = 8, kQ8ShortPitch = kWarp + 4;  // the pitch puts the 32 lanes' reads in 32 banks

// gemv_q8_short_kernel's static shared memory (its lane sums), besides q8_shared_bytes.
constexpr size_t q8_short_static_bytes(bool swiglu) {
    return sizeof(float) * (kThreads / kWarp) * (swiglu ? 2 : 1) * kQ8ShortRows * kQ8ShortPitch;
}

template <Type T, bool kSwiglu>
__global__ void __launch_bounds__(kThreads, kQ8ShortBlocksPerSm) gemv_q8_short_kernel(Q8Args a) {
    constexpr int kRows = kQ8ShortRows, kPitch = kQ8ShortPitch;
    constexpr int kParts = kSwiglu ? 2 : 1;
    extern __shared__ uint4 shared[];
    __shared__ float lane_sums[kThreads / kWarp][kParts][kRows * kPitch];
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    for (int i = lane; i < kParts * kRows * kPitch; i += kWarp)
        lane_sums[warp][i / (kRows * kPitch)][i % (kRows * kPitch)] = 0.0f;
    const Q8Shared sh = stage_q8<T>(a, shared);

    const int groups = (a.rows + kRows - 1) / kRows;
    const int warps = gridDim.x * (kThreads / kWarp);
    for (int item = blockIdx.x * (kThreads / kWarp) + warp; item < groups * a.slots; item += warps) {
        const int slot = item / groups, first = item % groups * kRows;
        const uint8_t* gate = sh.matrices[slot];
        const uint8_t* up = sh.matrices[a.slots + slot];
        const Act* xs = sh.act + (a.x_stride == 0 ? 0 : slot * a.units);
        for (int r = lane / a.units, u = lane % a.units; r < kRows;) {
            const int64_t offset = min(first + r, a.rows - 1) * a.row_bytes;
            const Act x = xs[u];
            float acc = 0.0f;
            acc += dot_unit<T>(gate + offset, u, x, sh.grid);
            lane_sums[warp][0][r * kPitch + u] = acc;
            if constexpr (kSwiglu) {
                float acc_up = 0.0f;
                acc_up += dot_unit<T>(up + offset, u, x, sh.grid);
                lane_sums[warp][kParts - 1][r * kPitch + u] = acc_up;
            }
            for (u += kWarp; u >= a.units; u -= a.units)
                ++r;
        }
        __syncwarp();
        const int r = lane / 4, q = lane % 4;
        float sums[8], sums_up[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            sums[k] = lane_sums[warp][0][r * kPitch + q + 4 * k];
            sums_up[k] = lane_sums[warp][kParts - 1][r * kPitch + q + 4 * k];
        }
        __syncwarp();  // the next item overwrites lane_sums
        const float sum = row_sum<4>(sums);
        const float sum_up = kSwiglu ? row_sum<4>(sums_up) : 0.0f;
        if (q == 0 && first + r < a.rows)
            a.y[int64_t{slot} * a.rows + first + r] = kSwiglu ? silu(sum) * sum_up : sum;
    }
}

// What one gemv_q8_group_kernel launch multiplies: the parts' row groups one after the other, items[p] being the
// first of part p (so a warp's rows never span two parts and its format is uniform).
constexpr int kGroupParts = 3;
struct Q8GroupArgs {
    Q8Part part[kGroupParts];
    int64_t row_bytes[kGroupParts];
    int items[kGroupParts + 1];
    int units;
    const BlockQ8* x;
};

// One row of format T on kLanes lanes, the per-row work of gemv_q8_kernel.
template <Type T, int kLanes>
__device__ __forceinline__ float q8_row(const uint8_t* row, const Act* xs, int units, int j) {
    float acc[kWarp / kLanes] = {};
    for (int base = j; base < units; base += kWarp) {
#pragma unroll
        for (int k = 0; k < kWarp / kLanes; ++k) {
            const int u = base + kLanes * k;
            if (u < units) {
                const Act x = xs[u];
                acc[k] += dot_unit<T>(row, u, x, nullptr);
            }
        }
    }
    return row_sum<kLanes>(acc);
}

// A row of any format gemv_q8_groups takes; `plain` holds the activations in value order, `nibbles` in IQ4 order.
template <int kLanes>
__device__ float group_row(Type type, const uint8_t* row, const Act* plain, const Act* nibbles, int units, int j) {
    switch (type) {
        case Type::Q4_K:
            return q8_row<Type::Q4_K, kLanes>(row, plain, units, j);
        case Type::Q5_K:
            return q8_row<Type::Q5_K, kLanes>(row, plain, units, j);
        case Type::Q6_K:
            return q8_row<Type::Q6_K, kLanes>(row, plain, units, j);
        default:
            return q8_row<Type::IQ4_XS, kLanes>(row, nibbles, units, j);
    }
}

// Several dense matrices with one input: each warp takes one row group of one part (16 lanes a row, 2 rows a warp)
// and decodes it in that part's format, picked at run time; the activations are staged once in both word orders.
// With kSwiglu the items are part 0's and each also runs part 1's same rows.
constexpr int kGroupLanes = 16;  // lanes a row in gemv_q8_group_kernel

template <bool kSwiglu>
__global__ void __launch_bounds__(kThreads, 4) gemv_q8_group_kernel(Q8GroupArgs a) {
    constexpr int kLanes = kGroupLanes, kGroups = kWarp / kLanes;
    extern __shared__ uint4 shared[];
    Act* plain = reinterpret_cast<Act*>(shared);
    Act* nibbles = plain + a.units;
    for (int i = threadIdx.x; i < a.units; i += kThreads) {
        plain[i] = act_of<Type::Q6_K>(a.x[i]);
        nibbles[i] = act_of<Type::IQ4_XS>(a.x[i]);
    }
    __syncthreads();
    const int item = blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
    const int parts = kSwiglu ? 1 : kGroupParts;
    if (item >= a.items[parts])
        return;  // the whole warp
    int p = 0;
    while (item >= a.items[p + 1])
        ++p;
    const int lane = threadIdx.x % kWarp, j = lane % kLanes;
    const Q8Part& part = a.part[p];
    const int row = (item - a.items[p]) * kGroups + lane / kLanes;
    const int64_t r = min(row, part.rows - 1);  // a group past the last row repeats it
    const float sum = group_row<kLanes>(part.type, part.w + r * a.row_bytes[p], plain, nibbles, a.units, j);
    float out = sum;
    if constexpr (kSwiglu)
        out =
            silu(sum) * group_row<kLanes>(a.part[1].type, a.part[1].w + r * a.row_bytes[1], plain, nibbles, a.units, j);
    if (j == 0 && row < part.rows)
        part.y[row] = out;
}

__device__ __forceinline__ float bf16_low(uint32_t v) {
    return __uint_as_float(v << 16);
}
__device__ __forceinline__ float bf16_high(uint32_t v) {
    return __uint_as_float(v & 0xffff0000u);
}

// acc += the eight products of one 16-byte chunk of BF16 weights with its eight activations, element by element.
__device__ __forceinline__ void bf16_chunk(float& acc, uint4 v, float4 a, float4 b) {
    acc += bf16_low(v.x) * a.x;
    acc += bf16_high(v.x) * a.y;
    acc += bf16_low(v.y) * a.z;
    acc += bf16_high(v.y) * a.w;
    acc += bf16_low(v.z) * b.x;
    acc += bf16_high(v.z) * b.y;
    acc += bf16_low(v.w) * b.z;
    acc += bf16_high(v.w) * b.w;
}

// Inject rows use F32 activations. Their four long dot products can run alongside the quantized down projection.
__global__ void __launch_bounds__(kThreads)
    mixer_down_q8_kernel(const uint8_t* w, const uint8_t* inject, int rows, int inject_rows, int cols,
                         const BlockQ8* xq, const float* x, float* low, float* skip, float scale) {
    const int column = blockIdx.x, lane = threadIdx.x % kWarp;
    const int index = blockIdx.y * kRowsPerBlock + threadIdx.x / kWarp;
    if (index >= rows + inject_rows)
        return;
    float sum = 0.0f;
    if (index < inject_rows) {
        const auto* weight = reinterpret_cast<const uint4*>(inject + int64_t{index} * cols * 2);
        const auto* input = reinterpret_cast<const float4*>(x + int64_t{column} * cols);
        for (int c = lane; c < cols / 8; c += kWarp)
            bf16_chunk(sum, weight[c], input[2 * c], input[2 * c + 1]);
        sum = warp_sum(sum);
        if (lane == 0)
            skip[int64_t{column} * inject_rows + index] = sum;
    } else {
        const int row = index - inject_rows, units = cols / 32;
        const uint8_t* weight = w + int64_t{row} * units * 34;
        xq += int64_t{column} * units;
        for (int u = lane; u < units; u += kWarp)
            sum += dot_unit<Type::Q8_0>(weight, u, act_of<Type::Q8_0>(xq[u]), nullptr);
        sum = warp_sum(sum);
        if (lane == 0)
            low[int64_t{column} * rows + row] = silu(sum * scale);
    }
}

// Rows r < rows come from w into y, rows r >= rows from w2 into y2: two matrices with one input share a launch.
// With `scale` non-zero, the first matrix's rows store silu(scale * sum) (gemv_bf16_pair).
struct Bf16Rows {
    const uint8_t *w, *w2;
    int rows, total;
    int64_t row_bytes;
    float *y, *y2;
    float scale;
    __device__ const uint4* row(int r) const {
        return reinterpret_cast<const uint4*>(r < rows ? w + r * row_bytes : w2 + (r - rows) * row_bytes);
    }
    __device__ void store(int r, float sum, int column = 0) const {
        if (r >= rows)
            y2[int64_t{column} * (total - rows) + r - rows] = sum;
        else
            y[int64_t{column} * rows + r] = scale != 0.0f ? silu(sum * scale) : sum;  // as silu_scale of the sum
    }
};

// One warp per row: lane l sums 8-element chunks l, l + 32, ... element by element, then the warp adds the lane sums.
// A lane loads kAhead chunks before it multiplies the first, so short launches keep more bytes in flight.
__global__ void __launch_bounds__(kThreads) gemv_bf16_kernel(Bf16Rows m, int chunks, const float* x) {
    constexpr int kAhead = 4;
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    if (row >= m.total)
        return;
    const uint4* r = m.row(row);
    const auto* x4 = reinterpret_cast<const float4*>(x);
    float acc = 0.0f;
    for (int first = lane; first < chunks; first += kAhead * kWarp) {
        uint4 v[kAhead];
#pragma unroll
        for (int i = 0; i < kAhead; ++i)
            v[i] = first + i * kWarp < chunks ? r[first + i * kWarp] : uint4{};
#pragma unroll
        for (int i = 0; i < kAhead; ++i) {
            const int c = first + i * kWarp;
            if (c < chunks)
                bf16_chunk(acc, v[i], x4[2 * c], x4[2 * c + 1]);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0)
        m.store(row, acc);
}

// Short rows (at most kShortLanes * kShortChunks chunks, 320 elements): a warp of one row would leave most lanes
// idle, so a row gets 8 lanes and a warp takes 4 rows at a time, kSteps times, with every step's weights loaded
// before the first product. Lane j holds its chunks' activations in registers and keeps the sums of canonical lanes
// j, j + 8, j + 16, j + 24 apart (chunk c belongs to lane c % 32), so row_sum gives the one-row-per-warp result.
constexpr int kShortLanes = 8, kShortChunks = 5, kShortSteps = 2;

template <int kSteps>
__global__ void __launch_bounds__(kThreads) gemv_bf16_short_kernel(Bf16Rows m, int chunks, const float* x) {
    constexpr int kGroups = kWarp / kShortLanes;
    const int lane = threadIdx.x % kWarp, j = lane % kShortLanes;
    const int first = (blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp) * kSteps * kGroups + lane / kShortLanes;
    const auto* x4 = reinterpret_cast<const float4*>(x);
    float4 xa[kShortChunks], xb[kShortChunks];
    uint4 v[kSteps][kShortChunks];
#pragma unroll
    for (int i = 0; i < kShortChunks; ++i) {
        const int c = j + kShortLanes * i;
        xa[i] = c < chunks ? x4[2 * c] : float4{};
        xb[i] = c < chunks ? x4[2 * c + 1] : float4{};
    }
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
        const uint4* r = m.row(min(first + s * kGroups, m.total - 1));
#pragma unroll
        for (int i = 0; i < kShortChunks; ++i)
            v[s][i] = j + kShortLanes * i < chunks ? r[j + kShortLanes * i] : uint4{};
    }
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
        float acc[kWarp / kShortLanes] = {};
#pragma unroll
        for (int i = 0; i < kShortChunks; ++i)
            if (j + kShortLanes * i < chunks)
                bf16_chunk(acc[i % (kWarp / kShortLanes)], v[s][i], xa[i], xb[i]);
        const float sum = row_sum<kShortLanes>(acc);
        if (j == 0 && first + s * kGroups < m.total)
            m.store(first + s * kGroups, sum);
    }
}

// ---- Several columns (a verify window) ----------------------------------------------------------------------------
//
// The kernels below read each row's weights once and multiply them with up to kCols columns of activations, x and y
// holding the columns one after the other. Per column they compute exactly what the one-column kernels compute for
// it: the same integer sums and F32 formula per unit, the same lane placement and row_sum, so every column is bitwise
// the one-column result (dot_q8.cuh: decode_unit and apply_unit). A column count rounds up to the kernel's kCols of 2,
// 4 or 8; the columns past it are skipped. One column runs the one-column kernels.

template <int kCols>
__device__ __forceinline__ bool column_on(int c, int columns) {
    return c < kCols && c < columns;
}

// Stages `columns` columns of `units` activation blocks (x[c * units + u]) as Act in the format's word order.
template <Type T>
__device__ void stage_columns(const BlockQ8* x, int units, int columns, Act* act) {
    for (int i = threadIdx.x; i < columns * units; i += kThreads)
        act[i] = act_of<T>(x[i]);
}

template <Type T>
__device__ void stage_grid(uint4* shared) {
    if constexpr (kColumnGridBytes<T> > 0) {
        const auto* source = static_cast<const uint4*>(column_grid_source<T>());
        for (int i = threadIdx.x; i < kColumnGridBytes<T> / static_cast<int>(sizeof(uint4)); i += kThreads)
            shared[i] = source[i];
    }
}

// What a multi-column dense launch multiplies.
struct ColumnsArgs {
    const uint8_t* w;
    int rows, units;
    int64_t row_bytes;
    const BlockQ8* x;  // columns x units blocks
    float* y;          // columns x rows
    int columns;
};

// A warp's run of 8 short rows (fewer than 32 units; gemv_q8_short_kernel's placement) from `first` times the columns
// c with bit c of `mask`, column c's activations at xs[c * units + u]: a lane's units decoded once and kept per column
// in registers, then each column's sums go through the warp's lane_sums (zero past the units) to their rows' lanes in
// turn; store(c, row, sum) for each row of the run.
template <Type T, int kCols, class Store>
__device__ __forceinline__ void short_run_columns(const uint8_t* w, int rows, int64_t row_bytes, int units, int first,
                                                  const Act* xs, unsigned mask, const uint32_t* grid, float* lane_sums,
                                                  Store store) {
    constexpr int kRows = kQ8ShortRows, kPitch = kQ8ShortPitch, kPairs = kRows;  // a lane's units: kRows * units / 32
    const int lane = threadIdx.x % kWarp;
    float v[kPairs][kCols];
    int at[kPairs];
    int r = lane / units, u = lane % units;
#pragma unroll
    for (int p = 0; p < kPairs; ++p) {
        at[p] = r < kRows ? r * kPitch + u : -1;
        if (r < kRows) {
            const Unit unit = decode_unit<T>(w + min(first + r, rows - 1) * row_bytes, u, grid);
#pragma unroll
            for (int c = 0; c < kCols; ++c) {
                float acc = 0.0f;
                if ((mask >> c & 1) != 0)
                    acc += apply_unit<T>(unit, xs[c * units + u]);
                v[p][c] = acc;
            }
        }
        for (u += kWarp; u >= units; u -= units)
            ++r;
    }
    const int row = lane / 4, q = lane % 4;
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
        if ((mask >> c & 1) == 0)
            continue;
#pragma unroll
        for (int p = 0; p < kPairs; ++p)
            if (at[p] >= 0)
                lane_sums[at[p]] = v[p][c];
        __syncwarp();
        float sums[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
            sums[k] = lane_sums[row * kPitch + q + 4 * k];
        __syncwarp();  // the next column overwrites lane_sums
        const float sum = row_sum<4>(sums);
        if (q == 0 && first + row < rows)
            store(c, first + row, sum);
    }
}

// Rows of fewer than 32 units: a warp's run of 8 rows (short_run_columns), the columns staged once per block and taken
// kCols at a time; a later pass rereads the run's rows, which its first pass has just brought into L1 (shared down
// 2560x640 at 6 columns: 27.3 us, against 33.1 as two launches).
template <Type T, int kCols>
__global__ void __launch_bounds__(kThreads) gemv_q8_short_columns_kernel(ColumnsArgs a) {
    extern __shared__ uint4 shared[];
    __shared__ float lane_sums[kThreads / kWarp][kQ8ShortRows * kQ8ShortPitch];
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    for (int i = lane; i < kQ8ShortRows * kQ8ShortPitch; i += kWarp)
        lane_sums[warp][i] = 0.0f;
    stage_grid<T>(shared);
    Act* act = reinterpret_cast<Act*>(shared + kColumnGridBytes<T> / sizeof(uint4));
    stage_columns<T>(a.x, a.units, a.columns, act);
    __syncthreads();
    const int first = (blockIdx.x * (kThreads / kWarp) + warp) * kQ8ShortRows;
    if (first >= a.rows)
        return;  // the whole warp
    for (int pass = 0; pass < a.columns; pass += kCols)
        short_run_columns<T, kCols>(a.w, a.rows, a.row_bytes, a.units, first, act + pass * a.units,
                                    (1u << min(kCols, a.columns - pass)) - 1, reinterpret_cast<const uint32_t*>(shared),
                                    lane_sums[warp],
                                    [&](int c, int row, float sum) { a.y[int64_t{pass + c} * a.rows + row] = sum; });
}

// An Act staged in value order, regrouped into IQ4_NL / IQ4_XS nibble order (act_of's permutation).
__device__ __forceinline__ Act nibble_order(const Act& a) {
    Act n = a;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        n.q[2 * k] = static_cast<int>(__byte_perm(a.q[k], a.q[k + 4], 0x5140));
        n.q[2 * k + 1] = static_cast<int>(__byte_perm(a.q[k], a.q[k + 4], 0x7362));
    }
    return n;
}

// kRows rows of format T on kLanes lanes for the columns c with bit c of `mask` (q8_row's work), row i's sums in
// out[i][c], the rows' units read in one loop (a SwiGLU's gate and up, as gemv_q8_kernel reads them); xs holds the
// columns' activations in value order (an IQ4_XS row regroups them unit by unit) or, without kValueOrder, in the
// format's order. With `at`, slot c multiplies column at[c].
template <Type T, int kLanes, int kCols, bool kValueOrder, int kRows>
__device__ __forceinline__ void q8_rows_columns(const uint8_t* const (&rows)[kRows], const Act* xs, int units, int j,
                                                unsigned mask, const uint32_t* grid, float (&out)[kRows][kCols],
                                                const int* at = nullptr) {
    constexpr int kSums = kWarp / kLanes;
    float acc[kRows][kCols][kSums] = {};
    for (int base = j; base < units; base += kWarp) {
#pragma unroll
        for (int k = 0; k < kSums; ++k) {
            const int u = base + kLanes * k;
            if (u < units) {
                Unit unit[kRows];
#pragma unroll
                for (int i = 0; i < kRows; ++i)
                    unit[i] = decode_unit<T>(rows[i], u, grid);
#pragma unroll
                for (int c = 0; c < kCols; ++c)
                    if ((mask >> c & 1) != 0) {
                        const Act& x = xs[(at != nullptr ? at[c] : c) * units + u];
                        const Act a = kValueOrder && T == Type::IQ4_XS ? nibble_order(x) : x;
#pragma unroll
                        for (int i = 0; i < kRows; ++i)
                            acc[i][c][k] += apply_unit<T>(unit[i], a);
                    }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kRows; ++i)
#pragma unroll
        for (int c = 0; c < kCols; ++c)
            out[i][c] = (mask >> c & 1) != 0 ? row_sum<kLanes>(acc[i][c]) : 0.0f;
}

// Rows of at least 32 units on kLanes lanes (gemv_q8_kernel's placement). A lane group takes kRowsPerGroup rows that
// share each unit's activations: one read of a column's activation block serves both rows, which halves the shared
// memory reads that bound the kernel at 3 and 4 columns (Q6_K 2560x6144 at 4 columns: 135 us with one row, see
// the commit). Each row's units still go lane by lane in the one-column order, so its sums are unchanged.
// Two columns keep the one-row form: its smaller blocks of work measured faster there (Q6_K 2560x6144, 88 us against
// 108). With kLoop a launch has only the blocks that stay resident and each warp takes every gridDim.x * 8-th run of
// rows.
template <int kCols>
inline constexpr int kColumnRows = kCols <= 2 ? 1 : 2;

template <Type T, int kLanes, int kCols, bool kLoop = false>
__global__ void __launch_bounds__(kThreads) gemv_q8_columns_kernel(ColumnsArgs a) {
    constexpr int kGroups = kWarp / kLanes;
    extern __shared__ uint4 shared[];
    stage_grid<T>(shared);
    Act* act = reinterpret_cast<Act*>(shared + kColumnGridBytes<T> / sizeof(uint4));
    stage_columns<T>(a.x, a.units, a.columns, act);
    __syncthreads();
    const int lane = threadIdx.x % kWarp, j = lane % kLanes;
    constexpr int kRowsEach = kColumnRows<kCols>;
    const auto work = [&](int warp_first) {
        const int first = warp_first + lane / kLanes * kRowsEach;
        const auto* grid = reinterpret_cast<const uint32_t*>(shared);
        const uint8_t* rows[kRowsEach];
#pragma unroll
        for (int i = 0; i < kRowsEach; ++i)
            rows[i] = a.w + int64_t{min(first + i, a.rows - 1)} * a.row_bytes;
        float sums[kRowsEach][kCols];
        q8_rows_columns<T, kLanes, kCols, false, kRowsEach>(rows, act, a.units, j, (1u << a.columns) - 1, grid, sums);
#pragma unroll
        for (int i = 0; i < kRowsEach; ++i)
#pragma unroll
            for (int c = 0; c < kCols; ++c)
                if (column_on<kCols>(c, a.columns) && j == 0 && first + i < a.rows)
                    a.y[int64_t{c} * a.rows + first + i] = sums[i][c];
    };
    const int warp_first = (blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp) * kGroups * kRowsEach;
    if constexpr (kLoop) {
        for (int at = warp_first; at < a.rows; at += gridDim.x * (kThreads / kWarp) * kGroups * kRowsEach)
            work(at);
    } else if (warp_first < a.rows) {
        work(warp_first);
    }
}

// kRowsEach rows of a part of any format gemv_q8_groups takes (q8_rows_columns), the columns' activations in value
// order.
template <int kLanes, int kCols, int kRowsEach>
__device__ void group_rows_columns(Type type, const uint8_t* const (&rows)[kRowsEach], const Act* plain, int units,
                                   int j, int columns, float (&out)[kRowsEach][kCols]) {
    const unsigned mask = (1u << columns) - 1;
    switch (type) {
        case Type::Q4_K:
            return q8_rows_columns<Type::Q4_K, kLanes, kCols, true, kRowsEach>(rows, plain, units, j, mask, nullptr,
                                                                               out);
        case Type::Q5_K:
            return q8_rows_columns<Type::Q5_K, kLanes, kCols, true, kRowsEach>(rows, plain, units, j, mask, nullptr,
                                                                               out);
        case Type::Q6_K:
            return q8_rows_columns<Type::Q6_K, kLanes, kCols, true, kRowsEach>(rows, plain, units, j, mask, nullptr,
                                                                               out);
        default:
            return q8_rows_columns<Type::IQ4_XS, kLanes, kCols, true, kRowsEach>(rows, plain, units, j, mask, nullptr,
                                                                                 out);
    }
}

// gemv_q8_group_kernel for several columns: the columns' activations staged once in value order (eight columns of
// 2,560 in 31 KB), part p's outputs at y[c * rows + r]. With kLoop a launch has only the blocks that stay resident and
// each warp takes every gridDim.x * 8-th row group.
template <bool kSwiglu, int kCols, int kLanes, int kRowsEach, bool kLoop = false>
__global__ void __launch_bounds__(kThreads) gemv_q8_group_columns_kernel(Q8GroupArgs a, int columns) {
    constexpr int kGroups = kWarp / kLanes;
    extern __shared__ uint4 shared[];
    Act* plain = reinterpret_cast<Act*>(shared);
    for (int i = threadIdx.x; i < columns * a.units; i += kThreads)
        plain[i] = act_of<Type::Q6_K>(a.x[i]);
    __syncthreads();
    const int end = a.items[kSwiglu ? 1 : kGroupParts];
    const auto work = [&](int item) {
        int p = 0;
        while (item >= a.items[p + 1])
            ++p;
        const int lane = threadIdx.x % kWarp, j = lane % kLanes;
        const Q8Part& part = a.part[p];
        const int first = ((item - a.items[p]) * kGroups + lane / kLanes) * kRowsEach;
        const uint8_t* rows[kRowsEach];
        const uint8_t* ups[kRowsEach];
#pragma unroll
        for (int i = 0; i < kRowsEach; ++i) {
            const int64_t r = min(first + i, part.rows - 1);
            rows[i] = part.w + r * a.row_bytes[p];
            ups[i] = kSwiglu ? a.part[1].w + r * a.row_bytes[1] : nullptr;
        }
        float sum[kRowsEach][kCols], up[kRowsEach][kCols];
        group_rows_columns<kLanes, kCols, kRowsEach>(part.type, rows, plain, a.units, j, columns, sum);
        if constexpr (kSwiglu)
            group_rows_columns<kLanes, kCols, kRowsEach>(a.part[1].type, ups, plain, a.units, j, columns, up);
#pragma unroll
        for (int i = 0; i < kRowsEach; ++i)
#pragma unroll
            for (int c = 0; c < kCols; ++c)
                if (column_on<kCols>(c, columns) && j == 0 && first + i < part.rows)
                    part.y[int64_t{c} * part.rows + first + i] = kSwiglu ? silu(sum[i][c]) * up[i][c] : sum[i][c];
    };
    const int item = blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
    if constexpr (kLoop) {
        for (int at = item; at < end; at += gridDim.x * (kThreads / kWarp))
            work(at);
    } else if (item < end) {
        work(item);
    }
}

// What gemv_q8_experts_window multiplies (gemv.h).
struct ExpertsWindowArgs {
    const uint8_t* const* gate;  // [slots]: unique expert u's gate matrix
    const uint8_t* const* up;    // [slots]
    const uint8_t* columns;      // [slots]: bit t when column t routed u
    const int8_t* rank;          // [slots][kMaxColumns]: the rank column t gave u
    const int32_t* count;        // unique experts, in device memory
    int slots, k, rows, units;
    int64_t row_bytes;
    const BlockQ8* x;  // window x units blocks
    float* h;          // (t * k + rank) * rows + r
    int window;
};

// The routed gate and up projections of a window's unique experts: a warp's 2 rows of expert u times the columns that
// routed u, kCols at a time (gemv_q8_kernel's SwiGLU per column: the same unit sums, lane placement and row_sum); a
// further pass over the same rows mostly hits L1. The columns' activations are staged once per block in the
// format's order; blocks loop over (expert, row group) items.
template <Type T, int kCols>
__global__ void __launch_bounds__(kThreads) gemv_q8_experts_window_kernel(ExpertsWindowArgs a) {
    constexpr int kLanes = 16, kGroups = kWarp / kLanes;
    extern __shared__ uint4 shared[];
    stage_grid<T>(shared);
    Act* act = reinterpret_cast<Act*>(shared + kColumnGridBytes<T> / sizeof(uint4));
    stage_columns<T>(a.x, a.units, a.window, act);
    __syncthreads();
    const auto* grid = reinterpret_cast<const uint32_t*>(shared);
    const int lane = threadIdx.x % kWarp, j = lane % kLanes;
    const int groups = (a.rows + kGroups - 1) / kGroups;
    const int items = min(*a.count, a.slots) * groups;
    for (int item = blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp; item < items;
         item += gridDim.x * (kThreads / kWarp)) {
        const int u = item / groups;
        const int row = item % groups * kGroups + lane / kLanes;
        const int64_t offset = min(row, a.rows - 1) * a.row_bytes;
        const uint8_t* const rows[2] = {a.gate[u] + offset, a.up[u] + offset};
        for (unsigned left = a.columns[u]; left != 0;) {
            int at[kCols];
            unsigned pass = 0;
#pragma unroll
            for (int c = 0; c < kCols; ++c) {
                at[c] = left != 0 ? __ffs(static_cast<int>(left)) - 1 : 0;
                pass |= left != 0 ? 1u << c : 0u;
                left &= left - 1;
            }
            float sums[2][kCols];
            q8_rows_columns<T, kLanes, kCols, false, 2>(rows, act, a.units, j, pass, grid, sums, at);
#pragma unroll
            for (int c = 0; c < kCols; ++c)
                if ((pass >> c & 1) != 0 && j == 0 && row < a.rows)
                    a.h[(int64_t{at[c]} * a.k + a.rank[u * kMaxColumns + at[c]]) * a.rows + row] =
                        silu(sums[0][c]) * sums[1][c];
        }
    }
}

// 8 short rows (fewer than 32 units) from `first` times the columns c with bit c of `mask`, column c's activations at
// xs[c * units + u], placed for row_sum<4> directly: lane 4i + q takes row first + i and its units q, q + 4, ... (the
// canonical lanes gemv_q8_short_kernel gathers there through shared memory), so each unit is decoded once and no
// sums move through shared memory; store(c, row, sum).
template <Type T, int kCols, class Store>
__device__ __forceinline__ void short_rows_columns(const uint8_t* w, int rows, int64_t row_bytes, int units, int first,
                                                   const Act* xs, unsigned mask, const uint32_t* grid, Store store) {
    const int lane = threadIdx.x % kWarp, q = lane % 4, row = first + lane / 4;
    const uint8_t* r = w + min(row, rows - 1) * row_bytes;
    float sums[kCols][8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int u = q + 4 * k;
        const bool on = u < units;
        Unit unit{};
        if (on)
            unit = decode_unit<T>(r, u, grid);
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            float acc = 0.0f;
            if (on && (mask >> c & 1) != 0)
                acc += apply_unit<T>(unit, xs[c * units + u]);
            sums[c][k] = acc;
        }
    }
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
        if ((mask >> c & 1) == 0)
            continue;
        const float sum = row_sum<4>(sums[c]);
        if (q == 0 && row < rows)
            store(c, row, sum);
    }
}

// What gemv_q8_experts_down_window multiplies (gemv.h).
struct DownWindowArgs {
    const uint8_t* const* down;  // [slots]: unique expert u's down matrix
    const uint8_t* columns;      // [slots]: bit t when column t routed u
    const int8_t* rank;          // [slots][kMaxColumns]
    const int32_t* count;        // unique experts, in device memory
    int slots, k, rows, units;
    int64_t row_bytes;
    const BlockQ8* x;  // [window * k][units]: column t's rank-r input at (t * k + r) * units
    float* y;          // [window * k][rows]
};

constexpr int kDownColumns = 2;  // columns a pass of gemv_q8_down_window_kernel
constexpr int kDownRuns = 2;     // runs of 8 rows a warp takes of one expert

// The routed down projections of a window's unique experts (short rows): a warp takes kDownRuns runs of 8 rows of
// expert u and multiplies each run, read once a pass, with the inputs of the columns that routed u, kDownColumns at a
// time, staged in the warp's own shared memory. Each (column, rank) slot is computed as gemv_q8_short_kernel computes
// it (short_rows_columns), so y equals the per-slot launch byte for byte. Measured at T = 3 and 8 (IQ4_NL, us a
// launch), against 4 columns and 4 runs through shared lane sums: 151 / 361 against 161 / 372; Q2_0 113 / 264
// against 141 / 323.
template <Type T>
__global__ void __launch_bounds__(kThreads) gemv_q8_down_window_kernel(DownWindowArgs a) {
    constexpr int kCols = kDownColumns;
    extern __shared__ uint4 shared[];
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    stage_grid<T>(shared);
    __syncthreads();
    const auto* grid = reinterpret_cast<const uint32_t*>(shared);
    Act* act = reinterpret_cast<Act*>(shared + kColumnGridBytes<T> / sizeof(uint4)) + warp * kCols * a.units;
    const int span = kQ8ShortRows * kDownRuns, chunks = (a.rows + span - 1) / span;
    const int items = min(*a.count, a.slots) * chunks;
    for (int item = blockIdx.x * (kThreads / kWarp) + warp; item < items; item += gridDim.x * (kThreads / kWarp)) {
        const int u = item / chunks, first = item % chunks * span;
        for (unsigned left = a.columns[u]; left != 0;) {
            int slot[kCols];  // the pass's (column, rank) slots
            unsigned pass = 0;
#pragma unroll
            for (int c = 0; c < kCols; ++c) {
                const int t = left != 0 ? __ffs(static_cast<int>(left)) - 1 : 0;
                slot[c] = t * a.k + a.rank[u * kMaxColumns + t];
                pass |= left != 0 ? 1u << c : 0u;
                left &= left - 1;
            }
            __syncwarp();  // the previous pass is done with act
            for (int i = lane; i < kCols * a.units; i += kWarp) {
                const int c = i / a.units;
                int s = slot[0];
#pragma unroll
                for (int k = 1; k < kCols; ++k)
                    s = c == k ? slot[k] : s;
                if ((pass >> c & 1) != 0)
                    act[i] = act_of<T>(a.x[int64_t{s} * a.units + i % a.units]);
            }
            __syncwarp();
            const auto store = [&](int c, int row, float sum) { a.y[int64_t{slot[c]} * a.rows + row] = sum; };
            for (int run = first; run < min(first + span, a.rows); run += kQ8ShortRows)
                short_rows_columns<T, kCols>(a.down[u], a.rows, a.row_bytes, a.units, run, act, pass, grid, store);
        }
    }
}

// Stages n chunks from `first` of each column's activations (x[c * 2 * chunks + 2 * k + h], h the chunk's half) as
// xs[(c * 2 + h) * pitch + k - first]: a warp's lanes reading consecutive chunks then read consecutive 16-byte words.
__device__ __forceinline__ void stage_bf16(const float* x, int chunks, int first, int n, int columns, int pitch,
                                           float4* xs) {
    const auto* x4 = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < columns * 2 * n; i += kThreads) {
        const int c = i / (2 * n), k = i % (2 * n);
        xs[(c * 2 + k % 2) * pitch + k / 2] = x4[int64_t{c} * 2 * chunks + 2 * first + k];
    }
}

// gemv_bf16_short_kernel for several columns: 4 rows a warp on 8 lanes each, kShortSteps times, the weights loaded
// before the columns' activations are staged; each staged chunk read serves the lane's kShortSteps rows, and each
// row and column sums in the one-column order.
template <int kCols>
__global__ void __launch_bounds__(kThreads)
    gemv_bf16_short_columns_kernel(Bf16Rows m, int chunks, const float* x, int columns) {
    constexpr int kGroups = kWarp / kShortLanes, kSums = kWarp / kShortLanes;
    extern __shared__ float4 xs[];  // columns x 2 halves x chunks
    const int lane = threadIdx.x % kWarp, j = lane % kShortLanes;
    const int first = (blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp) * kShortSteps * kGroups + lane / kShortLanes;
    uint4 v[kShortSteps][kShortChunks];
#pragma unroll
    for (int s = 0; s < kShortSteps; ++s) {
        const uint4* r = m.row(min(first + s * kGroups, m.total - 1));
#pragma unroll
        for (int i = 0; i < kShortChunks; ++i)
            v[s][i] = j + kShortLanes * i < chunks ? r[j + kShortLanes * i] : uint4{};
    }
    stage_bf16(x, chunks, 0, chunks, columns, chunks, xs);
    __syncthreads();
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
        if (!column_on<kCols>(c, columns))
            continue;
        float acc[kShortSteps][kSums] = {};
#pragma unroll
        for (int i = 0; i < kShortChunks; ++i) {
            const int k = j + kShortLanes * i;
            if (k < chunks) {
                const float4 lo = xs[2 * c * chunks + k], hi = xs[(2 * c + 1) * chunks + k];
#pragma unroll
                for (int s = 0; s < kShortSteps; ++s)
                    bf16_chunk(acc[s][i % kSums], v[s][i], lo, hi);
            }
        }
#pragma unroll
        for (int s = 0; s < kShortSteps; ++s) {
            const float sum = row_sum<kShortLanes>(acc[s]);
            if (j == 0 && first + s * kGroups < m.total)
                m.store(first + s * kGroups, sum, c);
        }
    }
}

// gemv_bf16_collapse: block b takes values e = 32 b .. 32 b + 31 of all 4 streams, 128 rows (row i of the block is
// stream i / 32, value 32 b + i % 32), each as gemv_bf16_short_columns_kernel computes it (8 lanes, the same chunks
// and row_sum), into shared memory; then warp c collapses column c's 32 values and quantizes them as one Q8 block, as
// hc_collapse_kernel does.
constexpr int kCollapseSteps = 4;  // 8 warps x 4 lane groups x 4 steps = 128 rows

__device__ void finish_collapse(const float* gate, const float* xn, float* mixed, BlockQ8* q8, int e0, int n,
                                int columns) {
    __syncthreads();
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    if (warp < columns) {
        const int64_t t = warp;
        const int e = e0 + lane;
        float sum = 0.0f;
        for (int s = 0; s < 4; ++s)
            sum += xn[(t * 4 + s) * n + e] * sigmoid(gate[(warp * 4 + s) * kWarp + lane]);
        const float m = sum * 0.25f;
        mixed[t * n + e] = m;
        quantize_q8_block(m, q8 + t * (n / 32) + e0 / 32);
    }
}

template <int kCols>
__global__ void __launch_bounds__(kThreads)
    mixer_up_q8_kernel(const uint8_t* w, int n, const float* low, const float* xn, float* mixed, BlockQ8* q8,
                       int columns) {
    constexpr int kUnits = 10, kLanes = 8;
    __shared__ BlockQ8 x[kCols * kUnits];
    __shared__ float gate[kCols][4 * kWarp];
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp, j = lane % kLanes;
    const int e0 = blockIdx.x * kWarp;
    for (int i = warp; i < columns * kUnits; i += kRowsPerBlock)
        quantize_q8_block(low[i * kWarp + lane], &x[i]);
    __syncthreads();
#pragma unroll
    for (int step = 0; step < kCollapseSteps; ++step) {
        const int i = (warp * kCollapseSteps + step) * (kWarp / kLanes) + lane / kLanes;
        const uint8_t* row = w + (int64_t{i / kWarp} * n + e0 + i % kWarp) * (kUnits * 34);
        const Unit first = decode<Type::Q8_0>(row, j, nullptr);
        Unit last{};
        if (j + kLanes < kUnits)
            last = decode<Type::Q8_0>(row, j + kLanes, nullptr);
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            if (c >= columns)
                continue;
            float acc[kWarp / kLanes] = {};
            acc[0] += apply<Type::Q8_0>(first, act_of<Type::Q8_0>(x[c * kUnits + j]));
            if (j + kLanes < kUnits)
                acc[1] += apply<Type::Q8_0>(last, act_of<Type::Q8_0>(x[c * kUnits + j + kLanes]));
            const float v = row_sum<kLanes>(acc);
            if (j == 0)
                gate[c][i] = v;
        }
    }
    finish_collapse(&gate[0][0], xn, mixed, q8, e0, n, columns);
}

template <int kCols>
__global__ void __launch_bounds__(kThreads)
    gemv_bf16_collapse_kernel(const uint8_t* w, int n, int chunks, const float* x, const float* xn, float* mixed,
                              BlockQ8* q8, int columns) {
    constexpr int kGroups = kWarp / kShortLanes, kSums = kWarp / kShortLanes, kHc = 4;
    extern __shared__ float4 xs[];  // columns x 2 halves x chunks
    __shared__ float gate[kCols][kHc * kWarp];
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp, j = lane % kShortLanes;
    const int e0 = static_cast<int>(blockIdx.x) * kWarp;
    uint4 v[kCollapseSteps][kShortChunks];
#pragma unroll
    for (int s = 0; s < kCollapseSteps; ++s) {
        const int i = (warp * kCollapseSteps + s) * kGroups + lane / kShortLanes;
        const auto* r = reinterpret_cast<const uint4*>(w + (int64_t{i / kWarp} * n + e0 + i % kWarp) * chunks * 16);
#pragma unroll
        for (int k = 0; k < kShortChunks; ++k)
            v[s][k] = j + kShortLanes * k < chunks ? r[j + kShortLanes * k] : uint4{};
    }
    stage_bf16(x, chunks, 0, chunks, columns, chunks, xs);
    __syncthreads();
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
        if (!column_on<kCols>(c, columns))
            continue;
        float acc[kCollapseSteps][kSums] = {};
#pragma unroll
        for (int k = 0; k < kShortChunks; ++k) {
            const int at = j + kShortLanes * k;
            if (at < chunks) {
                const float4 lo = xs[2 * c * chunks + at], hi = xs[(2 * c + 1) * chunks + at];
#pragma unroll
                for (int s = 0; s < kCollapseSteps; ++s)
                    bf16_chunk(acc[s][k % kSums], v[s][k], lo, hi);
            }
        }
#pragma unroll
        for (int s = 0; s < kCollapseSteps; ++s) {
            const float sum = row_sum<kShortLanes>(acc[s]);
            if (j == 0)
                gate[c][(warp * kCollapseSteps + s) * kGroups + lane / kShortLanes] = sum;
        }
    }
    finish_collapse(&gate[0][0], xn, mixed, q8, e0, n, columns);
}

// gemv_bf16_kernel for several columns, its activations read directly (L1-resident for rows of up to 4,096 values): one
// warp a row, kAhead chunks loaded before the first product, each column's lane sums in the one-column order, no
// staging and no block barriers, which cost the staged kernel most of a short matrix's time (the router, 513 x 2560,
// 23 against 17 us at 2 columns).
template <int kCols>
__global__ void __launch_bounds__(kThreads)
    gemv_bf16_direct_columns_kernel(Bf16Rows m, int chunks, const float* x, int columns) {
    constexpr int kAhead = 4;
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    if (row >= m.total)
        return;
    const uint4* r = m.row(row);
    const auto* x4 = reinterpret_cast<const float4*>(x);
    float acc[kCols] = {};
    for (int first = lane; first < chunks; first += kAhead * kWarp) {
        uint4 v[kAhead];
#pragma unroll
        for (int i = 0; i < kAhead; ++i)
            v[i] = first + i * kWarp < chunks ? r[first + i * kWarp] : uint4{};
#pragma unroll
        for (int i = 0; i < kAhead; ++i) {
            const int k = first + i * kWarp;
            if (k < chunks)
#pragma unroll
                for (int c = 0; c < kCols; ++c)
                    if (column_on<kCols>(c, columns))
                        bf16_chunk(acc[c], v[i], x4[int64_t{c} * 2 * chunks + 2 * k],
                                   x4[int64_t{c} * 2 * chunks + 2 * k + 1]);
        }
    }
#pragma unroll
    for (int c = 0; c < kCols; ++c)
        if (column_on<kCols>(c, columns)) {
            const float sum = warp_sum(acc[c]);
            if (lane == 0)
                m.store(row, sum, c);
        }
}

// gemv_bf16_kernel for several columns: a warp takes kR rows and a block's warps share each segment of the columns'
// activations, staged in shared memory (kSegment chunks of every column at a time); each staged chunk read serves
// the warp's kR rows, and lane l keeps chunks l, l + 32, ... in order per row and column. The next segment's weights
// load while the current one is multiplied.
constexpr int kSegment = 128;  // chunks of 8 values: 4 KB of each column

template <int kCols, int kR>
__global__ void __launch_bounds__(kThreads)
    gemv_bf16_columns_kernel(Bf16Rows m, int chunks, const float* x, int columns) {
    constexpr int kPer = kSegment / kWarp;
    __shared__ float4 xs[kCols * 2 * kSegment];
    const int first = (blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp) * kR;
    const int lane = threadIdx.x % kWarp;
    const uint4* r[kR];
#pragma unroll
    for (int q = 0; q < kR; ++q)
        r[q] = m.row(min(first + q, m.total - 1));
    uint4 next[kR][kPer];
    const auto load = [&](int segment) {
#pragma unroll
        for (int q = 0; q < kR; ++q)
#pragma unroll
            for (int i = 0; i < kPer; ++i) {
                const int k = segment + lane + kWarp * i;
                next[q][i] = k < chunks ? r[q][k] : uint4{};
            }
    };
    load(0);
    float acc[kR][kCols] = {};
    for (int segment = 0; segment < chunks; segment += kSegment) {
        const int n = min(kSegment, chunks - segment);
        uint4 v[kR][kPer];
#pragma unroll
        for (int q = 0; q < kR; ++q)
#pragma unroll
            for (int i = 0; i < kPer; ++i)
                v[q][i] = next[q][i];
        if (segment + kSegment < chunks)
            load(segment + kSegment);
        __syncthreads();  // the previous segment is used up
        stage_bf16(x, chunks, segment, n, columns, kSegment, xs);
        __syncthreads();
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            if (!column_on<kCols>(c, columns))
                continue;
#pragma unroll
            for (int i = 0; i < kPer; ++i) {
                const int k = lane + kWarp * i;
                if (k < n) {
                    const float4 lo = xs[2 * c * kSegment + k], hi = xs[(2 * c + 1) * kSegment + k];
#pragma unroll
                    for (int q = 0; q < kR; ++q)
                        bf16_chunk(acc[q][c], v[q][i], lo, hi);
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < kR; ++q) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            if (column_on<kCols>(c, columns)) {
                const float sum = warp_sum(acc[q][c]);
                if (lane == 0 && first + q < m.total)
                    m.store(first + q, sum, c);
            }
        }
    }
}

// Calls f.template operator()<T>() for the formats gemv_q8 takes; false for any other.
template <class F>
bool with_q8_type(Type type, F&& f) {
    switch (type) {
        case Type::Q8_0:
            f.template operator()<Type::Q8_0>();
            return true;
        case Type::Q2_0:
            f.template operator()<Type::Q2_0>();
            return true;
        case Type::Q4_K:
            f.template operator()<Type::Q4_K>();
            return true;
        case Type::Q5_K:
            f.template operator()<Type::Q5_K>();
            return true;
        case Type::Q6_K:
            f.template operator()<Type::Q6_K>();
            return true;
        case Type::IQ2_S:
            f.template operator()<Type::IQ2_S>();
            return true;
        case Type::IQ3_XXS:
            f.template operator()<Type::IQ3_XXS>();
            return true;
        case Type::IQ3_S:
            f.template operator()<Type::IQ3_S>();
            return true;
        case Type::IQ4_NL:
            f.template operator()<Type::IQ4_NL>();
            return true;
        case Type::IQ4_XS:
            f.template operator()<Type::IQ4_XS>();
            return true;
        default:
            return false;
    }
}

int64_t q8_row_bytes(Type type, int cols) {
    return static_cast<int64_t>(row_bytes(*type_info(static_cast<uint32_t>(type)), static_cast<uint64_t>(cols)));
}

// Blocks that stay resident together, `per_sm` on each SM: a launch whose warps loop over the rows needs no more.
int resident_blocks(int per_sm) {
    static const int sms = [] {
        int count = 0;
        TORCH_CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, 0));
        return count;
    }();
    return sms * per_sm;
}

// The blocks of `kernel` that stay resident together with `shared` bytes of dynamic shared memory, at most `blocks`:
// a launch whose warps loop over their work (kLoop) stages its activations once per resident block.
template <class K>
int looping_blocks(K kernel, size_t shared, int blocks) {
    int per_sm = 1;
    TORCH_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, shared));
    return std::min(blocks, resident_blocks(std::max(per_sm, 1)));
}

// A block of a grid format copies its grid (1-8 KB) once and its warps loop over the rows, in as many blocks as stay
// resident. Blocks of the other formats copy only the activations and take one row group a warp; the hardware
// schedules them, which spreads the work more evenly than a fixed share per warp (Q6_K output head: 165 against 158
// GB/s).
template <Type T, int kLanes, bool kSwiglu>
void launch_q8(const Q8Args& a, cudaStream_t stream) {
    const int64_t groups = (a.rows + kWarp / kLanes - 1) / (kWarp / kLanes) * int64_t{a.slots};
    const int blocks = kGridBytes<T> > 0
                           ? std::min(blocks_for(groups, kThreads / kWarp), resident_blocks(kQ8BlocksPerSm<T, kSwiglu>))
                           : blocks_for(groups, kThreads / kWarp);
    gemv_q8_kernel<T, kLanes, kSwiglu><<<blocks, kThreads, q8_shared_bytes<T>(a), stream>>>(a);
}

template <Type T, bool kSwiglu>
void launch_q8_direct(const Q8Args& a, cudaStream_t stream) {
    const int64_t rows = int64_t{a.rows} * a.slots;
    gemv_q8_direct_kernel<T, kSwiglu><<<blocks_for(rows, kThreads / kWarp), kThreads, 0, stream>>>(a);
}

template <Type T, bool kSwiglu>
void launch_q8_short(const Q8Args& a, cudaStream_t stream) {
    const int64_t groups = (a.rows + 7) / 8 * int64_t{a.slots};
    const int blocks = std::min(blocks_for(groups, kThreads / kWarp), resident_blocks(kQ8ShortBlocksPerSm));
    gemv_q8_short_kernel<T, kSwiglu><<<blocks, kThreads, q8_shared_bytes<T>(a), stream>>>(a);
}

// Q4_K, Q5_K and Q8_0 take the unstaged kernel (Q5_K 10240x2560: 154 against 150 GB/s on 32 lanes staged, Q8_0
// 2560x640: 141 against 127 as short rows). Rows of fewer than 32 units take the short-row kernel, others 16 lanes
// when their units fill them (2,560 and 6,144 columns), else a warp.
void launch_q8(Type type, const Q8Args& a, cudaStream_t stream) {
    with_q8_type(type, [&]<Type T>() {
        const bool swiglu = a.up_table != nullptr;
        if constexpr (T == Type::Q4_K || T == Type::Q5_K || T == Type::Q8_0)
            swiglu ? launch_q8_direct<T, true>(a, stream) : launch_q8_direct<T, false>(a, stream);
        else if (a.units < kWarp)
            swiglu ? launch_q8_short<T, true>(a, stream) : launch_q8_short<T, false>(a, stream);
        else if (a.units % 16 == 0)
            swiglu ? launch_q8<T, 16, true>(a, stream) : launch_q8<T, 16, false>(a, stream);
        else
            swiglu ? launch_q8<T, 32, true>(a, stream) : launch_q8<T, 32, false>(a, stream);
    });
}

// Calls f.template operator()<C>() with the smallest compiled column count C >= columns (2 to 8).
template <class F>
void with_columns(int columns, F&& f) {
    if (columns <= 2)
        f.template operator()<2>();
    else if (columns <= 4)
        f.template operator()<4>();
    else
        f.template operator()<8>();
}

// Lets `kernel` take `bytes` of dynamic shared memory: past the 48 KB default only wide rows at many columns ask.
template <class K>
void allow_shared(K kernel, size_t bytes) {
    if (bytes > 48 * 1024)
        TORCH_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
}

// Every format stages its columns' activations, Q4_K, Q5_K and Q8_0 too: their unstaged one-column kernel, which
// converts each column's activations per unit, took Q5_K 2560x6144 at 8 columns 886 us against 652 staged.
void launch_q8_columns(Type type, const ColumnsArgs& a, cudaStream_t stream) {
    with_q8_type(type, [&]<Type T>() {
        if (a.units < kWarp) {
            // runs of 8 rows keep each column's unit sums in registers: four columns a pass
            const size_t shared = kColumnGridBytes<T> + sizeof(Act) * static_cast<size_t>(a.columns) * a.units;
            const int blocks = blocks_for(a.rows, kThreads / kWarp * 8);
            if (a.columns <= 2)
                gemv_q8_short_columns_kernel<T, 2><<<blocks, kThreads, shared, stream>>>(a);
            else
                gemv_q8_short_columns_kernel<T, 4><<<blocks, kThreads, shared, stream>>>(a);
        } else if (a.units % 32 == 0 && a.columns > 2) {
            // a row on a whole warp: twice the blocks of 16 lanes a row, so 3-4 columns of a 2,560-row matrix leave
            // a shorter last wave (80 blocks for 64 resident). At 3-4 columns the resident blocks loop over the rows,
            // staging the columns once each (ssm_out Q6_K 2560x6144 at 4 columns: 112 against 117 us; slower at 5-8).
            const size_t shared = kColumnGridBytes<T> + sizeof(Act) * static_cast<size_t>(a.columns) * a.units;
            with_columns(a.columns, [&]<int C>() {
                const int blocks = blocks_for(a.rows, kThreads / kWarp * kColumnRows<C>);
                if constexpr (C == 4) {
                    const auto kernel = gemv_q8_columns_kernel<T, 32, C, true>;
                    allow_shared(kernel, shared);
                    kernel<<<looping_blocks(kernel, shared, blocks), kThreads, shared, stream>>>(a);
                } else {
                    allow_shared(gemv_q8_columns_kernel<T, 32, C>, shared);
                    gemv_q8_columns_kernel<T, 32, C><<<blocks, kThreads, shared, stream>>>(a);
                }
            });
        } else if (a.units % 16 == 0) {
            const size_t shared = kColumnGridBytes<T> + sizeof(Act) * static_cast<size_t>(a.columns) * a.units;
            with_columns(a.columns, [&]<int C>() {
                allow_shared(gemv_q8_columns_kernel<T, 16, C>, shared);
                gemv_q8_columns_kernel<T, 16, C><<<blocks_for(a.rows, kThreads / kWarp * (kWarp / 16) * kColumnRows<C>),
                                                   kThreads, shared, stream>>>(a);
            });
        } else {
            // shortcut: rows whose units do not fill 16 lanes (no matrix of the model) run one column a launch
            for (int c = 0; c < a.columns; ++c)
                launch_q8(type,
                          {a.w, nullptr, nullptr, 1, a.rows, a.units, a.row_bytes, a.x + int64_t{c} * a.units, 0,
                           a.y + int64_t{c} * a.rows},
                          stream);
        }
    });
}

}  // namespace

void quantize_q8(const float* x, int64_t n, BlockQ8* out, cudaStream_t stream) {
    const int64_t blocks = n / 32;
    quantize_q8_kernel<<<blocks_for(blocks, kRowsPerBlock), kThreads, 0, stream>>>(x, blocks, out);
}

void upload_iq_tables() {
    TORCH_CUDA_CHECK(cudaMemcpyToSymbol(g_iq2s_grid, kIq2sGrid, sizeof kIq2sGrid));
    TORCH_CUDA_CHECK(cudaMemcpyToSymbol(g_iq3xxs_grid, kIq3xxsGrid, sizeof kIq3xxsGrid));
    TORCH_CUDA_CHECK(cudaMemcpyToSymbol(g_iq3s_grid, kIq3sGrid, sizeof kIq3sGrid));
    static uint32_t signed_grid[256 * 16];
    for (int i = 0; i < 256; ++i)
        for (uint32_t n = 0; n < 16; ++n) {
            uint32_t m = 0;
            for (int b = 0; b < 4; ++b)
                m |= (n >> b & 1u) * (0xffu << 8 * b);
            signed_grid[16 * i + n] = (kIq3xxsGrid[i] ^ m) + (m & 0x01010101u);  // set_signed's word
        }
    TORCH_CUDA_CHECK(cudaMemcpyToSymbol(g_iq3xxs_signed, signed_grid, sizeof signed_grid));
}

bool gemv_q8_supports(Type type) {
    return with_q8_type(type, []<Type>() {});
}

void gemv_q8(Type type, const uint8_t* w, int rows, int cols, const BlockQ8* x, float* y, cudaStream_t stream,
             int columns) {
    if (columns <= 1)
        launch_q8(type, {w, nullptr, nullptr, 1, rows, cols / 32, q8_row_bytes(type, cols), x, 0, y}, stream);
    else
        launch_q8_columns(type, {w, rows, cols / 32, q8_row_bytes(type, cols), x, y, columns}, stream);
}

bool gemv_q8_groups(Type type) {
    return type == Type::Q4_K || type == Type::Q5_K || type == Type::Q6_K || type == Type::IQ4_XS;
}

void gemv_q8_group(const Q8Part* parts, int count, int cols, const BlockQ8* x, bool swiglu, cudaStream_t stream,
                   int columns) {
    Q8GroupArgs a{};
    a.units = cols / 32;
    a.x = x;
    for (int p = 0; p < kGroupParts; ++p) {
        a.items[p + 1] = a.items[p];
        if (p < count) {
            a.part[p] = parts[p];
            a.row_bytes[p] = q8_row_bytes(parts[p].type, cols);
            a.items[p + 1] += (parts[p].rows + kWarp / kGroupLanes - 1) / (kWarp / kGroupLanes);
        }
    }
    const size_t shared = sizeof(Act) * static_cast<size_t>(a.units) * (columns <= 1 ? 2 : columns);
    const int blocks = blocks_for(swiglu ? a.items[1] : a.items[count], kThreads / kWarp);
    if (columns <= 1) {
        if (swiglu)
            gemv_q8_group_kernel<true><<<blocks, kThreads, shared, stream>>>(a);
        else
            gemv_q8_group_kernel<false><<<blocks, kThreads, shared, stream>>>(a);
        return;
    }
    const auto run = [&]<int C, int R>() {
        Q8GroupArgs b = a;
        for (int p = 0; p < kGroupParts; ++p)
            b.items[p + 1] =
                b.items[p] +
                (p < count ? (parts[p].rows + kWarp / kGroupLanes * R - 1) / (kWarp / kGroupLanes * R) : 0);
        const int n = blocks_for(swiglu ? b.items[1] : b.items[count], kThreads / kWarp);
        if (R == 2 && !swiglu) {  // as many blocks as stay resident, each staging the columns once
            const auto kernel = gemv_q8_group_columns_kernel<false, C, kGroupLanes, R, true>;
            allow_shared(kernel, shared);
            kernel<<<looping_blocks(kernel, shared, n), kThreads, shared, stream>>>(b, columns);
            return;
        }
        if (swiglu) {
            allow_shared(gemv_q8_group_columns_kernel<true, C, kGroupLanes, R>, shared);
            gemv_q8_group_columns_kernel<true, C, kGroupLanes, R><<<n, kThreads, shared, stream>>>(b, columns);
        } else {
            allow_shared(gemv_q8_group_columns_kernel<false, C, kGroupLanes, R>, shared);
            gemv_q8_group_columns_kernel<false, C, kGroupLanes, R><<<n, kThreads, shared, stream>>>(b, columns);
        }
    };
    // From 3 columns a plain group's lane group takes two rows that share each activation read (DeltaNet qkv+z Q6_K at
    // 5 columns: 279 against 326 us; attention q+k+v at 4: 144 against 167). The shared expert's SwiGLU (640 rows)
    // keeps one: two measured slower (gate+up Q4_K at 4 columns 27.8 against 24.6 us), as do Q6_K-only groups at
    // exactly 3 columns (qkv+z: 224 against 236 us). Two-row launches run as many blocks as stay resident, their warps
    // looping over the row groups, so each block stages the columns once (qkv+z Q6_K+Q4_K at 4 columns: 217 against
    // 243 us; the loop made the one-row launches slower).
    const bool q6k_only = std::all_of(parts, parts + count, [](const Q8Part& p) { return p.type == Type::Q6_K; });
    with_columns(columns, [&]<int C>() {
        if (C >= 4 && !swiglu && !(columns == 3 && q6k_only))
            run.template operator()<C, 2>();
        else
            run.template operator()<C, 1>();
    });
}

void gemv_bf16(const uint8_t* w, int rows, int cols, const float* x, float* y, cudaStream_t stream, int columns) {
    gemv_bf16_pair(w, rows, nullptr, 0, cols, x, y, nullptr, stream, 0.0f, columns);
}

void gemv_bf16_pair(const uint8_t* w, int rows, const uint8_t* w2, int rows2, int cols, const float* x, float* y,
                    float* y2, cudaStream_t stream, float silu_scale, int columns) {
    const Bf16Rows m{w, w2, rows, rows + rows2, int64_t{cols} * 2, y, y2, silu_scale};
    const int chunks = cols / 8;
    if (columns > 1) {
        with_columns(columns, [&]<int C>() {
            if (chunks <= kShortLanes * kShortChunks)
                gemv_bf16_short_columns_kernel<C>
                    <<<blocks_for(m.total, kRowsPerBlock * kShortSteps * (kWarp / kShortLanes)), kThreads,
                       sizeof(float4) * 2 * static_cast<size_t>(chunks) * columns, stream>>>(m, chunks, x, columns);
            // short matrices of short rows, and at 2 columns of long rows too (a mixer's down projection and inject
            // rows, 324 x 10,240: 42.0 against 43.6 us; no faster at 3-6 columns)
            else if (m.total < 2048 && (cols <= 4096 ? columns <= 4 : columns <= 2))
                gemv_bf16_direct_columns_kernel<C>
                    <<<blocks_for(m.total, kRowsPerBlock), kThreads, 0, stream>>>(m, chunks, x, columns);
            else if (m.total >= 2048)  // two rows a warp: 486 against 596 us at 10240x2560, 8 columns
                gemv_bf16_columns_kernel<C, 2>
                    <<<blocks_for(m.total, kRowsPerBlock * 2), kThreads, 0, stream>>>(m, chunks, x, columns);
            else
                gemv_bf16_columns_kernel<C, 1>
                    <<<blocks_for(m.total, kRowsPerBlock), kThreads, 0, stream>>>(m, chunks, x, columns);
        });
        return;
    }
    if (chunks <= kShortLanes * kShortChunks) {
        constexpr int kRows = kRowsPerBlock * kShortSteps * (kWarp / kShortLanes);
        gemv_bf16_short_kernel<kShortSteps><<<blocks_for(m.total, kRows), kThreads, 0, stream>>>(m, chunks, x);
    } else {
        gemv_bf16_kernel<<<blocks_for(m.total, kRowsPerBlock), kThreads, 0, stream>>>(m, chunks, x);
    }
}

bool gemv_bf16_collapse(const uint8_t* w, int n, int hc, int cols, const float* x, const float* xn, float* mixed,
                        BlockQ8* q8, cudaStream_t stream, int columns) {
    const int chunks = cols / 8;
    if (hc != 4 || n % kWarp != 0 || cols % 8 != 0 || chunks > kShortLanes * kShortChunks || columns < 1 ||
        columns > kMaxColumns)
        return false;
    const size_t shared = sizeof(float4) * 2 * static_cast<size_t>(chunks) * columns;
    const auto run = [&]<int C>() {
        gemv_bf16_collapse_kernel<C><<<n / kWarp, kThreads, shared, stream>>>(w, n, chunks, x, xn, mixed, q8, columns);
    };
    if (columns == 1)
        run.template operator()<1>();
    else
        with_columns(columns, run);
    return true;
}

void mixer_down_q8(const uint8_t* w, const uint8_t* inject, int rows, int inject_rows, int cols, const BlockQ8* xq,
                   const float* x, float* low, float* skip, float scale, int columns, cudaStream_t stream) {
    mixer_down_q8_kernel<<<dim3(columns, blocks_for(rows + inject_rows, kRowsPerBlock)), kThreads, 0, stream>>>(
        w, inject, rows, inject_rows, cols, xq, x, low, skip, scale);
}

void mixer_up_q8(const uint8_t* w, int n, const float* low, const float* xn, float* mixed, BlockQ8* q8, int columns,
                 cudaStream_t stream) {
    auto run = [&]<int C>() {
        mixer_up_q8_kernel<C><<<n / kWarp, kThreads, 0, stream>>>(w, n, low, xn, mixed, q8, columns);
    };
    if (columns == 1)
        run.template operator()<1>();
    else
        with_columns(columns, run);
}

void gemv_q8_experts(Type type, const uint8_t* const* table, const uint8_t* const* up_table, int slots, int rows,
                     int cols, const BlockQ8* x, int64_t x_stride, float* y, cudaStream_t stream) {
    // Slots a launch stages within 48 KB of shared memory (q8_shared_bytes, and the short-row kernel's lane sums): a
    // window's down projection (8 x 10 slots, each with its own activations) runs as two launches; a slot's rows do
    // not depend on its launch.
    const int units = cols / 32;
    int per = slots;
    with_q8_type(type, [&]<Type T>() {
        const size_t fixed = kGridBytes<T> + (x_stride == 0 ? sizeof(Act) * units : 0) +
                             (units < kWarp ? q8_short_static_bytes(up_table != nullptr) : 0);
        const size_t each = 2 * sizeof(const uint8_t*) + (x_stride == 0 ? 0 : sizeof(Act) * units);
        per = static_cast<int>(std::min<size_t>(static_cast<size_t>(slots), (48 * 1024 - fixed) / each));
    });
    for (int s = 0; s < slots; s += per)
        launch_q8(type,
                  {nullptr, table + s, up_table != nullptr ? up_table + s : nullptr, std::min(per, slots - s), rows,
                   units, q8_row_bytes(type, cols), x + s * x_stride, x_stride, y + int64_t{s} * rows},
                  stream);
}

// Columns a pass of gemv_q8_experts_window (an expert routed by more columns takes further passes over its rows, mostly
// from L1): at 8 columns a pass needs 80 registers and fewer blocks stay resident (IQ2_S at T = 8, half of each
// column's experts shared: 14.4 us per unique expert in one pass, 11.2 in passes of 4, 12.2 in passes of 2).
constexpr int kWindowPass = 4;

void gemv_q8_experts_window(Type type, const uint8_t* const* gate, const uint8_t* const* up, const uint8_t* columns,
                            const int8_t* rank, const int32_t* count, int slots, int k, int rows, int cols,
                            const BlockQ8* x, float* h, int window, cudaStream_t stream) {
    if (window <= 1) {  // one column numbers its unique experts by rank: M3's launch
        gemv_q8_experts(type, gate, up, k, rows, cols, x, 0, h, stream);
        return;
    }
    const ExpertsWindowArgs a{gate, up, columns, rank, count, slots, k, rows, cols / 32, q8_row_bytes(type, cols),
                              x,    h,  window};
    const int64_t items = int64_t{slots} * ((rows + 1) / 2);
    with_q8_type(type, [&]<Type T>() {
        const size_t shared = kColumnGridBytes<T> + sizeof(Act) * static_cast<size_t>(window) * a.units;
        with_columns(std::min(window, kWindowPass), [&]<int C>() {
            const auto kernel = gemv_q8_experts_window_kernel<T, C>;
            allow_shared(kernel, shared);
            // as many blocks as are resident at once: a block past them would start its share of the items late
            static size_t cached_shared = 0;
            static int per_sm = 0;
            if (cached_shared != shared) {
                TORCH_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, shared));
                cached_shared = shared;
            }
            const int blocks = static_cast<int>(
                std::min<int64_t>(blocks_for(items, kThreads / kWarp), resident_blocks(std::max(per_sm, 1))));
            kernel<<<blocks, kThreads, shared, stream>>>(a);
        });
    });
}

void gemv_q8_experts_down_window(Type type, const uint8_t* const* down, const uint8_t* columns, const int8_t* rank,
                                 const int32_t* count, int slots, int k, int rows, int cols, const BlockQ8* x, float* y,
                                 int window, cudaStream_t stream) {
    const int units = cols / 32;
    if (window <= 1)  // one column numbers its unique experts by rank: M3's launch
        return gemv_q8_experts(type, down, nullptr, k, rows, cols, x, units, y, stream);
    if (units >= kWarp)
        throw Error("gemv_q8_experts_down_window takes rows of fewer than 1,024 values.");
    const DownWindowArgs a{down, columns, rank, count, slots, k, rows, units, q8_row_bytes(type, cols), x, y};
    const int64_t items = int64_t{slots} * ((rows + kQ8ShortRows * kDownRuns - 1) / (kQ8ShortRows * kDownRuns));
    with_q8_type(type, [&]<Type T>() {
        const size_t shared =
            kColumnGridBytes<T> + sizeof(Act) * kThreads / kWarp * kDownColumns * static_cast<size_t>(units);
        const auto kernel = gemv_q8_down_window_kernel<T>;
        allow_shared(kernel, shared);
        static size_t cached_shared = 0;
        static int per_sm = 0;
        if (cached_shared != shared) {
            TORCH_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, shared));
            cached_shared = shared;
        }
        const int blocks = static_cast<int>(
            std::min<int64_t>(blocks_for(items, kThreads / kWarp), resident_blocks(std::max(per_sm, 1))));
        kernel<<<blocks, kThreads, shared, stream>>>(a);
    });
}

}  // namespace torch::kernels
