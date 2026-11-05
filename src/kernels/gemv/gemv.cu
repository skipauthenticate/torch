#include <algorithm>

#include "core/cuda_check.h"
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
// shortcut: launches ask for no more than the default 48 KB (about 850 activation blocks; the model's largest is the
// down projection's 200), so a caller with far more slots or columns would need cudaFuncSetAttribute first.
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
template <Type T, bool kSwiglu>
__global__ void __launch_bounds__(kThreads, kQ8ShortBlocksPerSm) gemv_q8_short_kernel(Q8Args a) {
    constexpr int kRows = 8, kPitch = kWarp + 4;  // the pitch puts the 32 lanes' reads in 32 banks
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
    __device__ void store(int r, float sum) const {
        if (r >= rows)
            y2[r - rows] = sum;
        else
            y[r] = scale != 0.0f ? silu(sum * scale) : sum;  // as silu_scale (kernels/hc) of the stored sum
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

}  // namespace

void quantize_q8(const float* x, int64_t n, BlockQ8* out, cudaStream_t stream) {
    const int64_t blocks = n / 32;
    quantize_q8_kernel<<<blocks_for(blocks, kRowsPerBlock), kThreads, 0, stream>>>(x, blocks, out);
}

void upload_iq_tables() {
    TORCH_CUDA_CHECK(cudaMemcpyToSymbol(g_iq2s_grid, kIq2sGrid, sizeof kIq2sGrid));
    TORCH_CUDA_CHECK(cudaMemcpyToSymbol(g_iq3xxs_grid, kIq3xxsGrid, sizeof kIq3xxsGrid));
    TORCH_CUDA_CHECK(cudaMemcpyToSymbol(g_iq3s_grid, kIq3sGrid, sizeof kIq3sGrid));
}

bool gemv_q8_supports(Type type) {
    return with_q8_type(type, []<Type>() {});
}

void gemv_q8(Type type, const uint8_t* w, int rows, int cols, const BlockQ8* x, float* y, cudaStream_t stream) {
    launch_q8(type, {w, nullptr, nullptr, 1, rows, cols / 32, q8_row_bytes(type, cols), x, 0, y}, stream);
}

bool gemv_q8_groups(Type type) {
    return type == Type::Q4_K || type == Type::Q5_K || type == Type::Q6_K || type == Type::IQ4_XS;
}

void gemv_q8_group(const Q8Part* parts, int count, int cols, const BlockQ8* x, bool swiglu, cudaStream_t stream) {
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
    const size_t shared = 2 * sizeof(Act) * static_cast<size_t>(a.units);
    if (swiglu)
        gemv_q8_group_kernel<true><<<blocks_for(a.items[1], kThreads / kWarp), kThreads, shared, stream>>>(a);
    else
        gemv_q8_group_kernel<false><<<blocks_for(a.items[count], kThreads / kWarp), kThreads, shared, stream>>>(a);
}

void gemv_bf16(const uint8_t* w, int rows, int cols, const float* x, float* y, cudaStream_t stream) {
    gemv_bf16_pair(w, rows, nullptr, 0, cols, x, y, nullptr, stream);
}

void gemv_bf16_pair(const uint8_t* w, int rows, const uint8_t* w2, int rows2, int cols, const float* x, float* y,
                    float* y2, cudaStream_t stream, float silu_scale) {
    const Bf16Rows m{w, w2, rows, rows + rows2, int64_t{cols} * 2, y, y2, silu_scale};
    const int chunks = cols / 8;
    if (chunks <= kShortLanes * kShortChunks) {
        constexpr int kRows = kRowsPerBlock * kShortSteps * (kWarp / kShortLanes);
        gemv_bf16_short_kernel<kShortSteps><<<blocks_for(m.total, kRows), kThreads, 0, stream>>>(m, chunks, x);
    } else {
        gemv_bf16_kernel<<<blocks_for(m.total, kRowsPerBlock), kThreads, 0, stream>>>(m, chunks, x);
    }
}

void gemv_q8_experts(Type type, const uint8_t* const* table, const uint8_t* const* up_table, int slots, int rows,
                     int cols, const BlockQ8* x, int64_t x_stride, float* y, cudaStream_t stream) {
    launch_q8(type, {nullptr, table, up_table, slots, rows, cols / 32, q8_row_bytes(type, cols), x, x_stride, y},
              stream);
}

}  // namespace torch::kernels
