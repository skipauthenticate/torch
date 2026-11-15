#include <algorithm>

#include "core/cuda_check.h"
#include "core/error.h"
#include "formats/codebooks.h"
#include "kernels/common.cuh"
#include "kernels/gemm/gemm.h"
#include "kernels/gemv/q8.cuh"

namespace torch::kernels {
namespace {

constexpr int kStages = 2;  // each warp's private ring of activation stages
// An item's raw rows are staged 32 units at a time (two stages) through the same shared memory as the rings; the
// widest run is Q8_0's 32 blocks of 34 bytes.
constexpr int kRawMax = 32 * 34, kRawPitchMax = (kRawMax + 15 + 15) / 16 * 16 + 16;
// A warp's unit of one octet: 32 columns x 32 int8, their scales (F32) and their sums (int16).
constexpr int kWarpStage = 32 * 32 + 32 * 4 + 32 * 2;
// The tree's nodes [lane set][16 rows][pass columns], rows 6 floats more apart than a pass is wide: the warps' pair
// stores and the cross-set reads (16 rows of 2 columns a warp) then hit nearly distinct banks (at the pass width: 8-
// and 16-way conflicts).
__host__ __device__ constexpr int node_pitch(int threads) {
    return 32 * (threads / 256) + 6;
}
// The ring of a block of 512 threads (8 lane sets x 2 column halves, 64-column passes, one block an SM).
constexpr int kRing = std::max({16 * kStages * kWarpStage, 2 * 16 * kRawPitchMax, 8 * 16 * node_pitch(512) * 4});

// The 16 IQ4_NL levels as four words of int8 bytes, and the levels of the 4 nibbles held in bytes 0-3 of x (each below
// 16), in byte order (gemv's dot_q8.cuh does the same per 8 nibbles).
__host__ __device__ constexpr uint32_t iq4_word(int w) {
    uint32_t word = 0;
    for (int i = 0; i < 4; ++i)
        word |= static_cast<uint32_t>(static_cast<uint8_t>(kIq4nlValues[4 * w + i])) << 8 * i;
    return word;
}
__device__ __forceinline__ uint32_t iq4_plain(uint32_t x) {
    const uint32_t v = __byte_perm(x | x >> 4, 0, 0x0020);  // nibbles 0-3 into bytes 0 and 1
    const uint32_t index = v & 0x77777777u;
    const uint32_t pick = (v >> 1 & 0x44444444u) | 0x32103210u;
    return __byte_perm(__byte_perm(iq4_word(0), iq4_word(1), index), __byte_perm(iq4_word(2), iq4_word(3), index),
                       pick);
}

// Scale j (0-7) and its minimum from the 12 packed 6-bit values of Q4_K and Q5_K (as formats/dequant.h).
__device__ __forceinline__ void k_scale_min(const uint8_t* q, int j, int& scale, int& min) {
    if (j < 4) {
        scale = q[j] & 63;
        min = q[j + 4] & 63;
    } else {
        scale = (q[j + 4] & 0xf) | (q[j - 4] >> 6) << 4;
        min = (q[j + 4] >> 4) | (q[j] >> 6) << 4;
    }
}

__device__ __forceinline__ void mma16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(b), "r"(0));
}
__device__ __forceinline__ void mma32(int (&d)[4], uint4 a, uint2 b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(0));
}
__device__ __forceinline__ void copy16(void* shared, const void* global) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(shared))),
        "l"(global));
}
__device__ __forceinline__ void copy_commit() {
    asm volatile("cp.async.commit_group;");
}
template <int kPending>
__device__ __forceinline__ void copy_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(kPending));
}

// Word j (values 4j..4j+3) of sub-block s of the Q6_K block at b (2-byte aligned), as q - 32 in int8 bytes: the
// formula's -32 * sum16 folds into the exact sum.
__device__ __forceinline__ uint32_t q6k_word(const uint8_t* b, int s, int j) {
    const int n = s / 4, k = s % 4;
    const uint32_t low = load_u32_a2(b + 64 * n + 32 * (k % 2) + 4 * j) >> 4 * (k / 2) & 0x0f0f0f0fu;
    const uint32_t high = (load_u32_a2(b + 128 + 32 * n + 4 * j) >> 2 * k & 0x03030303u) << 4;
    const uint32_t x = (low | high) ^ 0x20202020u;  // q - 32 where q >= 32; q + 32 where q < 32
    return x | (x & 0x20202020u) * 6;               // ... then bits 6 and 7 set: q - 32 as int8
}

// Per format: units and bytes of a block; whether each 16-value half has a sum of its own (Q6_K) and whether the
// formula reads the activations' sums (Q4_K, Q5_K); `words` gives a row's words t and 4 + t of sub-unit s in plain
// value order (2-byte-aligned loads: decode runs once per item); `record` packs rows g (ba) and g + 8 (bb)'s scales of
// sub-unit s into 16 bytes; `value` is gemv_q8's per-unit formula (dot_q8.cuh) for row i (0: g, 1: g + 8) of the
// record, from the unit's integer sum(s), the column's scale and its activation sum, written as the one-column kernel
// writes it so the same compiler contraction applies.
template <Type T>
struct Fmt;

template <>
struct Fmt<Type::Q6_K> {
    static constexpr int kUnits = 8, kBytes = 210;
    static constexpr bool kHalves = true, kSums = false;
    __device__ static void words(const uint8_t* b, int s, int t, uint32_t& w0, uint32_t& w1) {
        w0 = q6k_word(b, s, t), w1 = q6k_word(b, s, 4 + t);
    }
    __device__ static uint4 record(const uint8_t* ba, const uint8_t* bb, int s) {
        const uint32_t sa = load_u16(ba + 192 + 8 * (s / 4) + 2 * (s % 4));
        const uint32_t sb = load_u16(bb + 192 + 8 * (s / 4) + 2 * (s % 4));
        return make_uint4(load_u16(ba + 208), load_u16(bb + 208), sa | sb << 16, 0);
    }
    __device__ static float value(const uint4& r, int i, int h0, int h1, float dx, int) {
        const uint32_t sc = r.z >> 16 * i;
        const float d = f16_to_f32(static_cast<uint16_t>(i == 0 ? r.x : r.y));
        return d * dx *
               static_cast<float>(static_cast<int8_t>(sc & 0xff) * h0 + static_cast<int8_t>(sc >> 8 & 0xff) * h1);
    }
};

template <Type T>
struct FmtK {  // Q4_K and Q5_K: d, dmin @0; 12 packed 6-bit scales and minima @4; Q5_K's high bits @16, nibbles @48
    static constexpr int kUnits = 8, kBytes = T == Type::Q4_K ? 144 : 176;
    static constexpr bool kHalves = false, kSums = true;
    __device__ static uint32_t word(const uint8_t* b, int s, int j) {
        const uint32_t low =
            load_u32_a2(b + (T == Type::Q4_K ? 16 : 48) + 32 * (s / 2) + 4 * j) >> 4 * (s % 2) & 0x0f0f0f0fu;
        if constexpr (T == Type::Q4_K)
            return low;
        else
            return low | (load_u32_a2(b + 16 + 4 * j) >> s & 0x01010101u) << 4;
    }
    __device__ static void words(const uint8_t* b, int s, int t, uint32_t& w0, uint32_t& w1) {
        w0 = word(b, s, t), w1 = word(b, s, 4 + t);
    }
    __device__ static uint4 record(const uint8_t* ba, const uint8_t* bb, int s) {
        int sa, ma, sb, mb;
        k_scale_min(ba + 4, s, sa, ma);
        k_scale_min(bb + 4, s, sb, mb);
        return make_uint4(load_u16(ba) | load_u16(ba + 2) << 16, load_u16(bb) | load_u16(bb + 2) << 16,
                          static_cast<uint32_t>(sa | ma << 8 | sb << 16 | mb << 24), 0);
    }
    __device__ static float value(const uint4& r, int i, int sum, int, float dx, int xsum) {
        const uint32_t dd = i == 0 ? r.x : r.y, sm = r.z >> 16 * i;
        const float d = f16_to_f32(static_cast<uint16_t>(dd & 0xffff)),
                    dmin = f16_to_f32(static_cast<uint16_t>(dd >> 16));
        const int scale = static_cast<int>(sm & 0xff), min = static_cast<int>(sm >> 8 & 0xff);
        return dx * (d * static_cast<float>(sum * scale) - dmin * static_cast<float>(xsum * min));
    }
};
template <>
struct Fmt<Type::Q4_K> : FmtK<Type::Q4_K> {};
template <>
struct Fmt<Type::Q5_K> : FmtK<Type::Q5_K> {};

template <>
struct Fmt<Type::IQ4_XS> {  // d @0, high scale bits @2, low scale nibbles @4, nibbles @8: byte i of sub-unit s holds
                            // values i (low nibble) and 16 + i (high)
    static constexpr int kUnits = 8, kBytes = 136;
    static constexpr bool kHalves = false, kSums = false;
    __device__ static void words(const uint8_t* b, int s, int t, uint32_t& w0, uint32_t& w1) {
        const uint32_t v = load_u32_a2(b + 8 + 16 * s + 4 * t);
        w0 = iq4_plain(v & 0x0f0f0f0fu), w1 = iq4_plain(v >> 4 & 0x0f0f0f0fu);
    }
    __device__ static int scale(const uint8_t* b, int s) {
        return static_cast<int>((b[4 + s / 2] >> 4 * (s % 2) & 0xf) | (load_u16(b + 2) >> 2 * s & 3) << 4);
    }
    __device__ static uint4 record(const uint8_t* ba, const uint8_t* bb, int s) {
        return make_uint4(load_u16(ba), load_u16(bb), static_cast<uint32_t>(scale(ba, s) | scale(bb, s) << 16), 0);
    }
    __device__ static float value(const uint4& r, int i, int sum, int, float dx, int) {
        const float d = f16_to_f32(static_cast<uint16_t>(i == 0 ? r.x : r.y));
        return d * dx * static_cast<float>(sum * (static_cast<int>(r.z >> 16 * i & 0xffff) - 32));
    }
};

template <>
struct Fmt<Type::IQ4_NL> {  // 32 values in 18 bytes: d @0, nibbles @2 as IQ4_XS's sub-units
    static constexpr int kUnits = 1, kBytes = 18;
    static constexpr bool kHalves = false, kSums = false;
    __device__ static void words(const uint8_t* b, int, int t, uint32_t& w0, uint32_t& w1) {
        const uint32_t v = load_u32_a2(b + 2 + 4 * t);
        w0 = iq4_plain(v & 0x0f0f0f0fu), w1 = iq4_plain(v >> 4 & 0x0f0f0f0fu);
    }
    __device__ static uint4 record(const uint8_t* ba, const uint8_t* bb, int) {
        return make_uint4(load_u16(ba), load_u16(bb), 0, 0);
    }
    __device__ static float value(const uint4& r, int i, int sum, int, float dx, int) {
        return f16_to_f32(static_cast<uint16_t>(i == 0 ? r.x : r.y)) * dx * static_cast<float>(sum);
    }
};

template <>
struct Fmt<Type::Q8_0> {  // 32 values in 34 bytes: d @0, int8 @2
    static constexpr int kUnits = 1, kBytes = 34;
    static constexpr bool kHalves = false, kSums = false;
    __device__ static void words(const uint8_t* b, int, int t, uint32_t& w0, uint32_t& w1) {
        w0 = load_u32_a2(b + 2 + 4 * t), w1 = load_u32_a2(b + 18 + 4 * t);
    }
    __device__ static uint4 record(const uint8_t* ba, const uint8_t* bb, int) {
        return make_uint4(load_u16(ba), load_u16(bb), 0, 0);
    }
    __device__ static float value(const uint4& r, int i, int sum, int, float dx, int) {
        return f16_to_f32(static_cast<uint16_t>(i == 0 ? r.x : r.y)) * dx * static_cast<float>(sum);
    }
};

// The IQ grids (formats/codebooks.h), copied by upload_gemm_tables(); decode reads them once per item.
__device__ __align__(16) uint64_t g_grid2s[1024];
__device__ __align__(16) uint32_t g_grid3xxs[256];
__device__ __align__(16) uint32_t g_grid3s[512];

// Four grid bytes g, negated where bits 0-3 of `signs` are set, as int8 bytes (gemv's set_signed).
__device__ __forceinline__ uint32_t signed_word(uint32_t g, uint32_t signs) {
    uint32_t m = 0;
    asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(m) : "r"((signs & 0xf) * 0x10204080u));
    return (g ^ m) + (m & 0x01010101u);
}

template <>
struct Fmt<Type::IQ2_S> {  // d @0, qs[32] @2, signs[32] @34, qh[8] @66, scales[8] @74: group j of sub-unit s is grid
                           // entry qs[4s + j] with 2 high bits from qh[s], signs[4s + j]; scales per 16 values
    static constexpr int kUnits = 8, kBytes = 82;
    static constexpr bool kHalves = true, kSums = false;
    __device__ static uint32_t word(const uint8_t* b, int s, int w) {  // w = 0..7: values 4w..4w+3
        const int j = w / 2;
        const uint32_t qs = b[2 + 4 * s + j], qh = b[66 + s], signs = b[34 + 4 * s + j];
        const uint2 entry = reinterpret_cast<const uint2*>(g_grid2s)[qs | (qh << (8 - 2 * j) & 0x300)];
        return signed_word(w % 2 == 0 ? entry.x : entry.y, w % 2 == 0 ? signs : signs >> 4);
    }
    __device__ static void words(const uint8_t* b, int s, int t, uint32_t& w0, uint32_t& w1) {
        w0 = word(b, s, t), w1 = word(b, s, 4 + t);
    }
    __device__ static uint4 record(const uint8_t* ba, const uint8_t* bb, int s) {
        return make_uint4(load_u16(ba), load_u16(bb), ba[74 + s] | static_cast<uint32_t>(bb[74 + s]) << 16, 0);
    }
    __device__ static float value(const uint4& r, int i, int h0, int h1, float dx, int) {
        const uint32_t sc = r.z >> 16 * i;
        const int total = (2 * static_cast<int>(sc & 0xf) + 1) * h0 + (2 * static_cast<int>(sc >> 4 & 0xf) + 1) * h1;
        return f16_to_f32(static_cast<uint16_t>(i == 0 ? r.x : r.y)) * dx * (0.125f * static_cast<float>(total));
    }
};

template <>
struct Fmt<Type::IQ3_XXS> {  // d @0, qs[64] @2, per sub-unit a u32 @66: four 7-bit sign indices and a 4-bit scale
    static constexpr int kUnits = 8, kBytes = 98;
    static constexpr bool kHalves = false, kSums = false;
    __device__ static uint32_t word(const uint8_t* b, int s, int w) {
        const int j = w / 2;
        const uint32_t aux = load_u32_a2(b + 66 + 4 * s), signs7 = aux >> 7 * j & 127;
        const uint32_t signs = signs7 | (__popc(signs7) & 1u) << 7;
        return signed_word(g_grid3xxs[b[2 + 8 * s + w]], w % 2 == 0 ? signs : signs >> 4);
    }
    __device__ static void words(const uint8_t* b, int s, int t, uint32_t& w0, uint32_t& w1) {
        w0 = word(b, s, t), w1 = word(b, s, 4 + t);
    }
    __device__ static uint4 record(const uint8_t* ba, const uint8_t* bb, int s) {
        return make_uint4(load_u16(ba), load_u16(bb),
                          load_u32_a2(ba + 66 + 4 * s) >> 28 | (load_u32_a2(bb + 66 + 4 * s) >> 28) << 16, 0);
    }
    __device__ static float value(const uint4& r, int i, int sum, int, float dx, int) {
        const int s0 = 2 * static_cast<int>(r.z >> 16 * i & 0xf) + 1;
        return f16_to_f32(static_cast<uint16_t>(i == 0 ? r.x : r.y)) * dx * (0.25f * static_cast<float>(s0 * sum));
    }
};

template <>
struct Fmt<Type::IQ3_S> {  // d @0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106: entry 2j + k of sub-unit s
                           // is qs[8s + 2j + k] with a ninth bit from qh[s], sign byte signs[4s + j]
    static constexpr int kUnits = 8, kBytes = 110;
    static constexpr bool kHalves = false, kSums = false;
    __device__ static uint32_t word(const uint8_t* b, int s, int w) {
        const int j = w / 2;
        const uint32_t qh = b[66 + s], signs = b[74 + 4 * s + j];
        const uint32_t index = b[2 + 8 * s + w] | (qh << (8 - w) & 256);
        return signed_word(g_grid3s[index], w % 2 == 0 ? signs : signs >> 4);
    }
    __device__ static void words(const uint8_t* b, int s, int t, uint32_t& w0, uint32_t& w1) {
        w0 = word(b, s, t), w1 = word(b, s, 4 + t);
    }
    __device__ static uint4 record(const uint8_t* ba, const uint8_t* bb, int s) {
        const uint32_t sa = ba[106 + s / 2] >> 4 * (s % 2) & 0xf, sb = bb[106 + s / 2] >> 4 * (s % 2) & 0xf;
        return make_uint4(load_u16(ba), load_u16(bb), sa | sb << 16, 0);
    }
    __device__ static float value(const uint4& r, int i, int sum, int, float dx, int) {
        const int s0 = 1 + 2 * static_cast<int>(r.z >> 16 * i & 0xf);
        return f16_to_f32(static_cast<uint16_t>(i == 0 ? r.x : r.y)) * dx * static_cast<float>(s0 * sum);
    }
};

template <>
struct Fmt<Type::Q2_0> {  // 64 values in 18 bytes: d @0, qs[16] @2; value 4i + k of sub-unit s is crumb k of byte
                          // 8s + i, meaning code - 1 (-1, 0, 1, 2): the formula's - sum folds into the exact sum
    static constexpr int kUnits = 2, kBytes = 18;
    static constexpr bool kHalves = false, kSums = false;
    __device__ static uint32_t word(const uint8_t* b, int s, int w) {
        const uint32_t x = b[2 + 8 * s + w];
        return __vsub4((x | x << 6 | x << 12 | x << 18) & 0x03030303u, 0x01010101u);
    }
    __device__ static void words(const uint8_t* b, int s, int t, uint32_t& w0, uint32_t& w1) {
        w0 = word(b, s, t), w1 = word(b, s, 4 + t);
    }
    __device__ static uint4 record(const uint8_t* ba, const uint8_t* bb, int) {
        return make_uint4(load_u16(ba), load_u16(bb), 0, 0);
    }
    __device__ static float value(const uint4& r, int i, int sum, int, float dx, int) {
        return f16_to_f32(static_cast<uint16_t>(i == 0 ? r.x : r.y)) * dx * static_cast<float>(sum);
    }
};

struct GemmArgs {
    const uint8_t* w;
    int rows, units;
    int64_t row_bytes;
    Q8Columns x;
    int range;  // columns an item (a multiple of 64)
    float* y;
    // Routed experts (gemm_q8_experts): an item is (expert, 16 rows) over the expert's bucket of slots; null for a
    // dense matrix. index: the output row of each slot (negative: a padding slot, not stored); null: the slot itself.
    const uint8_t* const* table;
    ExpertBuckets buckets;
    const int32_t* index;
};

// An item's raw rows: 32 units of a row at a time, each run 16-byte aligned with its offset kept.
template <Type T>
__host__ __device__ constexpr int raw_pitch() {
    return (32 / Fmt<T>::kUnits * Fmt<T>::kBytes + 15 + 15) / 16 * 16 + 16;
}
// The ring of a block of 256 threads (8 lane sets, 32-column passes): two such blocks share an SM when they fit, so one
// block's decode and tree run beside the other's products.
template <Type T>
constexpr int kRing256 = std::max({8 * kStages * kWarpStage, 2 * 16 * raw_pitch<T>(), 8 * 16 * node_pitch(256) * 4});

// The kernel (gemm.h) for format T at kT threads (512 or 256). Shared memory: the item's fragments [unit][32 lanes]
// [16 B] (a thread's a0..a3: words t of rows g and g + 8, then words 4 + t), its records [unit][8 g][16 B], then the
// ring.
template <Type T, int kT>
__global__ void __launch_bounds__(kT, 512 / kT) gemm_kernel(GemmArgs a) {
    using F = Fmt<T>;
    constexpr int kRunUnits = 32, kRun = kRunUnits / F::kUnits * F::kBytes;  // raw bytes of 32 units of a row
    constexpr int kRawPitch = raw_pitch<T>(), kRawChunks = kRawPitch / 16;
    constexpr int kPass = 32 * (kT / 256), kPitch = node_pitch(kT);  // a pass's columns
    static_assert(2 * 16 * kRawPitch <= (kT == 512 ? kRing : kRing256<T>), "the ring holds two raw stages");
    extern __shared__ uint4 smem[];
    uint8_t* frag = reinterpret_cast<uint8_t*>(smem);
    uint8_t* rec = frag + size_t{512} * a.units;
    uint8_t* ring = rec + size_t{128} * a.units;
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp, g = lane / 4, t = lane % 4;
    const int ls = warp % 8, half = warp / 8;
    const int octets = (a.units + 7) / 8, row_tiles = a.rows / 16;  // a last octet may be short (K = 640: 20 units)
    const bool routed = a.table != nullptr;
    const int ranges = routed ? *a.buckets.n_list : (a.x.columns + a.range - 1) / a.range, items = row_tiles * ranges;
    for (int item = blockIdx.x; item < items; item += gridDim.x) {
        const int row0 = item % row_tiles * 16, group = item / row_tiles;
        const int expert = routed ? a.buckets.list[group] : 0;
        const uint8_t* const w = routed ? a.table[expert] : a.w;
        const int first = routed ? a.buckets.first[expert] : group * a.range;
        const int last = routed ? first + a.buckets.counts[expert] : min(first + a.range, a.x.columns);
        if (first >= last)
            continue;  // an unrouted expert of a read-ahead pass (the whole block)
        // ---- decode the 16 rows, 32 units at a time
        const int chunks = (a.units + kRunUnits - 1) / kRunUnits;
        const auto run_of = [&](int i, int c) { return w + int64_t{row0 + i} * a.row_bytes + c * kRun; };
        const auto issue_raw = [&](int c) {
            uint8_t* stage = ring + (c % 2) * 16 * kRawPitch;
            const int bytes = min(kRunUnits, a.units - kRunUnits * c) / F::kUnits * F::kBytes;
            for (int f = threadIdx.x; f < 16 * kRawChunks; f += kT) {
                const int i = f / kRawChunks, j = f % kRawChunks;
                const uint8_t* run = run_of(i, c);
                const auto* src =
                    reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(run) & ~uintptr_t{15}) + 16 * j;
                if (src < run + bytes)
                    copy16(stage + i * kRawPitch + 16 * j, src);
            }
            copy_commit();
        };
        issue_raw(0);
        for (int c = 0; c < chunks; ++c) {
            if (c + 1 < chunks) {
                issue_raw(c + 1);
                copy_wait<1>();
            } else {
                copy_wait<0>();
            }
            __syncthreads();
            const uint8_t* stage = ring + (c % 2) * 16 * kRawPitch;
            const int units = min(kRunUnits, a.units - kRunUnits * c);
            for (int task = threadIdx.x; task < units * kWarp; task += kT) {
                const int lu = task / kWarp, l = task % kWarp, gg = l / 4, tt = l % 4;
                const int s = lu % F::kUnits, at = lu / F::kUnits * F::kBytes, u = kRunUnits * c + lu;
                const uint8_t* ra = stage + gg * kRawPitch + (reinterpret_cast<uintptr_t>(run_of(gg, c)) & 15) + at;
                const uint8_t* rb =
                    stage + (gg + 8) * kRawPitch + (reinterpret_cast<uintptr_t>(run_of(gg + 8, c)) & 15) + at;
                uint32_t a0, a1, a2, a3;
                F::words(ra, s, tt, a0, a2);
                F::words(rb, s, tt, a1, a3);
                *reinterpret_cast<uint4*>(frag + size_t{512} * u + 16 * l) = make_uint4(a0, a1, a2, a3);
                if (tt == 0)
                    *reinterpret_cast<uint4*>(rec + size_t{128} * u + 16 * gg) = F::record(ra, rb, s);
            }
            __syncthreads();  // the next issue overwrites this stage; the last hands the ring to the passes
        }
        // ---- passes of kPass columns
        for (int col0 = first; col0 < last; col0 += kPass) {
            uint8_t* wring = ring + warp * kStages * kWarpStage;
            const int wcol = col0 + 32 * half;
            // the output column of each of this thread's tree outputs (i = tid + kT m: column i / 16), read now so a
            // routed down projection's route index is not a load in the tree
            constexpr int kOuts = 16 * kPass / kT;
            int outs[kOuts];
#pragma unroll
            for (int m = 0; m < kOuts; ++m) {
                const int c = (static_cast<int>(threadIdx.x) + kT * m) / 16;
                outs[m] = col0 + c >= last ? -1 : a.index ? a.index[col0 + c] : col0 + c;
            }
            const auto issue = [&](int b) {  // this warp's unit of octet b (an empty group past the end)
                if (b < octets && 8 * b + ls < a.units) {
                    uint8_t* stage = wring + (b % kStages) * kWarpStage;
                    const int64_t at = int64_t{8 * b + ls} * a.x.stride + wcol;
                    for (int f = lane; f < 64 + 8 + (F::kSums ? 4 : 0); f += kWarp) {
                        if (f < 64)
                            copy16(stage + 16 * f, a.x.q + at * 32 + 16 * f);
                        else if (f < 72)
                            copy16(stage + 1024 + 16 * (f - 64), a.x.d + at + 4 * (f - 64));
                        else
                            copy16(stage + 1152 + 16 * (f - 72), a.x.sum + at + 8 * (f - 72));
                    }
                }
                copy_commit();
            };
            float acc[4][4][4] = {};  // [n-tile][pair][canonical lane ls + 8j]
#pragma unroll
            for (int b = 0; b < kStages - 1; ++b)
                issue(b);
            for (int b0 = 0; b0 < octets; b0 += 4)
#pragma unroll
                for (int j = 0; j < 4; ++j) {  // octet b0 + j's unit ls is canonical lane ls + 8j
                    const int b = b0 + j;
                    if (b >= octets)
                        break;
                    issue(b + kStages - 1);
                    copy_wait<kStages - 1>();
                    __syncwarp();
                    const int u = 8 * b + ls;
                    if (u >= a.units)  // this lane set has no unit in a short last octet
                        continue;
                    const uint4 fr = *reinterpret_cast<const uint4*>(frag + size_t{512} * u + 16 * lane);
                    const uint4 rc = *reinterpret_cast<const uint4*>(rec + size_t{128} * u + 16 * g);
                    const uint8_t* stage = wring + (b % kStages) * kWarpStage;
#pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        const uint2 xb = *reinterpret_cast<const uint2*>(stage + (8 * n + g) * 32 + 8 * t);
                        const float2 dx = *reinterpret_cast<const float2*>(stage + 1024 + (8 * n + 2 * t) * 4);
                        int xs[2] = {0, 0};
                        if constexpr (F::kSums) {
                            const uint32_t two = *reinterpret_cast<const uint32_t*>(stage + 1152 + (8 * n + 2 * t) * 2);
                            xs[0] = static_cast<int16_t>(two & 0xffff), xs[1] = static_cast<int16_t>(two >> 16);
                        }
                        int h0[4], h1[4] = {};
                        if constexpr (F::kHalves) {
                            mma16(h0, fr.x, fr.y, xb.x);
                            mma16(h1, fr.z, fr.w, xb.y);
                        } else {
                            mma32(h0, fr, xb);
                        }
#pragma unroll
                        for (int p = 0; p < 4; ++p)
                            acc[n][p][j] += F::value(rc, p / 2, h0[p], h1[p], p % 2 == 0 ? dx.x : dx.y, xs[p % 2]);
                    }
                    __syncwarp();  // this warp's next issue overwrites the stage it read
                }
            copy_wait<0>();
            __syncthreads();  // every warp is done with its ring: the tree takes it
            auto* node = reinterpret_cast<float(*)[16][kPitch]>(ring);
            const auto pair = [&](int n, int p) {  // ^16 and ^8 of the butterfly: lanes j, j + 2, then the pairs
                return (acc[n][p][0] + acc[n][p][2]) + (acc[n][p][1] + acc[n][p][3]);
            };
#pragma unroll
            for (int n = 0; n < 4; ++n)
#pragma unroll
                for (int h = 0; h < 2; ++h)  // rows g and g + 8: pairs 2h and 2h + 1 are adjacent columns
                    *reinterpret_cast<float2*>(&node[ls][g + 8 * h][32 * half + 8 * n + 2 * t]) =
                        make_float2(pair(n, 2 * h), pair(n, 2 * h + 1));
            __syncthreads();
#pragma unroll
            for (int m = 0; m < kOuts; ++m) {  // ^4, ^2, ^1 across the lane sets
                const int i = static_cast<int>(threadIdx.x) + kT * m, row = i % 16, c = i / 16, out = outs[m];
                if (out < 0)
                    continue;
                float v[8];
#pragma unroll
                for (int k = 0; k < 8; ++k)
                    v[k] = node[k][row][c];
                a.y[int64_t{out} * a.rows + row0 + row] =
                    ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
            }
            __syncthreads();  // the ring is the next pass's (or item's) again
        }
    }
}

template <Type T, int kT>
void run_gemm(const GemmArgs& a, size_t shared, int blocks, cudaStream_t stream) {
    static size_t allowed = 0;
    if (shared > allowed) {
        TORCH_CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel<T, kT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(shared)));
        allowed = shared;
    }
    gemm_kernel<T, kT><<<blocks, kT, shared, stream>>>(a);
}

template <Type T>
bool launch_short(const GemmArgs& a, cudaStream_t stream);

// K <= 640 in the formats gemm_short_kernel takes; else two blocks of 256 threads an SM when their shared memory fits
// (K up to ~4,000), else one of 512.
template <Type T>
void launch_gemm(const GemmArgs& a, cudaStream_t stream) {
    if (launch_short<T>(a, stream))
        return;
    static int sms = 0, per_sm = 0, reserved = 0;
    if (sms == 0) {
        TORCH_CUDA_CHECK(cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
        TORCH_CUDA_CHECK(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, 0));
        TORCH_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    }
    const size_t base = size_t{640} * a.units, two = base + kRing256<T>;
    if (2 * (two + static_cast<size_t>(reserved)) <= static_cast<size_t>(per_sm))
        return run_gemm<T, 256>(a, two, 2 * sms, stream);
    run_gemm<T, 512>(a, base + kRing, sms, stream);
}

// ---- K <= 640 (the experts' down projections, the shared expert's down): a warp runs an output's whole tree ----
//
// With at most 20 units, every canonical lane holds at most one unit: a lane's sum is that unit's value, and an output
// is warp_sum's butterfly over them. A block of 16 warps takes items of 128 rows x a column range or bucket: it stages
// the item's raw rows, decodes them into fragments as gemm_kernel does, then for each 32-column pass (the activations
// staged by the block, two stages) each warp takes 16 rows x 2 of the 4 n-tiles, runs their units in the butterfly's
// depth-first order (lanes 0, 16, 8, 24, ...: bitrev5) and folds each finished pair into the tree at once, as
// gemm_bf16_kernel folds its passes: no tree through shared memory. Measured (Orin, 2026-11-14) against gemm_kernel on
// the experts' IQ4_NL down projections of a 4,096-token prompt: 1,115 -> 707 ms.
constexpr int kShortUnits = 20, kShortRows = 128, kShortThreads = 512, kShortTilesN = 2;  // n-tiles a warp
constexpr int kShortActQ = kShortUnits * 32 * 32, kShortActD = kShortUnits * 32 * 4;
constexpr int kShortStage = kShortActQ + kShortActD + kShortUnits * 32 * 2;
template <Type T>
__host__ __device__ constexpr int short_pitch() {  // a row's raw run, 16-byte aligned with its offset kept
    return ((kShortUnits + Fmt<T>::kUnits - 1) / Fmt<T>::kUnits * Fmt<T>::kBytes + 15 + 15) / 16 * 16 + 16;
}
template <Type T>
constexpr int kShortShared =
    kShortRows / 16 * kShortUnits * 640 + std::max(2 * kShortStage, kShortRows* short_pitch<T>());

template <int L>
__device__ __forceinline__ void short_fold(float (&node)[5][kShortTilesN][4], float (&s)[kShortTilesN][4]) {
#pragma unroll
    for (int n = 0; n < kShortTilesN; ++n)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            float v = s[n][q];
#pragma unroll
            for (int t = 0; t < L; ++t)
                v = node[t][n][q] + v;
            if constexpr (L < 5)
                node[L][n][q] = v;
            else
                s[n][q] = v;
        }
}

template <Type T>
__global__ void __launch_bounds__(kShortThreads, 1) gemm_short_kernel(GemmArgs a) {
    using F = Fmt<T>;
    constexpr int kPitch = short_pitch<T>(), kRawChunks = kPitch / 16, kTiles = kShortRows / 16;
    extern __shared__ uint4 smem[];
    uint8_t* const frags = reinterpret_cast<uint8_t*>(smem);   // [tile][unit][32 lanes][16 B]
    uint8_t* const recs = frags + kTiles * kShortUnits * 512;  // [tile][unit][8 g][16 B]
    uint8_t* const act = recs + kTiles * kShortUnits * 128;    // raw rows, then the passes' two stages
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp, g = lane / 4, t = lane % 4;
    const int tile = warp % kTiles, n0 = warp / kTiles * kShortTilesN;  // the warp's 16 rows and 8-column n-tiles
    const uint8_t* const frag = frags + tile * kShortUnits * 512;
    const uint8_t* const rec = recs + tile * kShortUnits * 128;
    const bool routed = a.table != nullptr;
    const int row_items = a.rows / kShortRows;
    const int ranges = routed ? *a.buckets.n_list : (a.x.columns + a.range - 1) / a.range, items = row_items * ranges;
    const int run = (a.units + F::kUnits - 1) / F::kUnits * F::kBytes;
    for (int item = blockIdx.x; item < items; item += gridDim.x) {
        const int row0 = item % row_items * kShortRows, group = item / row_items;
        const int expert = routed ? a.buckets.list[group] : 0;
        const uint8_t* const w = routed ? a.table[expert] : a.w;
        const int first = routed ? a.buckets.first[expert] : group * a.range;
        const int last = routed ? first + a.buckets.counts[expert] : min(first + a.range, a.x.columns);
        if (first >= last)
            continue;  // an unrouted expert of a read-ahead pass (the whole block)
        // ---- the item's raw rows, then their fragments and records (gemm_kernel's decode)
        const auto row_of = [&](int i) { return w + int64_t{row0 + i} * a.row_bytes; };
        for (int f = threadIdx.x; f < kShortRows * kRawChunks; f += kShortThreads) {
            const int i = f / kRawChunks, j = f % kRawChunks;
            const uint8_t* r = row_of(i);
            const auto* src =
                reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(r) & ~uintptr_t{15}) + 16 * j;
            if (src < r + run)
                copy16(act + i * kPitch + 16 * j, src);
        }
        copy_commit();
        copy_wait<0>();
        __syncthreads();
        for (int task = threadIdx.x; task < kTiles * a.units * kWarp; task += kShortThreads) {
            const int l = task % kWarp, u = task / kWarp % a.units, m = task / kWarp / a.units;
            const int gg = l / 4, tt = l % 4, s = u % F::kUnits, at = u / F::kUnits * F::kBytes;
            const int ia = 16 * m + gg, ib = ia + 8;
            const uint8_t* ra = act + ia * kPitch + (reinterpret_cast<uintptr_t>(row_of(ia)) & 15) + at;
            const uint8_t* rb = act + ib * kPitch + (reinterpret_cast<uintptr_t>(row_of(ib)) & 15) + at;
            uint32_t a0, a1, a2, a3;
            F::words(ra, s, tt, a0, a2);
            F::words(rb, s, tt, a1, a3);
            *reinterpret_cast<uint4*>(frags + (m * kShortUnits + u) * 512 + 16 * l) = make_uint4(a0, a1, a2, a3);
            if (tt == 0)
                *reinterpret_cast<uint4*>(recs + (m * kShortUnits + u) * 128 + 16 * gg) = F::record(ra, rb, s);
        }
        __syncthreads();  // the fragments are ready; the passes' stages take the raw rows' memory
        // ---- 32-column passes
        const int passes = (last - first + 31) / 32;
        const auto issue = [&](int p) {  // pass p's activations (units x 32 columns) into stage p % 2
            uint8_t* st = act + (p % 2) * kShortStage;
            const int col0 = first + 32 * p, nq = 64 * a.units, nd = 8 * a.units, ns = F::kSums ? 4 * a.units : 0;
            for (int f = threadIdx.x; f < nq + nd + ns; f += kShortThreads) {
                if (f < nq)
                    copy16(st + 1024 * (f / 64) + 16 * (f % 64),
                           a.x.q + (int64_t{f / 64} * a.x.stride + col0) * 32 + 16 * (f % 64));
                else if (f < nq + nd)
                    copy16(st + kShortActQ + 128 * ((f - nq) / 8) + 16 * ((f - nq) % 8),
                           a.x.d + int64_t{(f - nq) / 8} * a.x.stride + col0 + 4 * ((f - nq) % 8));
                else
                    copy16(st + kShortActQ + kShortActD + 64 * ((f - nq - nd) / 4) + 16 * ((f - nq - nd) % 4),
                           a.x.sum + int64_t{(f - nq - nd) / 4} * a.x.stride + col0 + 8 * ((f - nq - nd) % 4));
            }
            copy_commit();
        };
        issue(0);
        for (int p = 0; p < passes; ++p) {
            if (p + 1 < passes) {
                issue(p + 1);
                copy_wait<1>();
            } else {
                copy_wait<0>();
            }
            __syncthreads();  // pass p's stage is in
            const uint8_t* st = act + (p % 2) * kShortStage;
            const int col0 = first + 32 * p;
            float node[5][kShortTilesN][4], s[kShortTilesN][4];  // [level][n-tile][rows g, g + 8 x columns 2t, 2t + 1]
#pragma unroll
            for (int i = 0; i < 32; ++i) {  // canonical lane bitrev5(i): unit u (or none)
                const int u = static_cast<int>(__brev(static_cast<unsigned>(i)) >> 27);
#pragma unroll
                for (int n = 0; n < kShortTilesN; ++n)
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                        s[n][q] = 0.0f;
                if (u < a.units) {
                    const uint4 fr = *reinterpret_cast<const uint4*>(frag + 512 * u + 16 * lane);
                    const uint4 rc = *reinterpret_cast<const uint4*>(rec + 128 * u + 16 * g);
#pragma unroll
                    for (int n = 0; n < kShortTilesN; ++n) {
                        const int c = 8 * (n0 + n);
                        const uint2 xb = *reinterpret_cast<const uint2*>(st + (32 * u + c + g) * 32 + 8 * t);
                        const float2 dx = *reinterpret_cast<const float2*>(st + kShortActQ + (32 * u + c + 2 * t) * 4);
                        int xs[2] = {0, 0};
                        if constexpr (F::kSums) {
                            const uint32_t two = *reinterpret_cast<const uint32_t*>(st + kShortActQ + kShortActD +
                                                                                    (32 * u + c + 2 * t) * 2);
                            xs[0] = static_cast<int16_t>(two & 0xffff), xs[1] = static_cast<int16_t>(two >> 16);
                        }
                        int h0[4], h1[4] = {};
                        if constexpr (F::kHalves) {
                            mma16(h0, fr.x, fr.y, xb.x);
                            mma16(h1, fr.z, fr.w, xb.y);
                        } else {
                            mma32(h0, fr, xb);
                        }
#pragma unroll
                        for (int q = 0; q < 4; ++q)  // the lane's sum: 0 + its one unit's value, as gemm_kernel's
                            s[n][q] += F::value(rc, q / 2, h0[q], h1[q], q % 2 == 0 ? dx.x : dx.y, xs[q % 2]);
                    }
                }
                switch (__ffs(i + 1) - 1) {  // known once unrolled: levels 0 .. L - 1 complete
                    case 0:
                        short_fold<0>(node, s);
                        break;
                    case 1:
                        short_fold<1>(node, s);
                        break;
                    case 2:
                        short_fold<2>(node, s);
                        break;
                    case 3:
                        short_fold<3>(node, s);
                        break;
                    case 4:
                        short_fold<4>(node, s);
                        break;
                    default:
                        short_fold<5>(node, s);
                        break;
                }
            }
#pragma unroll
            for (int n = 0; n < kShortTilesN; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int c = col0 + 8 * (n0 + n) + 2 * t + q % 2, row = row0 + 16 * tile + g + 8 * (q / 2);
                    const int out = c >= last ? -1 : a.index ? a.index[c] : c;
                    if (out >= 0)
                        a.y[int64_t{out} * a.rows + row] = s[n][q];
                }
            __syncthreads();  // every warp is done with this stage (and, after the last pass, the fragments)
        }
    }
}

// K <= 640 in IQ4_NL or Q2_0 with whole 128-row items: gemm_short_kernel, one block an SM.
template <Type T>
bool launch_short(const GemmArgs& a, cudaStream_t stream) {
    if constexpr (T != Type::IQ4_NL && T != Type::Q2_0) {
        return false;
    } else {
        if (a.units > kShortUnits || a.rows % kShortRows != 0)
            return false;
        static int sms = 0;
        if (sms == 0) {
            TORCH_CUDA_CHECK(cudaFuncSetAttribute(gemm_short_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  kShortShared<T>));
            TORCH_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        }
        gemm_short_kernel<T><<<sms, kShortThreads, kShortShared<T>, stream>>>(a);
        return true;
    }
}

// Buckets (gemm.h): block e counts expert e's routes; one block turns the counts into bucket starts (padded to
// multiples of 8 slots) and the list of routed experts; block e then writes its routes' slots in route order, marks its
// padding, adds its count to the route counts and stamps it with the window stamp of its last route.
constexpr int kBucketThreads = 256;

// h = silu(g) * u (as swiglu) of the real slots of each expert in b's list, quantized (as quantize_q8) into out's
// column layout (as q8_to_columns): one thread a (slot, unit), one block a listed expert. A thread quantizes its 32
// values as quantize_q8_block does across a warp: the same per-value arithmetic, and the maximum and the integer sum
// do not depend on order.
__global__ void expert_h_columns_kernel(const float* g, const float* u, ExpertBuckets b, Q8Columns out) {
    if (static_cast<int>(blockIdx.x) >= *b.n_list)
        return;
    const int e = b.list[blockIdx.x], first = b.first[e], tasks = b.counts[e] * out.units;
    for (int task = threadIdx.x; task < tasks; task += kBucketThreads) {
        const int s = first + task / out.units, unit = task % out.units;
        const int64_t i = (int64_t{s} * out.units + unit) * 32;
        const auto* g4 = reinterpret_cast<const float4*>(g + i);
        const auto* u4 = reinterpret_cast<const float4*>(u + i);
        float v[32];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float4 a = g4[k], c = u4[k];
            v[4 * k] = silu(a.x) * c.x, v[4 * k + 1] = silu(a.y) * c.y;
            v[4 * k + 2] = silu(a.z) * c.z, v[4 * k + 3] = silu(a.w) * c.w;
        }
        float amax = fabsf(v[0]);
#pragma unroll
        for (int k = 1; k < 32; ++k)
            amax = fmaxf(amax, fabsf(v[k]));
        const float d = amax / 127.0f;
        uint32_t q[8] = {};
        int sum = 0;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            const int x = amax == 0.0f ? 0 : static_cast<int>(roundf(v[k] / d));
            sum += x;
            q[k / 4] |= static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(x))) << 8 * (k % 4);
        }
        const int64_t at = int64_t{unit} * out.stride + s;
        auto* o = reinterpret_cast<uint4*>(out.q + at * 32);
        o[0] = make_uint4(q[0], q[4], q[1], q[5]);
        o[1] = make_uint4(q[2], q[6], q[3], q[7]);
        out.d[at] = f16_to_f32(__half_as_ushort(__float2half_rn(d)));
        out.sum[at] = static_cast<int16_t>(sum);
    }
}

__global__ void bucket_count_kernel(const int32_t* ids, int routes, int32_t* counts) {
    __shared__ int partial[kBucketThreads / kWarp];
    int n = 0;
    for (int r = threadIdx.x; r < routes; r += kBucketThreads)
        n += ids[r] == static_cast<int>(blockIdx.x);
    for (int offset = 16; offset > 0; offset >>= 1)
        n += __shfl_xor_sync(0xffffffffu, n, offset);
    if (threadIdx.x % kWarp == 0)
        partial[threadIdx.x / kWarp] = n;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < kBucketThreads / kWarp; ++w)
            total += partial[w];
        counts[blockIdx.x] = total;
    }
}

__global__ void bucket_scan_kernel(int n_expert, ExpertBuckets b) {
    if (threadIdx.x != 0)
        return;
    int at = 0, listed = 0;
    for (int e = 0; e < n_expert; ++e) {
        b.first[e] = at;
        at += (b.counts[e] + 7) / 8 * 8;
        if (b.counts[e] > 0)
            b.list[listed++] = e;
    }
    *b.n_list = listed;
}

__global__ void bucket_fill_kernel(const int32_t* ids, int routes, int k, ExpertBuckets b, int layer, int n_expert,
                                   const int32_t* pos, unsigned long long* route_counts, uint32_t* last_use) {
    __shared__ int offsets[kBucketThreads / kWarp];
    const int e = static_cast<int>(blockIdx.x), lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
    const int start = b.first[e], count = b.counts[e];
    if (count == 0)
        return;
    int placed = 0, last_route = 0;
    for (int base = 0; base < routes; base += kBucketThreads) {
        const int r = base + static_cast<int>(threadIdx.x);
        const bool mine = r < routes && ids[r] == e;
        const unsigned ballot = __ballot_sync(0xffffffffu, mine);
        if (lane == 0)
            offsets[warp] = __popc(ballot);
        __syncthreads();
        int before = placed;
        for (int w = 0; w < warp; ++w)
            before += offsets[w];
        if (mine) {
            b.slot_route[start + before + __popc(ballot & ((1u << lane) - 1u))] = r;
            last_route = r;
        }
        for (int w = 0; w < kBucketThreads / kWarp; ++w)
            placed += offsets[w];
        __syncthreads();  // offsets is the next round's
    }
    for (int i = count + static_cast<int>(threadIdx.x); i < (count + 7) / 8 * 8; i += kBucketThreads)
        b.slot_route[start + i] = -1;
    // the window stamp of the expert's last route: its 8-column group's first position + 1, as feeding windows of 8
    // from the chunk's start stamps it (expert_gate)
    for (int offset = 16; offset > 0; offset >>= 1)
        last_route = max(last_route, __shfl_xor_sync(0xffffffffu, last_route, offset));
    __syncthreads();
    if (lane == 0)
        offsets[warp] = last_route;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 0; w < kBucketThreads / kWarp; ++w)
            last_route = max(last_route, offsets[w]);
        const int64_t at = int64_t{layer} * n_expert + e;
        route_counts[at] += static_cast<unsigned long long>(count);
        last_use[at] = static_cast<uint32_t>(*pos + last_route / k / 8 * 8) + 1u;
    }
}

// A block gathers 32 slots x 8 units through shared memory: it reads each slot's 8 BlockQ8 (72 words in a row of its
// route's column), then writes each unit's 32 slots in a row. A null slot_route re-lays columns in order (slot s is
// column s: q8_to_columns).
constexpr int kGatherSlots = 32, kGatherUnits = 8, kGatherWords = kGatherUnits * 9, kGatherPitch = kGatherWords + 1;
__global__ void q8_gather_kernel(const BlockQ8* x, int units, const int32_t* slot_route, int k, Q8Columns out) {
    __shared__ uint32_t tile[kGatherSlots * kGatherPitch];  // [slot][BlockQ8 words], pitch 73: no bank conflicts
    __shared__ int32_t routes[kGatherSlots];
    const int s0 = static_cast<int>(blockIdx.x) * kGatherSlots, u0 = static_cast<int>(blockIdx.y) * kGatherUnits;
    const int t = static_cast<int>(threadIdx.x);
    if (t < kGatherSlots)
        routes[t] = s0 + t >= out.columns ? -1 : slot_route ? slot_route[s0 + t] : s0 + t;
    __syncthreads();
    const int words = min(kGatherUnits, units - u0) * 9;
    for (int w = t; w < kGatherSlots * kGatherWords; w += kBucketThreads) {
        const int j = w / kGatherWords, wi = w % kGatherWords, r = routes[j];
        if (r >= 0 && wi < words)
            tile[j * kGatherPitch + wi] = reinterpret_cast<const uint32_t*>(x + int64_t{r / k} * units + u0)[wi];
    }
    __syncthreads();
    const int j = t % kGatherSlots, s = s0 + j, u = u0 + t / kGatherSlots;  // a warp: one unit's 32 slots
    if (s >= out.columns || u >= units)
        return;
    const int64_t at = int64_t{u} * out.stride + s;
    auto* o = reinterpret_cast<uint4*>(out.q + at * 32);
    if (routes[j] < 0) {  // padding
        o[0] = o[1] = make_uint4(0, 0, 0, 0);
        out.d[at] = 0.0f;
        out.sum[at] = 0;
        return;
    }
    const uint32_t* b = tile + j * kGatherPitch + (u - u0) * 9;  // d and sum, then the 8 words of q
    o[0] = make_uint4(b[1], b[5], b[2], b[6]);
    o[1] = make_uint4(b[3], b[7], b[4], b[8]);
    out.d[at] = f16_to_f32(static_cast<uint16_t>(b[0] & 0xffff));
    out.sum[at] = static_cast<int16_t>(b[0] >> 16);
}

// ---- BF16 (gemm.h) ----
constexpr int kBfRows = 64, kBfCols = 64, kBfThreads = 256, kBfStage = 8;  // chunks staged at a time
constexpr uint16_t kBfLast = 0x8000;  // in the pass sequence: the chunk ends its pass

struct BfArgs {
    const uint8_t *w, *w2;    // rows r < rows from w (into y), the rest from w2 (into y2): a pair shares the launch
    int rows, total, chunks;  // chunks of 8 values a row
    int end;                  // gemm_bf16_kernel stores rows < end; gemm_bf16_few_kernel the rest
    const float* x;
    float *y, *y2;
    int columns;
    float silu_scale;  // non-zero: w's rows store silu(silu_scale * sum)
};

__device__ __forceinline__ float bf16_value(uint32_t v) {
    return __uint_as_float(v << 16);
}

// One stage of chunks in shared memory, value-major: a thread's 4 rows and 4 columns at one value are one 8-byte and
// one 16-byte load.
struct BfStage {
    uint16_t w[kBfStage * 8][kBfRows];
    float x[kBfStage * 8][kBfCols];
};

// A thread's share of a stage's loads: 2 weight chunks (16 B, rows f % 64 for f = tid, tid + 256) and 4 activation
// half-chunks (16 B, columns f % 64), consecutive threads on consecutive rows and columns so the transposed stores
// fall in different banks.
struct BfLoads {
    uint4 w[2];
    float4 x[4];
};

// A pass's sums into the tree when levels 0 .. L - 1 complete: they join the pending nodes in order and wait at
// level L (or, after pass 31, L = 5, are the butterfly's total); the next pass starts from 0.
template <int L>
__device__ __forceinline__ void bf_fold(float (&node)[5][4][4], float (&acc)[4][4]) {
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            float v = acc[r][c];
#pragma unroll
            for (int t = 0; t < L; ++t)
                v = node[t][r][c] + v;
            if constexpr (L < 5) {
                node[L][r][c] = v;
                acc[r][c] = 0.0f;
            } else {
                acc[r][c] = v;
            }
        }
}

// Thread (ty, tx) = (threadIdx.x % 16, threadIdx.x / 16) holds rows 4 ty .. 4 ty + 3 and columns 4 tx .. 4 tx + 3 of
// the block's tile: a warp's weight loads at one value are 128 contiguous bytes and its activation loads two
// broadcasts, one shared-memory wavefront each. Pass i runs canonical lane bitrev5(i) over its chunks in order; the 32
// passes' chunks form one sequence (shared memory after the stages: chunk index, kBfLast on a pass's last), which goes
// through two shared-memory stages kBfStage chunks at a time whatever the passes' bounds (the next stage's loads in
// flight while the current one is multiplied), multiplied in chunk then value order into one accumulator per output. At
// a pass's last chunk its sums join the tree: a binary counter over the passes, level t combining when 2^(t + 1) passes
// are done, which with the bit-reversed order pairs lanes l and l ^ 16 first, then ^ 8, ^ 4, ^ 2, ^ 1, as warp_sum
// does. Every lane has a chunk (chunks >= 32), so no pass is empty.
__global__ void __launch_bounds__(kBfThreads, 1) gemm_bf16_kernel(BfArgs a) {
    extern __shared__ uint4 bf_shared[];
    BfStage* stage = reinterpret_cast<BfStage*>(bf_shared);
    uint16_t* order = reinterpret_cast<uint16_t*>(stage + 2);
    if (threadIdx.x < 32) {  // pass i's chunks start after passes 0 .. i - 1's
        const auto count = [&](int i) {
            return (a.chunks - static_cast<int>(__brev(static_cast<unsigned>(i)) >> 27) + 31) / 32;
        };
        const int i = static_cast<int>(threadIdx.x), lane = static_cast<int>(__brev(threadIdx.x) >> 27), n = count(i);
        int f = 0;
        for (int p = 0; p < i; ++p)
            f += count(p);
        for (int j = 0; j < n; ++j)
            order[f + j] = static_cast<uint16_t>((lane + 32 * j) | (j == n - 1 ? kBfLast : 0));
    }
    __syncthreads();
    const int ty = threadIdx.x % 16, tx = threadIdx.x / 16;  // a warp: 16 row groups, 2 column groups
    const int row0 = blockIdx.y * kBfRows, col0 = blockIdx.x * kBfCols;
    const auto weights = [&](int row) {
        const int r = min(row, a.total - 1);
        return reinterpret_cast<const uint4*>(r < a.rows ? a.w + int64_t{r} * a.chunks * 16
                                                         : a.w2 + int64_t{r - a.rows} * a.chunks * 16);
    };
    const uint4* wrow[2] = {weights(row0 + static_cast<int>(threadIdx.x) % kBfRows),
                            weights(row0 + (static_cast<int>(threadIdx.x) + kBfThreads) % kBfRows)};
    const auto load = [&](int f0, int n, BfLoads& l) {  // sequence entries f0 .. f0 + n - 1
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int j = (static_cast<int>(threadIdx.x) + k * kBfThreads) / kBfRows;
            l.w[k] = j < n ? wrow[k][order[f0 + j] & ~kBfLast] : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int f = static_cast<int>(threadIdx.x) + k * kBfThreads, c = f % kBfCols, j = f / kBfCols / 2,
                      h = f / kBfCols % 2;
            const int col = min(col0 + c, a.columns - 1);
            l.x[k] = j < n ? *reinterpret_cast<const float4*>(
                                 a.x + (int64_t{col} * a.chunks + (order[f0 + j] & ~kBfLast)) * 8 + 4 * h)
                           : make_float4(0, 0, 0, 0);
        }
    };
    const auto store = [&](const BfLoads& l, BfStage& st) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int f = static_cast<int>(threadIdx.x) + k * kBfThreads, r = f % kBfRows, j = f / kBfRows;
            const uint32_t words[4] = {l.w[k].x, l.w[k].y, l.w[k].z, l.w[k].w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                st.w[8 * j + 2 * e][r] = static_cast<uint16_t>(words[e] & 0xffff);
                st.w[8 * j + 2 * e + 1][r] = static_cast<uint16_t>(words[e] >> 16);
            }
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int f = static_cast<int>(threadIdx.x) + k * kBfThreads, c = f % kBfCols, j = f / kBfCols / 2,
                      h = f / kBfCols % 2;
            st.x[8 * j + 4 * h][c] = l.x[k].x, st.x[8 * j + 4 * h + 1][c] = l.x[k].y;
            st.x[8 * j + 4 * h + 2][c] = l.x[k].z, st.x[8 * j + 4 * h + 3][c] = l.x[k].w;
        }
    };
    float node[5][4][4];  // the tree's pending node at each level
    float acc[4][4] = {};
    int pass = 0;
    BfLoads next;
    load(0, min(kBfStage, a.chunks), next);
    for (int f0 = 0, s = 0; f0 < a.chunks; f0 += kBfStage, s ^= 1) {
        const int n = min(kBfStage, a.chunks - f0);
        store(next, stage[s]);  // its last readers finished before the barrier at the end of the stage before last
        __syncthreads();
        if (f0 + kBfStage < a.chunks)  // the next stage's loads fly while this one is multiplied
            load(f0 + kBfStage, min(kBfStage, a.chunks - f0 - kBfStage), next);
        const BfStage& st = stage[s];
        for (int j = 0; j < n; ++j) {  // chunk by chunk, value by value: the lane's FMA chain
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int k = 8 * j + e;
                const uint2 wv = *reinterpret_cast<const uint2*>(&st.w[k][4 * ty]);
                const float4 xv = *reinterpret_cast<const float4*>(&st.x[k][4 * tx]);
                const float wr[4] = {bf16_value(wv.x & 0xffff), bf16_value(wv.x >> 16), bf16_value(wv.y & 0xffff),
                                     bf16_value(wv.y >> 16)};
                const float xc[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                for (int r = 0; r < 4; ++r)
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                        acc[r][c] += wr[r] * xc[c];  // as gemv's bf16_chunk writes it
            }
            if ((order[f0 + j] & kBfLast) == 0)
                continue;
            // fold the pass into the tree: levels 0 .. tz - 1 complete (tz: the trailing zeros of pass + 1, the same
            // for the whole block), the sum waits at level tz; after pass 31 (tz 5) it is the butterfly's total
            switch (__ffs(pass + 1) - 1) {  // a branch the whole block takes: only its levels' adds run
                case 0:
                    bf_fold<0>(node, acc);
                    break;
                case 1:
                    bf_fold<1>(node, acc);
                    break;
                case 2:
                    bf_fold<2>(node, acc);
                    break;
                case 3:
                    bf_fold<3>(node, acc);
                    break;
                case 4:
                    bf_fold<4>(node, acc);
                    break;
                default:
                    bf_fold<5>(node, acc);
                    break;
            }
            ++pass;
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int row = row0 + 4 * ty + r, col = col0 + 4 * tx + c;
            if (row >= a.end || col >= a.columns)
                continue;
            if (row < a.rows)
                a.y[int64_t{col} * a.rows + row] =
                    a.silu_scale != 0.0f ? silu(acc[r][c] * a.silu_scale) : acc[r][c];  // as gemv_bf16_pair
            else
                a.y2[int64_t{col} * (a.total - a.rows) + row - a.rows] = acc[r][c];
        }
}

// The last few rows of a product (at most kBfFewMax past its whole 64-row tiles: a mixer's inject rows, the router's
// shared gate), which would otherwise take a tile of their own: a warp a column and 4 rows, as gemv_bf16 runs a row
// (lane l's chunks l, l + 32, ... as one FMA chain each, then warp_sum), so each output is the same sum as the tiles'.
constexpr int kBfFewRows = 4, kBfFewMax = 16;
__global__ void __launch_bounds__(kBfThreads) gemm_bf16_few_kernel(BfArgs a) {
    const int lane = static_cast<int>(threadIdx.x) % kWarp;
    const int col = static_cast<int>(blockIdx.x) * (kBfThreads / kWarp) + static_cast<int>(threadIdx.x) / kWarp;
    const int row0 = a.end + static_cast<int>(blockIdx.y) * kBfFewRows;
    if (col >= a.columns)
        return;
    const uint4* w[kBfFewRows];
#pragma unroll
    for (int i = 0; i < kBfFewRows; ++i) {
        const int r = min(row0 + i, a.total - 1);
        w[i] = reinterpret_cast<const uint4*>(r < a.rows ? a.w + int64_t{r} * a.chunks * 16
                                                         : a.w2 + int64_t{r - a.rows} * a.chunks * 16);
    }
    const auto* x = reinterpret_cast<const float4*>(a.x + int64_t{col} * a.chunks * 8);
    float acc[kBfFewRows] = {};
    for (int j = lane; j < a.chunks; j += kWarp) {
        const float4 x0 = x[2 * j], x1 = x[2 * j + 1];
        const float xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int i = 0; i < kBfFewRows; ++i) {
            const uint4 v = w[i][j];
            const uint32_t ws[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int e = 0; e < 8; ++e)  // as gemv's bf16_chunk
                acc[i] += bf16_value(e % 2 == 0 ? ws[e / 2] & 0xffff : ws[e / 2] >> 16) * xs[e];
        }
    }
#pragma unroll
    for (int i = 0; i < kBfFewRows; ++i) {
        const float v = warp_sum(acc[i]);
        const int row = row0 + i;
        if (lane != 0 || row >= a.total)
            continue;
        if (row < a.rows)
            a.y[int64_t{col} * a.rows + row] = a.silu_scale != 0.0f ? silu(v * a.silu_scale) : v;
        else
            a.y2[int64_t{col} * (a.total - a.rows) + row - a.rows] = v;
    }
}

}  // namespace

void gemm_bf16(const uint8_t* w, int rows, int cols, const float* x, float* y, int columns, float silu_scale,
               cudaStream_t stream, const uint8_t* w2, int rows2, float* y2) {
    if (cols % 8 != 0 || cols < 256 || cols / 8 > kBfLast || rows < 1 || columns < 1 || rows2 < 0)
        throw Error("gemm_bf16 takes rows of 256 to 262,144 values (a multiple of 8) and at least one column.");
    BfArgs a{w, w2, rows, rows + rows2, cols / 8, rows + rows2, x, y, y2, columns, silu_scale};
    const int few = a.total % kBfRows;
    if (a.total > kBfRows && few <= kBfFewMax) {
        a.end = a.total - few;
        if (few > 0)
            gemm_bf16_few_kernel<<<dim3(
                                       static_cast<unsigned>((columns + kBfThreads / kWarp - 1) / (kBfThreads / kWarp)),
                                       static_cast<unsigned>((few + kBfFewRows - 1) / kBfFewRows)),
                                   kBfThreads, 0, stream>>>(a);
    }
    const size_t shared = 2 * sizeof(BfStage) + sizeof(uint16_t) * static_cast<size_t>(a.chunks);
    static size_t allowed = 0;
    if (shared > allowed) {
        TORCH_CUDA_CHECK(cudaFuncSetAttribute(gemm_bf16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(shared)));
        allowed = shared;
    }
    const dim3 grid(static_cast<unsigned>((columns + kBfCols - 1) / kBfCols),
                    static_cast<unsigned>((a.end + kBfRows - 1) / kBfRows));
    gemm_bf16_kernel<<<grid, kBfThreads, shared, stream>>>(a);
}

int expert_slots(int n_expert, int routes) {
    return routes + 7 * n_expert;
}

size_t expert_buckets_bytes(int n_expert, int routes) {
    return sizeof(int32_t) *
           (3 * static_cast<size_t>(n_expert) + 4 + static_cast<size_t>(expert_slots(n_expert, routes)));
}

ExpertBuckets expert_buckets_at(void* memory, int n_expert, int routes) {
    auto* p = static_cast<int32_t*>(memory);
    const size_t n = static_cast<size_t>(n_expert);
    return {p, p + n, p + 2 * n, p + 3 * n, p + 3 * n + 4, expert_slots(n_expert, routes)};
}

void expert_buckets(const int32_t* ids, int routes, int k, int n_expert, const ExpertBuckets& b, int layer,
                    const int32_t* pos, unsigned long long* route_counts, uint32_t* last_use, cudaStream_t stream) {
    TORCH_CUDA_CHECK(cudaMemsetAsync(b.slot_route, 0xff, sizeof(int32_t) * static_cast<size_t>(b.slots), stream));
    bucket_count_kernel<<<n_expert, kBucketThreads, 0, stream>>>(ids, routes, b.counts);
    bucket_scan_kernel<<<1, kWarp, 0, stream>>>(n_expert, b);
    bucket_fill_kernel<<<n_expert, kBucketThreads, 0, stream>>>(ids, routes, k, b, layer, n_expert, pos, route_counts,
                                                                last_use);
}

void q8_gather_columns(const BlockQ8* x, int units, const ExpertBuckets& b, int k, const Q8Columns& out,
                       cudaStream_t stream) {
    static_assert(kBucketThreads == kGatherSlots * kGatherUnits, "a thread writes one (unit, slot)");
    const dim3 grid(static_cast<unsigned>((out.columns + kGatherSlots - 1) / kGatherSlots),
                    static_cast<unsigned>((units + kGatherUnits - 1) / kGatherUnits));
    q8_gather_kernel<<<grid, kBucketThreads, 0, stream>>>(x, units, b.slot_route, k, out);
}

void upload_gemm_tables() {
    TORCH_CUDA_CHECK(cudaMemcpyToSymbol(g_grid2s, kIq2sGrid, sizeof kIq2sGrid));
    TORCH_CUDA_CHECK(cudaMemcpyToSymbol(g_grid3xxs, kIq3xxsGrid, sizeof kIq3xxsGrid));
    TORCH_CUDA_CHECK(cudaMemcpyToSymbol(g_grid3s, kIq3sGrid, sizeof kIq3sGrid));
}

size_t q8_columns_bytes(int units, int columns) {
    const size_t cells = static_cast<size_t>(units) * static_cast<size_t>(gemm_padded_columns(columns));
    return cells * 32 + (cells * sizeof(float) + 15) / 16 * 16 + (cells * sizeof(int16_t) + 15) / 16 * 16;
}

Q8Columns q8_columns_at(void* memory, int units, int columns) {
    const int stride = gemm_padded_columns(columns);
    const size_t cells = static_cast<size_t>(units) * static_cast<size_t>(stride);
    auto* q = static_cast<uint8_t*>(memory);
    auto* d = reinterpret_cast<float*>(q + cells * 32);
    auto* sum = reinterpret_cast<int16_t*>(reinterpret_cast<uint8_t*>(d) + (cells * sizeof(float) + 15) / 16 * 16);
    return {q, d, sum, units, columns, stride};
}

void expert_h_columns(const float* g, const float* u, const ExpertBuckets& b, int n_expert, const Q8Columns& out,
                      cudaStream_t stream) {
    expert_h_columns_kernel<<<n_expert, kBucketThreads, 0, stream>>>(g, u, b, out);
}

void q8_to_columns(const BlockQ8* x, const Q8Columns& out, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>((out.columns + kGatherSlots - 1) / kGatherSlots),
                    static_cast<unsigned>((out.units + kGatherUnits - 1) / kGatherUnits));
    q8_gather_kernel<<<grid, kBucketThreads, 0, stream>>>(x, out.units, nullptr, 1, out);
}

bool gemm_q8_supports(Type type) {
    return type == Type::Q6_K || type == Type::Q5_K || type == Type::Q4_K || type == Type::IQ4_XS ||
           type == Type::IQ4_NL || type == Type::Q8_0;
}

bool gemm_q8_experts_supports(Type type) {
    return type == Type::IQ2_S || type == Type::IQ3_XXS || type == Type::IQ3_S || type == Type::IQ4_XS ||
           type == Type::IQ4_NL || type == Type::Q2_0;
}

void gemm_q8_experts(Type type, const uint8_t* const* table, const ExpertBuckets& b, int n_expert, int rows, int cols,
                     const Q8Columns& x, const int32_t* index, float* y, cudaStream_t stream) {
    const TypeInfo& info = *type_info(static_cast<uint32_t>(type));
    if (!gemm_q8_experts_supports(type) || rows % 16 != 0 || cols % static_cast<int>(info.block_elements) != 0 ||
        x.units != cols / 32 || x.columns < b.slots)
        throw Error(
            "gemm_q8_experts takes IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL or Q2_0 experts (a multiple of 16 "
            "rows of whole blocks) and their buckets' slots.");
    (void)n_expert;
    const auto row = static_cast<int64_t>(row_bytes(info, static_cast<uint64_t>(cols)));
    const GemmArgs a{nullptr, rows, cols / 32, row, x, 64, y, table, b, index};
    switch (type) {
        case Type::IQ2_S:
            return launch_gemm<Type::IQ2_S>(a, stream);
        case Type::IQ3_XXS:
            return launch_gemm<Type::IQ3_XXS>(a, stream);
        case Type::IQ3_S:
            return launch_gemm<Type::IQ3_S>(a, stream);
        case Type::IQ4_XS:
            return launch_gemm<Type::IQ4_XS>(a, stream);
        case Type::IQ4_NL:
            return launch_gemm<Type::IQ4_NL>(a, stream);
        default:
            return launch_gemm<Type::Q2_0>(a, stream);
    }
}

void gemm_q8(Type type, const uint8_t* w, int rows, int cols, const Q8Columns& x, float* y, cudaStream_t stream) {
    const TypeInfo& info = *type_info(static_cast<uint32_t>(type));
    if (!gemm_q8_supports(type) || rows % 16 != 0 || cols % static_cast<int>(info.block_elements) != 0 ||
        x.units != cols / 32 || x.columns < 1)
        throw Error(
            "gemm_q8 takes Q6_K, Q5_K, Q4_K, IQ4_XS, IQ4_NL or Q8_0 rows (a multiple of 16 rows of whole "
            "blocks) and their columns.");
    // Items of one column range run on every SM at once, so the range's activations stay in the 4 MB L2: 256
    // columns at K = 6,144 (1.8 MB), 512 below.
    const int range = cols > 4096 ? 256 : 512;
    const auto row = static_cast<int64_t>(row_bytes(info, static_cast<uint64_t>(cols)));
    const GemmArgs a{w, rows, cols / 32, row, x, range, y, nullptr, {}, nullptr};
    switch (type) {
        case Type::Q6_K:
            return launch_gemm<Type::Q6_K>(a, stream);
        case Type::Q5_K:
            return launch_gemm<Type::Q5_K>(a, stream);
        case Type::Q4_K:
            return launch_gemm<Type::Q4_K>(a, stream);
        case Type::IQ4_XS:
            return launch_gemm<Type::IQ4_XS>(a, stream);
        case Type::IQ4_NL:
            return launch_gemm<Type::IQ4_NL>(a, stream);
        default:
            return launch_gemm<Type::Q8_0>(a, stream);
    }
}

}  // namespace torch::kernels
