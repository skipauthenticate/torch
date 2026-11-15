// The integer dot product of one 32-element unit of a weight row with one block of activations, for every quantized
// format of the model. Included only by gemv.cu, which owns the IQ grid tables declared here.
//
// dot_unit<T>(row, u, a, grid) returns the contribution of weights [32u, 32u + 32) of `row`. Block layouts are the
// ones src/formats/dequant.h documents and the CPU reference decodes (byte offsets within one block); each function
// turns 32 weights into int8 words, multiplies them with the activations by dp4a, applies the integer sub-scales in
// int32 and the F16 scales in F32: dw * da * float(integer sum), in that order. The integer sum is exact, so how a
// format pairs weight bytes with activation bytes is free: Act holds a block's activations in the word order its
// format's decode produces (act_of<T>), which saves the decode from regrouping weight bits. `grid` is the format's
// IQ grid in shared memory (unused by the other formats). Fields at 2-byte-aligned offsets are read as aligned words
// joined by funnel shifts (load_words), half the loads of 16-bit reads.
#pragma once

#include <cstdint>

#include "formats/codebooks.h"
#include "formats/types.h"
#include "kernels/common.cuh"
#include "kernels/gemv/gemv.h"

namespace torch::kernels {

// IQ grids, copied from the host tables in formats/codebooks.h by upload_iq_tables(); each kernel block copies its
// format's grid into shared memory.
__device__ __align__(16) uint64_t g_iq2s_grid[1024];
__device__ __align__(16) uint32_t g_iq3xxs_grid[256];
__device__ __align__(16) uint32_t g_iq3s_grid[512];
// IQ3_XXS grid entries already signed, for the multi-column decode: entry i with sign nibble n at 16 i + n, the bytes
// whose bit is set in n negated as set_signed negates them (one shared-memory load instead of the sign arithmetic).
__device__ __align__(16) uint32_t g_iq3xxs_signed[256 * 16];

// Bytes of the grid a format's decode reads from shared memory.
template <Type T>
inline constexpr int kGridBytes = T == Type::IQ2_S     ? sizeof g_iq2s_grid
                                  : T == Type::IQ3_XXS ? sizeof g_iq3xxs_grid
                                  : T == Type::IQ3_S   ? sizeof g_iq3s_grid
                                                       : 0;

template <Type T>
__device__ __forceinline__ const void* grid_source() {
    if constexpr (T == Type::IQ2_S)
        return g_iq2s_grid;
    else if constexpr (T == Type::IQ3_XXS)
        return g_iq3xxs_grid;
    else
        return g_iq3s_grid;
}

// The table the multi-column kernels stage for decode_unit: IQ3_XXS's signed grid, the plain grid otherwise.
template <Type T>
inline constexpr int kColumnGridBytes = T == Type::IQ3_XXS ? sizeof g_iq3xxs_signed : kGridBytes<T>;

template <Type T>
__device__ __forceinline__ const void* column_grid_source() {
    if constexpr (T == Type::IQ3_XXS)
        return g_iq3xxs_signed;
    else
        return grid_source<T>();
}

// One block of 32 activations as the kernels keep it in shared memory (48 bytes, three 16-byte loads).
struct alignas(16) Act {
    int q[8];      // the 32 int8 values as words, in the order act_of<T> gives
    float d;       // the F16 scale, as F32 (exact)
    int sum;       // sum of the 32 values
    int sum16[2];  // sums of values 0-15 and 16-31
};
static_assert(sizeof(Act) == 48, "Act is three 16-byte words");

// sum += the four grid bytes of g, negated where bits 0-3 of `signs` are set, times the four int8 activations in x.
// With m 0xff in the bytes whose sign is set, g ^ m is -g - 1 there, so the dot product is dp4a(g ^ m, x) - dp4a(m, x);
// the second term goes to `neg`, which the caller subtracts once. m is bit i moved to the top of byte i by one
// multiply, then copied through its byte by prmt's sign-replicating selectors (bit 3 of each selector nibble, which
// __byte_perm does not pass on).
__device__ __forceinline__ void signed_dot(uint32_t g, uint32_t signs, int x, int& sum, int& neg) {
    uint32_t m = 0;
    asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(m) : "r"((signs & 0xf) * 0x10204080u));
    sum = __dp4a(static_cast<int>(g ^ m), x, sum);
    neg = __dp4a(static_cast<int>(m), x, neg);
}

// The 16 IQ4_NL levels as four words of int8 bytes, for __byte_perm lookups.
__host__ __device__ constexpr uint32_t iq4_word(int w) {
    uint32_t word = 0;
    for (int i = 0; i < 4; ++i)
        word |= static_cast<uint32_t>(static_cast<uint8_t>(kIq4nlValues[4 * w + i])) << 8 * i;
    return word;
}
inline constexpr uint32_t kIq4Word0 = iq4_word(0), kIq4Word1 = iq4_word(1), kIq4Word2 = iq4_word(2),
                          kIq4Word3 = iq4_word(3);

// The eight 4-bit indices of v (nibbles, low first) as IQ4_NL levels: nibbles 0-3 into `first`, 4-7 into `second`,
// one int8 per byte. Each index's low 3 bits (a byte_perm selector) look up levels 0-7 and 8-15, and its bit 3 picks
// one of the two bytes.
__device__ __forceinline__ void iq4_levels(uint32_t v, int& first, int& second) {
    const uint32_t index = v & 0x77777777u;
    const uint32_t pick = (v >> 1 & 0x44444444u) | 0x32103210u;  // byte i of the low lookup, or byte i of the high
    first = static_cast<int>(
        __byte_perm(__byte_perm(kIq4Word0, kIq4Word1, index), __byte_perm(kIq4Word2, kIq4Word3, index), pick));
    second = static_cast<int>(__byte_perm(__byte_perm(kIq4Word0, kIq4Word1, index >> 16),
                                          __byte_perm(kIq4Word2, kIq4Word3, index >> 16), pick >> 16));
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

// A block of activations in its format's word order. Q2_0 decodes crumb k of every byte of a word at once, so its
// word k holds values k, k + 4, k + 8, k + 12 (and word 4 + k the same of values 16-31). IQ4_NL and IQ4_XS decode
// the nibbles of a weight word in order: low and high nibble of byte 0 are values i and i + 16, so word 2k holds
// values 4k, 4k + 16, 4k + 1, 4k + 17 and word 2k + 1 values 4k + 2, 4k + 18, 4k + 3, 4k + 19. The rest keep value
// order.
template <Type T>
__device__ __forceinline__ Act act_of(const BlockQ8& b) {
    Act a;
    int x[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
        x[k] = reinterpret_cast<const int*>(b.q)[k];
    a.sum16[0] =
        __dp4a(x[0], 0x01010101, __dp4a(x[1], 0x01010101, __dp4a(x[2], 0x01010101, __dp4a(x[3], 0x01010101, 0))));
    a.sum16[1] =
        __dp4a(x[4], 0x01010101, __dp4a(x[5], 0x01010101, __dp4a(x[6], 0x01010101, __dp4a(x[7], 0x01010101, 0))));
    if constexpr (T == Type::Q2_0) {
#pragma unroll
        for (int h = 0; h < 8; h += 4) {
            const uint32_t t0 = __byte_perm(x[h], x[h + 1], 0x5140), t1 = __byte_perm(x[h + 2], x[h + 3], 0x5140);
            const uint32_t t2 = __byte_perm(x[h], x[h + 1], 0x7362), t3 = __byte_perm(x[h + 2], x[h + 3], 0x7362);
            a.q[h] = static_cast<int>(__byte_perm(t0, t1, 0x5410));
            a.q[h + 1] = static_cast<int>(__byte_perm(t0, t1, 0x7632));
            a.q[h + 2] = static_cast<int>(__byte_perm(t2, t3, 0x5410));
            a.q[h + 3] = static_cast<int>(__byte_perm(t2, t3, 0x7632));
        }
    } else if constexpr (T == Type::IQ4_NL || T == Type::IQ4_XS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            a.q[2 * k] = static_cast<int>(__byte_perm(x[k], x[k + 4], 0x5140));
            a.q[2 * k + 1] = static_cast<int>(__byte_perm(x[k], x[k + 4], 0x7362));
        }
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k)
            a.q[k] = x[k];
    }
    a.d = f16_to_f32(b.d);
    a.sum = b.sum;
    return a;
}

// kWords little-endian words from a 2-byte-aligned address, as kWords + 1 aligned loads (the last one only when the
// address is not 4-byte aligned, so no word past the field is read) joined by funnel shifts.
template <int kWords>
__device__ __forceinline__ void load_words(const uint8_t* p, uint32_t (&out)[kWords]) {
    const auto* w = reinterpret_cast<const uint32_t*>(reinterpret_cast<uintptr_t>(p) & ~uintptr_t{3});
    const uint32_t shift = (reinterpret_cast<uintptr_t>(p) & 2) * 8;
    uint32_t prev = w[0];
#pragma unroll
    for (int i = 0; i < kWords; ++i) {
        const uint32_t next = i + 1 < kWords || shift != 0 ? w[i + 1] : 0;
        out[i] = __funnelshift_r(prev, next, shift);
        prev = next;
    }
}

template <Type T>
__device__ float dot_unit(const uint8_t* row, int u, const Act& a, const uint32_t* grid);

// 32 elements in 34 bytes: d @0, qs[32] (int8) @2.
template <>
__device__ __forceinline__ float dot_unit<Type::Q8_0>(const uint8_t* row, int u, const Act& a, const uint32_t*) {
    const uint8_t* b = row + u * 34;
    uint32_t qs[8];
    load_words(b + 2, qs);
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k)
        sum = __dp4a(static_cast<int>(qs[k]), a.q[k], sum);
    return f16_to_f32(load_u16(b)) * a.d * static_cast<float>(sum);
}

// 64 elements in 18 bytes: d @0, qs[16] @2, four 2-bit codes per byte meaning -1, 0, 1, 2. Unit u is half u % 2.
template <>
__device__ __forceinline__ float dot_unit<Type::Q2_0>(const uint8_t* row, int u, const Act& a, const uint32_t*) {
    const uint8_t* b = row + u / 2 * 18;
    const uint8_t* qs = b + 2 + 8 * (u % 2);
    uint32_t words[2];
    load_words(qs, words);
    const uint32_t low = words[0], high = words[1];
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        sum = __dp4a(static_cast<int>(low >> 2 * k & 0x03030303u), a.q[k], sum);
        sum = __dp4a(static_cast<int>(high >> 2 * k & 0x03030303u), a.q[k + 4], sum);
    }
    return f16_to_f32(load_u16(b)) * a.d * static_cast<float>(sum - a.sum);
}

// 32 elements in 18 bytes: d @0, qs[16] @2; element e < 16 is the low nibble of qs[e], e >= 16 the high nibble of
// qs[e - 16]; each nibble indexes the IQ4_NL levels.
template <>
__device__ __forceinline__ float dot_unit<Type::IQ4_NL>(const uint8_t* row, int u, const Act& a, const uint32_t*) {
    const uint8_t* b = row + u * 18;
    uint32_t qs[4];
    load_words(b + 2, qs);
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        int first = 0, second = 0;
        iq4_levels(qs[k], first, second);
        sum = __dp4a(first, a.q[2 * k], sum);
        sum = __dp4a(second, a.q[2 * k + 1], sum);
    }
    return f16_to_f32(load_u16(b)) * a.d * static_cast<float>(sum);
}

// 256 elements in 136 bytes: d @0, scales_h (u16) @2, scales_l[4] @4, qs[128] @8 (4-byte aligned, as rows are).
// Sub-block s is laid out as IQ4_NL at qs + 16s, with the 6-bit scale (scales_l nibble s, 2 bits s of scales_h)
// minus 32.
template <>
__device__ __forceinline__ float dot_unit<Type::IQ4_XS>(const uint8_t* row, int u, const Act& a, const uint32_t*) {
    const uint8_t* b = row + u / 8 * 136;
    const int s = u % 8;
    const int scale = (b[4 + s / 2] >> 4 * (s % 2) & 0xf) | (load_u16(b + 2) >> 2 * s & 3) << 4;
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        int first = 0, second = 0;
        iq4_levels(load_u32(b + 8 + 16 * s + 4 * k), first, second);
        sum = __dp4a(first, a.q[2 * k], sum);
        sum = __dp4a(second, a.q[2 * k + 1], sum);
    }
    return f16_to_f32(load_u16(b)) * a.d * static_cast<float>(sum * (scale - 32));
}

// 256 elements in 144 bytes: d @0, dmin @2, scales[12] @4, qs[128] @16. Sub-block s is the low (s even) or high
// nibbles of qs[32(s / 2), +32); value d * scale * q - dmin * min.
template <>
__device__ __forceinline__ float dot_unit<Type::Q4_K>(const uint8_t* row, int u, const Act& a, const uint32_t*) {
    const uint8_t* b = row + u / 8 * 144;
    const int s = u % 8;
    int scale = 0, min = 0;
    k_scale_min(b + 4, s, scale, min);
    const uint8_t* qs = b + 16 + 32 * (s / 2);
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k)
        sum = __dp4a(static_cast<int>(load_u32(qs + 4 * k) >> 4 * (s % 2) & 0x0f0f0f0fu), a.q[k], sum);
    const float d = f16_to_f32(load_u16(b)), dmin = f16_to_f32(load_u16(b + 2));
    return a.d * (d * static_cast<float>(sum * scale) - dmin * static_cast<float>(a.sum * min));
}

// 256 elements in 176 bytes: d @0, dmin @2, scales[12] @4, qh[32] @16, qs[128] @48. As Q4_K, plus a fifth bit:
// bit s of qh[l] for element l of sub-block s.
template <>
__device__ __forceinline__ float dot_unit<Type::Q5_K>(const uint8_t* row, int u, const Act& a, const uint32_t*) {
    const uint8_t* b = row + u / 8 * 176;
    const int s = u % 8;
    int scale = 0, min = 0;
    k_scale_min(b + 4, s, scale, min);
    const uint8_t* qs = b + 48 + 32 * (s / 2);
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint32_t low = load_u32(qs + 4 * k) >> 4 * (s % 2) & 0x0f0f0f0fu;
        const uint32_t high = (load_u32(b + 16 + 4 * k) >> s & 0x01010101u) << 4;
        sum = __dp4a(static_cast<int>(low | high), a.q[k], sum);
    }
    const float d = f16_to_f32(load_u16(b)), dmin = f16_to_f32(load_u16(b + 2));
    return a.d * (d * static_cast<float>(sum * scale) - dmin * static_cast<float>(a.sum * min));
}

// 256 elements in 210 bytes: ql[128] @0, qh[64] @128, scales[16] (int8) @192, d @208. Sub-block s = 4n + k takes its
// low 4 bits from ql[64n + 32(k % 2) + l] (low nibbles for k < 2) and its high 2 bits from bits 2k of qh[32n + l];
// value d * scale * (q - 32), one scale per 16 elements. The offset is taken out of the integer sums once:
// sum (q - 32) x = sum q x - 32 sum x.
template <>
__device__ __forceinline__ float dot_unit<Type::Q6_K>(const uint8_t* row, int u, const Act& a, const uint32_t*) {
    const uint8_t* b = row + u / 8 * 210;
    const int s = u % 8, n = s / 4, k = s % 4;
    const uint8_t* ql = b + 64 * n + 32 * (k % 2);
    const uint8_t* qh = b + 128 + 32 * n;
    uint32_t lows[8], highs[8];
    load_words(ql, lows);
    load_words(qh, highs);
    int sum[2] = {-32 * a.sum16[0], -32 * a.sum16[1]};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t low = lows[j] >> 4 * (k / 2) & 0x0f0f0f0fu;
        const uint32_t high = (highs[j] >> 2 * k & 0x03030303u) << 4;
        sum[j / 4] = __dp4a(static_cast<int>(low | high), a.q[j], sum[j / 4]);
    }
    const uint32_t scales = load_u16(b + 192 + 8 * n + 2 * k);  // two int8
    const int total = static_cast<int8_t>(scales & 0xff) * sum[0] + static_cast<int8_t>(scales >> 8) * sum[1];
    return f16_to_f32(load_u16(b + 208)) * a.d * static_cast<float>(total);
}

// 256 elements in 82 bytes: d @0, qs[32] @2, signs[32] @34, qh[8] @66, scales[8] @74. Group g of eight elements is
// grid entry qs[g] plus 2 high bits from qh[g / 4], with sign byte signs[g]; each sub-block has two 4-bit scales,
// one per 16 elements, worth (2s + 1) / 8.
template <>
__device__ __forceinline__ float dot_unit<Type::IQ2_S>(const uint8_t* row, int u, const Act& a, const uint32_t* grid) {
    const uint8_t* b = row + u / 8 * 82;
    const int s = u % 8;
    const uint32_t qs = load_u32_a2(b + 2 + 4 * s), signs = load_u32_a2(b + 34 + 4 * s);
    const uint32_t qh = b[66 + s], scales = b[74 + s];
    int sum[2] = {0, 0}, neg[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint2 entry = reinterpret_cast<const uint2*>(grid)[(qs >> 8 * j & 0xff) | (qh << (8 - 2 * j) & 0x300)];
        const uint32_t sign = signs >> 8 * j;
        signed_dot(entry.x, sign, a.q[2 * j], sum[j / 2], neg[j / 2]);
        signed_dot(entry.y, sign >> 4, a.q[2 * j + 1], sum[j / 2], neg[j / 2]);
    }
    const int total = (2 * static_cast<int>(scales & 0xf) + 1) * (sum[0] - neg[0]) +
                      (2 * static_cast<int>(scales >> 4) + 1) * (sum[1] - neg[1]);
    return f16_to_f32(load_u16(b)) * a.d * (0.125f * static_cast<float>(total));
}

// 256 elements in 98 bytes: d @0, qs[64] @2, then one u32 per sub-block @66 with four 7-bit sign indices (bits 0-27)
// and a 4-bit scale (bits 28-31) worth (2s + 1) / 4. Group g is grid entries qs[2g] and qs[2g + 1]; the eighth sign
// bit makes the number of negative signs even.
template <>
__device__ __forceinline__ float dot_unit<Type::IQ3_XXS>(const uint8_t* row, int u, const Act& a,
                                                         const uint32_t* grid) {
    const uint8_t* b = row + u / 8 * 98;
    const int s = u % 8;
    uint32_t qs[2];
    load_words(b + 2 + 8 * s, qs);
    const uint32_t aux = load_u32_a2(b + 66 + 4 * s);
    int sum = 0, neg = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t signs7 = aux >> 7 * j & 127;
        const uint32_t signs = signs7 | (__popc(signs7) & 1u) << 7;
        const uint32_t pair = qs[j / 2] >> 16 * (j % 2);
        signed_dot(grid[pair & 0xff], signs, a.q[2 * j], sum, neg);
        signed_dot(grid[pair >> 8 & 0xff], signs >> 4, a.q[2 * j + 1], sum, neg);
    }
    const int scale = static_cast<int>(aux >> 28);
    return f16_to_f32(load_u16(b)) * a.d * (0.25f * static_cast<float>((2 * scale + 1) * (sum - neg)));
}

// 256 elements in 110 bytes: d @0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106. Group g is grid entries
// qs[2g] and qs[2g + 1], each with a ninth index bit from qh[g / 4], and sign byte signs[g]; each sub-block has a
// 4-bit scale worth 1 + 2s.
template <>
__device__ __forceinline__ float dot_unit<Type::IQ3_S>(const uint8_t* row, int u, const Act& a, const uint32_t* grid) {
    const uint8_t* b = row + u / 8 * 110;
    const int s = u % 8;
    uint32_t qs[2];
    load_words(b + 2 + 8 * s, qs);
    const uint32_t qh = b[66 + s], signs = load_u32_a2(b + 74 + 4 * s);
    int sum = 0, neg = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t pair = qs[j / 2] >> 16 * (j % 2);
        const uint32_t first = (pair & 0xff) | (qh << (8 - 2 * j) & 256);
        const uint32_t second = (pair >> 8 & 0xff) | (qh << (7 - 2 * j) & 256);
        const uint32_t sign = signs >> 8 * j;
        signed_dot(grid[first], sign, a.q[2 * j], sum, neg);
        signed_dot(grid[second], sign >> 4, a.q[2 * j + 1], sum, neg);
    }
    const int scale = b[106 + s / 2] >> 4 * (s % 2) & 0xf;
    return f16_to_f32(load_u16(b)) * a.d * static_cast<float>((1 + 2 * scale) * (sum - neg));
}

// ---- Several columns ----------------------------------------------------------------------------------------------
//
// The multi-column kernels decode a unit's weights once (decode_unit) and multiply them with each column's
// activations (apply_unit): the 32 weights as eight int8 words in the order the format's Act holds activations (IQ
// grid bytes already signed: (g ^ m) + (m & 1) is the two's complement of the negated bytes), and the scales the
// format's formula needs. apply_unit forms the same exact integer sums as dot_unit and applies the same F32 formula in
// the same order, so a column's value is bitwise dot_unit's (gemv_test checks every format at 1 to 8 columns).
struct Unit {
    int w[8];
    int s0, s1;     // integer sub-scales (or scale and minimum)
    float d, dmin;  // the F16 scales as F32
};

template <Type T>
__device__ Unit decode(const uint8_t* row, int u, const uint32_t* grid);

template <Type T>
__device__ float apply(const Unit& w, const Act& a);

// Word i of w: four grid bytes g, negated where bits 0-3 of `signs` are set (signed_dot's masks m).
__device__ __forceinline__ void set_signed(Unit& w, int i, uint32_t g, uint32_t signs) {
    uint32_t m = 0;
    asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(m) : "r"((signs & 0xf) * 0x10204080u));
    w.w[i] = static_cast<int>((g ^ m) + (m & 0x01010101u));
}

// sum over words [first, first + 4) of w with a's words.
__device__ __forceinline__ int dot4(const Unit& w, const Act& a, int first, int sum) {
#pragma unroll
    for (int k = first; k < first + 4; ++k)
        sum = __dp4a(w.w[k], a.q[k], sum);
    return sum;
}

// 32 elements in 34 bytes: d @0, qs[32] (int8) @2.
template <>
__device__ __forceinline__ Unit decode<Type::Q8_0>(const uint8_t* row, int u, const uint32_t*) {
    const uint8_t* b = row + u * 34;
    Unit w;
    load_words(b + 2, reinterpret_cast<uint32_t(&)[8]>(w.w));
    w.d = f16_to_f32(load_u16(b));
    return w;
}
template <>
__device__ __forceinline__ float apply<Type::Q8_0>(const Unit& w, const Act& a) {
    return w.d * a.d * static_cast<float>(dot4(w, a, 4, dot4(w, a, 0, 0)));
}

// 64 elements in 18 bytes: d @0, qs[16] @2, four 2-bit codes per byte meaning -1, 0, 1, 2. Unit u is half u % 2.
template <>
__device__ __forceinline__ Unit decode<Type::Q2_0>(const uint8_t* row, int u, const uint32_t*) {
    const uint8_t* b = row + u / 2 * 18;
    uint32_t words[2];
    load_words(b + 2 + 8 * (u % 2), words);
    Unit w;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        w.w[k] = static_cast<int>(words[0] >> 2 * k & 0x03030303u);
        w.w[k + 4] = static_cast<int>(words[1] >> 2 * k & 0x03030303u);
    }
    w.d = f16_to_f32(load_u16(b));
    return w;
}
template <>
__device__ __forceinline__ float apply<Type::Q2_0>(const Unit& w, const Act& a) {
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        sum = __dp4a(w.w[k], a.q[k], sum);
        sum = __dp4a(w.w[k + 4], a.q[k + 4], sum);
    }
    return w.d * a.d * static_cast<float>(sum - a.sum);
}

// 32 elements in 18 bytes: d @0, qs[16] @2; element e < 16 is the low nibble of qs[e], e >= 16 the high nibble of
// qs[e - 16]; each nibble indexes the IQ4_NL levels.
template <>
__device__ __forceinline__ Unit decode<Type::IQ4_NL>(const uint8_t* row, int u, const uint32_t*) {
    const uint8_t* b = row + u * 18;
    uint32_t qs[4];
    load_words(b + 2, qs);
    Unit w;
#pragma unroll
    for (int k = 0; k < 4; ++k)
        iq4_levels(qs[k], w.w[2 * k], w.w[2 * k + 1]);
    w.d = f16_to_f32(load_u16(b));
    return w;
}
template <>
__device__ __forceinline__ float apply<Type::IQ4_NL>(const Unit& w, const Act& a) {
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        sum = __dp4a(w.w[2 * k], a.q[2 * k], sum);
        sum = __dp4a(w.w[2 * k + 1], a.q[2 * k + 1], sum);
    }
    return w.d * a.d * static_cast<float>(sum);
}

// 256 elements in 136 bytes: d @0, scales_h (u16) @2, scales_l[4] @4, qs[128] @8 (4-byte aligned, as rows are).
// Sub-block s is laid out as IQ4_NL at qs + 16s, with the 6-bit scale (scales_l nibble s, 2 bits s of scales_h)
// minus 32.
template <>
__device__ __forceinline__ Unit decode<Type::IQ4_XS>(const uint8_t* row, int u, const uint32_t*) {
    const uint8_t* b = row + u / 8 * 136;
    const int s = u % 8;
    Unit w;
    w.s0 = (b[4 + s / 2] >> 4 * (s % 2) & 0xf) | (load_u16(b + 2) >> 2 * s & 3) << 4;
#pragma unroll
    for (int k = 0; k < 4; ++k)
        iq4_levels(load_u32(b + 8 + 16 * s + 4 * k), w.w[2 * k], w.w[2 * k + 1]);
    w.d = f16_to_f32(load_u16(b));
    return w;
}
template <>
__device__ __forceinline__ float apply<Type::IQ4_XS>(const Unit& w, const Act& a) {
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        sum = __dp4a(w.w[2 * k], a.q[2 * k], sum);
        sum = __dp4a(w.w[2 * k + 1], a.q[2 * k + 1], sum);
    }
    return w.d * a.d * static_cast<float>(sum * (w.s0 - 32));
}

// 256 elements in 144 bytes: d @0, dmin @2, scales[12] @4, qs[128] @16. Sub-block s is the low (s even) or high
// nibbles of qs[32(s / 2), +32); value d * scale * q - dmin * min.
template <>
__device__ __forceinline__ Unit decode<Type::Q4_K>(const uint8_t* row, int u, const uint32_t*) {
    const uint8_t* b = row + u / 8 * 144;
    const int s = u % 8;
    Unit w;
    k_scale_min(b + 4, s, w.s0, w.s1);
    const uint8_t* qs = b + 16 + 32 * (s / 2);
#pragma unroll
    for (int k = 0; k < 8; ++k)
        w.w[k] = static_cast<int>(load_u32(qs + 4 * k) >> 4 * (s % 2) & 0x0f0f0f0fu);
    w.d = f16_to_f32(load_u16(b)), w.dmin = f16_to_f32(load_u16(b + 2));
    return w;
}

// 256 elements in 176 bytes: d @0, dmin @2, scales[12] @4, qh[32] @16, qs[128] @48. As Q4_K, plus a fifth bit:
// bit s of qh[l] for element l of sub-block s.
template <>
__device__ __forceinline__ Unit decode<Type::Q5_K>(const uint8_t* row, int u, const uint32_t*) {
    const uint8_t* b = row + u / 8 * 176;
    const int s = u % 8;
    Unit w;
    k_scale_min(b + 4, s, w.s0, w.s1);
    const uint8_t* qs = b + 48 + 32 * (s / 2);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint32_t low = load_u32(qs + 4 * k) >> 4 * (s % 2) & 0x0f0f0f0fu;
        const uint32_t high = (load_u32(b + 16 + 4 * k) >> s & 0x01010101u) << 4;
        w.w[k] = static_cast<int>(low | high);
    }
    w.d = f16_to_f32(load_u16(b)), w.dmin = f16_to_f32(load_u16(b + 2));
    return w;
}

// Q4_K and Q5_K: d * scale * sum - dmin * min * sum(x), then the activation scale.
template <>
__device__ __forceinline__ float apply<Type::Q4_K>(const Unit& w, const Act& a) {
    const int sum = dot4(w, a, 4, dot4(w, a, 0, 0));
    return a.d * (w.d * static_cast<float>(sum * w.s0) - w.dmin * static_cast<float>(a.sum * w.s1));
}
template <>
__device__ __forceinline__ float apply<Type::Q5_K>(const Unit& w, const Act& a) {
    return apply<Type::Q4_K>(w, a);
}

// 256 elements in 210 bytes: ql[128] @0, qh[64] @128, scales[16] (int8) @192, d @208. Sub-block s = 4n + k takes its
// low 4 bits from ql[64n + 32(k % 2) + l] (low nibbles for k < 2) and its high 2 bits from bits 2k of qh[32n + l];
// value d * scale * (q - 32), one scale per 16 elements. The offset is taken out of the integer sums once:
// sum (q - 32) x = sum q x - 32 sum x.
template <>
__device__ __forceinline__ Unit decode<Type::Q6_K>(const uint8_t* row, int u, const uint32_t*) {
    const uint8_t* b = row + u / 8 * 210;
    const int s = u % 8, n = s / 4, k = s % 4;
    uint32_t lows[8], highs[8];
    load_words(b + 64 * n + 32 * (k % 2), lows);
    load_words(b + 128 + 32 * n, highs);
    Unit w;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        w.w[j] = static_cast<int>((lows[j] >> 4 * (k / 2) & 0x0f0f0f0fu) | (highs[j] >> 2 * k & 0x03030303u) << 4);
    const uint32_t scales = load_u16(b + 192 + 8 * n + 2 * k);  // two int8
    w.s0 = static_cast<int8_t>(scales & 0xff), w.s1 = static_cast<int8_t>(scales >> 8);
    w.d = f16_to_f32(load_u16(b + 208));
    return w;
}
template <>
__device__ __forceinline__ float apply<Type::Q6_K>(const Unit& w, const Act& a) {
    const int total = w.s0 * dot4(w, a, 0, -32 * a.sum16[0]) + w.s1 * dot4(w, a, 4, -32 * a.sum16[1]);
    return w.d * a.d * static_cast<float>(total);
}

// 256 elements in 82 bytes: d @0, qs[32] @2, signs[32] @34, qh[8] @66, scales[8] @74. Group g of eight elements is
// grid entry qs[g] plus 2 high bits from qh[g / 4], with sign byte signs[g]; each sub-block has two 4-bit scales,
// one per 16 elements, worth (2s + 1) / 8.
__device__ __forceinline__ Unit decode_iq2s(const uint8_t* row, int u, const uint32_t* grid) {
    const uint8_t* b = row + u / 8 * 82;
    const int s = u % 8;
    const uint32_t qs = load_u32_a2(b + 2 + 4 * s), signs = load_u32_a2(b + 34 + 4 * s);
    const uint32_t qh = b[66 + s], scales = b[74 + s];
    Unit w;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint2 entry = reinterpret_cast<const uint2*>(grid)[(qs >> 8 * j & 0xff) | (qh << (8 - 2 * j) & 0x300)];
        const uint32_t sign = signs >> 8 * j;
        set_signed(w, 2 * j, entry.x, sign);
        set_signed(w, 2 * j + 1, entry.y, sign >> 4);
    }
    w.s0 = 2 * static_cast<int>(scales & 0xf) + 1, w.s1 = 2 * static_cast<int>(scales >> 4) + 1;
    w.d = f16_to_f32(load_u16(b));
    return w;
}
__device__ __forceinline__ float apply_iq2s(const Unit& w, const Act& a) {
    const int total = w.s0 * dot4(w, a, 0, 0) + w.s1 * dot4(w, a, 4, 0);
    return w.d * a.d * (0.125f * static_cast<float>(total));
}

// 256 elements in 98 bytes: d @0, qs[64] @2, then one u32 per sub-block @66 with four 7-bit sign indices (bits 0-27)
// and a 4-bit scale (bits 28-31) worth (2s + 1) / 4. Group g is grid entries qs[2g] and qs[2g + 1]; the eighth sign
// bit makes the number of negative signs even.
__device__ __forceinline__ Unit decode_iq3xxs(const uint8_t* row, int u, const uint32_t* grid) {
    const uint8_t* b = row + u / 8 * 98;
    const int s = u % 8;
    uint32_t qs[2];
    load_words(b + 2 + 8 * s, qs);
    const uint32_t aux = load_u32_a2(b + 66 + 4 * s);
    Unit w;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t signs7 = aux >> 7 * j & 127;
        const uint32_t signs = signs7 | (__popc(signs7) & 1u) << 7;
        const uint32_t pair = qs[j / 2] >> 16 * (j % 2);
        w.w[2 * j] = static_cast<int>(grid[(pair & 0xff) << 4 | (signs & 0xf)]);  // grid: g_iq3xxs_signed
        w.w[2 * j + 1] = static_cast<int>(grid[(pair >> 8 & 0xff) << 4 | (signs >> 4 & 0xf)]);
    }
    w.s0 = 2 * static_cast<int>(aux >> 28) + 1;
    w.d = f16_to_f32(load_u16(b));
    return w;
}
__device__ __forceinline__ float apply_iq3xxs(const Unit& w, const Act& a) {
    const int sum = dot4(w, a, 0, 0) + dot4(w, a, 4, 0);
    return w.d * a.d * (0.25f * static_cast<float>(w.s0 * sum));
}

// 256 elements in 110 bytes: d @0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106. Group g is grid entries
// qs[2g] and qs[2g + 1], each with a ninth index bit from qh[g / 4], and sign byte signs[g]; each sub-block has a
// 4-bit scale worth 1 + 2s.
__device__ __forceinline__ Unit decode_iq3s(const uint8_t* row, int u, const uint32_t* grid) {
    const uint8_t* b = row + u / 8 * 110;
    const int s = u % 8;
    uint32_t qs[2];
    load_words(b + 2 + 8 * s, qs);
    const uint32_t qh = b[66 + s], signs = load_u32_a2(b + 74 + 4 * s);
    Unit w;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t pair = qs[j / 2] >> 16 * (j % 2);
        const uint32_t sign = signs >> 8 * j;
        set_signed(w, 2 * j, grid[(pair & 0xff) | (qh << (8 - 2 * j) & 256)], sign);
        set_signed(w, 2 * j + 1, grid[(pair >> 8 & 0xff) | (qh << (7 - 2 * j) & 256)], sign >> 4);
    }
    w.s0 = 1 + 2 * (b[106 + s / 2] >> 4 * (s % 2) & 0xf);
    w.d = f16_to_f32(load_u16(b));
    return w;
}
__device__ __forceinline__ float apply_iq3s(const Unit& w, const Act& a) {
    const int sum = dot4(w, a, 0, 0) + dot4(w, a, 4, 0);
    return w.d * a.d * static_cast<float>(w.s0 * sum);
}

template <Type T>
__device__ __forceinline__ Unit decode_unit(const uint8_t* row, int u, const uint32_t* grid) {
    if constexpr (T == Type::IQ2_S)
        return decode_iq2s(row, u, grid);
    else if constexpr (T == Type::IQ3_XXS)
        return decode_iq3xxs(row, u, grid);
    else if constexpr (T == Type::IQ3_S)
        return decode_iq3s(row, u, grid);
    else
        return decode<T>(row, u, grid);
}
template <Type T>
__device__ __forceinline__ float apply_unit(const Unit& w, const Act& a) {
    if constexpr (T == Type::IQ2_S)
        return apply_iq2s(w, a);
    else if constexpr (T == Type::IQ3_XXS)
        return apply_iq3xxs(w, a);
    else if constexpr (T == Type::IQ3_S)
        return apply_iq3s(w, a);
    else
        return apply<T>(w, a);
}

}  // namespace torch::kernels
