// The integer dot product of one 32-element unit of a weight row with one BlockQ8 of activations, for every
// quantized format of the model. Included only by gemv.cu, which owns the IQ grid tables declared here.
//
// dot_unit<T>(row, u, a) returns the contribution of weights [32u, 32u + 32) of `row`. Block layouts are the ones
// src/formats/dequant.h documents and the CPU reference decodes (byte offsets within one block); each function
// turns 32 weights into int8 words, multiplies them with the activations by dp4a, applies the integer sub-scales in
// int32 and the F16 scales in F32: dw * da * float(integer sum), in that order.
#pragma once

#include <cstdint>

#include "formats/codebooks.h"
#include "formats/types.h"
#include "kernels/common.cuh"
#include "kernels/gemv/gemv.h"

namespace torch::kernels {

// IQ grids, copied from the host tables in formats/codebooks.h by upload_iq_tables().
__device__ uint64_t g_iq2s_grid[1024];
__device__ uint32_t g_iq3xxs_grid[256];
__device__ uint32_t g_iq3s_grid[512];

// The 16 IQ4_NL levels as four words of int8 bytes, for __byte_perm lookups.
__host__ __device__ constexpr uint32_t iq4_word(int w) {
    uint32_t word = 0;
    for (int i = 0; i < 4; ++i)
        word |= static_cast<uint32_t>(static_cast<uint8_t>(kIq4nlValues[4 * w + i])) << 8 * i;
    return word;
}
inline constexpr uint32_t kIq4Word0 = iq4_word(0), kIq4Word1 = iq4_word(1), kIq4Word2 = iq4_word(2),
                          kIq4Word3 = iq4_word(3);

// Four 4-bit indices, one per byte, to the four IQ4_NL levels as int8 bytes.
__device__ __forceinline__ int iq4_levels(uint32_t indices) {
    const uint32_t select = (indices & 0x7) | (indices >> 4 & 0x70) | (indices >> 8 & 0x700) | (indices >> 12 & 0x7000);
    const uint32_t low = __byte_perm(kIq4Word0, kIq4Word1, select);
    const uint32_t high = __byte_perm(kIq4Word2, kIq4Word3, select);
    const uint32_t upper = (indices >> 3 & 0x01010101u) * 0xffu;  // bytes whose index is 8-15
    return static_cast<int>((high & upper) | (low & ~upper));
}

// Four magnitudes (one per byte) negated where the matching bit of `signs` (bits 0-3) is set.
__device__ __forceinline__ int with_signs(uint32_t magnitudes, uint32_t signs) {
    const uint32_t mask = __vcmpne4((signs & 1) | (signs & 2) << 7 | (signs & 4) << 14 | (signs & 8) << 21, 0);
    return static_cast<int>(__vsub4(magnitudes ^ mask, mask));
}

// Four 2-bit codes of a byte, low bits first, as four bytes.
__device__ __forceinline__ int spread_crumbs(uint32_t byte) {
    return static_cast<int>((byte & 0x03) | (byte & 0x0c) << 6 | (byte & 0x30) << 12 | (byte & 0xc0) << 18);
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

__device__ __forceinline__ const int* words(const BlockQ8& a) {
    return reinterpret_cast<const int*>(a.q);
}

template <Type T>
__device__ float dot_unit(const uint8_t* row, int u, const BlockQ8& a);

// 32 elements in 34 bytes: d @0, qs[32] (int8) @2.
template <>
__device__ __forceinline__ float dot_unit<Type::Q8_0>(const uint8_t* row, int u, const BlockQ8& a) {
    const uint8_t* b = row + u * 34;
    const int* x = words(a);
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k)
        sum = __dp4a(static_cast<int>(load_u32_a2(b + 2 + 4 * k)), x[k], sum);
    return f16_to_f32(load_u16(b)) * f16_to_f32(a.d) * static_cast<float>(sum);
}

// 64 elements in 18 bytes: d @0, qs[16] @2, four 2-bit codes per byte meaning -1, 0, 1, 2. Unit u is half u % 2.
template <>
__device__ __forceinline__ float dot_unit<Type::Q2_0>(const uint8_t* row, int u, const BlockQ8& a) {
    const uint8_t* b = row + u / 2 * 18;
    const uint8_t* qs = b + 2 + 8 * (u % 2);
    const int* x = words(a);
    const uint32_t low = load_u32_a2(qs), high = load_u32_a2(qs + 4);
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        sum = __dp4a(spread_crumbs(low >> 8 * k & 0xff), x[k], sum);
        sum = __dp4a(spread_crumbs(high >> 8 * k & 0xff), x[k + 4], sum);
    }
    return f16_to_f32(load_u16(b)) * f16_to_f32(a.d) * static_cast<float>(sum - a.sum);
}

// 32 elements in 18 bytes: d @0, qs[16] @2; element e < 16 is the low nibble of qs[e], e >= 16 the high nibble of
// qs[e - 16]; each nibble indexes the IQ4_NL levels.
template <>
__device__ __forceinline__ float dot_unit<Type::IQ4_NL>(const uint8_t* row, int u, const BlockQ8& a) {
    const uint8_t* b = row + u * 18;
    const int* x = words(a);
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t v = load_u32_a2(b + 2 + 4 * k);
        sum = __dp4a(iq4_levels(v & 0x0f0f0f0fu), x[k], sum);
        sum = __dp4a(iq4_levels(v >> 4 & 0x0f0f0f0fu), x[k + 4], sum);
    }
    return f16_to_f32(load_u16(b)) * f16_to_f32(a.d) * static_cast<float>(sum);
}

// 256 elements in 136 bytes: d @0, scales_h (u16) @2, scales_l[4] @4, qs[128] @8. Sub-block s is laid out as
// IQ4_NL at qs + 16s, with the 6-bit scale (scales_l nibble s, 2 bits s of scales_h) minus 32.
template <>
__device__ __forceinline__ float dot_unit<Type::IQ4_XS>(const uint8_t* row, int u, const BlockQ8& a) {
    const uint8_t* b = row + u / 8 * 136;
    const int s = u % 8;
    const int scale = (b[4 + s / 2] >> 4 * (s % 2) & 0xf) | (load_u16(b + 2) >> 2 * s & 3) << 4;
    const int* x = words(a);
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t v = load_u32_a2(b + 8 + 16 * s + 4 * k);
        sum = __dp4a(iq4_levels(v & 0x0f0f0f0fu), x[k], sum);
        sum = __dp4a(iq4_levels(v >> 4 & 0x0f0f0f0fu), x[k + 4], sum);
    }
    return f16_to_f32(load_u16(b)) * f16_to_f32(a.d) * static_cast<float>(sum * (scale - 32));
}

// 256 elements in 144 bytes: d @0, dmin @2, scales[12] @4, qs[128] @16. Sub-block s is the low (s even) or high
// nibbles of qs[32(s / 2), +32); value d * scale * q - dmin * min.
template <>
__device__ __forceinline__ float dot_unit<Type::Q4_K>(const uint8_t* row, int u, const BlockQ8& a) {
    const uint8_t* b = row + u / 8 * 144;
    const int s = u % 8;
    int scale = 0, min = 0;
    k_scale_min(b + 4, s, scale, min);
    const uint8_t* qs = b + 16 + 32 * (s / 2);
    const int* x = words(a);
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k)
        sum = __dp4a(static_cast<int>(load_u32(qs + 4 * k) >> 4 * (s % 2) & 0x0f0f0f0fu), x[k], sum);
    const float d = f16_to_f32(load_u16(b)), dmin = f16_to_f32(load_u16(b + 2));
    return f16_to_f32(a.d) * (d * static_cast<float>(sum * scale) - dmin * static_cast<float>(a.sum * min));
}

// 256 elements in 176 bytes: d @0, dmin @2, scales[12] @4, qh[32] @16, qs[128] @48. As Q4_K, plus a fifth bit:
// bit s of qh[l] for element l of sub-block s.
template <>
__device__ __forceinline__ float dot_unit<Type::Q5_K>(const uint8_t* row, int u, const BlockQ8& a) {
    const uint8_t* b = row + u / 8 * 176;
    const int s = u % 8;
    int scale = 0, min = 0;
    k_scale_min(b + 4, s, scale, min);
    const uint8_t* qs = b + 48 + 32 * (s / 2);
    const int* x = words(a);
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint32_t low = load_u32(qs + 4 * k) >> 4 * (s % 2) & 0x0f0f0f0fu;
        const uint32_t high = (load_u32(b + 16 + 4 * k) >> s & 0x01010101u) << 4;
        sum = __dp4a(static_cast<int>(low | high), x[k], sum);
    }
    const float d = f16_to_f32(load_u16(b)), dmin = f16_to_f32(load_u16(b + 2));
    return f16_to_f32(a.d) * (d * static_cast<float>(sum * scale) - dmin * static_cast<float>(a.sum * min));
}

// 256 elements in 210 bytes: ql[128] @0, qh[64] @128, scales[16] (int8) @192, d @208. Sub-block s = 4n + k takes its
// low 4 bits from ql[64n + 32(k % 2) + l] (low nibbles for k < 2) and its high 2 bits from bits 2k of qh[32n + l];
// value d * scale * (q - 32), one scale per 16 elements.
template <>
__device__ __forceinline__ float dot_unit<Type::Q6_K>(const uint8_t* row, int u, const BlockQ8& a) {
    const uint8_t* b = row + u / 8 * 210;
    const int s = u % 8, n = s / 4, k = s % 4;
    const uint8_t* ql = b + 64 * n + 32 * (k % 2);
    const uint8_t* qh = b + 128 + 32 * n;
    const int* x = words(a);
    int sum[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t low = load_u32_a2(ql + 4 * j) >> 4 * (k / 2) & 0x0f0f0f0fu;
        const uint32_t high = (load_u32_a2(qh + 4 * j) >> 2 * k & 0x03030303u) << 4;
        sum[j / 4] = __dp4a(static_cast<int>(__vsub4(low | high, 0x20202020u)), x[j], sum[j / 4]);
    }
    const auto* scales = reinterpret_cast<const int8_t*>(b + 192 + 8 * n + 2 * k);
    return f16_to_f32(load_u16(b + 208)) * f16_to_f32(a.d) *
           static_cast<float>(scales[0] * sum[0] + scales[1] * sum[1]);
}

// 256 elements in 82 bytes: d @0, qs[32] @2, signs[32] @34, qh[8] @66, scales[8] @74. Group g of eight elements is
// grid entry qs[g] plus 2 high bits from qh[g / 4], with sign byte signs[g]; each sub-block has two 4-bit scales,
// one per 16 elements, worth (2s + 1) / 8.
template <>
__device__ __forceinline__ float dot_unit<Type::IQ2_S>(const uint8_t* row, int u, const BlockQ8& a) {
    const uint8_t* b = row + u / 8 * 82;
    const int s = u % 8;
    const uint32_t qs = load_u32_a2(b + 2 + 4 * s), signs = load_u32_a2(b + 34 + 4 * s);
    const uint32_t qh = b[66 + s], scales = b[74 + s];
    const int* x = words(a);
    int sum[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint64_t grid = g_iq2s_grid[(qs >> 8 * j & 0xff) | (qh << (8 - 2 * j) & 0x300)];
        const uint32_t sign = signs >> 8 * j;
        sum[j / 2] = __dp4a(with_signs(static_cast<uint32_t>(grid), sign & 0xf), x[2 * j], sum[j / 2]);
        sum[j / 2] = __dp4a(with_signs(static_cast<uint32_t>(grid >> 32), sign >> 4 & 0xf), x[2 * j + 1], sum[j / 2]);
    }
    const int total =
        (2 * static_cast<int>(scales & 0xf) + 1) * sum[0] + (2 * static_cast<int>(scales >> 4) + 1) * sum[1];
    return f16_to_f32(load_u16(b)) * f16_to_f32(a.d) * (0.125f * static_cast<float>(total));
}

// 256 elements in 98 bytes: d @0, qs[64] @2, then one u32 per sub-block @66 with four 7-bit sign indices (bits 0-27)
// and a 4-bit scale (bits 28-31) worth (2s + 1) / 4. Group g is grid entries qs[2g] and qs[2g + 1]; the eighth sign
// bit makes the number of negative signs even.
template <>
__device__ __forceinline__ float dot_unit<Type::IQ3_XXS>(const uint8_t* row, int u, const BlockQ8& a) {
    const uint8_t* b = row + u / 8 * 98;
    const int s = u % 8;
    const uint32_t qs[2] = {load_u32_a2(b + 2 + 8 * s), load_u32_a2(b + 6 + 8 * s)};
    const uint32_t aux = load_u32_a2(b + 66 + 4 * s);
    const int* x = words(a);
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t signs7 = aux >> 7 * j & 127;
        const uint32_t signs = signs7 | (__popc(signs7) & 1u) << 7;
        const uint32_t pair = qs[j / 2] >> 16 * (j % 2);
        sum = __dp4a(with_signs(g_iq3xxs_grid[pair & 0xff], signs & 0xf), x[2 * j], sum);
        sum = __dp4a(with_signs(g_iq3xxs_grid[pair >> 8 & 0xff], signs >> 4), x[2 * j + 1], sum);
    }
    const int scale = static_cast<int>(aux >> 28);
    return f16_to_f32(load_u16(b)) * f16_to_f32(a.d) * (0.25f * static_cast<float>((2 * scale + 1) * sum));
}

// 256 elements in 110 bytes: d @0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106. Group g is grid entries
// qs[2g] and qs[2g + 1], each with a ninth index bit from qh[g / 4], and sign byte signs[g]; each sub-block has a
// 4-bit scale worth 1 + 2s.
template <>
__device__ __forceinline__ float dot_unit<Type::IQ3_S>(const uint8_t* row, int u, const BlockQ8& a) {
    const uint8_t* b = row + u / 8 * 110;
    const int s = u % 8;
    const uint32_t qs[2] = {load_u32_a2(b + 2 + 8 * s), load_u32_a2(b + 6 + 8 * s)};
    const uint32_t qh = b[66 + s], signs = load_u32_a2(b + 74 + 4 * s);
    const int* x = words(a);
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t pair = qs[j / 2] >> 16 * (j % 2);
        const uint32_t first = (pair & 0xff) | (qh << (8 - 2 * j) & 256);
        const uint32_t second = (pair >> 8 & 0xff) | (qh << (7 - 2 * j) & 256);
        const uint32_t sign = signs >> 8 * j;
        sum = __dp4a(with_signs(g_iq3s_grid[first], sign & 0xf), x[2 * j], sum);
        sum = __dp4a(with_signs(g_iq3s_grid[second], sign >> 4 & 0xf), x[2 * j + 1], sum);
    }
    const int scale = b[106 + s / 2] >> 4 * (s % 2) & 0xf;
    return f16_to_f32(load_u16(b)) * f16_to_f32(a.d) * static_cast<float>((1 + 2 * scale) * sum);
}

}  // namespace torch::kernels
