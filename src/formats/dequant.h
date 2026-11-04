// Dequantization of the 13 formats in the model's two shards: the single definition of every block layout, shared by
// the CPU reference and (from M2) the GPU kernels.
//
// dequant8<T>(row, i, out) writes elements [i, i + 8) of one row as F32, for i % 8 == 0. In every format here eight
// consecutive elements share one scale, so eight is the unit both the CPU and the GPU decode. Block layouts and value
// formulas are the file format's, as docs/formats.md writes them out. Byte offsets in the comments are within one
// block.
#pragma once

#include <bit>
#include <cstdint>
#include <cstring>

#include "formats/codebooks.h"
#include "formats/types.h"

#if defined(__CUDACC__)
#define TORCH_HD __host__ __device__ __forceinline__
#else
#define TORCH_HD inline
#endif

namespace torch {

TORCH_HD uint16_t load_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | p[1] << 8);
}

TORCH_HD uint32_t load_u32(const uint8_t* p) {
    return p[0] | static_cast<uint32_t>(p[1]) << 8 | static_cast<uint32_t>(p[2]) << 16 |
           static_cast<uint32_t>(p[3]) << 24;
}

// IEEE half to float, exact for every input, including subnormals, infinities and NaN.
TORCH_HD float fp16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    if (exponent == 0) {  // zero or subnormal: mantissa * 2^-24, exact in float
        const float magnitude = static_cast<float>(mantissa) * 0x1p-24f;
        return sign ? -magnitude : magnitude;
    }
    const uint32_t bits = exponent == 0x1f ? sign | 0x7f800000 | mantissa << 13  // infinity or NaN
                                           : sign | (exponent + 112) << 23 | mantissa << 13;
    return std::bit_cast<float>(bits);
}

TORCH_HD float bf16_to_f32(uint16_t h) {
    return std::bit_cast<float>(static_cast<uint32_t>(h) << 16);
}

// Element e (0-31) of a 32-element block whose 16 bytes hold elements 0-15 in the low nibbles and 16-31 in the high
// nibbles (Q4_0's layout, used by IQ4_NL and IQ4_XS).
TORCH_HD int nibble(const uint8_t* qs, int e) {
    return e < 16 ? qs[e] & 0xf : qs[e - 16] >> 4;
}

// Scale `j` (0-7) and its minimum from the 12 packed 6-bit values of Q4_K and Q5_K.
TORCH_HD void k_scale_min(const uint8_t* q, int j, int& scale, int& min) {
    if (j < 4) {
        scale = q[j] & 63;
        min = q[j + 4] & 63;
    } else {
        scale = (q[j + 4] & 0xf) | (q[j - 4] >> 6) << 4;
        min = (q[j + 4] >> 4) | (q[j] >> 6) << 4;
    }
}

// The step the IQ2/IQ3 formats share: eight grid bytes times the scale, negated where the sign bit is set.
TORCH_HD void grid8(uint64_t grid, unsigned signs, float scale, float* out) {
    for (int j = 0; j < 8; ++j) {
        const float v = scale * static_cast<float>((grid >> 8 * j) & 0xff);
        out[j] = (signs >> j) & 1 ? -v : v;
    }
}

template <Type T>
TORCH_HD void dequant8(const uint8_t* row, int64_t i, float* out);

template <>
TORCH_HD void dequant8<Type::F32>(const uint8_t* row, int64_t i, float* out) {
    std::memcpy(out, row + 4 * i, 8 * sizeof(float));
}

template <>
TORCH_HD void dequant8<Type::F16>(const uint8_t* row, int64_t i, float* out) {
    for (int t = 0; t < 8; ++t)
        out[t] = fp16_to_f32(load_u16(row + 2 * (i + t)));
}

template <>
TORCH_HD void dequant8<Type::BF16>(const uint8_t* row, int64_t i, float* out) {
    for (int t = 0; t < 8; ++t)
        out[t] = bf16_to_f32(load_u16(row + 2 * (i + t)));
}

// 32 elements in 34 bytes: d (f16) @0, qs[32] (int8) @2.
template <>
TORCH_HD void dequant8<Type::Q8_0>(const uint8_t* row, int64_t i, float* out) {
    const uint8_t* b = row + i / 32 * 34;
    const int e = static_cast<int>(i % 32);
    const float d = fp16_to_f32(load_u16(b));
    for (int t = 0; t < 8; ++t)
        out[t] = d * static_cast<float>(static_cast<int8_t>(b[2 + e + t]));
}

// 64 elements in 18 bytes: d (f16) @0, qs[16] @2 with four 2-bit codes per byte, low bits first. Codes 0-3 are the
// values -1, 0, 1, 2.
template <>
TORCH_HD void dequant8<Type::Q2_0>(const uint8_t* row, int64_t i, float* out) {
    const uint8_t* b = row + i / 64 * 18;
    const uint8_t* qs = b + 2 + i % 64 / 4;
    const float d = fp16_to_f32(load_u16(b));
    for (int t = 0; t < 8; ++t)
        out[t] = d * static_cast<float>((qs[t / 4] >> 2 * (t % 4) & 3) - 1);
}

// 32 elements in 18 bytes: d (f16) @0, qs[16] @2 as in nibble(); each nibble indexes kIq4nlValues.
template <>
TORCH_HD void dequant8<Type::IQ4_NL>(const uint8_t* row, int64_t i, float* out) {
    const uint8_t* b = row + i / 32 * 18;
    const int e = static_cast<int>(i % 32);
    const float d = fp16_to_f32(load_u16(b));
    for (int t = 0; t < 8; ++t)
        out[t] = d * static_cast<float>(kIq4nlValues[nibble(b + 2, e + t)]);
}

// 256 elements in 136 bytes: d (f16) @0, scales_h (u16) @2, scales_l[4] @4, qs[128] @8. Eight 32-element sub-blocks,
// each laid out as IQ4_NL with a 6-bit scale (4 low bits in scales_l, 2 high bits in scales_h) minus 32.
template <>
TORCH_HD void dequant8<Type::IQ4_XS>(const uint8_t* row, int64_t i, float* out) {
    const uint8_t* b = row + i / 256 * 136;
    const int sub = static_cast<int>(i % 256 / 32);
    const int e = static_cast<int>(i % 32);
    const int scale = (b[4 + sub / 2] >> 4 * (sub % 2) & 0xf) | (load_u16(b + 2) >> 2 * sub & 3) << 4;
    const float d = fp16_to_f32(load_u16(b)) * static_cast<float>(scale - 32);
    for (int t = 0; t < 8; ++t)
        out[t] = d * static_cast<float>(kIq4nlValues[nibble(b + 8 + 16 * sub, e + t)]);
}

// 256 elements in 144 bytes: d (f16) @0, dmin (f16) @2, scales[12] @4, qs[128] @16. Four 64-element chunks; chunk c
// keeps elements 0-31 in the low nibbles of qs[32c, 32c + 32) and 32-63 in the high nibbles. Value d * scale * q -
// dmin * min, with scale and min 6-bit, one pair per 32 elements.
template <>
TORCH_HD void dequant8<Type::Q4_K>(const uint8_t* row, int64_t i, float* out) {
    const uint8_t* b = row + i / 256 * 144;
    const int c = static_cast<int>(i % 256 / 64);
    const int w = static_cast<int>(i % 64);
    int scale = 0, min = 0;
    k_scale_min(b + 4, 2 * c + w / 32, scale, min);
    const float d = fp16_to_f32(load_u16(b)) * static_cast<float>(scale);
    const float m = fp16_to_f32(load_u16(b + 2)) * static_cast<float>(min);
    const uint8_t* qs = b + 16 + 32 * c + w % 32;
    for (int t = 0; t < 8; ++t)
        out[t] = d * static_cast<float>(w < 32 ? qs[t] & 0xf : qs[t] >> 4) - m;
}

// 256 elements in 176 bytes: d, dmin (f16) @0, @2, scales[12] @4, qh[32] @16, qs[128] @48. As Q4_K, plus a fifth bit
// per element: bit (2c + half) of qh[l] for element l of half `half` of chunk c.
template <>
TORCH_HD void dequant8<Type::Q5_K>(const uint8_t* row, int64_t i, float* out) {
    const uint8_t* b = row + i / 256 * 176;
    const int c = static_cast<int>(i % 256 / 64);
    const int half = static_cast<int>(i % 64 / 32);
    const int l = static_cast<int>(i % 32);
    int scale = 0, min = 0;
    k_scale_min(b + 4, 2 * c + half, scale, min);
    const float d = fp16_to_f32(load_u16(b)) * static_cast<float>(scale);
    const float m = fp16_to_f32(load_u16(b + 2)) * static_cast<float>(min);
    const uint8_t* qh = b + 16 + l;
    const uint8_t* qs = b + 48 + 32 * c + l;
    for (int t = 0; t < 8; ++t) {
        const int q = (half ? qs[t] >> 4 : qs[t] & 0xf) | (qh[t] >> (2 * c + half) & 1) << 4;
        out[t] = d * static_cast<float>(q) - m;
    }
}

// 256 elements in 210 bytes: ql[128] @0, qh[64] @128, scales[16] (int8) @192, d (f16) @208. Two 128-element halves of
// four 32-element quarters; quarter k of half n takes its low 4 bits from ql[64n + 32(k % 2) + l] (low nibble for
// k < 2) and its high 2 bits from bits 2k of qh[32n + l]. Value d * scale * (q - 32), one scale per 16 elements.
template <>
TORCH_HD void dequant8<Type::Q6_K>(const uint8_t* row, int64_t i, float* out) {
    const uint8_t* b = row + i / 256 * 210;
    const int n = static_cast<int>(i % 256 / 128);
    const int k = static_cast<int>(i % 128 / 32);
    const int l = static_cast<int>(i % 32);
    const uint8_t* ql = b + 64 * n + 32 * (k % 2) + l;
    const uint8_t* qh = b + 128 + 32 * n + l;
    const int scale = static_cast<int8_t>(b[192 + 8 * n + 2 * k + l / 16]);
    const float d = fp16_to_f32(load_u16(b + 208)) * static_cast<float>(scale);
    for (int t = 0; t < 8; ++t) {
        const int q = (k < 2 ? ql[t] & 0xf : ql[t] >> 4) | (qh[t] >> 2 * k & 3) << 4;
        out[t] = d * static_cast<float>(q - 32);
    }
}

// 256 elements in 82 bytes: d (f16) @0, qs[32] @2, signs[32] @34, qh[8] @66, scales[8] @74. Group g of eight elements
// is grid entry qs[g] plus 2 high bits from qh[g / 4], with sign byte signs[g]; each 32-element sub-block has two
// 4-bit scales, one per 16 elements, worth d * (0.5 + s) / 4.
template <>
TORCH_HD void dequant8<Type::IQ2_S>(const uint8_t* row, int64_t i, float* out) {
    const uint8_t* b = row + i / 256 * 82;
    const int g = static_cast<int>(i % 256 / 8);
    const int scale = b[74 + g / 4] >> 4 * (g % 4 / 2) & 0xf;
    const float d = fp16_to_f32(load_u16(b)) * (0.5f + static_cast<float>(scale)) * 0.25f;
    const int index = b[2 + g] | (b[66 + g / 4] << (8 - 2 * (g % 4)) & 0x300);
    grid8(kIq2sGrid[index], b[34 + g], d, out);
}

// 256 elements in 98 bytes: d (f16) @0, qs[64] @2, then one u32 per 32-element sub-block @66 holding four 7-bit sign
// indices (bits 0-27) and a 4-bit scale (bits 28-31) worth d * (0.5 + s) / 2. Group g of eight elements is grid
// entries qs[2g] and qs[2g + 1].
template <>
TORCH_HD void dequant8<Type::IQ3_XXS>(const uint8_t* row, int64_t i, float* out) {
    const uint8_t* b = row + i / 256 * 98;
    const int g = static_cast<int>(i % 256 / 8);
    const uint32_t aux = load_u32(b + 66 + 4 * (g / 4));
    const float d = fp16_to_f32(load_u16(b)) * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
    // Seven sign bits are stored; the eighth makes the number of negative signs even.
    const unsigned signs7 = aux >> 7 * (g % 4) & 127;
    const unsigned signs = signs7 | (std::popcount(signs7) & 1u) << 7;
    grid8(kIq3xxsGrid[b[2 + 2 * g]] | static_cast<uint64_t>(kIq3xxsGrid[b[3 + 2 * g]]) << 32, signs, d, out);
}

// 256 elements in 110 bytes: d (f16) @0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106. Group g of eight
// elements is grid entries qs[2g] and qs[2g + 1], each with a ninth index bit from qh[g / 4], and sign byte signs[g];
// each 32-element sub-block has a 4-bit scale worth d * (1 + 2s).
template <>
TORCH_HD void dequant8<Type::IQ3_S>(const uint8_t* row, int64_t i, float* out) {
    const uint8_t* b = row + i / 256 * 110;
    const int g = static_cast<int>(i % 256 / 8);
    const int sub = g / 4;
    const int scale = b[106 + sub / 2] >> 4 * (sub % 2) & 0xf;
    const float d = fp16_to_f32(load_u16(b)) * static_cast<float>(1 + 2 * scale);
    const int qh = b[66 + sub];
    const int index1 = b[2 + 2 * g] | (qh << (8 - 2 * (g % 4)) & 256);
    const int index2 = b[3 + 2 * g] | (qh << (7 - 2 * (g % 4)) & 256);
    grid8(kIq3sGrid[index1] | static_cast<uint64_t>(kIq3sGrid[index2]) << 32, b[74 + g], d, out);
}

// The CPU reference: n elements (n % 8 == 0) of one row. Returns false for a type the engine does not dequantize.
inline bool dequant_row(Type type, const uint8_t* row, int64_t n, float* out) {
    return with_type(type, [&]<Type T>() {
        for (int64_t i = 0; i < n; i += 8)
            dequant8<T>(row, i, out + i);
    });
}

}  // namespace torch
