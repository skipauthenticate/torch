# Tensor formats

The model's two shards store their tensors in 13 formats. This page writes out each one: how many values a block
holds, where every field sits, how the bits are packed and how a value is computed. `src/formats/dequant.h` (C++) and
`tests/fixtures/reference.py` (numpy) both implement this page, and `tests/unit/dequant_test.cpp` checks that they
agree.

Conventions:

- A tensor row is a sequence of whole blocks. Byte offsets (`@n`) are within one block. Multi-byte fields are
  little-endian.
- `f16` is an IEEE half, `u8`/`u16`/`u32` unsigned integers, `i8` a signed byte.
- Element `i` is the position within the block, 0 first. `>>` and `&` are bit shifts and masks; `x[k]` is byte `k`
  of field `x`.
- All arithmetic is in F32 in the order the formula gives, after converting `d` and `dmin` from F16.
- The four codebooks (`kIq4nlValues`, `kIq2sGrid`, `kIq3xxsGrid`, `kIq3sGrid`) are in `src/formats/codebooks.h`;
  [Codebooks](#codebooks) says what they hold and where they come from.

| Format | Type id | Values per block | Bytes per block | Bits per value |
| --- | ---: | ---: | ---: | ---: |
| F32 | 0 | 1 | 4 | 32 |
| F16 | 1 | 1 | 2 | 16 |
| BF16 | 30 | 1 | 2 | 16 |
| Q8_0 | 8 | 32 | 34 | 8.5 |
| Q2_0 | 42 | 64 | 18 | 2.25 |
| IQ4_NL | 20 | 32 | 18 | 4.5 |
| IQ4_XS | 23 | 256 | 136 | 4.25 |
| Q4_K | 12 | 256 | 144 | 4.5 |
| Q5_K | 13 | 256 | 176 | 5.5 |
| Q6_K | 14 | 256 | 210 | 6.5625 |
| IQ2_S | 22 | 256 | 82 | 2.5625 |
| IQ3_XXS | 18 | 256 | 98 | 3.0625 |
| IQ3_S | 21 | 256 | 110 | 3.4375 |

## Unquantized

- **F32**: IEEE single precision.
- **F16**: IEEE half precision, including subnormals, infinities and NaN.
- **BF16**: the upper 16 bits of an IEEE single; the value is those bits followed by 16 zero bits.

## Q8_0

| Field | Offset | Type |
| --- | --- | --- |
| `d` | @0 | f16 |
| `qs` | @2 | i8[32] |

`x[i] = d * qs[i]`.

## Q2_0

| Field | Offset | Type |
| --- | --- | --- |
| `d` | @0 | f16 |
| `qs` | @2 | u8[16], four 2-bit codes per byte |

Element `i` is code `c = qs[i / 4] >> 2 * (i % 4) & 3` (low bits first). `x[i] = d * (c - 1)`, so the four codes are
the values -1, 0, 1 and 2 times `d`.

## IQ4_NL

| Field | Offset | Type |
| --- | --- | --- |
| `d` | @0 | f16 |
| `qs` | @2 | u8[16], two 4-bit indices per byte |

Element `i < 16` is the low nibble of `qs[i]`, element `i >= 16` the high nibble of `qs[i - 16]`. `x[i] = d *
kIq4nlValues[nibble]`.

## IQ4_XS

| Field | Offset | Type |
| --- | --- | --- |
| `d` | @0 | f16 |
| `scales_h` | @2 | u16 |
| `scales_l` | @4 | u8[4] |
| `qs` | @8 | u8[128] |

Eight sub-blocks of 32 elements. Sub-block `j` has the 6-bit scale `s = (scales_l[j / 2] >> 4 * (j % 2) & 15) |
(scales_h >> 2 * j & 3) << 4` and stores its 32 indices in `qs[16j, 16j + 16)` the way IQ4_NL does. `x = d * (s - 32)
* kIq4nlValues[nibble]`.

## Q4_K

| Field | Offset | Type |
| --- | --- | --- |
| `d` | @0 | f16 |
| `dmin` | @2 | f16 |
| `scales` | @4 | u8[12], eight 6-bit (scale, min) pairs |
| `qs` | @16 | u8[128], 4-bit values |

Pair `j` (0-7), with `q = scales`:

- `j < 4`: `scale = q[j] & 63`, `min = q[j + 4] & 63`.
- `j >= 4`: `scale = (q[j + 4] & 15) | (q[j - 4] >> 6) << 4`, `min = (q[j + 4] >> 4) | (q[j] >> 6) << 4`.

Four chunks of 64 elements. In chunk `c`, element `l < 32` is the low nibble of `qs[32c + l]` and element `32 + l`
the high nibble of the same byte. Element `i` uses pair `j = i / 32`. `x = d * scale_j * q - dmin * min_j`.

## Q5_K

| Field | Offset | Type |
| --- | --- | --- |
| `d` | @0 | f16 |
| `dmin` | @2 | f16 |
| `scales` | @4 | u8[12], as Q4_K |
| `qh` | @16 | u8[32], the fifth bit of every value |
| `qs` | @48 | u8[128], the low 4 bits, as Q4_K |

Element `i` takes its low 4 bits from `qs` as Q4_K does and bit 4 from bit `i / 32` of `qh[i % 32]`. `x = d *
scale_j * q - dmin * min_j` with `j = i / 32`.

## Q6_K

| Field | Offset | Type |
| --- | --- | --- |
| `ql` | @0 | u8[128], the low 4 bits |
| `qh` | @128 | u8[64], the high 2 bits |
| `scales` | @192 | i8[16], one per 16 elements |
| `d` | @208 | f16 |

Two halves of 128 elements, each four quarters of 32. For element `i`, with `n = i / 128`, `k = i % 128 / 32` and
`l = i % 32`:

- low bits `ql[64n + 32 * (k % 2) + l] >> 4 * (k / 2) & 15` (low nibble for quarters 0 and 1),
- high bits `qh[32n + l] >> 2k & 3`,
- `q = low | high << 4`, scale `scales[8n + 2k + l / 16]`.

`x = d * scale * (q - 32)`.

## IQ2_S

| Field | Offset | Type |
| --- | --- | --- |
| `d` | @0 | f16 |
| `qs` | @2 | u8[32], low 8 bits of each group's grid index |
| `signs` | @34 | u8[32], one sign byte per group |
| `qh` | @66 | u8[8], high 2 bits of four groups' indices |
| `scales` | @74 | u8[8], two 4-bit scales per byte |

Thirty-two groups of 8 elements. Group `g`:

- grid index `k = qs[g] | (qh[g / 4] >> 2 * (g % 4) & 3) << 8` (10 bits, 0-1023),
- magnitudes: entry `k` of `kIq2sGrid`, eight bytes, byte `t` for element `t` of the group,
- signs: element `t` is negative when bit `t` of `signs[g]` is set,
- scale `s = scales[g / 4] >> 4 * (g % 4 / 2) & 15`, so each 16 elements share one scale.

`x = d * (0.5 + s) / 4 * magnitude`, negated where the sign bit is set.

## IQ3_XXS

| Field | Offset | Type |
| --- | --- | --- |
| `d` | @0 | f16 |
| `qs` | @2 | u8[64], two grid indices per group |
| `aux` | @66 | u32[8], one per 32 elements |

Thirty-two groups of 8 elements; group `g` belongs to sub-block `j = g / 4`.

- magnitudes: elements 0-3 are entry `qs[2g]` of `kIq3xxsGrid`, elements 4-7 entry `qs[2g + 1]`; each entry is four
  bytes, byte `t` for element `t`,
- signs: bits 0-6 are `aux[j] >> 7 * (g % 4) & 127`; bit 7 is set when bits 0-6 have an odd number of ones, so every
  group has an even number of negative elements,
- scale `s = aux[j] >> 28`.

`x = d * (0.5 + s) / 2 * magnitude`, negated where the sign bit is set.

## IQ3_S

| Field | Offset | Type |
| --- | --- | --- |
| `d` | @0 | f16 |
| `qs` | @2 | u8[64], low 8 bits of two grid indices per group |
| `qh` | @66 | u8[8], the ninth index bit of eight indices |
| `signs` | @74 | u8[32], one sign byte per group |
| `scales` | @106 | u8[4], two 4-bit scales per byte |

Thirty-two groups of 8 elements; group `g` belongs to sub-block `j = g / 4`.

- indices `k1 = qs[2g] | (qh[j] >> 2 * (g % 4) & 1) << 8` for elements 0-3 and `k2 = qs[2g + 1] | (qh[j] >> (2 *
  (g % 4) + 1) & 1) << 8` for elements 4-7 (9 bits, 0-511),
- magnitudes: entries `k1` and `k2` of `kIq3sGrid`, four bytes each, byte `t` for element `t`,
- signs: element `t` is negative when bit `t` of `signs[g]` is set,
- scale `s = scales[j / 2] >> 4 * (j % 2) & 15`.

`x = d * (1 + 2s) * magnitude`, negated where the sign bit is set.

## Codebooks

| Name | Entries | Entry | Used by |
| --- | ---: | --- | --- |
| `kIq4nlValues` | 16 | i8, one level | IQ4_NL, IQ4_XS |
| `kIq2sGrid` | 1,024 | u64, eight magnitude bytes, element 0 in the low byte | IQ2_S |
| `kIq3xxsGrid` | 256 | u32, four magnitude bytes, element 0 in the low byte | IQ3_XXS |
| `kIq3sGrid` | 512 | u32, four magnitude bytes, element 0 in the low byte | IQ3_S |

The codebooks are the formats' fixed tables. They are in `src/formats/codebooks.h` under the MIT License; that
file carries their copyright and permission notice (see also `NOTICE`).
