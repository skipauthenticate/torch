"""Reference quantization and dequantization of the 13 tensor formats, in numpy, written from docs/formats.md.

dequantize(raw, type) decodes whole tensors at once with array operations; the field decoders (iq2s_fields and so
on) return a format's scales, indices and signs before the codebook lookup. quantize(x, type) writes the five formats
test inputs are made of (F32, F16, BF16, Q8_0, Q2_0).

The IQ codebooks are read from src/formats/codebooks.h, so the C++ code and this file share one copy of them.
Arithmetic is float32 in the order docs/formats.md gives, so results equal the C++ reference bit for bit.
"""

import re
from functools import cache
from pathlib import Path

import numpy as np

# Tensor type ids of the 13 formats and (elements, bytes) per block.
F32, F16, Q8_0, Q4_K, Q5_K, Q6_K, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, BF16, Q2_0 = (
    0, 1, 8, 12, 13, 14, 18, 20, 21, 22, 23, 30, 42)  # fmt: skip
BLOCK = {
    F32: (1, 4), F16: (1, 2), BF16: (1, 2), Q8_0: (32, 34), Q2_0: (64, 18), IQ4_NL: (32, 18), IQ4_XS: (256, 136),
    Q4_K: (256, 144), Q5_K: (256, 176), Q6_K: (256, 210), IQ2_S: (256, 82), IQ3_XXS: (256, 98), IQ3_S: (256, 110),
}  # fmt: skip
NAMES = {F32: "F32", F16: "F16", Q8_0: "Q8_0", Q4_K: "Q4_K", Q5_K: "Q5_K", Q6_K: "Q6_K", IQ3_XXS: "IQ3_XXS",
         IQ4_NL: "IQ4_NL", IQ3_S: "IQ3_S", IQ2_S: "IQ2_S", IQ4_XS: "IQ4_XS", BF16: "BF16", Q2_0: "Q2_0"}  # fmt: skip

CODEBOOKS_H = Path(__file__).resolve().parents[2] / "src" / "formats" / "codebooks.h"


@cache
def codebook(name: str) -> np.ndarray:
    """One table of codebooks.h as an array (int8 for kIq4nlValues, uint8 bytes per entry for the grids)."""
    text = CODEBOOKS_H.read_text()
    match = re.search(r"(u?int\d+)_t " + name + r"\[(\d+)\] = \{([^}]*)\}", text)
    if match is None:
        raise ValueError(f"{name} is not in {CODEBOOKS_H}")
    ctype, count, body = match.groups()
    values = [int(v, 0) for v in re.findall(r"-?(?:0x[0-9a-fA-F]+|\d+)", body)]
    assert len(values) == int(count), name
    if ctype == "int8":
        return np.array(values, dtype=np.int8)
    width = {"uint32": 4, "uint64": 8}[ctype]
    return np.array(values, dtype=f"<u{width}").view(np.uint8).reshape(-1, width)


def blocks(raw, qtype: int) -> np.ndarray:
    """The bytes as a [blocks, block_bytes] uint8 array."""
    raw = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray, memoryview)) else raw
    return raw.reshape(-1, BLOCK[qtype][1])


def f16(b: np.ndarray, at: int) -> np.ndarray:
    """The little-endian half at byte `at` of every block, as float32 [blocks, 1]."""
    return np.ascontiguousarray(b[:, at : at + 2]).view("<f2").astype(np.float32)


def u16(b: np.ndarray, at: int) -> np.ndarray:
    return np.ascontiguousarray(b[:, at : at + 2]).view("<u2").astype(np.int32)


def nibbles(qs: np.ndarray) -> np.ndarray:
    """[..., 16] bytes to [..., 32] values: elements 0-15 in the low nibbles, 16-31 in the high nibbles."""
    return np.concatenate([qs & 15, qs >> 4], axis=-1)


def bits(byte: np.ndarray) -> np.ndarray:
    """[...] bytes to [..., 8] booleans, bit 0 first."""
    return (byte[..., None] >> np.arange(8, dtype=np.uint8) & 1).astype(bool)


def k_scales(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The eight 6-bit (scale, min) pairs packed in 12 bytes, as int32 [blocks, 8] each."""
    q = q.astype(np.int32)
    scale = np.concatenate([q[:, 0:4] & 63, (q[:, 8:12] & 15) | (q[:, 0:4] >> 6) << 4], axis=1)
    low = np.concatenate([q[:, 4:8] & 63, (q[:, 8:12] >> 4) | (q[:, 4:8] >> 6) << 4], axis=1)
    return scale, low


def iq4nl_fields(raw) -> tuple[np.ndarray, np.ndarray]:
    """IQ4_NL: d [blocks, 1] and the 4-bit level index of every element [blocks, 32]."""
    b = blocks(raw, IQ4_NL)
    return f16(b, 0), nibbles(b[:, 2:18])


def iq4xs_fields(raw) -> tuple[np.ndarray, np.ndarray]:
    """IQ4_XS: the scale of each 32-element sub-block [blocks, 8] and the level index of every element [blocks, 256]."""
    b = blocks(raw, IQ4_XS)
    sub = np.arange(8)
    low = b[:, 4 + sub // 2].astype(np.int32) >> 4 * (sub % 2) & 15
    high = u16(b, 2) >> 2 * sub & 3
    scale = f16(b, 0) * ((low | high << 4) - 32).astype(np.float32)
    return scale, nibbles(b[:, 8:136].reshape(-1, 8, 16)).reshape(-1, 256)


def iq2s_fields(raw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IQ2_S: per group of eight elements [blocks, 32], its scale, its 10-bit grid index, and its signs [.., 8]."""
    b = blocks(raw, IQ2_S)
    g = np.arange(32)
    s = b[:, 74 + g // 4].astype(np.int32) >> 4 * (g % 4 // 2) & 15
    scale = f16(b, 0) * (np.float32(0.5) + s.astype(np.float32)) * np.float32(0.25)
    index = b[:, 2:34].astype(np.int32) | (b[:, 66 + g // 4].astype(np.int32) >> 2 * (g % 4) & 3) << 8
    return scale, index, bits(b[:, 34:66])


def iq3xxs_fields(raw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IQ3_XXS: per group of eight [blocks, 32], its scale, its two 8-bit grid indices [.., 2] and its signs [.., 8]."""
    b = blocks(raw, IQ3_XXS)
    aux = np.ascontiguousarray(b[:, 66:98]).view("<u4").astype(np.int64)  # [blocks, 8], one per 32 elements
    scale = f16(b, 0) * (np.float32(0.5) + (aux >> 28).astype(np.float32)) * np.float32(0.5)
    signs7 = (aux[:, :, None] >> 7 * np.arange(4) & 127).reshape(-1, 32).astype(np.uint8)
    parity = np.bitwise_count(signs7) & 1  # the eighth sign makes the number of negative signs even
    return np.repeat(scale, 4, axis=1), b[:, 2:66].reshape(-1, 32, 2), bits(signs7 | parity << 7)


def iq3s_fields(raw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IQ3_S: per group of eight [blocks, 32], its scale, its two 9-bit grid indices [.., 2], and its signs [.., 8]."""
    b = blocks(raw, IQ3_S)
    sub = np.arange(8)
    s = b[:, 106 + sub // 2].astype(np.int32) >> 4 * (sub % 2) & 15
    scale = f16(b, 0) * (1 + 2 * s).astype(np.float32)
    qh = np.repeat(b[:, 66:74].astype(np.int32), 4, axis=1)  # [blocks, 32]: the high-bit byte of each group
    shift = 2 * (np.arange(32) % 4)
    high = np.stack([qh >> shift & 1, qh >> (shift + 1) & 1], axis=-1)
    index = b[:, 2:66].reshape(-1, 32, 2).astype(np.int32) | high << 8
    return np.repeat(scale, 4, axis=1), index, bits(b[:, 74:106])


def grid_values(scale: np.ndarray, entries: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """scale [.., groups] times the grid bytes [.., groups, 8], negated where signs is set; flattened per block."""
    v = scale[..., None] * entries.astype(np.float32)
    return np.where(signs, -v, v).reshape(scale.shape[0], -1)


def dequantize(raw, qtype: int) -> np.ndarray:
    """The tensor's values as float32 [n]."""
    raw = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray, memoryview)) else raw
    if qtype == F32:
        return raw.view("<f4").astype(np.float32)
    if qtype == F16:
        return raw.view("<f2").astype(np.float32)
    if qtype == BF16:
        return (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
    b = blocks(raw, qtype)
    if qtype == Q8_0:
        out = f16(b, 0) * b[:, 2:34].view(np.int8).astype(np.float32)
    elif qtype == Q2_0:
        codes = (b[:, 2:18, None] >> 2 * np.arange(4, dtype=np.uint8) & 3).reshape(-1, 64)
        out = f16(b, 0) * (codes.astype(np.float32) - np.float32(1))
    elif qtype == IQ4_NL:
        d, index = iq4nl_fields(raw)
        out = d * codebook("kIq4nlValues")[index].astype(np.float32)
    elif qtype == IQ4_XS:
        scale, index = iq4xs_fields(raw)
        out = (np.repeat(scale, 32, axis=1) * codebook("kIq4nlValues")[index].astype(np.float32))
    elif qtype in (Q4_K, Q5_K):
        q = nibbles(b[:, 16:144].reshape(-1, 4, 32)) if qtype == Q4_K else nibbles(b[:, 48:176].reshape(-1, 4, 32))
        q = q.reshape(-1, 256).astype(np.int32)  # chunk c: elements 0-31 low nibbles, 32-63 high, of 32 bytes
        if qtype == Q5_K:
            pair = np.arange(8).repeat(32)  # element 32j + l takes bit j of qh[l]
            q |= (b[:, 16:48][:, np.arange(256) % 32].astype(np.int32) >> pair & 1) << 4
        scale, low = k_scales(b[:, 4:16])
        d = f16(b, 0) * scale.astype(np.float32)
        m = f16(b, 2) * low.astype(np.float32)
        out = np.repeat(d, 32, axis=1) * q.astype(np.float32) - np.repeat(m, 32, axis=1)
    elif qtype == Q6_K:
        e = np.arange(256)
        n, k, l = e // 128, e % 128 // 32, e % 32
        low = b[:, 64 * n + 32 * (k % 2) + l].astype(np.int32) >> 4 * (k // 2) & 15
        high = b[:, 128 + 32 * n + l].astype(np.int32) >> 2 * k & 3
        scale = b[:, 192 + 8 * n + 2 * k + l // 16].view(np.int8).astype(np.float32)
        out = f16(b, 208) * scale * ((low | high << 4) - 32).astype(np.float32)
    elif qtype == IQ2_S:
        scale, index, signs = iq2s_fields(raw)
        out = grid_values(scale, codebook("kIq2sGrid")[index], signs)
    elif qtype == IQ3_XXS:
        scale, index, signs = iq3xxs_fields(raw)
        out = grid_values(scale, codebook("kIq3xxsGrid")[index].reshape(-1, 32, 8), signs)
    elif qtype == IQ3_S:
        scale, index, signs = iq3s_fields(raw)
        out = grid_values(scale, codebook("kIq3sGrid")[index].reshape(-1, 32, 8), signs)
    else:
        raise ValueError(f"type {qtype} is not one of the 13 formats")
    return out.reshape(-1)


def quantize(x: np.ndarray, qtype: int) -> np.ndarray:
    """x (float32, size a multiple of the block) in one of the five formats test inputs use, as uint8 [bytes]."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if qtype == F32:
        return x.astype("<f4").view(np.uint8)
    if qtype == F16:
        return x.astype("<f2").view(np.uint8)
    if qtype == BF16:  # round to nearest, ties to even; inputs are finite
        u = x.view(np.uint32)
        return ((u + 0x7FFF + (u >> 16 & 1)) >> 16).astype("<u2").view(np.uint8)
    if qtype == Q8_0:
        x = x.reshape(-1, 32)
        d = np.abs(x).max(axis=1, keepdims=True) / np.float32(127)
        q = np.round(np.divide(x, d, out=np.zeros_like(x), where=d > 0))
        return np.concatenate([d.astype("<f2").view(np.uint8), q.astype(np.int8).view(np.uint8)], axis=1).reshape(-1)
    if qtype == Q2_0:  # levels -1, 0, 1, 2 times d: d covers both the largest value (as 2) and the smallest (as -1)
        x = x.reshape(-1, 64)
        d = np.maximum(x.max(axis=1, keepdims=True) / np.float32(2), -x.min(axis=1, keepdims=True))
        q = np.clip(np.round(np.divide(x, d, out=np.zeros_like(x), where=d > 0)), -1, 2).astype(np.uint8) + 1
        packed = (q.reshape(-1, 16, 4) << 2 * np.arange(4, dtype=np.uint8)).sum(axis=2, dtype=np.uint8)
        return np.concatenate([d.astype("<f2").view(np.uint8), packed], axis=1).reshape(-1)
    raise ValueError(f"cannot quantize to type {qtype}")
