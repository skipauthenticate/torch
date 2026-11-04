#!/usr/bin/env python3
"""Writes the test fixtures from this directory's numpy reference of the formats (reference.py, gguf_writer.py).

quant_blocks.bin  For each of the 13 formats, one record: u32 type, u32 n, u32 nbytes, bytes[nbytes], f32[n], where
                  the floats are reference.py's dequantization of the bytes. Read by tests/unit/dequant_test.cpp.
tiny.gguf         A small shard of a split file (metadata, an F32 tensor and a 3-D Q2_0 "experts" tensor), written
                  by gguf_writer.py. Read by tests/unit/gguf_test.cpp and the `torch inspect` tests.

Every byte is synthetic and comes from a seeded generator, so each run writes identical files. CI regenerates them
and compares the result with the committed copies.

usage: python tests/fixtures/make_fixtures.py [--out DIR]   (DIR defaults to this script's directory)
"""

import argparse
import struct
from pathlib import Path

import numpy as np

import gguf_writer as G
import reference as R

N = 512  # elements per record: two blocks of the 256-element formats

# The records of these hold seeded normal floats, quantized by reference.quantize. Its Q2_0 rule is this project's own
# (the block scale covers the largest value as 2 and the smallest as -1), so these Q2_0 blocks, like the experts in
# tiny.gguf, need not match another quantizer's blocks for the same floats; the dequantization they test is fixed.
QUANTIZABLE = [R.F32, R.F16, R.BF16, R.Q8_0, R.Q2_0]

# The records of these hold seeded random bytes, with every f16 scale replaced by a finite value in [0.5, 2). Every
# other field (grid indices, signs, 4- and 6-bit scales, nibbles) is valid for any byte value, so random bytes reach
# every kind of codebook entry. The offsets are the scales' positions in docs/formats.md.
SCALE_OFFSETS = {
    R.Q4_K: (0, 2),  # d, dmin
    R.Q5_K: (0, 2),  # d, dmin
    R.Q6_K: (208,),  # d
    R.IQ3_XXS: (0,),
    R.IQ4_NL: (0,),
    R.IQ3_S: (0,),
    R.IQ2_S: (0,),
    R.IQ4_XS: (0,),
}


def quant_record(qtype: int) -> bytes:
    rng = np.random.default_rng(qtype)  # PCG64: the same stream on every platform and numpy 2.x version
    block_elements, block_bytes = R.BLOCK[qtype]
    n_blocks = N // block_elements
    if qtype in QUANTIZABLE:
        raw = R.quantize(rng.standard_normal(N, dtype=np.float32), qtype)
    else:
        raw = rng.integers(0, 256, size=(n_blocks, block_bytes), dtype=np.uint8)
        for offset in SCALE_OFFSETS[qtype]:
            scales = rng.uniform(0.5, 2.0, size=n_blocks).astype(np.float16)
            raw[:, offset : offset + 2] = scales.view(np.uint8).reshape(n_blocks, 2)
    raw = raw.reshape(-1)
    assert raw.size == n_blocks * block_bytes
    values = R.dequantize(raw, qtype).astype("<f4")
    assert values.size == N and np.isfinite(values).all()
    return struct.pack("<III", qtype, N, raw.size) + raw.tobytes() + values.tobytes()


def write_quant_blocks(path: Path) -> None:
    path.write_bytes(b"".join(quant_record(qtype) for qtype in sorted([*QUANTIZABLE, *SCALE_OFFSETS])))


def write_tiny(path: Path) -> None:
    rng = np.random.default_rng(0)
    metadata = [  # general.architecture first; the key types match the real model's shards
        ("general.architecture", G.STRING, "qwen4exp"),
        ("split.no", G.U16, 0),
        ("split.count", G.U16, 2),
        ("split.tensors.count", G.I32, 3),
        ("qwen4exp.block_count", G.U32, 48),
        ("qwen4exp.attention.layer_norm_rms_epsilon", G.F32, 1e-6),
        ("tokenizer.ggml.tokens", G.ARRAY, (G.STRING, ["a", "bc", "def"])),
    ]
    # Shapes are fastest-varying first: [3, 2] F32 and [64, 2, 3] Q2_0 (64-element rows, 3 experts).
    norm = np.arange(6, dtype=np.float32)
    experts = R.quantize(rng.standard_normal((3, 2, 64), dtype=np.float32), R.Q2_0)
    tensors = [
        ("output_norm.weight", [3, 2], R.F32, R.quantize(norm, R.F32)),
        ("blk.0.ffn_down_exps.weight", [64, 2, 3], R.Q2_0, experts),
    ]
    G.write(path, metadata, tensors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent, help="output directory")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    write_quant_blocks(out / "quant_blocks.bin")
    write_tiny(out / "tiny.gguf")


if __name__ == "__main__":
    main()
