#!/usr/bin/env python3
"""Checks tools/make_draft.py's Q8_0, IQ4_NL and IQ4_XS blocks, decoded by tests/fixtures/reference.py (the CPU
dequantization's twin), and its step-2 writer, on synthetic tensors (numpy only).

usage: python3 tests/tools/make_draft_test.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests" / "fixtures")]
import gguf_writer as G  # noqa: E402
import make_draft as D  # noqa: E402
import reference as R  # noqa: E402


def test_q8_0() -> None:
    """Each block is the F16 scale max|x| / 127 and 32 int8 codes round(x / scale), and decodes within half a step
    plus the scale's F16 rounding."""
    rng = np.random.default_rng(8)
    x = rng.standard_normal((6, 32), dtype=np.float32) * np.float32(3)
    x[1] = 0
    raw = np.frombuffer(D.convert(R.quantize(x, R.BF16).tobytes(), R.BF16, R.Q8_0), np.uint8).reshape(-1, 34)
    x = R.dequantize(R.quantize(x, R.BF16), R.BF16).reshape(x.shape)  # the BF16 values the quantizer saw
    y = R.dequantize(raw.reshape(-1), R.Q8_0).reshape(x.shape)
    for block, xs, ys in zip(raw, x, y):
        d = np.abs(xs).max() / np.float32(127)
        q = np.round(xs / d) if d > 0 else np.zeros(32)
        d16 = block[:2].view("<f2")[0]
        assert d16 == np.float16(d) and (block[2:].view(np.int8) == q).all()
        assert d == 0 or np.abs(q).max() == 127
        assert (np.abs(ys - xs) <= (d / 2 + 127 * abs(np.float32(d16) - d)) * (1 + 1e-6)).all()


def test_iq4() -> None:
    """Values that are a power-of-two scale times levels (the largest at level -127) decode exactly; Gaussian values
    decode with less squared error than the scale that maps the largest value to -127 gives."""
    rng = np.random.default_rng(4)
    levels = D.LEVELS
    q = rng.integers(0, 16, size=(64, 32))
    q[:, rng.integers(0, 32)] = 0  # one value at -127 in every block
    x = (np.float32(2.0**-9) * levels[q]).reshape(-1)
    assert (R.dequantize(D.quantize_iq4_nl(x), R.IQ4_NL) == x).all()
    ls = rng.integers(-32, 32, size=(8, 8))
    ls[:, 0] = -32  # every super-block's largest sub-block scale
    y = (np.float32(2.0**-12) * ls.reshape(-1, 1).astype(np.float32) * levels[q]).reshape(-1)
    assert (R.dequantize(D.quantize_iq4_xs(y), R.IQ4_XS) == y).all()

    g = rng.standard_normal(1 << 16).astype(np.float32) * np.float32(0.02)
    blocks = g.reshape(-1, 32)
    peak = np.take_along_axis(blocks, np.abs(blocks).argmax(1)[:, None], 1)
    d = (peak / np.float32(-127)).astype(np.float16).astype(np.float32)
    naive = ((d * levels[D.nearest(blocks / d)] - blocks) ** 2).sum()
    for quantize, ttype in ((D.quantize_iq4_nl, R.IQ4_NL), (D.quantize_iq4_xs, R.IQ4_XS)):
        raw = quantize(g)
        assert raw.size == D.nbytes([g.size], ttype)
        assert ((R.dequantize(raw, ttype) - g) ** 2).sum() < naive
    assert (R.dequantize(D.quantize_iq4_nl(np.zeros(64, np.float32)), R.IQ4_NL) == 0).all()
    assert (R.dequantize(D.quantize_iq4_xs(np.zeros(256, np.float32)), R.IQ4_XS) == 0).all()


def test_write_draft() -> None:
    """Step 2, read in chunks that end inside rows, writes the same bytes as converting each whole tensor and writing
    the file in memory; a router value BF16 does not hold is refused."""
    rng = np.random.default_rng(48)

    def bf16(n):
        return R.quantize(rng.standard_normal(n, dtype=np.float32), R.BF16)

    router = R.dequantize(bf16(192), R.BF16)
    source = [  # name, shape fastest first, type in step 1, raw; then the draft's type and raw
        ("blk.48.attn_q_norm.weight", [36], R.F32, R.quantize(router[:36] + 1, R.F32)),
        ("blk.48.ffn_gate_inp.weight", [64, 3], R.F32, R.quantize(router, R.F32)),
        ("blk.48.hc_attn_down.weight", [96, 2], R.BF16, bf16(192)),
        ("blk.48.attn_k.weight", [96, 5], R.BF16, bf16(480)),
        ("blk.48.ffn_gate_exps.weight", [64, 3, 2], R.BF16, bf16(384)),
    ]
    want = [
        source[0],
        (*source[1][:2], R.BF16, R.quantize(router, R.BF16)),
        source[2],
        *((name, s, R.Q8_0, R.quantize(R.dequantize(raw, R.BF16), R.Q8_0)) for name, s, _, raw in source[3:]),
    ]
    meta = D.metadata()
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, ref = (Path(tmp) / name for name in ("src.gguf", "dst.gguf", "ref.gguf"))
        G.write(src, meta, source)
        D.write_draft(src, meta, [t[:3] for t in source], dst, meta, chunk=64)
        G.write(ref, meta, want)
        assert dst.read_bytes() == ref.read_bytes()
    # With iq4, the routed experts take IQ4_XS (gate, up) and IQ4_NL (down); the rest is as above.
    experts = [("blk.48.ffn_gate_exps.weight", [256, 2, 2], R.BF16, bf16(1024)),
               ("blk.48.ffn_down_exps.weight", [64, 4, 2], R.BF16, bf16(512)), source[3]]
    want = [(experts[0][0], experts[0][1], R.IQ4_XS, D.quantize_iq4_xs(R.dequantize(experts[0][3], R.BF16))),
            (experts[1][0], experts[1][1], R.IQ4_NL, D.quantize_iq4_nl(R.dequantize(experts[1][3], R.BF16))),
            (*source[3][:2], R.Q8_0, R.quantize(R.dequantize(source[3][3], R.BF16), R.Q8_0))]
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, ref = (Path(tmp) / name for name in ("src.gguf", "dst.gguf", "ref.gguf"))
        G.write(src, meta, experts)
        D.write_draft(src, meta, [t[:3] for t in experts], dst, meta, chunk=256, iq4=True)
        G.write(ref, meta, want)
        assert dst.read_bytes() == ref.read_bytes()
    try:
        D.convert(np.float32([1 + 2**-20]).tobytes(), R.F32, R.BF16)
    except AssertionError:
        pass
    else:
        raise AssertionError("a router value that BF16 rounds was accepted")


if __name__ == "__main__":
    test_q8_0()
    test_iq4()
    test_write_draft()
    print("ok")
