#!/usr/bin/env python3
"""Builds the draft-layer GGUF (docs/architecture.md, decision 12) from the MTP tensors of the original checkpoint.

Step 1 reads the 31 BF16 `mtp.*` tensors of Qwen/Qwen3.8-Flash-Next at a pinned revision with HTTP range requests:
each shard's safetensors header, then the tensors' bytes in pieces of at most 64 MiB, so nothing else of the 360 GB
checkpoint is downloaded. Every read must come back as 206 with the asked Content-Range. It writes OUT/draft-bf16.gguf
with the main file's names and layout: blk.48.* and blk.48.nextn.*, shapes fastest first, eh_proj = [fc_embedding |
fc_hidden] per output row, the indexer's fused projection split into q (all but the last key-width rows) and k, the
fused expert gate_up split into gate (first half of each expert's rows) and up, matrices BF16, the router F32, and
the nine norm gammas F32 with the 1 that the checkpoint's norms add at run time (x * (1 + w)) baked in. The piece
list in OUT/draft-bf16.done makes it resumable: run the same command again after an interruption.

Step 2 writes OUT/draft-q8_0.gguf from it: routed and shared experts, attention projections and eh_proj in Q8_0
(tests/fixtures/reference.py's quantizer), the router BF16 (exact, as it came from BF16), hyper-connection mixers and
the indexer BF16, norms F32. With --experts iq4 it writes OUT/draft-iq4.gguf instead, the same but for the routed
experts: gate and up IQ4_XS, down IQ4_NL (their rows are 640 long, which IQ4_XS's 256-value blocks do not divide),
from this file's quantizers (search_scale).

usage: python3 tools/make_draft.py OUT [--experts q8_0|iq4] [--bf16-only]   (needs numpy, tests/fixtures/requirements.txt)
"""

import argparse
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "fixtures"))
import gguf_writer as G  # noqa: E402
import reference as R  # noqa: E402

REPO, REVISION = "Qwen/Qwen3.8-Flash-Next", "de4b8e4d43b917e7706784d8bb445c9af86a3540"
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
PIECE = 64 << 20  # bytes per range read
B, N, M = "blk.48.", "blk.48.nextn.", "mtp.layers.0."

NORMS = {  # stored as 1 + w, F32
    "mtp.pre_fc_norm_embedding.weight": N + "enorm.weight",
    "mtp.pre_fc_norm_hidden.weight": N + "hnorm.weight",
    "mtp.hyper_connection_mixer.hc_norm.weight": N + "hc_head_norm.weight",
    M + "attn_hyper_connection.hc_norm.weight": B + "hc_attn_norm.weight",
    M + "mlp_hyper_connection.hc_norm.weight": B + "hc_ffn_norm.weight",
    M + "self_attn.q_norm.weight": B + "attn_q_norm.weight",
    M + "self_attn.k_norm.weight": B + "attn_k_norm.weight",
    M + "self_attn.indexer.q_layernorm.weight": B + "indexer.q_norm.weight",
    M + "self_attn.indexer.k_layernorm.weight": B + "indexer.k_norm.weight",
}
ROUTERS = {  # F32 in step 1, BF16 in the draft
    M + "mlp.gate.weight": B + "ffn_gate_inp.weight",
    M + "mlp.shared_expert_gate.weight": B + "ffn_gate_inp_shexp.weight",
}
MIXERS = {  # hyper-connection mixers: BF16 bytes as they are, also in the draft
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": N + "hc_head_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": N + "hc_head_up.weight",
} | {
    f"{M}{hf}_hyper_connection.{part}.weight": f"{B}hc_{gg}_{name}.weight"
    for hf, gg in (("attn", "attn"), ("mlp", "ffn"))
    for part, name in (("input_mix_weight_down", "down"), ("input_mix_weight_up", "up"),
                       ("block_inject_weight", "inject"))
}
MATRICES = {  # BF16 bytes as they are, Q8_0 in the draft
    M + "self_attn.q_proj.weight": B + "attn_q.weight",  # each head's query rows then its output-gate rows
    M + "self_attn.k_proj.weight": B + "attn_k.weight",
    M + "self_attn.v_proj.weight": B + "attn_v.weight",
    M + "self_attn.o_proj.weight": B + "attn_output.weight",
    M + "mlp.shared_expert.gate_proj.weight": B + "ffn_gate_shexp.weight",
    M + "mlp.shared_expert.up_proj.weight": B + "ffn_up_shexp.weight",
    M + "mlp.shared_expert.down_proj.weight": B + "ffn_down_shexp.weight",
    M + "mlp.experts.down_proj": B + "ffn_down_exps.weight",
}
EH = ("mtp.fc_embedding.weight", "mtp.fc_hidden.weight")
INDEX_QK, INDEX_K_NORM = M + "self_attn.indexer.index_qk_proj.weight", M + "self_attn.indexer.k_layernorm.weight"
GATE_UP = M + "mlp.experts.gate_up_proj"
COPIES = MIXERS | MATRICES
SOURCES = {*NORMS, *ROUTERS, *COPIES, *EH, INDEX_QK, GATE_UP}
assert len(SOURCES) == 31
Q8_0 = {*MATRICES.values(), N + "eh_proj.weight", B + "ffn_gate_exps.weight", B + "ffn_up_exps.weight"}
IQ4 = {B + "ffn_gate_exps.weight": R.IQ4_XS, B + "ffn_up_exps.weight": R.IQ4_XS, B + "ffn_down_exps.weight": R.IQ4_NL}

# IQ4_NL and IQ4_XS (docs/formats.md): each value is a scale times one of 16 non-linear levels.
LEVELS = R.codebook("kIq4nlValues").astype(np.float32)  # ascending, -127 to 113
MIDS = (LEVELS[1:] + LEVELS[:-1]) / 2
STEP = 8  # the scale search assigns levels from a table of x / d at 1 / STEP resolution; the final assignment is exact
LUT = np.searchsorted(MIDS, np.arange(-128 * STEP, 128 * STEP + 1, dtype=np.float32) / STEP).astype(np.uint8)


def metadata() -> list:
    link = f"https://huggingface.co/{REPO}"
    return [
        ("general.architecture", G.STRING, "qwen4exp"),
        ("general.type", G.STRING, "model"),
        ("general.name", G.STRING, "Qwen3.8-Flash-Next draft layer"),
        ("general.license", G.STRING, "other"),
        ("general.license.name", G.STRING, "qwen-community-1.0"),
        ("general.license.link", G.STRING, f"{link}/blob/{REVISION}/LICENSE"),
        ("general.source.url", G.STRING, f"{link}/tree/{REVISION}"),
        ("qwen4exp.block_count", G.U32, 49),
        ("qwen4exp.nextn_predict_layers", G.U32, 1),
    ]


def nbytes(shape, ttype: int) -> int:
    elements, block_bytes = R.BLOCK[ttype]
    return int(np.prod(shape)) // elements * block_bytes


def fetch(url: str, start: int | None = None, stop: int | None = None) -> bytes:
    """The bytes [start, stop) of url (all of it without a range), retrying transient failures with backoff."""
    headers = {"User-Agent": "torch-make-draft"} | ({"Range": f"bytes={start}-{stop - 1}"} if start is not None else {})
    for attempt in range(10):
        wait = min(120, 2**attempt)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                if start is not None:  # checked before the body is read: a server that ignores Range sends everything
                    got = r.headers.get("Content-Range", "")
                    if r.status != 206 or not got.startswith(f"bytes {start}-{stop - 1}/"):
                        raise SystemExit(f"{url}: asked for bytes {start}-{stop - 1}, got HTTP {r.status} '{got}'.")
                data = r.read()
            if start is None or len(data) == stop - start:
                return data
            error = f"{len(data)} of {stop - start} bytes"
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise
            error, wait = f"HTTP {e.code}", max(wait, int(e.headers.get("Retry-After") or 0))
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            error = repr(e)
        print(f"retrying in {wait} s after {error}: {url} {start}", file=sys.stderr, flush=True)
        time.sleep(wait)
    raise SystemExit(f"{url}: gave up after 10 attempts.")


def sources() -> dict:
    """Each MTP tensor's URL, the file offset of its data and its shape, from the shard index and headers."""
    index = json.loads(fetch(BASE + "model.safetensors.index.json"))["weight_map"]
    shards = {shard for name, shard in index.items() if name.startswith("mtp.")}
    found = {}
    for shard in sorted(shards):
        size = int.from_bytes(fetch(BASE + shard, 0, 8), "little")
        for name, t in json.loads(fetch(BASE + shard, 8, 8 + size)).items():
            if name.startswith("mtp."):
                if t["dtype"] != "BF16" or name not in SOURCES:
                    raise SystemExit(f"{shard} has {name} ({t['dtype']}), which this tool does not map.")
                begin, end = t["data_offsets"]
                assert end - begin == nbytes(t["shape"], R.BF16)
                found[name] = (BASE + shard, 8 + size + begin, t["shape"])
    if set(found) != SOURCES:
        raise SystemExit(f"The checkpoint lacks {sorted(SOURCES - set(found))}.")
    return found


def plan(src: dict) -> tuple[list, list]:
    """The step-1 tensors [(name, shape fastest first, type)] and the reads that fill them: [(key, [(source, start,
    stop)], make)], where start and stop are offsets in the source tensor and make(list of bytes) returns the writes
    [(tensor, offset in the tensor, bytes)]."""
    tensors, jobs = [], []
    shape = {hf: s for hf, (_, _, s) in src.items()}

    def copy(hf, name, start, stop):  # source bytes [start, stop) to the start of tensor name
        for a in range(start, stop, PIECE):
            write = lambda b, name=name, at=a - start: [(name, at, b[0])]  # noqa: E731
            jobs.append((f"{hf}@{a}", [(hf, a, min(a + PIECE, stop))], write))

    def f32(raw, add=0):
        return (R.dequantize(raw, R.BF16) + np.float32(add)).tobytes()

    for hf, name in NORMS.items():
        tensors.append((name, shape[hf][::-1], R.F32))
        jobs.append((hf, [(hf, 0, nbytes(shape[hf], R.BF16))], lambda b, name=name: [(name, 0, f32(b[0], 1))]))
    for hf, name in ROUTERS.items():
        tensors.append((name, shape[hf][::-1], R.F32))
        jobs.append((hf, [(hf, 0, nbytes(shape[hf], R.BF16))], lambda b, name=name: [(name, 0, f32(b[0]))]))
    for hf, name in COPIES.items():
        tensors.append((name, shape[hf][::-1], R.BF16))
        copy(hf, name, 0, nbytes(shape[hf], R.BF16))

    (emb, hid) = (shape[hf] for hf in EH)
    assert emb[0] == hid[0]
    tensors.append((N + "eh_proj.weight", [emb[1] + hid[1], emb[0]], R.BF16))

    def eh(b):  # each output row: its fc_embedding row, then its fc_hidden row
        rows = [np.frombuffer(raw, "<u2").reshape(s) for raw, s in zip(b, (emb, hid))]
        return [(N + "eh_proj.weight", 0, np.concatenate(rows, axis=1).tobytes())]

    jobs.append((EH[0], [(hf, 0, nbytes(shape[hf], R.BF16)) for hf in EH], eh))

    (qk, cols), (k,) = shape[INDEX_QK], shape[INDEX_K_NORM]  # [heads x key width | key width] rows
    split = nbytes([qk - k, cols], R.BF16)
    tensors += [(B + "indexer.q_proj.weight", [cols, qk - k], R.BF16), (B + "indexer.k_proj.weight", [cols, k], R.BF16)]
    copy(INDEX_QK, B + "indexer.q_proj.weight", 0, split)
    copy(INDEX_QK, B + "indexer.k_proj.weight", split, nbytes(shape[INDEX_QK], R.BF16))

    experts, rows, cols = shape[GATE_UP]  # per expert: gate rows, then up rows
    gate, up, half = B + "ffn_gate_exps.weight", B + "ffn_up_exps.weight", nbytes([rows // 2, cols], R.BF16)
    tensors += [(gate, [cols, rows // 2, experts], R.BF16), (up, [cols, rows // 2, experts], R.BF16)]
    step = max(1, PIECE // (2 * half))
    for e in range(0, experts, step):
        n = min(step, experts - e)

        def make(b, e=e, n=n):
            pair = np.frombuffer(b[0], np.uint8).reshape(n, 2, half)
            return [(gate, e * half, pair[:, 0].tobytes()), (up, e * half, pair[:, 1].tobytes())]

        jobs.append((f"{GATE_UP}@{e}", [(GATE_UP, e * 2 * half, (e + n) * 2 * half)], make))
    return tensors, jobs


def fetch_bf16(out: Path) -> list:
    """Step 1: writes (or finishes) OUT/draft-bf16.gguf and returns its tensors."""
    src = sources()
    tensors, jobs = plan(src)
    header, at = G.layout(metadata(), [(name, s, t, nbytes(s, t)) for name, s, t in tensors])
    where = {name: offset for (name, _, _), offset in zip(tensors, at)}
    path, log_path = out / "draft-bf16.gguf", out / "draft-bf16.done"
    stamp = hashlib.sha256(header).hexdigest()
    done = log_path.read_text().split() if log_path.exists() else []
    if done and done[0] != stamp:
        raise SystemExit(f"{path} was started with another layout; delete it and {log_path} to start over.")
    if not done:
        with open(path, "wb") as f:
            f.write(header)
            f.truncate(at[-1])
        log_path.write_text(stamp + "\n")
    total = sum(b - a for key, reads, _ in jobs if key not in done for _, a, b in reads)
    t0, got = time.time(), 0
    with open(path, "r+b") as f, open(log_path, "a") as log:
        for i, (key, reads, make) in enumerate(jobs):
            if key in done:
                continue
            parts = [fetch(src[hf][0], src[hf][1] + a, src[hf][1] + b) for hf, a, b in reads]
            for name, offset, data in make(parts):
                os.pwrite(f.fileno(), data, where[name] + offset)
            os.fsync(f.fileno())
            log.write(key + "\n")
            log.flush()
            os.fsync(log.fileno())
            got += sum(len(p) for p in parts)
            rate = got / 1e6 / (time.time() - t0)
            print(f"{i + 1}/{len(jobs)} {got / 1e6:,.0f} of {total / 1e6:,.0f} MB, {rate:.1f} MB/s  {key}", flush=True)
    return tensors


def draft_type(name: str, ttype: int, iq4: bool = False) -> int:
    if iq4 and name in IQ4:
        return IQ4[name]
    return R.Q8_0 if name in Q8_0 else R.BF16 if name in ROUTERS.values() else ttype


def nearest(v: np.ndarray) -> np.ndarray:
    """The index of the level nearest to each v (ties to the lower level)."""
    return np.searchsorted(MIDS, v).astype(np.uint8)


def search_scale(x: np.ndarray, rounds: int = 2) -> np.ndarray:
    """Per row of x [n, 32], a scale d with a small sum of (x - d * level)^2 over nearest-level assignments: starts with
    the largest |x| at the lowest or the highest level (d may be negative) and 0.9, 1 and 1.1 times that scale; each
    start's assignment gives its least-squares scale, which assigns again, `rounds` times; the least-squares scale with
    the least error of all. 0 for an all-zero row."""
    peak = np.take_along_axis(x, np.abs(x).argmax(1)[:, None], 1)[:, 0]
    xx = (x * x).sum(1)
    best_err, best = np.full(len(x), np.inf, np.float32), np.zeros(len(x), np.float32)
    for top in (LEVELS[0], LEVELS[-1]):
        for f in (0.9, 1.0, 1.1):
            d = np.where(peak != 0, peak / (top * np.float32(f)), 1).astype(np.float32)
            for _ in range(rounds + 1):
                t = LEVELS[LUT[np.rint((np.clip(x / d[:, None], -128, 128) + 128) * STEP).astype(np.int32)]]
                xt, tt = (x * t).sum(1), (t * t).sum(1)  # tt > 0: no level is 0
                ls = (xt / tt).astype(np.float32)  # the assignment's least-squares scale, and its error
                err = xx - xt * xt / tt
                better = err < best_err
                best_err, best = np.where(better, err, best_err), np.where(better, ls, best)
                d = np.where(xt != 0, ls, d)
    return np.where(peak != 0, best, 0).astype(np.float32)


def nibbles(q: np.ndarray) -> np.ndarray:
    """[n, 32] level indices as [n, 16] bytes: element i in the low nibble of byte i, element 16 + i in the high one."""
    return (q[:, :16] | q[:, 16:] << 4).astype(np.uint8)


def quantize_iq4_nl(x: np.ndarray) -> np.ndarray:
    """Blocks of 32: an F16 scale d and the nearest level to each x / d."""
    x = np.asarray(x, np.float32).reshape(-1, 32)
    d = search_scale(x).astype("<f2")
    q = nearest(x / np.where(d == 0, 1, d).astype(np.float32)[:, None])
    return np.concatenate([d.view(np.uint8).reshape(-1, 2), nibbles(q)], axis=1).reshape(-1)


def quantize_iq4_xs(x: np.ndarray) -> np.ndarray:
    """Blocks of 256: eight 32-value sub-blocks, each with the scale d * (s - 32) for one F16 d and a 6-bit s, and the
    nearest level to each value over its sub-block's scale. d puts the largest sub-block scale found by search_scale at
    s - 32 = -32; each s is the rounded ratio of its sub-block's own scale to d."""
    x = np.asarray(x, np.float32).reshape(-1, 8, 32)
    sub = search_scale(x.reshape(-1, 32)).reshape(-1, 8)
    peak = np.take_along_axis(sub, np.abs(sub).argmax(1)[:, None], 1)[:, 0]
    d = (peak / np.float32(-32)).astype("<f2")
    df = d.astype(np.float32)
    ls = np.clip(np.rint(sub / np.where(df == 0, 1, df)[:, None]), -32, 31).astype(np.int32)
    scale = df[:, None] * ls.astype(np.float32)
    q = nearest(x / np.where(scale == 0, 1, scale)[:, :, None]).reshape(-1, 32)
    s = (ls + 32).astype(np.uint32)
    high = (((s >> 4) & 3) << (2 * np.arange(8, dtype=np.uint32))).sum(axis=1).astype("<u2")
    low = (s & 15).astype(np.uint8)
    return np.concatenate([d.view(np.uint8).reshape(-1, 2), high.view(np.uint8).reshape(-1, 2),
                           (low[:, 0::2] | low[:, 1::2] << 4).astype(np.uint8), nibbles(q).reshape(-1, 128)],
                          axis=1).reshape(-1)


def convert(raw: bytes, src: int, dst: int) -> bytes:
    if src == dst:
        return raw
    x = R.dequantize(raw, src)
    if dst == R.BF16:
        assert not (x.view(np.uint32) & 0xFFFF).any(), "a value that BF16 does not hold exactly"
        return R.quantize(x, dst).tobytes()
    assert np.isfinite(x).all()
    quantize = {R.Q8_0: lambda v: R.quantize(v, R.Q8_0), R.IQ4_NL: quantize_iq4_nl, R.IQ4_XS: quantize_iq4_xs}[dst]
    return quantize(x).tobytes()


def write_draft(src: Path, src_meta: list, tensors: list, dst: Path, dst_meta: list, chunk: int = 1 << 22,
                iq4: bool = False) -> None:
    """Step 2: copies the GGUF at src (metadata src_meta, tensors [(name, shape, type)]) to dst with each tensor in
    draft_type, reading `chunk` elements (a multiple of 256) at a time."""
    _, at = G.layout(src_meta, [(name, s, t, nbytes(s, t)) for name, s, t in tensors])
    out = [(name, s, draft_type(name, t, iq4)) for name, s, t in tensors]
    header, _ = G.layout(dst_meta, [(name, s, t, nbytes(s, t)) for name, s, t in out])
    part = dst.with_name(dst.name + ".part")
    with open(src, "rb") as fi, open(part, "wb") as fo:
        fo.write(header)
        for (name, s, t), (_, _, u), offset in zip(tensors, out, at):
            fi.seek(offset)
            n = int(np.prod(s))
            for i in range(0, n, chunk):
                raw = fi.read(nbytes([min(chunk, n - i)], t))
                fo.write(convert(raw, t, u))
            fo.write(bytes(G.pad(nbytes(s, u))))
    os.replace(part, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path, help="output directory")
    parser.add_argument("--experts", choices=["q8_0", "iq4"], default="q8_0", help="the routed experts' formats")
    parser.add_argument("--bf16-only", action="store_true",
                        help="stop after step 1 (the installer checks the BF16 file's sha256 before quantizing)")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tensors = fetch_bf16(out)
    if args.bf16_only:
        return
    draft = out / f"draft-{args.experts}.gguf"
    write_draft(out / "draft-bf16.gguf", metadata(), tensors, draft, metadata(), iq4=args.experts == "iq4")
    print(f"wrote {draft} ({draft.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
