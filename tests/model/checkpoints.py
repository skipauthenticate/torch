"""Check a saved prefix against a fresh run. Keep the last five logit vectors in memory.

Run a reference first for each length and prefill mode. Use the same results directory
and model options for the arena, memory and disk checks. Each invocation loads one model.
See docs/development.md for an example.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("tokens", type=Path, help="comma-separated input ids")
    parser.add_argument("length", type=int)
    parser.add_argument("tier", choices=("fresh", "arena", "memory", "disk"))
    parser.add_argument("results", type=Path)
    parser.add_argument("options", nargs=argparse.REMAINDER, help="torch run options after --")
    args = parser.parse_args()
    flags = args.options
    if flags[:1] == ["--"]:
        flags = flags[1:]
    if args.length < 1 or "--draft" not in flags:
        parser.error("give a positive prefix length and a draft model")
    args.results.mkdir(mode=0o700, parents=True, exist_ok=True)
    mode = "fast" if "--fast-prefill" in flags else "exact"
    stem = f"{mode}-{args.length}"
    reference = args.results / f"{stem}-fresh.json"
    with tempfile.TemporaryDirectory(dir=args.results) as tmp:
        work = Path(tmp)
        ids = args.tokens.read_text().strip().split(",")
        if not ids or not all(t.isdecimal() for t in ids):
            parser.error("input must contain comma-separated token ids")
        prefix = (ids * ((args.length + len(ids) - 1) // len(ids)))[:args.length]
        first, target, side = (work / name for name in ("first", "target", "side"))
        first.write_text(",".join(prefix + ["410", "105", "966", "23"] * 3))
        target.write_text(",".join(prefix + ["816", "587", "302", "69"] * 4))
        side.write_text(",".join(["71" if prefix[0] != "71" else "72", "78", "922", "891"] * 19))
        # The last position in the prompt and all four generated positions each have one raw F32 vector.
        keep = 5 * 248320 * 4
        cmd = [str(args.executable.resolve()), "run", str(args.model), *flags, "--top", "1", "-n", "4",
               "--draft-depth", "1"]
        if args.tier == "fresh":
            cmd.extend(["--tokens", f"@{target}"])
        else:
            memory = 0 if args.tier == "disk" else (3072 if args.length >= 65536 else 1024)
            cmd.extend(["--prefix-memory", str(memory), "--checkpoint-at", str(args.length),
                        "--check-draft-kv", "--tokens", f"@{first}"])
            if args.tier == "disk":
                cmd.extend(["--prefix-disk", str(work / "disk"), "--prefix-disk-gib", "4"])
            if args.tier != "arena":
                cmd.extend(["--then", f"@{side}"])
            cmd.extend(["--then", f"@{target}"])
        rd, wr = os.pipe()
        tail = bytearray()
        log = args.results / f"{stem}-{args.tier}.log"
        with log.open("wb") as output, os.fdopen(rd, "rb") as pipe:
            try:
                child = subprocess.Popen(cmd + ["--logits", f"/proc/self/fd/{wr}"], pass_fds=(wr,),
                                         stdout=output, stderr=subprocess.STDOUT)
            finally:
                os.close(wr)
            with child:
                while part := pipe.read(1024 * 1024):
                    tail.extend(part)
                    if len(tail) > keep:
                        del tail[:-keep]
                if child.wait():
                    raise SystemExit(f"{log}\n{log.read_text()[-1800:]}")
        lines = log.read_text()
        if len(tail) != keep:
            raise SystemExit(f"Missing logits in {log}")
        digest = hashlib.sha256(tail).hexdigest()
        chosen = [int(t) for t in re.findall(r"^pos\s+\d+\s+in\s+(\d+)\s+->", lines, re.M)[-4:]]
        result = {"length": args.length, "mode": mode, "tier": args.tier, "sha256": digest,
                  "ids": chosen, "logit_bytes": len(tail)}
        if args.tier != "fresh":
            want = json.loads(reference.read_text())
            hits = re.findall(r"prefix checkpoint: cached (\d+), tier (\d+), restore ([.\d]+)", lines)
            rows = re.findall(r"rows (\d+)-(\d+) equal, bit for bit, a fresh KV-only pass", lines)
            expected = (str(args.length), str(("fresh", "arena", "memory", "disk").index(args.tier)))
            if not hits or hits[-1][:2] != expected or not rows or (digest, chosen) != (want["sha256"], want["ids"]):
                raise SystemExit(f"Prefix mismatch in {log}: {hits[-1:]}, {rows}, {digest}, {chosen}; expected {want}")
            result.update(cached=int(hits[-1][0]), restore_seconds=float(hits[-1][2]), draft_rows=rows[-1])
        path = args.results / f"{stem}-{args.tier}.json"
        path.write_text(json.dumps(result) + "\n")
        print(path, digest, chosen, flush=True)


if __name__ == "__main__":
    main()
