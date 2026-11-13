#!/usr/bin/env python3
"""Needle in a haystack: can the model find one sentence in a long prompt of synthetic text?

The haystack is the prose generator of bench/workloads/make_workloads.py (our own synthetic sentences, no copied text).
One needle sentence carrying a random 7-digit number is inserted between paragraphs at a given depth (0 = start,
1 = end), and the prompt ends with a question about it, in the model's chat format with thinking off. The answer is
checked on the generated token ids: the tokenizer splits numbers into single digits, so the number's digit ids appear
in the greedy answer exactly when the number does.

  needle.py make --context N --depth D [--seed S] [--tokenizer tokenizer.json] [--out DIR]
      writes DIR/needle-<N>-<D>.tokens (comma-separated ids, at most N - 64 tokens) and needle-<N>-<D>.json (the
      needle, the question, the answer and its digit ids). Needs the `tokenizers` package and the pinned tokenizer.
  needle.py check --meta needle-<N>-<D>.json --log RUN.log
      reads the output of `torch run --tokens @needle-<N>-<D>.tokens -n 32 --context N --window 8 --quiet` and prints
      found or not found, the prompt time and the generated ids. Standard library only (runs on the board).
"""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ANSWER_TOKENS = 64  # room left in the context for the answer
NEEDLE = "The vault code written inside the blue ledger is {code}."
QUESTION = ("What is the vault code written inside the blue ledger, according to the text above? Answer with the "
            "code only.")


def make(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "workloads"))
    from tokenizers import Tokenizer
    from make_workloads import TOKENIZER_SHA256, chat, prose_text

    if hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        sys.exit(f"{args.tokenizer} is not the pinned tokenizer (sha256 differs)")
    if not 0.0 <= args.depth <= 1.0:
        sys.exit("--depth is a fraction from 0 to 1.")
    tok = Tokenizer.from_file(str(args.tokenizer))
    rng = random.Random(args.seed)
    code = str(rng.randint(1_000_000, 9_999_999))
    needle = NEEDLE.format(code=code)

    def prompt(paragraphs):
        text = prose_text(random.Random(args.seed * 7919), paragraphs).split("\n\n")
        text.insert(round(args.depth * len(text)), needle)
        return chat("\n\n".join(text) + "\n\n" + QUESTION)

    limit = args.context - ANSWER_TOKENS
    lo, hi, best = 1, max(2, args.context // 10), None  # paragraphs are ~100 tokens
    while lo <= hi:
        mid = (lo + hi) // 2
        ids = tok.encode(prompt(mid), add_special_tokens=False).ids
        if len(ids) <= limit:
            best, lo = (mid, ids), mid + 1
        else:
            hi = mid - 1
    if best is None:
        sys.exit("The context is too short for a haystack.")
    paragraphs, ids = best
    name = f"needle-{args.context}-{args.depth:g}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.tokens").write_text(",".join(map(str, ids)) + "\n")
    meta = {"context": args.context, "depth": args.depth, "seed": args.seed, "prompt_tokens": len(ids),
            "paragraphs": paragraphs, "needle": needle, "question": QUESTION, "answer": code,
            "answer_ids": tok.encode(code, add_special_tokens=False).ids}
    (args.out / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{name}: {len(ids)} tokens, {paragraphs} paragraphs, answer {code}")


def contains(seq, sub):
    return any(seq[i:i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


def check(args):
    meta = json.loads(args.meta.read_text())
    out = args.log.read_text()
    m = re.search(r"^generated tokens: ([0-9,]*)$", out, re.M)
    if not m:
        sys.exit(f"{args.log} has no generated tokens; run torch with --quiet.")
    ids = [int(x) for x in m.group(1).split(",") if x]
    t = re.search(r"^prompt in ([0-9.]+) s", out, re.M)
    found = contains(ids, meta["answer_ids"])
    print(f"{'found' if found else 'NOT FOUND'}: context {meta['context']}, depth {meta['depth']:g}, "
          f"{meta['prompt_tokens']} prompt tokens, answer {meta['answer']}, prompt in "
          f"{t.group(1) if t else '?'} s; generated {ids}")
    sys.exit(0 if found else 1)


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = p.add_subparsers(dest="cmd", required=True)
mk = sub.add_parser("make")
mk.add_argument("--context", type=int, required=True)
mk.add_argument("--depth", type=float, required=True)
mk.add_argument("--seed", type=int, default=1)
mk.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
mk.add_argument("--out", type=Path, default=Path("."))
ck = sub.add_parser("check")
ck.add_argument("--meta", type=Path, required=True)
ck.add_argument("--log", type=Path, required=True)
args = p.parse_args()
{"make": make, "check": check}[args.cmd](args)
