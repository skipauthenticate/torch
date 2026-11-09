#!/usr/bin/env python3
"""The draft layer's acceptance per depth on the benchmark workloads (M4 R3), and the draft head's vocabulary.

  draft_acceptance.py vocab --torch BUILD/torch --model SHARD1 --ids DIR --out VOCAB [--workload review] [-n 256]
      Decodes the held-out calibration prompts (seeds 101-103) greedily and keeps each one's generated ids in DIR (a
      prompt already there is skipped, so the workloads can run in separate sessions). Once all nine are there, writes
      VOCAB: the generated ids by frequency (ties to the lower id), then every other id in ascending order, one per
      line. The draft head's N-row subsets are its first N lines. The benchmark prompts are never used.

  draft_acceptance.py run --torch BUILD/torch --model SHARD1 --draft DRAFT --out RESULT.jsonl [--workload review]
                          [-n 128] [--depth 7] [--variants LIST] [--vocab VOCAB]
      For each benchmark prompt (seeds 1-3): `torch run --draft ... --acceptance-report`, appending the report's lines,
      tagged with the workload and seed, to RESULT.jsonl.

  draft_acceptance.py table RESULT.jsonl
      Sums the counts per workload and variant and prints, as Markdown, the acceptance at each depth given the earlier
      ones, the mean accepted drafts with 3 drafts and with the run's depth, and the eager drafter's times; then the
      acceptance by draft probability over all workloads.

`--placement COUNTS` is passed to torch run in both modes (it changes speed only). Standard library only.
"""
import argparse
import collections
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

WORKLOADS = Path(__file__).resolve().parent / "workloads"
NAMES = ["review", "prose", "repetition"]
N_VOCAB = 248320
PROB_BOUNDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]  # tools/run.cpp's buckets


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"{' '.join(map(str, cmd[:3]))} ... failed ({proc.returncode}):\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}")
    return proc.stdout


def vocab(args, names, extra):
    ids = Path(args.ids)
    ids.mkdir(parents=True, exist_ok=True)
    for name in names:
        for seed in (101, 102, 103):
            out = ids / f"{name}-calib-{seed}.generated"
            if out.exists():
                continue
            text = run([args.torch, "run", args.model, "--tokens", f"@{WORKLOADS / f'{name}-calib-{seed}.tokens'}",
                        "-n", str(args.n), "--quiet", *extra])
            out.write_text(re.search(r"generated tokens: ([0-9,]+)", text).group(1) + "\n")
            print(f"generated {args.n} tokens from {name}-calib-{seed}", flush=True)
    files = sorted(ids.glob("*-calib-*.generated"))
    if len(files) < 3 * len(NAMES):
        print(f"{len(files)} of {3 * len(NAMES)} calibration outputs so far; run the other workloads to write the vocabulary")
        return
    counts = collections.Counter(int(t) for f in files for t in f.read_text().strip().split(","))
    ranked = sorted(counts, key=lambda t: (-counts[t], t))
    ranked += [t for t in range(N_VOCAB) if t not in counts]
    Path(args.out).write_text("\n".join(map(str, ranked)) + "\n")
    print(f"wrote {args.out}: {len(counts)} distinct ids in {sum(counts.values())} generated tokens, then the rest by id")


def measure(args, names, extra):
    for name in names:
        for seed in (1, 2, 3):
            with tempfile.NamedTemporaryFile("r", suffix=".jsonl") as report:
                cmd = [args.torch, "run", args.model, "--tokens", f"@{WORKLOADS / f'{name}-bench-{seed}.tokens'}",
                       "-n", str(args.n), "--quiet", "--draft", args.draft, "--draft-depth", str(args.depth),
                       "--draft-variants", args.variants, "--acceptance-report", report.name, *extra]
                if args.vocab:
                    cmd += ["--draft-vocab", args.vocab]
                text = run(cmd)
                lines = [json.loads(line) for line in report.read().splitlines()]
            tok_s = re.search(r"([0-9.]+) tok/s", text)
            with open(args.out, "a") as out:
                for d in lines:
                    d.update(workload=name, seed=seed, trunk_tok_s=float(tok_s.group(1)) if tok_s else None)
                    out.write(json.dumps(d) + "\n")
            print(f"{name}-bench-{seed}:", *(line for line in text.splitlines() if line.startswith(("draft", "  mean"))),
                  sep="\n", flush=True)


def mean_accepted(lengths, cap):
    return sum(min(n, cap) * c for n, c in enumerate(lengths)) / max(sum(lengths), 1)


def table(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    keys = ["steps", "tried", "accepted", "lengths", "prob_tried", "prob_accepted", "kv_ms", "kv_positions",
            "draft_ms", "draft_steps"]
    sums = {}
    for r in rows:
        for group in (r["workload"], "all"):
            s = sums.setdefault((group, r["variant"]), {"depth": r["depth"], "prompts": 0})
            s["prompts"] += 1
            for k in keys:
                if isinstance(r[k], list):
                    s[k] = [a + b for a, b in zip(s[k], r[k])] if k in s else list(r[k])
                else:
                    s[k] = s.get(k, 0) + r[k]
    depth = max(s["depth"] for s in sums.values())
    print("| Workload | Variant | Prompts | Steps | " + " | ".join(f"d{j + 1}" for j in range(depth)) +
          " | Mean accepted (3) | Mean accepted (" + str(depth) + ") | KV-only ms/position | Draft ms/step (eager) |")
    print("|" + " --- |" * 4 + " ---: |" * (depth + 4))
    order = {name: i for i, name in enumerate(NAMES + ["all"])}
    for (group, variant), s in sorted(sums.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])):
        acc = [f"{100 * a / t:.1f}" if t else "-" for a, t in zip(s["accepted"], s["tried"])]
        print(f"| {group} | {variant} | {s['prompts']} | {s['steps']} | " + " | ".join(acc) +
              f" | {mean_accepted(s['lengths'], 3):.2f} | {mean_accepted(s['lengths'], s['depth']):.2f} | "
              f"{s['kv_ms'] / max(s['kv_positions'], 1):.2f} | {s['draft_ms'] / max(s['draft_steps'], 1):.1f} |")
    bounds = [0.0, *PROB_BOUNDS, 1.0]
    print("\nAcceptance by the draft's probability, all workloads and depths (drafts made with every earlier one right):\n")
    print("| Variant | " + " | ".join(f"{a:g}-{b:g}" for a, b in zip(bounds, bounds[1:])) + " |")
    print("| --- |" + " ---: |" * (len(bounds) - 1))
    for (group, variant), s in sorted(sums.items()):
        if group == "all":
            cells = [f"{100 * a / t:.0f}% of {t}" if t else "-" for a, t in zip(s["prob_accepted"], s["prob_tried"])]
            print(f"| {variant} | " + " | ".join(cells) + " |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["vocab", "run", "table"])
    p.add_argument("result", nargs="?", help="table: the RESULT.jsonl to sum")
    p.add_argument("--torch")
    p.add_argument("--model")
    p.add_argument("--workload", choices=NAMES + ["all"], default="all")
    p.add_argument("--placement")
    p.add_argument("--ids")
    p.add_argument("--out")
    p.add_argument("--draft")
    p.add_argument("--vocab")
    p.add_argument("--variants", default="stream:full")
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("-n", type=int)
    args = p.parse_args()
    if args.mode == "table":
        return table(args.result)
    names = NAMES if args.workload == "all" else [args.workload]
    extra = ["--placement", args.placement] if args.placement else []
    if args.mode == "vocab":
        args.n = args.n or 256
        return vocab(args, names, extra)
    args.n = args.n or 128
    measure(args, names, extra)


if __name__ == "__main__":
    main()
