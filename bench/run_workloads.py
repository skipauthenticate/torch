#!/usr/bin/env python3
"""Runs the benchmark workloads (bench/workloads/, docs/benchmarking.md) with torch run and writes a result file.

  run_workloads.py --torch BUILD/torch --model SHARD1 --out RESULT.json [--workload review] [--placement COUNTS]
      For each benchmark prompt of the workload (or all three): feeds the prompt and generates 256 tokens greedily
      with `torch run --quiet --profile`, and records decode tok/s (generated tokens over their decode time; the
      prompt is excluded), ms per token, expert hits, misses and hedged reads, and the sha256 of the generated ids.
      An existing RESULT.json is updated, so the workloads can run in separate sessions.

  run_workloads.py --torch BUILD/torch --model SHARD1 --calibrate COUNTS [--workload review]
      Decodes the held-out calibration prompts (seeds 101-103) and adds their route counts to COUNTS, the input of
      `--placement` (core/placement.h). The benchmark prompts are never used for calibration.

Standard library only.
"""
import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

WORKLOADS = Path(__file__).resolve().parent / "workloads"
NAMES = ["review", "prose", "repetition"]


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"{' '.join(map(str, cmd[:3]))} ... failed ({proc.returncode}):\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}")
    return proc.stdout


def number(pattern, text, kind=float):
    m = re.search(pattern, text)
    return kind(m.group(1)) if m else None


def board():
    def read(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, shell=True).stdout.strip()
        except OSError:
            return ""
    return {"host": platform.node(), "jetpack": read("head -1 /etc/nv_tegra_release"),
            "power_mode": read("nvpmodel -q 2>/dev/null | head -1"),
            "gpu_max_mhz": read("cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/max_freq 2>/dev/null")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--torch", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--workload", choices=NAMES + ["all"], default="all")
    p.add_argument("--placement")
    p.add_argument("--calibrate")
    p.add_argument("--out")
    p.add_argument("--commit", default="", help="the engine commit the binary was built from")
    p.add_argument("-n", type=int, default=256)
    args = p.parse_args()
    names = NAMES if args.workload == "all" else [args.workload]

    if args.calibrate:
        for name in names:
            for seed in (101, 102, 103):
                out = run([args.torch, "run", args.model, "--tokens", f"@{WORKLOADS / f'{name}-calib-{seed}.tokens'}",
                           "-n", "64", "--quiet", "--route-counts", args.calibrate])
                print(f"calibrated on {name}-calib-{seed}: {number(r'([0-9.]+) tok/s', out)} tok/s", flush=True)
        return

    if not args.out:
        sys.exit("--out is required unless --calibrate is given")
    result = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    binary = Path(args.torch).read_bytes()
    result.update({"engine_commit": args.commit, "torch_sha256": hashlib.sha256(binary).hexdigest(),
                   "model": Path(args.model).name, "model_bytes": Path(args.model).stat().st_size,
                   "placement": args.placement or "", "generated_per_request": args.n, "board": board()})
    if args.placement:
        result["placement_sha256"] = hashlib.sha256(Path(args.placement).read_bytes()).hexdigest()
    result.setdefault("workloads", {})
    for name in names:
        requests = []
        for seed in (1, 2, 3):
            cmd = [args.torch, "run", args.model, "--tokens", f"@{WORKLOADS / f'{name}-bench-{seed}.tokens'}",
                   "-n", str(args.n), "--quiet", "--profile"]
            if args.placement:
                cmd += ["--placement", args.placement]
            start = time.time()
            out = run(cmd)
            ids = re.search(r"generated tokens: ([0-9,]+)", out).group(1)
            requests.append({
                "seed": seed,
                "prompt_tokens": number(r"prompt (\d+) tokens", out, int),
                "decode_tok_s": number(r"([0-9.]+) tok/s", out),
                "ms_per_token": number(r"\(([0-9.]+) ms per token\)", out),
                "resident": number(r"experts: (\d+) of \d+ resident", out, int),
                "hits": number(r"; (\d+) routed hits", out, int),
                "misses": number(r", (\d+) misses", out, int),
                "hedged": number(r"(\d+) hedged reads", out, int),
                "mem_available_end_gib": number(r"MemAvailable at the end ([0-9.]+) GiB", out),
                "wall_s": round(time.time() - start, 1),
                "output_sha256": hashlib.sha256(ids.encode()).hexdigest(),
                "profile": re.search(r"decode time per token.*", out).group(0),
            })
            r = requests[-1]
            hits = r["hits"] / max(r["hits"] + r["misses"], 1)
            print(f"{name}-{seed}: {r['decode_tok_s']} tok/s, {100 * hits:.2f}% hits, {r['hedged']} hedged", flush=True)
        result["workloads"][name] = {
            "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in requests),
            "requests": requests,
        }
        print(f"{name}: median {result['workloads'][name]['median_decode_tok_s']} tok/s", flush=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
