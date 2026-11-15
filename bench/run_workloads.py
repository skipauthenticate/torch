#!/usr/bin/env python3
"""Runs the benchmark workloads (bench/workloads/, docs/benchmarking.md) with torch run and writes a result file.

  run_workloads.py --torch BUILD/torch --model SHARD1 --out RESULT.json [--workload review] [--placement COUNTS]
                   [--spec-args "ARGS"]
      For each benchmark prompt of the workload (or all three): feeds the prompt and generates 256 tokens greedily
      with `torch run --quiet --profile`, and records decode tok/s (generated tokens over their decode time; the
      prompt is excluded), ms per token, the resident experts, expert hits, misses and hedged reads (all of the run
      and those of the generation), MemAvailable after loading, at its lowest while the run went (sampled every
      0.5 s) and at the end, and the sha256 of the generated ids. ARGS are passed to torch run as they are, for
      speculative decoding, for example:
        --draft DRAFT.gguf --draft-variants stream:40960 --draft-vocab VOCAB --spec 5 --spec-policy adaptive-stop
        --swap 32 --swap-prompt 0
      Then each request also records the speculative summary: tokens per window, drafts verified on average, ms per
      window and its split into verify and accept, drafting and host, and the swaps. An existing RESULT.json is
      updated, so the workloads can run in separate sessions.

  run_workloads.py --torch BUILD/torch --model SHARD1 --calibrate COUNTS [--workload review] [--placement START]
                   [--spec-args "ARGS"] [--calib-n 256]
      Decodes the held-out calibration prompts (seeds 101-103), generating --calib-n tokens each from placement START
      and with ARGS when given, and adds the route counts of the generated tokens (not the prompts) to COUNTS, the input
      of `--placement` (core/placement.h). The benchmark prompts are never used for calibration. The counts for
      speculative decoding come from the plain ones: --calibrate calib-spec.txt --placement calib.txt --spec-args "the
      draft file, variant, vocabulary and --spec 5 --spec-policy adaptive-stop".

Standard library only.
"""
import argparse
import hashlib
import json
import platform
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

WORKLOADS = Path(__file__).resolve().parent / "workloads"
NAMES = ["review", "prose", "repetition"]


def mem_available_gib():
    try:
        m = re.search(r"MemAvailable:\s+(\d+) kB", Path("/proc/meminfo").read_text())
    except OSError:
        return None
    return int(m.group(1)) / 1048576 if m else None


def run(cmd):
    """cmd's output, and the lowest MemAvailable in GiB while it ran (sampled every 0.5 s; None where there is no
    /proc/meminfo)."""
    low, done = [mem_available_gib()], threading.Event()

    def sample():
        while not done.wait(0.5):
            v = mem_available_gib()
            low[0] = v if low[0] is None or (v is not None and v < low[0]) else low[0]
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    done.set()
    sampler.join()
    if proc.returncode != 0:
        sys.exit(f"{' '.join(map(str, cmd[:3]))} ... failed ({proc.returncode}):\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}")
    return proc.stdout, None if low[0] is None else round(low[0], 2)


def number(pattern, text, kind=float):
    m = re.search(pattern, text)
    return kind(m.group(1)) if m else None


SPECULATIVE = re.compile(
    r"speculative \(([^)]*)\): \d+ tokens in (\d+) windows of the last token and up to (\d+) drafts, ([0-9.]+) "
    r"verified on average: ([0-9.]+) tokens per window, ([0-9.]+) tok/s \(([0-9.]+) ms per window: verify and "
    r"accept ([0-9.]+), draft ([0-9.]+), host ([0-9.]+)\)")
GENERATION = re.compile(r"generation: (\d+) routed hits, (\d+) misses(?:, read in ([0-9.]+) ms)?; (\d+) swaps"
                        r"(?: \((\d+) when the staging filled\))?, ([0-9.]+) ms \(([0-9.]+) ms per window\); the "
                        r"prompt's: (\d+) swaps, ([0-9.]+) ms")  # read time and swaps when full: torch run since M8


def speculative(out):
    """The speculative summary torch run prints with --spec, or None."""
    m = SPECULATIVE.search(out)
    if not m:
        return None
    keys = ["policy", "windows", "max_drafts", "drafts_verified", "tokens_per_window", "tok_s", "ms_per_window",
            "verify_accept_ms", "draft_ms", "host_ms"]
    kinds = [str, int, int, float, float, float, float, float, float, float]
    return {k: kind(v) for k, kind, v in zip(keys, kinds, m.groups())}


def generation(out):
    """The generation's own expert hits and misses and the swaps, or None."""
    m = GENERATION.search(out)
    if not m:
        return None
    g = m.groups()
    return {"hits": int(g[0]), "misses": int(g[1]), "read_ms": float(g[2]) if g[2] else None, "swaps": int(g[3]),
            "swaps_when_full": int(g[4]) if g[4] else None, "swap_ms": float(g[5]), "swap_ms_per_window": float(g[6]),
            "prompt_swaps": int(g[7]), "prompt_swap_ms": float(g[8])}


def decode_tok_s(out):
    s = speculative(out)
    return s["tok_s"] if s else number(r"generated \d+: ([0-9.]+) tok/s", out)


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
    p.add_argument("--calib-n", type=int, default=256, help="tokens generated per calibration prompt")
    p.add_argument("--spec-args", default="", help="more torch run arguments, for speculative decoding")
    args = p.parse_args()
    names = NAMES if args.workload == "all" else [args.workload]
    extra = (["--placement", args.placement] if args.placement else []) + shlex.split(args.spec_args)

    if args.calibrate:
        for name in names:
            for seed in (101, 102, 103):
                out, _ = run([args.torch, "run", args.model, "--tokens",
                              f"@{WORKLOADS / f'{name}-calib-{seed}.tokens'}", "-n", str(args.calib_n), "--quiet",
                              "--route-counts", args.calibrate, *extra])
                print(f"calibrated on {name}-calib-{seed}: {decode_tok_s(out)} tok/s", flush=True)
        return

    if not args.out:
        sys.exit("--out is required unless --calibrate is given")
    result = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    binary = Path(args.torch).read_bytes()
    result.update({"engine_commit": args.commit, "torch_sha256": hashlib.sha256(binary).hexdigest(),
                   "model": Path(args.model).name, "model_bytes": Path(args.model).stat().st_size,
                   "placement": Path(args.placement).name if args.placement else "",
                   "spec_args": " ".join(a if "/" not in a else Path(a).name for a in args.spec_args.split()), "generated_per_request": args.n,
                   "board": board()})
    if args.placement:
        result["placement_sha256"] = hashlib.sha256(Path(args.placement).read_bytes()).hexdigest()
    result.setdefault("workloads", {})
    for name in names:
        requests = []
        for seed in (1, 2, 3):
            cmd = [args.torch, "run", args.model, "--tokens", f"@{WORKLOADS / f'{name}-bench-{seed}.tokens'}",
                   "-n", str(args.n), "--quiet", "--profile", *extra]
            start = time.time()
            out, lowest = run(cmd)
            ids = re.search(r"generated tokens: ([0-9,]+)", out).group(1)
            tok_s, spec = decode_tok_s(out), speculative(out)
            profile = re.search(r"decode time per token.*", out)
            requests.append({
                "seed": seed,
                "prompt_tokens": number(r"prompt (\d+) tokens", out, int),
                "decode_tok_s": tok_s,
                "ms_per_token": round(1000 / tok_s, 1) if spec and tok_s
                else number(r"\(([0-9.]+) ms per token\)", out),
                "resident": number(r"experts: (\d+) of \d+ resident", out, int),
                "hits": number(r"experts: \d+ of \d+ resident; (\d+) routed hits", out, int),
                "misses": number(r"experts: \d+ of \d+ resident; \d+ routed hits, (\d+) misses", out, int),
                "hedged": number(r"(\d+) hedged reads", out, int),
                "mem_available_after_load_gib": number(r"loaded in [0-9.]+ s; MemAvailable now ([0-9.]+) GiB", out),
                "mem_available_lowest_gib": lowest,
                "mem_available_end_gib": number(r"MemAvailable at the end ([0-9.]+) GiB", out),
                "generation": generation(out),
                "speculative": spec,
                "wall_s": round(time.time() - start, 1),
                "output_sha256": hashlib.sha256(ids.encode()).hexdigest(),
                "profile": profile.group(0) if profile else None,
            })
            r = requests[-1]
            hits = r["hits"] / max(r["hits"] + r["misses"], 1)
            window = f", {spec['tokens_per_window']} tokens per window" if spec else ""
            print(f"{name}-{seed}: {r['decode_tok_s']} tok/s{window}, {100 * hits:.2f}% hits, {r['hedged']} hedged, "
                  f"lowest MemAvailable {lowest} GiB", flush=True)
        result["workloads"][name] = {
            "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in requests),
            "requests": requests,
        }
        print(f"{name}: median {result['workloads'][name]['median_decode_tok_s']} tok/s", flush=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
