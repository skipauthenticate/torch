#!/usr/bin/env python3
"""Checks bench/run_workloads.py's parsing and arguments with a fake torch binary that prints torch run's summary
lines (copied from runs on the Orin) and records how it was called, so no GPU is needed. Standard library only.

usage: python3 tests/tools/run_workloads_test.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "run_workloads.py"

FAKE = r'''import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps(args) + "\n")
if "--route-counts" in args:
    with open(args[args.index("--route-counts") + 1], "a") as counts:
        counts.write("counts\n")
spec = "--spec" in args
print("loaded in 15.5 s; MemAvailable now 7.48 GiB")
print("generated tokens: 1919,1970,41643")
if spec:
    print("generation: 145646 routed hits, 1234 misses; 980 swaps, 21.5 ms (0.27 ms per window); the prompt's: 0 "
          "swaps, 0.0 ms")
    print("prompt 1491 tokens, generated 256")
else:
    print("prompt 1491 tokens, generated 256: 22.65 tok/s (44.2 ms per token)")
print("experts: 20110 of 24576 resident; 857591 routed hits, 4969 misses (99.4% hits); 2 hedged reads")
print("MemAvailable at the end 9.10 GiB")
if spec:
    print("speculative (adaptive-stop): 256 tokens in 80 windows of the last token and up to 5 drafts, 2.83 verified "
          "on average: 3.21 tokens per window, 36.96 tok/s (86.4 ms per window: verify and accept 79.4, draft 7.1, "
          "host 0.0)")
if "--profile" in args:
    print("decode time per token over 1747 steps: 39.99 ms; PLE rows 0.33")
'''

SPEC_ARGS = ("--draft d.gguf --draft-variants stream:40960 --spec 5 --spec-policy adaptive-stop --swap 32 "
             "--swap-prompt 0")


def run(tmp: Path, *args: str) -> list:
    """Runs run_workloads.py with the fake binary; returns the argument lists the fake was called with."""
    log = tmp / "calls.jsonl"
    log.unlink(missing_ok=True)
    env = dict(os.environ, FAKE_LOG=str(log))
    proc = subprocess.run([sys.executable, str(SCRIPT), "--torch", str(tmp / "torch"), "--model", str(tmp / "model"),
                           *args], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return [json.loads(line) for line in log.read_text().splitlines()]


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        fake = tmp / "torch"
        fake.write_text(f"#!{sys.executable}\n" + FAKE)
        fake.chmod(0o755)
        (tmp / "model").write_bytes(b"gguf")
        placement = tmp / "p.txt"
        placement.write_text("torch route counts\n")

        # speculative: the summary, the generation's own counts and the memory, per request
        out = tmp / "spec.json"
        calls = run(tmp, "--out", str(out), "--workload", "prose", "--placement", str(placement), "--spec-args",
                    SPEC_ARGS)
        assert len(calls) == 3 and all(c[-14:] == ["--placement", str(placement), *SPEC_ARGS.split()] for c in calls)
        result = json.loads(out.read_text())
        assert result["spec_args"] == SPEC_ARGS and result["placement"] == placement.name  # file names only, no local paths
        requests = result["workloads"]["prose"]["requests"]
        assert [r["seed"] for r in requests] == [1, 2, 3]
        r = requests[0]
        assert r["decode_tok_s"] == 36.96 and r["ms_per_token"] == 27.1
        assert r["speculative"] == {"policy": "adaptive-stop", "windows": 80, "max_drafts": 5, "drafts_verified": 2.83,
                                    "tokens_per_window": 3.21, "tok_s": 36.96, "ms_per_window": 86.4,
                                    "verify_accept_ms": 79.4, "draft_ms": 7.1, "host_ms": 0.0}
        assert r["generation"] == {"hits": 145646, "misses": 1234, "swaps": 980, "swap_ms": 21.5,
                                   "swap_ms_per_window": 0.27, "prompt_swaps": 0, "prompt_swap_ms": 0.0}
        assert (r["resident"], r["hits"], r["misses"], r["hedged"]) == (20110, 857591, 4969, 2)  # the run's totals
        assert r["mem_available_after_load_gib"] == 7.48 and r["mem_available_end_gib"] == 9.10
        lowest = r["mem_available_lowest_gib"]
        assert lowest is None if not Path("/proc/meminfo").exists() else lowest > 0
        assert r["profile"].startswith("decode time per token") and r["prompt_tokens"] == 1491
        assert result["workloads"]["prose"]["median_decode_tok_s"] == 36.96

        # one token at a time: no speculative summary
        out = tmp / "plain.json"
        calls = run(tmp, "--out", str(out), "--workload", "review")
        assert all("--spec" not in c and "--placement" not in c for c in calls)
        r = json.loads(out.read_text())["workloads"]["review"]["requests"][0]
        assert r["decode_tok_s"] == 22.65 and r["ms_per_token"] == 44.2
        assert r["speculative"] is None and r["generation"] is None

        # calibration with speculation: placement, ARGS and --calib-n reach every calibration prompt
        counts = tmp / "calib-spec.txt"
        calls = run(tmp, "--calibrate", str(counts), "--workload", "prose", "--placement", "calib.txt",
                    "--calib-n", "128", "--spec-args", SPEC_ARGS)
        assert [Path(c[3][1:]).name for c in calls] == [f"prose-calib-{s}.tokens" for s in (101, 102, 103)]
        for c in calls:
            assert c[c.index("-n") + 1] == "128" and c[c.index("--route-counts") + 1] == str(counts)
            assert c[c.index("--placement") + 1] == "calib.txt" and c[-12:] == SPEC_ARGS.split()
        assert counts.read_text() == "counts\n" * 3
    print("ok")


if __name__ == "__main__":
    main()
