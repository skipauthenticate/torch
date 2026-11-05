#!/usr/bin/env python3
"""Checks a kernel harness result (torch_bench_kernels --json) against the decode gates: every launch that moves 1 MiB
or more runs at >= 85% of the streaming ceiling, and the launches under 1 MiB take <= 2 ms and number <= 600 per token.
Prints each failure and the totals; exits 1 if a gate fails.

usage: kernel_gates.py RESULT.json
"""
import json
import sys


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    d = json.load(open(sys.argv[1]))
    t = d["totals"]
    slow = [c for c in d["cases"] if c["bytes"] >= 1 << 20 and c["percent_of_ceiling"] < 85.0]
    for c in slow:
        print(f"FAIL {c['kernel']} {c['case']} {c['format']} {c['shape']}: {c['gbs']:.1f} GB/s, "
              f"{c['percent_of_ceiling']:.0f}% of {d['ceiling_gbs']:.0f}")
    small_ok = t["small_launch_ms_per_token"] <= 2.0 and t["small_launches_per_token"] <= 600
    print(f"{'ok  ' if small_ok else 'FAIL'} launches under 1 MiB: {t['small_launches_per_token']} per token taking "
          f"{t['small_launch_ms_per_token']:.2f} ms (gate: <= 600 and <= 2 ms)")
    print(f"{'ok  ' if not slow else 'FAIL'} launches of 1 MiB or more: {len(slow)} below 85% of the ceiling")
    print(f"kernels {t['kernel_ms_per_token']:.1f} ms per token, {t['launches_per_token']} launches")
    sys.exit(1 if slow or not small_ok else 0)


if __name__ == "__main__":
    main()
