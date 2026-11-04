#!/usr/bin/env python3
"""Model parity: Torch's next-token log-probabilities against a second implementation of the model on the same file.

  parity.py compare --torch BUILD/torch --model SHARD1 --ref tests/model/reference.json [--resident K] [--out FILE]
      Runs `torch run` on each reference prompt followed by the reference's own greedy continuation (teacher
      forcing, so one near-tie cannot derail the rest of the comparison). The gate:
        - top-1: at every generated position, Torch's most likely token is the reference's greedy token, or the
          reference itself ranks Torch's choice within --tie of its own (a near-tie);
        - KL: the mean over all positions of KL(reference || Torch), on the reference's top-5 tokens plus the rest of
          the probability mass, is at most --max-mean-kl.
      It also reports, without gating, the largest |log p_torch - log p_ref| on tokens both top-5 lists share: that
      number is dominated by improbable tokens and varies as much between two runs of the reference itself.

  parity.py spread --a REF_A.json --b REF_B.json
      The same numbers between two reference runs made with different numerics (for example with part of the model
      computed on the CPU): the reference's own noise, from which the thresholds are set.

A reference file is a JSON list of {"prompt", "tokens", "steps": [{"id", "top": [[token, logprob], ...]}], "text"},
one entry per prompt. Standard library only.
"""
import argparse
import json
import math
import subprocess
import sys




def kl(ref_top, other_top):
    """KL(ref || other) over the reference's top-5 tokens plus the remaining mass; a token missing from the other
    top-5 gets the other's fifth log-probability (an upper bound on its probability)."""
    floor = min(lp for _, lp in other_top)
    other = dict(other_top)
    p = [math.exp(lp) for _, lp in ref_top]
    q = [math.exp(other.get(t, floor)) for t, _ in ref_top]
    p_rest, q_rest = max(1 - sum(p), 1e-12), max(1 - sum(q), 1e-12)
    return sum(a * math.log(a / b) for a, b in zip(p, q) if a > 0) + p_rest * math.log(p_rest / q_rest)


def check(ref_steps, ours, tie):
    """Per position: (agree, near_tie, gap). ours[j] is {token: logprob} predicting reference step j."""
    rows = []
    for step, mine in zip(ref_steps, ours):
        ref_lp = {t: lp for t, lp in step["top"]}
        our_top = max(mine, key=lambda t: (mine[t], -t))
        agree = our_top == step["id"]
        near_tie = not agree and our_top in ref_lp and ref_lp[step["id"]] - ref_lp[our_top] <= tie
        gaps = [abs(mine[t] - lp) for t, lp in step["top"] if t in mine]
        rows.append((agree, near_tie, max(gaps) if gaps else float("inf"), kl(step["top"], list(mine.items()))))
    return rows


def summarize(name, rows):
    agree = sum(r[0] for r in rows)
    ties = sum(r[1] for r in rows)
    gap = max(r[2] for r in rows)
    mean_kl = sum(r[3] for r in rows) / len(rows)
    ok = agree + ties == len(rows)
    print(f"{'ok  ' if ok else 'FAIL'} {agree}/{len(rows)} top-1 agree, {ties} near-ties, mean KL {mean_kl:.5f}, "
          f"max top-5 logprob gap {gap:.4f}  {name!r}")
    return ok, agree, ties, gap, [r[3] for r in rows]


def compare(args):
    with open(args.ref) as f:
        ref = json.load(f)
    results, failures, all_kl = [], 0, []
    for i, case in enumerate(ref):
        forced = case["tokens"] + [s["id"] for s in case["steps"][:-1]]
        jsonl = f"{args.work}/parity-{i}.jsonl"
        cmd = [args.torch, "run", args.model, "--tokens", ",".join(map(str, forced)), "-n", "0", "--top", "5",
               "--json", jsonl]
        if args.resident is not None:
            cmd += ["--resident", str(args.resident)]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            sys.exit(f"torch run failed ({run.returncode}):\n{run.stdout[-3000:]}{run.stderr[-3000:]}")
        if i == 0:
            print(run.stdout.split("\npos")[0])  # the budget table
        speed = [l for l in run.stdout.splitlines() if l.startswith("experts:")]
        with open(jsonl) as f:
            lines = [json.loads(l) for l in f]
        first = len(case["tokens"]) - 1  # the position that predicts the first generated token
        ours = [{t: lp for t, lp in line["top"]} for line in lines[first:]]
        ok, agree, ties, gap, kls = summarize(case["prompt"][:40], check(case["steps"], ours, args.tie))
        failures += not ok
        all_kl += kls
        results.append({"prompt": case["prompt"], "positions": len(case["steps"]), "agree": agree,
                        "near_ties": ties, "max_gap": gap, "mean_kl": sum(kls) / len(kls), "ok": ok,
                        "experts": speed[0] if speed else ""})
    mean_kl = sum(all_kl) / len(all_kl)
    kl_ok = mean_kl <= args.max_mean_kl
    print(f"mean KL over all {len(all_kl)} positions {mean_kl:.5f} (gate {args.max_mean_kl}) {'ok' if kl_ok else 'FAIL'}; "
          f"max KL {max(all_kl):.5f}; max top-5 logprob gap {max(r['max_gap'] for r in results):.4f} (not gated)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"tie": args.tie, "max_mean_kl": args.max_mean_kl, "mean_kl": mean_kl, "results": results}, f,
                      indent=1)
    sys.exit(1 if failures or not kl_ok else 0)


def spread(args):
    with open(args.a) as f:
        a = json.load(f)
    with open(args.b) as f:
        b = json.load(f)
    worst, all_kl = 0.0, []
    for ca, cb in zip(a, b):
        if ca["tokens"] != cb["tokens"]:
            sys.exit("the two references tokenized a prompt differently")
        # where the two greedy runs diverge, teacher forcing is impossible: compare up to the first difference
        n = next((j for j, (x, y) in enumerate(zip(ca["steps"], cb["steps"])) if x["id"] != y["id"]), len(ca["steps"]))
        ours = [{t: lp for t, lp in s["top"]} for s in cb["steps"][:n + 1]]
        rows = check(ca["steps"][:n + 1], ours, float("inf"))
        _, _, _, gap, kls = summarize(ca["prompt"][:40], rows)
        print(f"     greedy runs agree on {n}/{len(ca['steps'])} tokens")
        worst = max(worst, gap)
        all_kl += kls
    print(f"between the references: mean KL {sum(all_kl) / len(all_kl):.5f} over {len(all_kl)} positions, "
          f"max KL {max(all_kl):.5f}, max top-5 logprob gap {worst:.4f}")


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = p.add_subparsers(dest="cmd", required=True)
c = sub.add_parser("compare")
c.add_argument("--torch", required=True)
c.add_argument("--model", required=True)
c.add_argument("--ref", required=True)
c.add_argument("--resident", type=int)
c.add_argument("--work", default=".")
c.add_argument("--out")
c.add_argument("--tie", type=float, default=0.38)
c.add_argument("--max-mean-kl", type=float, default=0.0041)
s = sub.add_parser("spread")
s.add_argument("--a", required=True)
s.add_argument("--b", required=True)
args = p.parse_args()
{"compare": compare, "spread": spread}[args.cmd](args)
