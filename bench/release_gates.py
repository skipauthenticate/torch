#!/usr/bin/env python3
"""Check the paired Orin release report. A loss or an incomplete report gives a nonzero exit code."""
import argparse
import json
import math
import re
from pathlib import Path
from statistics import median


HIGHER = {
    "input_tokens_s", "output_tokens_s", "resident_experts", "expert_hit_percent",
    "decode_expert_hit_percent", "tokens_per_window", "available_gib", "cached_tokens",
}
LOWER = {
    "wall_s", "prompt_s", "visible_first_output_s", "first_callback_s", "max_callback_ms",
    "intervals_over_200ms", "max_visible_gap_ms", "visible_gaps_over_200ms", "max_verify_ms",
    "checks_over_200ms", "hedged_reads", "grouped_reads", "verify_ms_per_output", "draft_ms_per_output",
}


STAGES = {"forward", "reverse", "stream", "timed_reverse", "fixed_residency", "fresh", "multi_turn", "fast"}
STANDARD = {f"{group}-{n}" for group in ("review", "prose", "repetition") for n in (1, 2, 3)}
WINDOW = {"max_verify_ms", "checks_over_200ms", "hedged_reads", "grouped_reads",
          "verify_ms_per_output", "draft_ms_per_output", "decode_expert_hit_percent"}
STREAM = {"max_visible_gap_ms", "visible_gaps_over_200ms", "visible_first_output_s"}


def expected_requests(stage):
    if stage == "multi_turn":
        return {f"{length}-{turn}" for length in (2073, 10000) for turn in ("A1", "B1", "S1", "A2", "B2", "S2", "A3")}
    if stage == "fresh":
        return {"review-4906", "prose-4906", "repetition-4906"}
    return STANDARD


def failures(report, speed_targets=False):
    if report["reference_code"] != "b51633482e78fca4014753903e26df51d31a4e5d":
        raise ValueError("The report needs the v0.4.0 reference.")
    if set(report["models"]) != {"IQ3_S", "UD-IQ3_XXS"}:
        raise ValueError("Both model files need results.")
    for model, result in report["models"].items():
        board = result["board_checks"]
        large = board["large_operations"]
        if len(large) != board["large_cases"] or not large or any(v["bytes"] < 2**20 for v in large):
            raise ValueError(f"{model} needs all large GPU operations.")
        small = board["small_operations"]
        if (board["exit_code"] or board["large_cases_below_153_gbs"] or any(not math.isfinite(v["gb_per_second"]) or v["gb_per_second"] < 153 for v in large)
                or not small["passed"] or not 0 <= small["count"] <= 600 or not 0 <= small["ms_per_token"] <= 2):
            yield f"{model} failed the board bandwidth or launch checks."
        limit = 0.00328 if model == "IQ3_S" else 0.00596
        if min(board["unit_tests_passed"], board["gpu_parity_tests_passed"]) < 13:
            raise ValueError(f"{model} needs the complete unit and GPU test suites.")
        if not (board["model_reference_positions"] >= 160 and 0 <= board["mean_kl"] <= limit):
            yield f"{model} failed the model reference check."
        if not result["all_standard_tokens_and_answers_match_v0.4.0_and_single_token_generation"]:
            yield f"{model} changed a benchmark output."
        reference = result["model_reference_comparison"]
        if not reference["all_five_top_ids_and_scores_match"]:
            yield f"{model} changed a reference token or its score."
        for metric in ("mean_kl", "maximum_kl", "maximum_top5_log_probability_gap"):
            before, after = (reference[arm][metric] for arm in ("v0.4.0", "candidate"))
            if not all(math.isfinite(v) and v >= 0 for v in (before, after)):
                raise ValueError(f"{model} has an invalid reference score for {metric}.")
            if after > before:
                yield f"{model} moved farther from the model reference on {metric}."
        vectors = result["last_four_full_logit_vectors"]["sha256_by_prompt_length_and_width"]
        widths = {"300": {"-1", "0", "37", "300"}, "2200": {"-1", "0", "333", "4096"}}
        if set(vectors) != set(widths):
            raise ValueError(f"{model} needs both prompt lengths for the full score check.")
        for length, required in widths.items():
            hashes = vectors[length]
            if set(hashes) != required or any(not re.fullmatch("[a-f0-9]{64}", v) for v in hashes.values()):
                raise ValueError(f"{model} needs all four full score hashes at length {length}.")
            if len(set(hashes.values())) != 1:
                yield f"{model} changed a full score vector at length {length}."
        sampled = result["fixed_seed_sampling"]
        hashes = sampled["output_ids_sha256_le32"]
        if set(hashes) != {"release", "candidate"} or any(not re.fullmatch("[a-f0-9]{64}", v) for v in hashes.values()):
            raise ValueError(f"{model} needs both fixed-seed output hashes.")
        if hashes["release"] != hashes["candidate"] or not sampled["draft_state_matches_fresh_pass"]:
            yield f"{model} changed fixed-seed output or draft state."
        pairs = result["pairs"]
        if set(pairs) != STAGES:
            raise ValueError(f"{model} needs all eight comparisons.")
        measured = set()
        for case, pair in pairs.items():
            order = pair["metric_order"]
            if not order or set(order) - HIGHER - LOWER or len(set(order)) != len(order):
                raise ValueError(f"{model} {case} has an unknown or duplicate metric.")
            measured.update(order)
            required = {"prompt_s", "cached_tokens", "resident_experts", "wall_s", "first_callback_s", "max_callback_ms",
                        "intervals_over_200ms", "available_gib"}
            if case != "multi_turn":
                required |= {"input_tokens_s", "output_tokens_s", "expert_hit_percent", "tokens_per_window"}
            if case in ("multi_turn", "stream"):
                required |= STREAM
            if case in ("timed_reverse", "fixed_residency"):
                required |= WINDOW
            if not required.issubset(order):
                raise ValueError(f"{model} {case} is missing {sorted(required - set(order))}.")
            expected = expected_requests(case)
            if set(pair["requests"]) != expected or pair["compared_requests"] != len(expected):
                raise ValueError(f"{model} {case} has incomplete requests.")
            if pair["matching_requests"] != len(expected) or pair["different_outputs"]:
                yield f"{model} {case} changed a benchmark answer or token."
            if case != "multi_turn" and pair["matching_output_ids"] != len(expected) * 256:
                yield f"{model} {case} has fewer matching output tokens."
            startup = pair["startup_s"]
            if not all(math.isfinite(startup[arm]) and startup[arm] >= 0 for arm in ("v0.4.0", "candidate")):
                raise ValueError(f"{model} {case} has an invalid startup time.")
            if startup["worse"] or startup["candidate"] > startup["v0.4.0"]:
                yield f"{model} {case} took longer to start."
            # The saved list includes losses smaller than the precision of the printed rates.
            slower = set(pair["requests_with_lower_results"])
            for request, values in pair["requests"].items():
                if case != "multi_turn":
                    if values["prompt_tokens"] != 4096 or values["output_tokens"] != 256:
                        raise ValueError(f"{model} {case} {request} has the wrong request length.")
                    index = order.index("cached_tokens")
                    if any(values[arm][index] != 0 for arm in ("v0.4.0", "candidate")):
                        yield f"{model} {case} {request} used cached input tokens."
                hashes = values["output_hashes"]
                if any(set(hashes[arm]) != {"ids", "answer"}
                       or any(not re.fullmatch("[a-f0-9]{64}", digest) for digest in hashes[arm].values())
                       for arm in ("v0.4.0", "candidate")):
                    raise ValueError(f"{model} {case} {request} needs both output hashes.")
                if hashes["candidate"] != hashes["v0.4.0"]:
                    yield f"{model} {case} {request} changed its output."
                for name, before, after in zip(order, values["v0.4.0"], values["candidate"], strict=True):
                    if not all(math.isfinite(v) and v >= 0 for v in (before, after)):
                        raise ValueError(f"{model} {case} {request} has an invalid value for {name}.")
                    if (after < before if name in HIGHER else after > before):
                        slower.add(request)
            if slower:
                yield f"{model} {case} has a lower result on {len(slower)} requests."
            if case in ("forward", "reverse", "fresh", "stream"):
                if set(pair["workload_medians"]) != {"review", "prose", "repetition"}:
                    raise ValueError(f"{model} {case} needs results for all three workloads.")
                for workload, values in pair["workload_medians"].items():
                    rows = [row for name, row in pair["requests"].items() if name.startswith(workload + "-")]
                    for arm in ("v0.4.0", "candidate"):
                        rates = dict(zip(order, values[arm], strict=True))
                        for metric in ("input_tokens_s", "output_tokens_s"):
                            i = order.index(metric)
                            if rates[metric] != median(row[arm][i] for row in rows):
                                raise ValueError(f"{model} {case} {workload} has an incorrect median for {metric}.")
                    rates = dict(zip(order, values["candidate"], strict=True))
                    if speed_targets and (rates["input_tokens_s"] < 2000 or rates["output_tokens_s"] < 200):
                        yield f"{model} {case} {workload} missed a speed target."
        if measured != HIGHER | LOWER:
            raise ValueError(f"{model} needs these metrics ({sorted((HIGHER | LOWER) - measured)}).")
        kernels = result["paired_kernels"]
        if kernels["matched_cases"] < 1:
            raise ValueError(f"{model} needs paired kernel results.")
        slower = matched = 0
        for kernel in kernels["cases"]:
            before, after, second_after, second_before = kernel["us"]
            if None not in kernel["us"]:
                matched += 1
                if not all(math.isfinite(v) and v > 0 for v in kernel["us"]):
                    raise ValueError(f"{model} has an invalid GPU time.")
                if after > before or second_after > second_before:
                    slower += 1
        if matched != kernels["matched_cases"]:
            raise ValueError(f"{model} has incomplete GPU pairs.")
        for name, (before, after, second_after, second_before) in kernels["totals"].items():
            if not all(math.isfinite(v) and v >= 0 for v in (before, after, second_after, second_before)):
                raise ValueError(f"{model} has an invalid GPU total for {name}.")
            if name == "tokens_per_second":
                worse = after < before or second_after < second_before
            else:
                worse = after > before or second_after > second_before
            if worse:
                yield f"{model} has a lower result for the GPU total {name}."
        if slower:
            yield f"{model} measured a longer time for {slower} GPU operations."
    checks = ["observed_outputs_match", "official_orin", "all_compared_metrics"]
    if speed_targets:
        checks.append("both_speed_targets_on_every_workload")
    for name in checks:
        if not report["release_checks"][name]:
            yield f"The comparison reports a failed check for {name}."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--speed-targets", action="store_true", help="also require 200 generated and 2,000 uncached prompt tokens per second")
    args = parser.parse_args()
    try:
        issues = list(failures(json.loads(args.report.read_text()), args.speed_targets))
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(2, f"Incomplete release report ({error}).\n")
    for issue in issues:
        print(issue)
    if issues:
        return 1
    print("All release checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
