"""Check that a lower result, a changed answer, and a missing measurement block a release."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import release_gates as gates


def passing_report():
    order = sorted(gates.HIGHER | gates.LOWER)
    numbers = [2000 if metric == "input_tokens_s" else 200 if metric == "output_tokens_s" else 1 for metric in order]
    numbers[order.index("cached_tokens")] = 0
    hashes = {"ids": "a" * 64, "answer": "b" * 64}
    value = {"prompt_tokens": 4096, "output_tokens": 256, "v0.4.0": numbers[:], "candidate": numbers[:],
             "output_hashes": {"v0.4.0": hashes.copy(), "candidate": hashes.copy()}}
    pairs = {}
    for stage in gates.STAGES:
        requests = {key: copy.deepcopy(value) for key in gates.expected_requests(stage)}
        pairs[stage] = {
            "requests": requests, "compared_requests": len(requests), "matching_requests": len(requests),
            "matching_output_ids": len(requests) * 256, "different_outputs": {}, "requests_with_lower_results": {},
            "startup_s": {"v0.4.0": 1, "candidate": 1, "worse": False}, "metric_order": order,
            "workload_medians": {g: copy.deepcopy(value) for g in ("review", "prose", "repetition")},
        }
    model = {"pairs": pairs, "all_standard_tokens_and_answers_match_v0.4.0_and_single_token_generation": True,
             "board_checks": {"large_operations": [{"bytes": 2**20, "gb_per_second": 153}], "large_cases": 1,
                              "large_cases_below_153_gbs": 0, "small_operations": {"count": 600, "ms_per_token": 2, "passed": True},
                              "unit_tests_passed": 13, "gpu_parity_tests_passed": 13,
                              "exit_code": 0, "mean_kl": 0, "model_reference_positions": 160},
             "paired_kernels": {"matched_cases": 1, "cases": [{"us": [1, 1, 1, 1]}],
                                "totals": {"tokens_per_second": [100, 100, 100, 100]}}}
    reference = {"mean_kl": 0, "maximum_kl": 0, "maximum_top5_log_probability_gap": 0}
    model["model_reference_comparison"] = {"all_five_top_ids_and_scores_match": True,
                                           "v0.4.0": reference.copy(), "candidate": reference.copy()}
    model["last_four_full_logit_vectors"] = {"sha256_by_prompt_length_and_width":
        {length: dict.fromkeys(widths, "a" * 64) for length, widths in
         (("300", ("-1", "0", "37", "300")), ("2200", ("-1", "0", "333", "4096")))}}
    model["fixed_seed_sampling"] = {"output_ids_sha256_le32": dict.fromkeys(("release", "candidate"), "a" * 64),
                                    "draft_state_matches_fresh_pass": True}
    return {"models": {name: copy.deepcopy(model) for name in ("IQ3_S", "UD-IQ3_XXS")},
            "reference_code": "b51633482e78fca4014753903e26df51d31a4e5d",
            "release_checks": dict(observed_outputs_match=True, official_orin=True, all_compared_metrics=True,
                                   both_speed_targets_on_every_workload=True)}


class ReleaseGatesTest(unittest.TestCase):
    def test_each_direction(self):
        report = passing_report()
        self.assertEqual(list(gates.failures(report)), [])
        pair = report["models"]["IQ3_S"]["pairs"]["multi_turn"]
        row = pair["requests"]["2073-A1"]
        for arm in ("v0.4.0", "candidate"):
            row[arm][pair["metric_order"].index("cached_tokens")] = 1
        values = row["candidate"]
        for i, metric in enumerate(pair["metric_order"]):
            with self.subTest(metric=metric):
                original = values[i]
                values[i] += -1 if metric in gates.HIGHER else 1
                self.assertTrue(list(gates.failures(report)))
                values[i] = original

    def test_speed_targets(self):
        report = passing_report()
        pair = report["models"]["IQ3_S"]["pairs"]["forward"]
        index = pair["metric_order"].index("output_tokens_s")
        for arm in ("v0.4.0", "candidate"):
            for row in pair["requests"].values():
                row[arm][index] = 60
            for row in pair["workload_medians"].values():
                row[arm][index] = 60
        report["release_checks"]["both_speed_targets_on_every_workload"] = False
        self.assertEqual(list(gates.failures(report)), [])
        self.assertTrue(any("missed a speed target" in failure for failure in gates.failures(report, True)))

    def test_quality_measurements_override_a_passing_summary(self):
        report = passing_report()
        model = report["models"]["IQ3_S"]
        model["model_reference_comparison"]["candidate"]["maximum_kl"] = 0.1
        model["last_four_full_logit_vectors"]["sha256_by_prompt_length_and_width"]["300"]["37"] = "c" * 64
        model["fixed_seed_sampling"]["output_ids_sha256_le32"]["candidate"] = "d" * 64
        errors = list(gates.failures(report))
        for metric in ("maximum_kl", "full score vector", "fixed-seed"):
            self.assertTrue(any(metric in error for error in errors))

    def test_medians_match_measured_requests(self):
        report = passing_report()
        pair = report["models"]["IQ3_S"]["pairs"]["forward"]
        index = pair["metric_order"].index("output_tokens_s")
        for name in ("review-1", "review-2", "review-3"):
            pair["requests"][name]["candidate"][index] = 60
        with self.assertRaisesRegex(ValueError, "incorrect median"):
            list(gates.failures(report))

    def test_integrity_and_completeness(self):
        report = passing_report()
        board = report["models"]["UD-IQ3_XXS"]["board_checks"]
        for suite in ("unit_tests_passed", "gpu_parity_tests_passed"):
            board[suite] = 12
            with self.assertRaisesRegex(ValueError, "test suites"):
                list(gates.failures(report))
            board[suite] = 13
        pair = report["models"]["UD-IQ3_XXS"]["pairs"]["forward"]
        pair["requests"]["prose-2"]["output_hashes"]["candidate"]["answer"] = "c" * 64
        self.assertTrue(list(gates.failures(report)))
        pair["requests"]["prose-2"]["output_hashes"]["candidate"]["answer"] = "b" * 64
        index = pair["metric_order"].index("cached_tokens")
        pair["requests"]["prose-2"]["candidate"][index] = 2048
        self.assertTrue(any("cached input" in text for text in gates.failures(report)))
        pair["requests"]["prose-2"]["candidate"][index] = 0
        pair["requests"]["prose-2"]["prompt_tokens"] = 100
        with self.assertRaises(ValueError):
            list(gates.failures(report))
        del pair["requests"]["prose-2"]
        with self.assertRaises(ValueError):
            list(gates.failures(report))


if __name__ == "__main__":
    unittest.main()
