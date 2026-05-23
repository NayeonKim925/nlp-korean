import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "validity_gated_exp" / "compare_results.py"

spec = importlib.util.spec_from_file_location("compare_results", MODULE_PATH)
compare_results = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(compare_results)


class CompareResultsTest(unittest.TestCase):
    def test_tradeoff_claim_is_suggested_when_naive_has_higher_pairacc(self):
        results = {
            "Baseline": {
                "f1": [0.793, 0.798, 0.786],
                "flip_rate": [0.050, 0.059, 0.042],
                "strict_flip_rate": [0.054, 0.052, 0.055],
                "prob_gap": [0.040, 0.037, 0.042],
                "strict_prob_gap": [0.042, 0.039, 0.044],
            },
            "Naive Swap": {
                "f1": [0.792, 0.796, 0.788],
                "pair_accuracy": [0.823, 0.827, 0.820],
                "strict_pair_accuracy": [0.828, 0.832, 0.824],
                "flip_rate": [0.018, 0.019, 0.017],
                "strict_flip_rate": [0.018, 0.021, 0.016],
                "prob_gap": [0.020, 0.023, 0.019],
                "strict_prob_gap": [0.021, 0.023, 0.018],
                "train_valid_cf_ratio": [0.045, 0.045, 0.045],
                "cons_batch_ratio": [0.95, 0.96, 0.94],
                "avg_valid_cf_per_batch": [2.9, 2.8, 2.9],
            },
            "Strict-Gated": {
                "f1": [0.794, 0.797, 0.790],
                "pair_accuracy": [0.818, 0.821, 0.815],
                "strict_pair_accuracy": [0.822, 0.825, 0.819],
                "flip_rate": [0.021, 0.028, 0.015],
                "strict_flip_rate": [0.024, 0.031, 0.016],
                "prob_gap": [0.019, 0.018, 0.019],
                "strict_prob_gap": [0.018, 0.016, 0.020],
                "train_valid_cf_ratio": [0.035, 0.035, 0.035],
                "cons_batch_ratio": [0.89, 0.90, 0.88],
                "avg_valid_cf_per_batch": [2.2, 2.3, 2.2],
            },
        }

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            compare_results.print_interpretation_notes(results)
        text = out.getvalue()

        self.assertIn("invariance-validity tradeoff", text)
        self.assertIn("Use the tradeoff claim", text)
        self.assertIn("Naive=4.50% vs Strict=3.50%", text)
        self.assertIn("Regularized batches: Naive=95.00% vs Strict=89.00%", text)
        self.assertIn("Valid CF per batch: Naive=2.87 vs Strict=2.23", text)

    def test_loading_and_markdown_table_from_json(self):
        payload = {
            "Baseline": {"f1": [0.79], "flip_rate": [0.05]},
            "metadata": {"ignored": True},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            results = compare_results.load_results([path])

        self.assertEqual(list(results), ["Baseline"])

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            compare_results.print_markdown_table(results)
        self.assertIn("| Baseline | 0.7900 |", out.getvalue())


if __name__ == "__main__":
    unittest.main()
