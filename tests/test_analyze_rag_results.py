"""Statistical checks using synthetic pairs and a read-only saved-results integration."""

import ast
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments import analyze_rag_results as analysis


def fixture(n=150):
    test = [dict(question_id=f"q{i:03}", question=f"who is alpha {i}",
                 answers={"text": ["alpha"]}, relevant_passage_id=f"p{i}") for i in range(n)]
    rows = []
    for method, k in analysis.CONFIGS:
        for question in test:
            rows.append(dict(question_id=question["question_id"], question=question["question"],
                             relevant_passage_id=question["relevant_passage_id"],
                             references=["alpha"], generated_answer="alpha", status="completed",
                             api_error=None, retrieval_method=method, top_k=k,
                             exact_match=1.0, f1=1.0, abstained=False, input_tokens=20,
                             output_tokens=2, generation_latency_ms=10.0, retrieval_latency_ms=1.0))
    return test, rows


class PairedAnalysisTests(unittest.TestCase):
    def test_all_150_questions_are_retained_and_aligned(self):
        test, rows = fixture()
        ids, groups = analysis.align_records(list(reversed(rows)), test)
        self.assertEqual(len(ids), 150)
        for config in analysis.CONFIGS:
            self.assertEqual([r["question_id"] for r in groups[config]], ids)
            self.assertEqual(analysis.summary(config, groups[config])["questions"], 150)

    def test_missing_duplicate_and_unexpected_questions_fail(self):
        test, rows = fixture()
        for bad_rows in (rows[:-1], rows + [rows[0]], [{**rows[0], "question_id": "unknown"}] + rows[1:]):
            with self.assertRaises(ValueError):
                analysis.align_records(bad_rows, test)
        with self.assertRaisesRegex(ValueError, "150"):
            analysis.align_records(rows, test[:-1])

    def test_missing_invalid_metrics_and_score_changes_fail(self):
        test, rows = fixture()
        for field, value in (("f1", None), ("f1", float("nan")), ("f1", 1.1),
                             ("exact_match", .5), ("input_tokens", -1),
                             ("generation_latency_ms", "unknown"), ("abstained", None),
                             ("f1", .2)):
            altered = copy.deepcopy(rows)
            altered[0][field] = value
            with self.assertRaises(ValueError):
                analysis.align_records(altered, test)
        del rows[0]["output_tokens"]
        with self.assertRaisesRegex(ValueError, "output_tokens"):
            analysis.align_records(rows, test)

    def test_pair_ids_and_order_must_match(self):
        _, rows = fixture()
        left, right = rows[:150], copy.deepcopy(rows[:150])
        idx = analysis.bootstrap_indices(150, 50)
        for invalid in (list(reversed(right)), right[:-1]):
            with self.assertRaisesRegex(ValueError, "question IDs"):
                analysis.paired_comparison(left, invalid, idx, "test", ("semantic", 3), ("bm25", 3))

    def test_shared_resampling_preserves_identical_condition_correlation(self):
        _, rows = fixture()
        left = rows[:150]
        for i, row in enumerate(left):
            row["exact_match"] = float(i % 2)
            row["f1"] = (i % 5) / 4
        # Any independent resampling would give nonzero uncertainty here.
        idx = analysis.bootstrap_indices(150)
        result, cis = analysis.paired_comparison(left, copy.deepcopy(left), idx,
                                               "test", ("semantic", 3), ("bm25", 3))
        for row in cis:
            self.assertEqual(row["estimate"], 0)
            self.assertEqual(row["ci_low"], 0)
            self.assertEqual(row["ci_high"], 0)
        self.assertEqual(result["f1_ties"], 150)

    def test_bootstrap_matches_explicit_question_pair_resampling(self):
        _, rows = fixture()
        left, right = copy.deepcopy(rows[:150]), copy.deepcopy(rows[:150])
        for i, (a, b) in enumerate(zip(left, right)):
            a["f1"], b["f1"] = (i % 7) / 6, (i % 3) / 2
        idx = analysis.bootstrap_indices(150, 200)
        result, _ = analysis.paired_comparison(left, right, idx, "test", ("semantic", 3), ("bm25", 3))
        sampled_diffs = [np.mean([left[i]["f1"] for i in sample]) -
                         np.mean([right[i]["f1"] for i in sample]) for sample in idx]
        low, high = np.percentile(sampled_diffs, [2.5, 97.5])
        self.assertAlmostEqual(result["f1_ci_low"], low)
        self.assertAlmostEqual(result["f1_ci_high"], high)

    def test_seed_reproducibility(self):
        first = analysis.bootstrap_indices(150)
        second = analysis.bootstrap_indices(150)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, analysis.bootstrap_indices(150, seed=43)))

    def test_exact_match_units_and_absolute_effect(self):
        _, rows = fixture()
        left, right = copy.deepcopy(rows[:150]), copy.deepcopy(rows[:150])
        for row in right[:15]:
            row["exact_match"] = row["f1"] = 0.0
        result, cis = analysis.paired_comparison(left, right, analysis.bootstrap_indices(150),
                                               "test", ("semantic", 3), ("bm25", 3))
        self.assertAlmostEqual(result["delta_em_pp"], 10)
        self.assertAlmostEqual(result["delta_f1"], .1)
        self.assertEqual(cis[0]["units"], "percentage_points")
        self.assertEqual(cis[1]["units"], "fraction")
        self.assertEqual(cis[0]["estimate"], cis[0]["absolute_effect_magnitude"])

    def test_snapshots_include_audit_and_exclude_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results/validation").mkdir(parents=True)
            (root / "results/analysis").mkdir()
            original = root / "results/validation/audit.json"
            original.write_text("original")
            before = analysis.artifact_snapshot(root)
            (root / "results/analysis/new.csv").write_text("new analysis")
            self.assertEqual(before, analysis.artifact_snapshot(root))
            original.write_text("changed")
            self.assertNotEqual(before, analysis.artifact_snapshot(root))

    def test_no_inference_imports_or_fit_calls(self):
        tree = ast.parse(Path(analysis.__file__).read_text())
        forbidden = {"openai", "httpx", "requests", "sklearn", "joblib", "torch", "faiss",
                     "sentence_transformers", "src.generation", "src.reranker", "src.semantic"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(n.name.split(".")[0] not in forbidden for n in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, forbidden)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"fit", "fit_transform", "generate_answer"})

    def test_verify_only_detects_altered_analysis_and_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "results/analysis"
            output.mkdir(parents=True)
            original = root / "results/original.json"
            original.write_text("original")
            analysis_file = output / "summary.csv"
            analysis_file.write_text("unchanged")
            snapshot = analysis.artifact_snapshot(root)
            manifest = dict(protected_snapshot_before=snapshot, protected_snapshot_after=snapshot,
                            analysis_source_sha256=analysis.digest(Path(analysis.__file__)),
                            output_sha256={analysis_file.name: analysis.digest(analysis_file)},
                            question_count=150, record_count=1500, api_calls=0, models_fitted=0)
            analysis.write_json(output / "analysis_manifest.json", manifest)
            with redirect_stdout(io.StringIO()):
                analysis.verify_outputs(root)
            analysis_file.write_text("modified")
            with self.assertRaisesRegex(ValueError, "output hash mismatch"):
                analysis.verify_outputs(root)
            analysis_file.write_text("unchanged")
            original.write_text("modified")
            with self.assertRaisesRegex(ValueError, "Original artifacts differ"):
                analysis.verify_outputs(root)

    def test_critical_audit_inconsistency_stops_before_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "results/validation"
            audit_dir.mkdir(parents=True)
            analysis.write_json(audit_dir / "experiment_audit.json",
                                dict(summary=dict(overall_readiness="CORRECTIONS_REQUIRED")))
            (audit_dir / "experiment_audit.md").write_text("CORRECTIONS_REQUIRED")
            with self.assertRaisesRegex(ValueError, "Critical audit readiness"):
                analysis.run(root)
            self.assertFalse((root / "results/analysis").exists())

    def test_real_analysis_preserves_originals_and_is_repeatable_without_network(self):
        root = analysis.ROOT
        if not (root / "results/validation/experiment_audit.json").exists():
            self.skipTest("Saved experiment unavailable")
        before = analysis.artifact_snapshot(root)
        with patch.object(socket.socket, "connect", side_effect=AssertionError("No network allowed")):
            with redirect_stdout(io.StringIO()):
                first = analysis.run(root)
                second = analysis.run(root)
                analysis.verify_outputs(root)
        self.assertEqual(before, analysis.artifact_snapshot(root))
        self.assertEqual(first["output_sha256"], second["output_sha256"])
        self.assertEqual(first["bootstrap_matrix_sha256"], second["bootstrap_matrix_sha256"])
        self.assertTrue(second["original_artifacts_unchanged"])
        self.assertEqual(second["api_calls"], 0)
        self.assertEqual(second["models_fitted"], 0)
        self.assertEqual(second["question_count"], 150)
        summary = json.loads((root / "results/analysis/configuration_summary.json").read_text())
        self.assertEqual(len(summary["configurations"]), 10)


if __name__ == "__main__":
    unittest.main()
