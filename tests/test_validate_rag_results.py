"""Manually verified metric and integrity examples; no APIs or model fitting."""

import json
from pathlib import Path
import tempfile
import unittest

from experiments.validate_rag_results import (
    Audit, canonical_hash, combination_audit, expected_request_key,
    independent_normalize, independent_scores, protected_snapshot,
)
from src.answer_evaluation import score_answer


class ManualMetricTests(unittest.TestCase):
    def test_case_punctuation_articles_whitespace(self):
        self.assertEqual(independent_normalize("  THE, U.S. President! "), "us president")
        # Unicode dash is retained under the declared ASCII-punctuation policy.
        self.assertEqual(independent_normalize("red—a—fox"), "red— —fox")
        for scorer in (independent_scores, score_answer):
            self.assertEqual(scorer("The RED fox!", ["red fox"])["exact_match"], 1)
            self.assertEqual(scorer("The RED fox!", ["red fox"])["f1"], 1)

    def test_multiset_f1_precision_recall(self):
        for scorer in (independent_scores, score_answer):
            # Two shared tokens, prediction length three, reference length three.
            self.assertAlmostEqual(scorer("red red blue", ["red blue blue"])["f1"], 2 / 3)
            # Precision 1, recall 1/2, harmonic mean 2/3.
            result = scorer("Alexander", ["Alexander Fleming"])
            self.assertEqual(result["exact_match"], 0)
            self.assertAlmostEqual(result["f1"], 2 / 3)
            self.assertEqual(scorer("London", ["Paris"])["f1"], 0)

    def test_multiple_references_take_best_not_first(self):
        for scorer in (independent_scores, score_answer):
            result = scorer("Fleming", ["Alexander Fleming", "Fleming"])
            self.assertEqual(result["exact_match"], 1)
            self.assertEqual(result["f1"], 1)

    def test_missing_empty_abstained_and_invalid_inputs(self):
        for scorer in (independent_scores, score_answer):
            for prediction in (None, "", "  ", "Insufficient evidence.", "I don't know.", "I do not know."):
                result = scorer(prediction, ["Paris"])
                self.assertEqual(result["exact_match"], 0)
                self.assertEqual(result["f1"], 0)
            for references in ([], [None], [""], "Paris"):
                with self.assertRaises(ValueError):
                    scorer("Paris", references)
            with self.assertRaises(ValueError):
                scorer({"answer": "Paris"}, ["Paris"])


class IntegrityExamples(unittest.TestCase):
    def test_missing_duplicate_and_unexpected_are_distinct(self):
        configs = [("llm_only", 0), ("bm25", 3)]
        rows = [{"question_id": qid, "retrieval_method": method, "top_k": k}
                for qid in ("q1", "q2") for method, k in configs]
        correct = combination_audit(rows, ["q1", "q2"], configs)
        self.assertEqual(len(correct["expected"]), 4)
        self.assertFalse(correct["missing"])
        corrupted = rows[:-1] + [rows[0], {"question_id": "val_id", "retrieval_method": "bm25", "top_k": 3}]
        result = combination_audit(corrupted, ["q1", "q2"], configs)
        self.assertEqual(result["missing"], [("q2", "bm25", 3)])
        self.assertEqual(result["duplicate_excess"], 1)
        self.assertEqual(result["unexpected"], [("val_id", "bm25", 3)])

    def fixture(self):
        settings = {"provider": "openai", "model": "fixed-model", "base_url": "https://example.test/v1",
                    "temperature": 0, "max_output_tokens": 64, "seed": 42, "max_tokens_field": "max_completion_tokens"}
        manifest = {"frozen_spec": {"settings": settings, "prompt_version": "v1",
                                   "system_prompt_sha256": canonical_hash("constant system"),
                                   "corpus_sha256": "corpus-hash", "reranker_sha256": "model-hash"},
                    "phases": {"test": {"question_ids": ["q1"], "dataset_sha256": "test-hash"}}}
        question = {"question_id": "q1", "question": "Which city?", "answers": {"text": ["Paris"]}, "relevant_passage_id": "p1"}
        content = {"question": question["question"], "condition": "no-retrieval",
                   "evidence_policy": "No external sources are supplied. Answer from existing knowledge, or respond 'I don't know.'"}
        row = {"_audit_line": 1, "question_id": "q1", "question": question["question"], "retrieval_method": "llm_only", "top_k": 0,
               "generated_answer": "I don't know.", "status": "completed", "api_error": None,
               "finish_reason": "stop", "model_settings": settings, "model_name": "fixed-model", "provider": "openai", "prompt_version": "v1",
               "retrieval_latency_ms": 0.0, "generation_latency_ms": 100.0, "generation_total_latency_ms": 101.0, "rate_limit_wait_ms": 1.0,
               "input_tokens": 10, "output_tokens": 5, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
               "request_attempts": 1, "attempt_errors": [], "cache_hit": False, "estimated_cost_usd": None,
               "retrieved_passage_ids": [], "retrieval_ranking": None, "passage_count": 0,
               "messages": [{"role": "system", "content": "constant system"}, {"role": "user", "content": json.dumps(content)}],
               "provenance": {"test_sha256": "test-hash", "corpus_sha256": "corpus-hash", "reranker_sha256": "model-hash",
                              "selected_question_ids_sha256": canonical_hash(["q1"])},
               "references": ["Paris"], "relevant_passage_id": "p1", "exact_match": 0.0, "f1": 0.0,
               "empty_answer": False, "abstained": True}
        row["cache_key"] = expected_request_key(row)
        return manifest, question, row

    def test_api_success_is_not_answer_correctness(self):
        manifest, question, row = self.fixture()
        audit = Audit()
        scored, _ = audit.record_checks([row], {"q1": question}, {"p1": "Paris is a city."}, manifest)
        self.assertEqual(scored[0]["exact_match"], 0)
        self.assertFalse(audit.failed_rows)
        self.assertTrue(all(check["status"] == "PASS" for check in audit.checks))

    def test_empty_completed_response_and_reference_leak_are_detected(self):
        manifest, question, row = self.fixture()
        row["generated_answer"] = ""
        row["empty_answer"], row["abstained"] = True, False
        content = json.loads(row["messages"][1]["content"])
        content["evidence_policy"] += " The reference answer is Paris."
        row["messages"][1]["content"] = json.dumps(content)
        audit = Audit()
        audit.record_checks([row], {"q1": question}, {"p1": "Paris is a city."}, manifest)
        codes = {issue["issue_code"] for issue in audit.issues}
        self.assertIn("empty_or_malformed_answer", codes)
        self.assertIn("prompt_integrity", codes)
        self.assertIn("request_identity_mismatch", codes)
        self.assertEqual(audit.failed_rows, {1})

    def test_aggregate_denominator_includes_abstentions(self):
        _, _, row = self.fixture()
        correct = {**row, "question_id": "q2", "generated_answer": "Paris", "abstained": False}
        independent = [{"row": row, "exact_match": 0.0, "f1": 0.0},
                       {"row": correct, "exact_match": 1.0, "f1": 1.0}]
        saved = {"method": "llm_only", "k": 0, "questions": 2, "completed": 2,
                 "exact_match": 0.5, "f1": 0.5, "llm_calls_this_run": 2,
                 "original_llm_calls_including_cache": 2, "cache_hits": 0,
                 "api_failures": 0, "api_error_attempts_this_run": 0,
                 "avg_generation_latency_ms": 100.0, "avg_retrieval_latency_ms": 0.0,
                 "avg_input_tokens": 10.0, "avg_output_tokens": 5.0}
        audit = Audit()
        table = audit.recompute_aggregates(independent, [("llm_only", 0)], ["q1", "q2"],
                                          {"aggregate": [saved]}, [{"method": "llm_only", "k": "0", "exact_match": "0.5", "f1": "0.5"}])
        self.assertEqual(table[0]["recomputed_exact_match"], 0.5)
        self.assertEqual(table[0]["recomputed_f1"], 0.5)
        self.assertEqual(table[0]["status"], "PASS")
        self.assertTrue(all(check["status"] == "PASS" for check in audit.checks))

    def test_missing_ground_truth_is_explicit_not_a_scored_answer(self):
        manifest, question, row = self.fixture()
        question["answers"] = None
        audit = Audit()
        recomputed, _ = audit.record_checks([row], {"q1": question}, {"p1": "Paris is a city."}, manifest)
        self.assertEqual(recomputed, [])
        self.assertIn("invalid_references", {issue["issue_code"] for issue in audit.issues})
        self.assertIn(1, audit.failed_rows)

    def test_read_only_snapshot_excludes_only_new_audit_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protected = root / "results" / "original.json"
            protected.parent.mkdir()
            protected.write_text('{"original":true}')
            before = protected_snapshot(root)
            audit_dir = root / "results" / "validation"
            audit_dir.mkdir()
            (audit_dir / "experiment_audit.json").write_text('{"audit":true}')
            self.assertEqual(before, protected_snapshot(root))
            protected.write_text('{"original":false}')
            self.assertNotEqual(before, protected_snapshot(root))


if __name__ == "__main__":
    unittest.main()
