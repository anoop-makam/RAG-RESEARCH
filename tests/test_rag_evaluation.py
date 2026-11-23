"""Controlled experiment logic without API calls or test-set performance tuning."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import evaluate_rag as rag
from src.answer_evaluation import normalize_answer, score_answer
from src.generation import AnswerGenerator, GenerationConfig, build_messages


class AnswerMetricTests(unittest.TestCase):
    def test_normalization_and_multiple_references(self):
        self.assertEqual(normalize_answer(" The, U.S. President! "), "us president")
        self.assertEqual(score_answer("Fleming", ["Alexander Fleming", "Fleming"])["exact_match"], 1)
        self.assertEqual(score_answer("red red", ["red blue"])["f1"], 0.5)
        self.assertAlmostEqual(score_answer("Alexander", ["Alexander Fleming"])["f1"], 2 / 3)

    def test_empty_abstention_and_invalid_references(self):
        for answer in (None, "", "  ", "Insufficient evidence.", "I don't know."):
            result = score_answer(answer, ["Alexander Fleming"])
            self.assertEqual(result["exact_match"], 0)
            self.assertEqual(result["f1"], 0)
        self.assertTrue(score_answer("Insufficient evidence.", ["evidence"])["abstained"])
        with self.assertRaises(ValueError):
            score_answer("answer", [])


class ControlledRagTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.results = self.root / "results"
        self.model = self.root / "model.joblib"
        self.model.write_bytes(b"frozen saved model")
        self.text = "Alexander Fleming discovered penicillin."
        (self.data / "corpus.jsonl").write_text(json.dumps({"passage_id": "p1", "text": self.text}) + "\n")
        for split, size in (("val", 50), ("test", 150)):
            rows = [{"question_id": f"{split}_{i}", "question": f"Who discovered penicillin example {split} {i}?",
                     "answers": {"text": ["Alexander Fleming", "Fleming"], "answer_start": [0, 10]},
                     "relevant_passage_id": "p1"} for i in range(size)]
            (self.data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.config = GenerationConfig(model="fixed-model", api_key="fake-key", requests_per_minute=0)
        self.calls = []
        self.patches = [patch.object(rag, "DATA", self.data), patch.object(rag, "MODEL", self.model),
                        patch.object(rag, "plots", lambda *args: self.fake_plots(*args))]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def fake_plots(self, report, path):
        for name in ("cs391_accuracy_comparison.png", "cs391_f1_by_k.png", "cs391_latency_by_method.png", "cs391_retrieval_vs_answer_quality.png"):
            (path / name).write_bytes(b"fake chart for orchestration tests")

    def generator(self, fail=False):
        def transport(url, headers, payload, timeout):
            self.calls.append(payload)
            if fail:
                from urllib.error import HTTPError
                body = io.BytesIO(b'{"error":{"code":"insufficient_quota","type":"insufficient_quota"}}')
                raise HTTPError(url, 429, "quota", {}, body)
            user = json.loads(payload["messages"][1]["content"])
            self.assertNotIn("answers", user)
            self.assertNotIn("relevant_passage_id", user)
            return ({"model": "fixed-model", "choices": [{"message": {"content": "Alexander Fleming"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 50, "completion_tokens": 2}}, {})
        return AnswerGenerator(self.config, self.root / "cache", transport=transport)

    def retrieval(self):
        text = self.text
        class Retrieval:
            def retrieve(self, question, top_k):
                return {"llm_only": {"passages": None, "retrieval_latency_ms": 0.0},
                        **{method: {"passages": [{"passage_id": "p1", "text": text, "score": 1.0}],
                                    "retrieval_latency_ms": 1.0, "candidate_passage_ids": ["p1"]}
                           for method in rag.METHODS}}
        return Retrieval()

    def run_phase(self, phase="pilot", **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return rag.run_experiment(phase=phase, test_size=kwargs.pop("test_size", 2), config=self.config,
                                      results_dir=self.results, generator=kwargs.pop("generator", self.generator()),
                                      retrieval=self.retrieval(), **kwargs)

    def test_pilot_gate_final_subset_freeze_and_rescore(self):
        with self.assertRaisesRegex(ValueError, "successful five-question"):
            self.run_phase("test")
        report, manifest = self.run_phase()
        self.assertTrue(manifest["pilot_passed"])
        self.assertEqual(len(self.calls), 50)
        self.assertTrue(report["all_completed"])
        report, manifest = self.run_phase("test")
        self.assertEqual(report["question_count"], 2)
        self.assertEqual(len(self.calls), 70)
        self.assertTrue(manifest["test_started"])
        ids = manifest["phases"]["test"]["question_ids"]
        self.assertEqual(len(ids), 2)
        before = len(self.calls)
        self.run_phase("test")
        self.assertEqual(len(self.calls), before)
        self.run_phase("test", rescore=True)
        self.assertEqual(len(self.calls), before)
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.run_phase("test", test_size=3)
        table = report["aggregate"]
        self.assertEqual(len(table), 10)
        self.assertTrue(all(row["exact_match"] == 1 and row["f1"] == 1 for row in table))

    def test_api_failure_is_explicit_and_does_not_pass_pilot(self):
        report, manifest = self.run_phase(generator=self.generator(fail=True))
        self.assertFalse(manifest["pilot_passed"])
        self.assertFalse(report["all_completed"])
        records = rag.read_jsonl(self.results / "cs391_per_question_results.jsonl")
        self.assertEqual(len(records), 50)
        self.assertEqual(sum(r["status"] == "error" for r in records), 1)
        self.assertEqual(sum(r["status"] == "not_attempted" for r in records), 49)
        self.assertTrue(all(r["exact_match"] == 0 for r in records))

    def test_dry_run_cannot_unlock_test_and_metrics_are_null(self):
        report, manifest = self.run_phase(dry_run=True)
        self.assertFalse(manifest["pilot_passed"])
        self.assertEqual(len(self.calls), 0)
        self.assertTrue(all(row["exact_match"] is None for row in report["aggregate"]))
        with self.assertRaises(ValueError):
            self.run_phase("test")

    def test_test_subset_selection_is_seeded_and_stable(self):
        a, _, _ = rag.load_questions("test", 7, 42)
        b, _, _ = rag.load_questions("test", 7, 42)
        c, _, _ = rag.load_questions("test", 7, 43)
        self.assertEqual([r["question_id"] for r in a], [r["question_id"] for r in b])
        self.assertNotEqual([r["question_id"] for r in a], [r["question_id"] for r in c])

    def test_prompt_and_ranking_integrity(self):
        questions, corpus, _ = rag.load_questions("pilot")
        question = questions[0]
        record = {"question_id": question["question_id"], "question": question["question"], "retrieval_method": "bm25",
                  "top_k": 3, "retrieved_passage_ids": ["p1"],
                  "retrieval_ranking": [{"passage_id": "p1", "score": 1}],
                  "messages": build_messages(question["question"], [{"passage_id": "p1", "text": self.text}])}
        rag.validate_prompt(record, question, corpus)
        record["messages"][1]["content"] += " reference answer injection"
        with self.assertRaises(ValueError):
            rag.validate_prompt(record, question, corpus)


if __name__ == "__main__":
    unittest.main()
