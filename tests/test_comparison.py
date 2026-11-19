"""Validation-only comparison checks with known rankings and no model downloads."""

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.compare_retrievers import (
    check_rankings, combine_results, evaluate_method, print_wins,
    save_results, validate_inputs,
)
from src.evaluation import reciprocal_rank_at_k


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [{"passage_id": "p_a", "text": "answer alpha"},
                       {"passage_id": "p_b", "text": "answer beta"}]
        self.questions = [
            {"question_id": "q_a", "question": "alpha question", "relevant_passage_id": "p_a",
             "answers": {"text": ["alpha"], "answer_start": [7]}},
            {"question_id": "q_b", "question": "beta question", "relevant_passage_id": "p_b",
             "answers": {"text": ["beta"], "answer_start": [7]}},
        ]
        self.passages = {row["passage_id"]: row["text"] for row in self.corpus}
        self.results = [{**row, "score": 2.0 - i} for i, row in enumerate(self.corpus)]

    def test_input_schema_duplicate_text_and_mapping_checks(self):
        self.assertEqual(validate_inputs(self.corpus, self.questions, 2), self.passages)
        with self.assertRaisesRegex(ValueError, "Expected 50"):
            validate_inputs(self.corpus, self.questions)
        for mutate in [lambda q: q[0].pop("answers"),
                       lambda q: q[0].update(relevant_passage_id="missing"),
                       lambda q: q[1].update(question="ALPHA question?"),
                       lambda q: q[0].update(answers={"text": ["wrong"], "answer_start": [0]})]:
            questions = copy.deepcopy(self.questions)
            mutate(questions)
            with self.assertRaises(ValueError):
                validate_inputs(self.corpus, questions, 2)
        for corpus in [self.corpus + [self.corpus[0]],
                       self.corpus + [{"passage_id": "p_c", "text": "answer alpha"}],
                       [{"passage_id": "p", "text": " "}]]:
            with self.assertRaises(ValueError):
                validate_inputs(corpus, self.questions, 2)

    def test_rank_checks_reject_unknown_duplicate_unsorted_and_wrong_text(self):
        check_rankings(self.results, self.passages, 10)
        for mutate in [lambda r: r[0].update(passage_id="missing"),
                       lambda r: r[1].update(passage_id="p_a"),
                       lambda r: r[0].update(text="wrong"),
                       lambda r: r[1].update(score=3.0),
                       lambda r: r[0].update(score=float("nan"))]:
            results = copy.deepcopy(self.results)
            mutate(results)
            with self.assertRaises(ValueError):
                check_rankings(results, self.passages, 10)

    def test_mrr_boundaries(self):
        ranking = [{"passage_id": f"p_{i}"} for i in range(11)]
        self.assertEqual(reciprocal_rank_at_k(ranking, "p_0"), 1)
        self.assertEqual(reciprocal_rank_at_k(ranking, "p_9"), 0.1)
        self.assertEqual(reciprocal_rank_at_k(ranking, "p_10"), 0)
        self.assertEqual(reciprocal_rank_at_k([], "p_0"), 0)
        with self.assertRaises(ValueError):
            reciprocal_rank_at_k(ranking, "p_0", 0)

    def test_existing_evaluator_reused_with_query_only_timing_and_inputs(self):
        calls, results = [], self.results

        class Retriever:
            # No answer or reference ID is accepted by the retrieval API.
            def retrieve(self, question, top_k):
                calls.append((question, top_k))
                return results

        with patch("experiments.compare_retrievers.perf_counter", side_effect=[1, 1.002, 2, 2.004]):
            metrics, details = evaluate_method(self.questions, Retriever(), self.passages)
        self.assertEqual(calls, [("alpha question", 10), ("beta question", 10)])
        self.assertEqual(metrics["Hit@1"], 0.5)
        self.assertEqual(metrics["Hit@3"], 1)
        self.assertEqual(metrics["MRR@10"], 0.75)
        self.assertAlmostEqual(metrics["avg_query_latency_ms"], 3.0)
        self.assertEqual([row["relevant_rank"] for row in details], [1, 2])

        semantic = copy.deepcopy(details)
        semantic[0]["relevant_rank"] = None
        semantic[1]["relevant_rank"] = 1
        records = combine_results(self.questions, {"bm25": details, "semantic": semantic})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_wins(records, "semantic", "bm25")
            print_wins(records, "bm25", "semantic")
        self.assertIn("semantic: 1, bm25: 2", output.getvalue())
        self.assertIn("bm25: 1, semantic: not in top 10", output.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            report = {"metrics": {"bm25": metrics, "semantic": metrics}}
            save_results(path, records, report)
            saved = [json.loads(line) for line in (path / "retrieval_comparison_val.jsonl").read_text().splitlines()]
            self.assertEqual(saved, records)
            with (path / "retrieval_comparison_val.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(json.loads(rows[0]["bm25_ranked_passage_ids"]), ["p_a", "p_b"])
            self.assertEqual(rows[0]["semantic_relevant_rank"], "")
            self.assertTrue((path / "retrieval_comparison.png").read_bytes().startswith(b"\x89PNG"))
            self.assertEqual(json.loads((path / "retrieval_metrics_val.json").read_text()), report)


if __name__ == "__main__":
    unittest.main()
