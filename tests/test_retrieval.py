"""Offline retrieval ranking, Hit@K boundaries, and saved-artifact checks."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from src.bm25 import BM25Retriever
from src.evaluation import evaluate_retrieval, hit_at_k, run_evaluation


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.corpus = self.root / "corpus.jsonl"
        records = [
            {"passage_id": "p_z", "text": "Alexander Fleming discovered penicillin medicine"},
            {"passage_id": "p_b", "text": "Mars planets orbit space astronomy"},
            {"passage_id": "p_c", "text": "Bread flour yeast dough baking"},
            {"passage_id": "p_a", "text": "Ocean waves fish coral water"},
        ]
        self.corpus.write_text("".join(json.dumps(row) + "\n" for row in records))
        self.retriever = BM25Retriever(self.corpus)

    def test_retrieval_returns_ranked_serializable_ids_scores_and_text(self):
        results = self.retriever.retrieve(question="Who discovered penicillin?", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["passage_id"], "p_z")
        self.assertGreater(results[0]["score"], results[1]["score"])
        self.assertEqual(set(results[0]), {"passage_id", "score", "text"})
        json.dumps(results, allow_nan=False)
        self.assertEqual(len(self.retriever.retrieve("penicillin", top_k=10)), 4)

    def test_ties_have_deterministic_passage_id_order(self):
        results = self.retriever.retrieve("unseenword", top_k=4)
        self.assertEqual([r["passage_id"] for r in results], ["p_a", "p_b", "p_c", "p_z"])

    def test_invalid_inputs(self):
        for question in (None, "", "   ", "???"):
            with self.assertRaises(ValueError):
                self.retriever.retrieve(question)
        for k in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                self.retriever.retrieve("penicillin", k)
        self.corpus.write_text("")
        with self.assertRaisesRegex(ValueError, "empty"):
            BM25Retriever(self.corpus)

    def test_hit_at_k_uses_ids_and_exact_rank_boundary(self):
        ranking = [{"passage_id": f"p_{i}", "score": 10 - i} for i in range(10)]
        self.assertEqual(hit_at_k(ranking, "p_2", 1), 0)
        self.assertEqual(hit_at_k(ranking, "p_2", 3), 1)
        self.assertEqual(hit_at_k(ranking, "p_9", 10), 1)
        self.assertEqual(hit_at_k(ranking, "absent", 10), 0)
        self.assertEqual(hit_at_k([], "p_0", 5), 0)

    def test_average_and_single_retrieval_per_question(self):
        calls = []

        def retrieve(question, top_k):
            calls.append((question, top_k))
            return [{"passage_id": f"p_{i}", "score": 10 - i} for i in range(top_k)]

        questions = [
            {"question_id": "q1", "question": "first", "relevant_passage_id": "p_2"},
            {"question_id": "q2", "question": "second", "relevant_passage_id": "absent"},
        ]
        table, details = evaluate_retrieval(questions, retrieve)
        self.assertEqual([r["hit_at_k"] for r in table], [0, 0.5, 0.5, 0.5])
        self.assertEqual(calls, [("first", 10), ("second", 10)])
        self.assertEqual(details[0]["hits"]["Hit@3"], 1)
        with self.assertRaises(ValueError):
            evaluate_retrieval([], retrieve)
        with self.assertRaises(ValueError):
            evaluate_retrieval(questions, retrieve, ks=[0])

    def test_evaluation_writes_table_predictions_and_provenance(self):
        questions_path = self.root / "test.jsonl"
        question = {"question_id": "q1", "question": "penicillin", "relevant_passage_id": "p_z"}
        questions_path.write_text(json.dumps(question) + "\n")
        output = self.root / "results"
        with contextlib.redirect_stdout(io.StringIO()):
            table, _ = run_evaluation(self.corpus, questions_path, output)
        self.assertEqual([r["hit_at_k"] for r in table], [1, 1, 1, 1])
        with (output / "bm25_hit_at_k.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 4)
        self.assertEqual(len((output / "bm25_predictions.jsonl").read_text().splitlines()), 1)
        metadata = json.loads((output / "bm25_evaluation.json").read_text())
        self.assertEqual(metadata["corpus_size"], 4)
        question["relevant_passage_id"] = "missing"
        questions_path.write_text(json.dumps(question) + "\n")
        with self.assertRaisesRegex(ValueError, "outside this corpus"):
            run_evaluation(self.corpus, questions_path, output)


if __name__ == "__main__":
    unittest.main()
