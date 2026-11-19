"""Check FAISS row-to-passage alignment without downloading a model."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.semantic import SemanticRetriever, retrieve


class FakeEncoder:
    """Known unit vectors let tests check rankings independently of the model."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        vectors = {"first context": [0, 1, 0], "second context": [1, 0, 0],
                   "third context": [0, 0, 1], "query": [0.8, 0.6, 0]}
        return np.array([vectors[text] for text in texts], dtype=np.float64)


class SemanticTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "corpus.jsonl"
        # Deliberately unsorted IDs ensure an accidental sort breaks the test.
        self.records = [{"passage_id": "p_z", "text": "first context"},
                        {"passage_id": "p_a", "text": "second context"},
                        {"passage_id": "p_m", "text": "third context"}]
        self.write(self.records)
        self.model = FakeEncoder()

    def write(self, records):
        self.path.write_text("".join(json.dumps(record) + "\n" for record in records))

    def test_exact_index_order_and_ranked_ids_scores_text(self):
        retriever = SemanticRetriever(self.path, model=self.model)
        self.assertEqual(retriever.passage_ids, ("p_z", "p_a", "p_m"))
        self.assertEqual(retriever.index.ntotal, 3)
        self.assertEqual(self.model.calls[0][0], [record["text"] for record in self.records])
        self.assertTrue(self.model.calls[0][1]["normalize_embeddings"])
        results = retriever.retrieve(question="query", top_k=3)
        self.assertEqual([row["passage_id"] for row in results], ["p_a", "p_z", "p_m"])
        self.assertEqual([row["text"] for row in results], ["second context", "first context", "third context"])
        np.testing.assert_allclose([row["score"] for row in results], [0.8, 0.6, 0], atol=1e-6)
        json.dumps(results, allow_nan=False)
        # Corpus was encoded once; subsequent searches only encode queries.
        retriever.retrieve("query", top_k=1)
        self.assertEqual([len(call[0]) for call in self.model.calls], [3, 1, 1])

    def test_top_k_larger_than_corpus_has_no_missing_neighbors(self):
        retriever = SemanticRetriever(self.path, model=self.model)
        self.assertEqual(len(retriever.retrieve("query", top_k=10)), 3)

    def test_invalid_questions_k_and_corpora(self):
        retriever = SemanticRetriever(self.path, model=self.model)
        for question in (None, "", "   "):
            with self.assertRaises(ValueError):
                retriever.retrieve(question)
        for k in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                retriever.retrieve("query", top_k=k)
        for records in ([], [self.records[0], self.records[0]],
                        [{"passage_id": "p", "text": " "}]):
            self.write(records)
            with self.assertRaises(ValueError):
                SemanticRetriever(self.path, model=self.model)

    def test_default_wrapper(self):
        retriever = SemanticRetriever(self.path, model=self.model)
        with patch("src.semantic._default_retriever", return_value=retriever):
            self.assertEqual(retrieve(question="query", top_k=1)[0]["passage_id"], "p_a")

    def test_reject_invalid_embeddings(self):
        with patch.object(self.model, "encode", return_value=np.zeros((3, 3))):
            with self.assertRaisesRegex(ValueError, "unit length"):
                SemanticRetriever(self.path, model=self.model)
        with patch.object(self.model, "encode", return_value=np.full((3, 3), np.nan)):
            with self.assertRaisesRegex(ValueError, "invalid embeddings"):
                SemanticRetriever(self.path, model=self.model)


if __name__ == "__main__":
    unittest.main()
