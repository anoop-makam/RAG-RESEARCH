"""Offline checks for feature identity, candidate labeling, and leakage boundaries."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.prepare_reranker_data import write_and_verify_csv
from src.bm25 import BM25Retriever, tokenize
from src.reranker_features import (
    FEATURE_COLUMNS, ROW_COLUMNS, RerankerFeatureExtractor, build_feature_rows,
    candidate_union, dataset_statistics, validate_feature_rows,
    validate_split_isolation, word_overlap,
)
from src.semantic import SemanticRetriever


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        mapping = {"gold copper metal": [1, 0, 0], "river water stream": [0, 1, 0],
                   "forest green trees": [0, 0, 1], "find gold": [0, 0.6, 0.8]}
        return np.array([mapping[text] for text in texts], dtype=np.float32)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / "corpus.jsonl"
        self.corpus = [{"passage_id": "p_z", "text": "gold copper metal"},
                       {"passage_id": "p_a", "text": "river water stream"},
                       {"passage_id": "p_m", "text": "forest green trees"}]
        self.path.write_text("".join(json.dumps(row) + "\n" for row in self.corpus))
        self.model = Encoder()
        self.bm25 = BM25Retriever(self.path)
        self.semantic = SemanticRetriever(self.path, model=self.model)
        self.extractor = RerankerFeatureExtractor(self.bm25, self.semantic)
        self.questions = [{"question_id": "q1", "question": "find gold", "relevant_passage_id": "p_m",
                           "answers": {"text": ["forest"], "answer_start": [0]}}]

    def retrieval_patches(self):
        return (patch.object(self.bm25, "retrieve", return_value=[{"passage_id": "p_z"}]),
                patch.object(self.semantic, "retrieve", return_value=[{"passage_id": "p_a"}]))

    def build(self, split):
        left, right = self.retrieval_patches()
        with left as bm25_call, right as semantic_call:
            rows, audit = build_feature_rows(self.questions, self.extractor, split, top_k=1)
            bm25_call.assert_called_once_with(question="find gold", top_k=1)
            semantic_call.assert_called_once_with(question="find gold", top_k=1)
        return rows, audit

    def test_union_source_and_deduplication(self):
        left = [{"passage_id": "a"}, {"passage_id": "b"}]
        right = [{"passage_id": "b"}, {"passage_id": "c"}]
        self.assertEqual(candidate_union(left, right), {"a": "BM25", "b": "both", "c": "semantic"})
        with self.assertRaises(ValueError):
            candidate_union(left + [left[0]], right)

    def test_features_use_stored_vectors_and_correct_bm25_positions(self):
        self.assertEqual(len(self.model.calls), 1)
        rows = self.extractor.extract("find gold", ["p_m", "p_z", "p_a"])
        # Feature extraction adds a query encode, never another passage encode.
        self.assertEqual(self.model.calls[1:], [["find gold"]])
        np.testing.assert_allclose([row["semantic_similarity"] for row in rows], [0.8, 0, 0.6])
        scores = self.bm25.index.get_scores(tokenize("find gold"))
        for row in rows:
            pos = self.extractor.bm25_positions[row["passage_id"]]
            self.assertEqual(row["bm25_score"], float(scores[pos]))
        self.assertEqual(rows[1]["word_overlap"], 0.5)
        self.assertEqual(rows[1]["passage_length"], 3)
        self.assertEqual(word_overlap([], ["gold"]), 0)
        self.assertEqual(word_overlap(["gold", "gold", "metal"], ["gold"]), 0.5)

    def test_training_forces_positive_but_does_not_inflate_retrieval(self):
        rows, audit = self.build("train")
        self.assertEqual(len(rows), 3)
        forced = [row for row in rows if row["force_included_positive"]]
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0]["passage_id"], "p_m")
        self.assertEqual(forced[0]["candidate_source"], "forced_positive")
        self.assertEqual(forced[0]["label"], 1)
        stats = dataset_statistics(rows, audit)
        self.assertEqual(stats["positive_labels"], 1)
        self.assertEqual(stats["negative_labels"], 2)
        self.assertEqual(stats["natural_union_hit_at_10"], 0)
        self.assertEqual(stats["force_included_positives"], 1)
        broken = copy.deepcopy(audit)
        broken[0]["natural_candidate_hit"] = 1
        with self.assertRaisesRegex(ValueError, "retrieval success"):
            validate_feature_rows(rows, self.questions, self.extractor.passages, "train", broken)

    def test_validation_never_inserts_positive_and_answers_are_not_features(self):
        rows, audit = self.build("val")
        self.assertEqual({row["passage_id"] for row in rows}, {"p_z", "p_a"})
        self.assertEqual(sum(row["label"] for row in rows), 0)
        self.assertEqual(sum(row["force_included_positive"] for row in rows), 0)
        self.assertEqual(dataset_statistics(rows, audit)["questions_missing_natural_positive"], 1)
        self.questions[0]["answers"] = {"text": ["different reference answer"], "answer_start": [123]}
        again, _ = self.build("val")
        self.assertEqual(rows, again)
        self.assertEqual(FEATURE_COLUMNS, ("bm25_score", "semantic_similarity", "word_overlap", "passage_length"))
        with self.assertRaises(ValueError):
            build_feature_rows(self.questions, self.extractor, "test")

    def test_naturally_retrieved_positive_is_not_forced(self):
        self.questions[0]["relevant_passage_id"] = "p_a"
        rows, audit = self.build("train")
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(row["force_included_positive"] for row in rows), 0)
        self.assertEqual(audit[0]["natural_candidate_hit"], 1)

    def test_feature_checks_csv_order_and_roundtrip(self):
        rows, audit = self.build("train")
        path = self.root / "features.csv"
        write_and_verify_csv(path, rows, self.questions, self.extractor.passages, "train", audit)
        with path.open() as handle:
            self.assertEqual(tuple(csv.DictReader(handle).fieldnames), ROW_COLUMNS)
        for mutate in [lambda data: data[0].update(bm25_score=float("inf")),
                       lambda data: data[0].update(label=2),
                       lambda data: data.append(data[0]),
                       lambda data: data[0].update(answer_text="leak")]:
            broken = copy.deepcopy(rows)
            mutate(broken)
            with self.assertRaises(ValueError):
                validate_feature_rows(broken, self.questions, self.extractor.passages, "train", audit)
        val = copy.deepcopy(self.questions)
        with self.assertRaises(ValueError):
            validate_split_isolation(self.questions, val)
        val[0]["question_id"] = "q2"
        with self.assertRaises(ValueError):
            validate_split_isolation(self.questions, val)
        val[0]["question"] = "different question"
        validate_split_isolation(self.questions, val)


if __name__ == "__main__":
    unittest.main()
