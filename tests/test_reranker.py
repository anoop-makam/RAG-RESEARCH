"""Train-only scaling, feature schema, ranking, and serialization checks."""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from experiments.train_reranker import C_VALUES, ranking_metrics, tune_pipeline
from src.reranker import (
    LogisticReranker, feature_matrix, load_feature_rows, make_pipeline,
    rank_feature_rows, rerank,
)
from src.reranker_features import FEATURE_COLUMNS, ROW_COLUMNS


def record(pid, strength, label=0, question_id="train_q"):
    return {"question_id": question_id, "passage_id": pid,
            "bm25_score": float(strength), "semantic_similarity": strength / 10,
            "word_overlap": min(strength / 10, 1.0), "passage_length": 10 + strength,
            "label": label, "candidate_source": "both", "force_included_positive": 0}


class RerankerTests(unittest.TestCase):
    def setUp(self):
        self.training = [record("a", 0), record("b", 1), record("c", 2), record("d", 7, 1)]
        self.pipeline = make_pipeline(C=0.1)
        self.pipeline.fit(feature_matrix(self.training), [row["label"] for row in self.training])

    def test_feature_order_is_explicit_and_metadata_is_excluded(self):
        row = record("p", 3)
        reversed_row = dict(reversed(list(row.items())))
        np.testing.assert_array_equal(feature_matrix([row]), feature_matrix([reversed_row]))
        expected = [row[name] for name in FEATURE_COLUMNS]
        np.testing.assert_array_equal(feature_matrix([row])[0], expected)
        row.update(label=999, candidate_source="forced_positive", force_included_positive=1, answers="do not use")
        np.testing.assert_array_equal(feature_matrix([row])[0], expected)

    def test_missing_and_nonfinite_features_fail(self):
        for bad in (None, "", "3", float("nan"), float("inf"), True):
            row = record("p", 3)
            row["semantic_similarity"] = bad
            with self.assertRaises(ValueError):
                feature_matrix([row])
        row = record("p", 3)
        row.pop("bm25_score")
        with self.assertRaises(ValueError):
            feature_matrix([row])

    def test_csv_schema_and_missing_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS)
                writer.writeheader()
                writer.writerows(self.training)
            self.assertEqual(load_feature_rows(path), self.training)
            text = path.read_text()
            path.write_text(text.replace("semantic_similarity", "wrong_feature", 1))
            with self.assertRaises(ValueError):
                load_feature_rows(path)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS)
                writer.writeheader()
                row = self.training[0].copy()
                row["bm25_score"] = ""
                writer.writerow(row)
            with self.assertRaises(ValueError):
                load_feature_rows(path)

    def test_probabilities_rank_descending_and_ties_use_passage_id(self):
        candidates = [record("low", 0), record("high", 7)]
        ranked = rank_feature_rows(self.pipeline, candidates)
        self.assertEqual([row["passage_id"] for row in ranked], ["high", "low"])
        self.assertTrue(0 <= ranked[1]["score"] < ranked[0]["score"] <= 1)
        ties = rank_feature_rows(self.pipeline, [record("z", 2), record("a", 2)])
        self.assertEqual([row["passage_id"] for row in ties], ["a", "z"])
        self.assertEqual(ties, rank_feature_rows(self.pipeline, [record("a", 2), record("z", 2)]))
        with self.assertRaises(ValueError):
            rank_feature_rows(self.pipeline, [record("same", 0), record("same", 1)])

    def test_online_api_and_saved_model_match_precomputed_ranking(self):
        candidates = [record("low", 0), record("high", 7)]
        features = {row["passage_id"]: row for row in candidates}
        calls = []

        class Extractor:
            passages = {"low": "low text", "high": "high text"}

            def extract(self, question, passage_ids):
                calls.append((question, passage_ids))
                return [features[pid] for pid in passage_ids]

        model = LogisticReranker(self.pipeline, extractor=Extractor())
        result = model.rerank("question", [{"passage_id": "low", "text": "low text"}, {"passage_id": "high"}])
        expected = rank_feature_rows(self.pipeline, candidates)
        self.assertEqual([(row["passage_id"], row["score"]) for row in result],
                         [(row["passage_id"], row["score"]) for row in expected])
        self.assertEqual(calls, [("question", ["low", "high"])])
        self.assertEqual(result[0]["text"], "high text")
        with patch("src.reranker._default_reranker", return_value=model):
            self.assertEqual(rerank("question", ["low", "high"]), result)
        self.assertEqual(model.rerank("question", []), [])
        with self.assertRaises(ValueError):
            model.rerank("question", [{"passage_id": "low", "text": "wrong"}])
        with self.assertRaises(ValueError):
            model.rerank("question", ["unknown"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            model.save(path)
            restored = LogisticReranker.load(path, extractor=Extractor())
            self.assertEqual(restored.rerank("question", ["low", "high"]), result)
            bundle = joblib.load(path)
            bundle["feature_columns"] = tuple(reversed(FEATURE_COLUMNS))
            joblib.dump(bundle, path)
            with self.assertRaisesRegex(ValueError, "ordering"):
                LogisticReranker.load(path)

    def test_tuning_fits_scaler_only_on_train_and_chooses_mrr(self):
        val = [record("v_positive", 9, 1, "val_q"), record("v_negative", 5, 0, "val_q")]
        questions = [{"question_id": "val_q", "question": "validation", "relevant_passage_id": "v_positive"}]
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline, C, metrics, rankings, tuning, weights = tune_pipeline(self.training, val, questions)
        np.testing.assert_array_equal(pipeline.named_steps["scaler"].mean_, feature_matrix(self.training).mean(axis=0))
        self.assertEqual(pipeline.named_steps["scaler"].n_samples_seen_, len(self.training))
        self.assertEqual(pipeline.named_steps["logistic"].class_weight, "balanced")
        self.assertEqual(weights, {0: 4 / 6, 1: 2.0})
        self.assertEqual([row["C"] for row in tuning], list(C_VALUES))
        maximum = max(row["MRR@10"] for row in tuning)
        self.assertEqual(C, min(row["C"] for row in tuning if row["MRR@10"] == maximum))
        self.assertEqual(metrics["MRR@10"], maximum)
        self.assertEqual(set(rankings), {"val_q"})

    def test_missing_source_still_counts_in_denominator(self):
        questions = [{"question_id": "q1", "question": "first", "relevant_passage_id": "a"},
                     {"question_id": "q2", "question": "second", "relevant_passage_id": "missing"}]
        rankings = {"q1": [{"passage_id": "a", "score": 1.0}],
                    "q2": [{"passage_id": "other", "score": 1.0}]}
        metrics, _ = ranking_metrics(questions, rankings)
        self.assertEqual(metrics["MRR@10"], 0.5)
        self.assertEqual(metrics["Hit@10"], 0.5)


if __name__ == "__main__":
    unittest.main()
