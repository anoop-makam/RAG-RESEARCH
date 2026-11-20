"""Logistic regression relevance ranking with an explicit four-feature allowlist.

Natural Questions labels can be incomplete. The class-weighted predicted scores
are ranking signals, not calibrated probabilities of factual relevance.
"""

import csv
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.reranker_features import FEATURE_COLUMNS, ROW_COLUMNS


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "logistic_reranker.joblib"


def feature_matrix(records):
    """Select feature names explicitly; never use bookkeeping or label columns."""
    values = []
    for row in records:
        vector = []
        for column in FEATURE_COLUMNS:
            value = row.get(column)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"Feature {column!r} must be present and numeric.")
            if not np.isfinite(value):
                raise ValueError(f"Feature {column!r} must be finite.")
            vector.append(float(value))
        values.append(vector)
    return np.asarray(values, dtype=np.float64).reshape(-1, len(FEATURE_COLUMNS))


def load_feature_rows(path):
    """Load the actual preparation schema, rejecting missing/extra/invalid cells."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != ROW_COLUMNS:
                raise ValueError(f"Unexpected feature columns in {path}.")
            rows = list(reader)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing {path}; run python -m experiments.prepare_reranker_data first.") from exc
    if not rows:
        raise ValueError(f"Empty feature dataset: {path}")
    for number, row in enumerate(rows, 2):
        try:
            if None in row or any(value is None or value == "" for value in row.values()):
                raise ValueError("missing or extra cells")
            for column in FEATURE_COLUMNS:
                row[column] = int(row[column]) if column == "passage_length" else float(row[column])
            for column in ("label", "force_included_positive"):
                row[column] = int(row[column])
                if row[column] not in (0, 1):
                    raise ValueError("labels and flags must be 0 or 1")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid feature row {path}:{number}: {exc}") from exc
    feature_matrix(rows)
    return rows


def make_pipeline(C=1.0, class_weight="balanced"):
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be positive and finite.")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(C=C, class_weight=class_weight,
                                        random_state=42, max_iter=1000)),
    ])


def rank_feature_rows(pipeline, records):
    """Score precomputed candidates; resolve probability ties by passage ID."""
    records = list(records)
    if not records:
        return []
    ids = [row.get("passage_id") for row in records]
    if any(not isinstance(pid, str) or not pid.strip() for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Candidate passage IDs must be nonempty and unique.")
    check_is_fitted(pipeline)
    classes = list(pipeline.classes_)
    if classes != [0, 1]:
        raise ValueError("The reranker must be fitted to binary labels 0 and 1.")
    probabilities = pipeline.predict_proba(feature_matrix(records))[:, classes.index(1)]
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite reranker probabilities.")
    ranked = [{"passage_id": row["passage_id"], "score": float(probability),
               "relevance_probability": float(probability)}
              for row, probability in zip(records, probabilities)]
    return sorted(ranked, key=lambda row: (-row["score"], row["passage_id"]))


class LogisticReranker:
    def __init__(self, pipeline, extractor=None, metadata=None):
        check_is_fitted(pipeline)
        if not isinstance(pipeline, Pipeline) or tuple(pipeline.named_steps) != ("scaler", "logistic"):
            raise ValueError("Expected the StandardScaler/LogisticRegression pipeline.")
        if pipeline.n_features_in_ != len(FEATURE_COLUMNS):
            raise ValueError("Model feature count does not match the feature allowlist.")
        self.pipeline = pipeline
        self.extractor = extractor
        self.metadata = metadata or {}

    def _feature_extractor(self):
        if self.extractor is None:
            from src.bm25 import BM25Retriever, CORPUS_PATH
            from src.reranker_features import RerankerFeatureExtractor
            from src.semantic import SemanticRetriever
            corpus_path = Path(self.metadata.get("corpus_path", CORPUS_PATH))
            expected = self.metadata.get("corpus_sha256")
            if expected and sha256(corpus_path.read_bytes()).hexdigest() != expected:
                raise ValueError("Corpus differs from the trained reranker's corpus.")
            self.extractor = RerankerFeatureExtractor(
                BM25Retriever(corpus_path),
                SemanticRetriever(corpus_path, revision=self.metadata.get("model_revision"),
                                  local_files_only=self.metadata.get("local_files_only", False)),
            )
        return self.extractor

    def rerank(self, question, candidate_passages):
        """Return candidates ranked by predicted label-1 probability.

        Candidates may be corpus passage IDs or dictionaries with passage_id and
        optional text. Reference annotations are neither accepted nor required.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be nonempty.")
        candidates = list(candidate_passages)
        if not candidates:
            return []
        extractor = self._feature_extractor()
        ids = []
        for candidate in candidates:
            passage_id = candidate if isinstance(candidate, str) else candidate.get("passage_id") if isinstance(candidate, dict) else None
            if not isinstance(passage_id, str) or passage_id not in extractor.passages:
                raise ValueError("Candidate does not reference a corpus passage.")
            if isinstance(candidate, dict) and "text" in candidate and candidate["text"] != extractor.passages[passage_id]:
                raise ValueError("Candidate text does not match its corpus passage ID.")
            ids.append(passage_id)
        features = extractor.extract(question, ids)
        ranked = rank_feature_rows(self.pipeline, features)
        return [{**row, "text": extractor.passages[row["passage_id"]]} for row in ranked]

    def save(self, path=MODEL_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "feature_columns": FEATURE_COLUMNS,
                     "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path=MODEL_PATH, extractor=None):
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or tuple(bundle.get("feature_columns", ())) != FEATURE_COLUMNS:
            raise ValueError("Saved model has incompatible feature ordering.")
        return cls(bundle["pipeline"], extractor=extractor, metadata=bundle.get("metadata"))


@lru_cache(maxsize=1)
def _default_reranker():
    return LogisticReranker.load()


def rerank(question, candidate_passages):
    return _default_reranker().rerank(question, candidate_passages)
