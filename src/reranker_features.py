"""Label-free query/passage features and prototype candidate-pair preparation.

Only FEATURE_COLUMNS should be supplied to a future scaler/classifier. Labels,
source flags, IDs, and force-inclusion flags are bookkeeping, never ML inputs.
Label 0 means not annotated as relevant, not guaranteed factually irrelevant.
"""

from collections import Counter
import math

import numpy as np

from src.bm25 import tokenize
from src.dataset import normalize_question
from src.evaluation import hit_at_k


FEATURE_COLUMNS = ("bm25_score", "semantic_similarity", "word_overlap", "passage_length")
ROW_COLUMNS = ("question_id", "passage_id", *FEATURE_COLUMNS, "label",
               "candidate_source", "force_included_positive")
SOURCE_VALUES = {"BM25", "semantic", "both", "forced_positive"}


def word_overlap(query_tokens, passage_tokens):
    query = set(query_tokens)
    return len(query.intersection(passage_tokens)) / len(query) if query else 0.0


def candidate_union(bm25_results, semantic_results):
    """Preserve BM25 rank order, followed by unseen semantic candidates."""
    bm25_ids = [row["passage_id"] for row in bm25_results]
    semantic_ids = [row["passage_id"] for row in semantic_results]
    if len(set(bm25_ids)) != len(bm25_ids) or len(set(semantic_ids)) != len(semantic_ids):
        raise ValueError("A retrieval ranking contains duplicate passage IDs.")
    left, right = set(bm25_ids), set(semantic_ids)
    return {passage_id: "both" if passage_id in left and passage_id in right
            else "BM25" if passage_id in left else "semantic"
            for passage_id in dict.fromkeys(bm25_ids + semantic_ids)}


class RerankerFeatureExtractor:
    """Reuse the existing BM25 index and vectors already stored in FAISS.

    Full-context word features use the BM25 tokenizer. Semantic vectors retain
    the embedding model's existing token truncation behavior.
    """

    def __init__(self, bm25, semantic):
        self.bm25 = bm25
        self.semantic = semantic
        self.passages = {row["passage_id"]: row["text"] for row in bm25.passages}
        other = {row["passage_id"]: row["text"] for row in semantic.passages}
        if (self.passages != other or len(self.passages) != len(bm25.passages)
                or len(other) != len(semantic.passages)):
            raise ValueError("Retrievers must use identical, unique corpus passages.")
        if tuple(semantic.passage_ids) != tuple(row["passage_id"] for row in semantic.passages):
            raise ValueError("Semantic passage ID order does not match FAISS input order.")
        self.bm25_positions = {row["passage_id"]: i for i, row in enumerate(bm25.passages)}
        self.semantic_positions = {passage_id: i for i, passage_id in enumerate(semantic.passage_ids)}
        if semantic.index.ntotal != len(other):
            raise ValueError("FAISS corpus size does not match passage IDs.")
        # IndexFlatIP stores the original float32 passage vectors; no re-encoding.
        self.passage_embeddings = semantic.index.reconstruct_n(0, semantic.index.ntotal)
        self.passage_tokens = {passage_id: tokenize(text) for passage_id, text in self.passages.items()}

    def extract(self, question, passage_ids):
        """Extract features using only question text and candidate IDs, without labels."""
        if not isinstance(question, str) or not tokenize(question):
            raise ValueError("Question needs searchable tokens.")
        if len(set(passage_ids)) != len(passage_ids) or any(pid not in self.passages for pid in passage_ids):
            raise ValueError("Candidate IDs must be unique and present in the corpus.")
        query_tokens = tokenize(question)
        bm25_scores = self.bm25.index.get_scores(query_tokens)
        # Encode only the query once for all candidate feature records.
        query_embedding = self.semantic._encode([question])[0]
        similarities = self.passage_embeddings @ query_embedding
        return [{"passage_id": passage_id,
                 "bm25_score": float(bm25_scores[self.bm25_positions[passage_id]]),
                 "semantic_similarity": float(similarities[self.semantic_positions[passage_id]]),
                 "word_overlap": word_overlap(query_tokens, self.passage_tokens[passage_id]),
                 "passage_length": len(self.passage_tokens[passage_id])}
                for passage_id in passage_ids]


def build_feature_rows(questions, extractor, split, top_k=10):
    """Training includes missing positives; validation uses natural candidates only."""
    if split not in {"train", "val"}:
        raise ValueError("Only train and val splits are supported; no test preparation.")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be positive.")
    rows, audit = [], []
    for question in questions:
        text = question["question"]
        # Reference labels and answers are never arguments to either retriever.
        bm25_results = extractor.bm25.retrieve(question=text, top_k=top_k)
        semantic_results = extractor.semantic.retrieve(question=text, top_k=top_k)
        candidates = candidate_union(bm25_results, semantic_results)
        relevant_id = question["relevant_passage_id"]
        bm25_hit = hit_at_k(bm25_results, relevant_id, top_k)
        semantic_hit = hit_at_k(semantic_results, relevant_id, top_k)
        natural_hit = bool(bm25_hit or semantic_hit)
        forced = split == "train" and not natural_hit
        if forced:
            candidates[relevant_id] = "forced_positive"
        for features in extractor.extract(text, list(candidates)):
            passage_id = features["passage_id"]
            row = {"question_id": question["question_id"], **features,
                   "label": int(passage_id == relevant_id),
                   "candidate_source": candidates[passage_id],
                   "force_included_positive": int(forced and passage_id == relevant_id)}
            rows.append({column: row[column] for column in ROW_COLUMNS})
        # Preserve actual retrieval membership so force-insertion cannot inflate hits.
        audit.append({"question_id": question["question_id"],
                      "bm25_passage_ids": [row["passage_id"] for row in bm25_results],
                      "semantic_passage_ids": [row["passage_id"] for row in semantic_results],
                      "bm25_hit": bm25_hit, "semantic_hit": semantic_hit,
                      "natural_candidate_hit": int(natural_hit), "force_included_positive": int(forced)})
    validate_feature_rows(rows, questions, extractor.passages, split, audit)
    return rows, audit


def validate_feature_rows(rows, questions, passages, split, audit):
    """Check features and labels against natural candidate membership independently."""
    if split not in {"train", "val"}:
        raise ValueError("Unsupported split.")
    question_map = {q["question_id"]: q for q in questions}
    audits = {row["question_id"]: row for row in audit}
    if len(question_map) != len(questions) or set(audits) != set(question_map) or len(audits) != len(audit):
        raise ValueError("Question/audit IDs are duplicate or misaligned.")
    seen, counts, positives = set(), Counter(), Counter()
    for row in rows:
        if tuple(row) != ROW_COLUMNS:
            raise ValueError("Feature CSV schema or column order differs from the allowlist.")
        question_id, passage_id = row["question_id"], row["passage_id"]
        if question_id not in question_map or passage_id not in passages:
            raise ValueError("Feature pair references an unknown question or passage.")
        pair = (question_id, passage_id)
        if pair in seen:
            raise ValueError("Duplicate question-passage pair.")
        seen.add(pair)
        for column in FEATURE_COLUMNS:
            value = row[column]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Feature {column} must be numeric and finite.")
        if not 0 <= row["word_overlap"] <= 1 or row["passage_length"] < 0:
            raise ValueError("Invalid overlap or passage length.")
        if row["label"] not in (0, 1) or row["force_included_positive"] not in (0, 1):
            raise ValueError("Labels and force flags must be binary.")
        if row["candidate_source"] not in SOURCE_VALUES:
            raise ValueError("Unknown candidate source.")
        relevant_id = question_map[question_id]["relevant_passage_id"]
        if row["label"] != int(passage_id == relevant_id):
            raise ValueError("Feature label does not match source-passage annotation.")
        entry = audits[question_id]
        natural = candidate_union([{"passage_id": pid} for pid in entry["bm25_passage_ids"]],
                                  [{"passage_id": pid} for pid in entry["semantic_passage_ids"]])
        expected_force = int(split == "train" and passage_id == relevant_id and passage_id not in natural)
        expected_source = "forced_positive" if expected_force else natural.get(passage_id)
        if row["force_included_positive"] != expected_force or row["candidate_source"] != expected_source:
            raise ValueError("Candidate source or force-inclusion flag is incorrect.")
        counts[question_id] += 1
        positives[question_id] += row["label"]
    if set(counts) != set(question_map):
        raise ValueError("A question has no candidate pairs.")
    for question_id, entry in audits.items():
        relevant_id = question_map[question_id]["relevant_passage_id"]
        left, right = set(entry["bm25_passage_ids"]), set(entry["semantic_passage_ids"])
        natural = left | right
        actual_hit = int(relevant_id in natural)
        expected_force = int(split == "train" and not actual_hit)
        if (entry["natural_candidate_hit"] != actual_hit or entry["bm25_hit"] != int(relevant_id in left)
                or entry["semantic_hit"] != int(relevant_id in right)
                or entry["force_included_positive"] != expected_force):
            raise ValueError("Force-inclusion was incorrectly counted as retrieval success.")
        expected_ids = natural | ({relevant_id} if expected_force else set())
        actual_ids = {passage_id for qid, passage_id in seen if qid == question_id}
        if actual_ids != expected_ids or positives[question_id] != int(split == "train" or actual_hit):
            raise ValueError("Candidate union or positive coverage is incorrect.")
    if split == "train" and {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Training features require positive and negative examples.")


def validate_split_isolation(train, val):
    train_ids = {row["question_id"] for row in train}
    val_ids = {row["question_id"] for row in val}
    train_questions = {normalize_question(row["question"]) for row in train}
    val_questions = {normalize_question(row["question"]) for row in val}
    if train_ids & val_ids or train_questions & val_questions:
        raise ValueError("Training and validation questions overlap.")


def dataset_statistics(rows, audit):
    positives = sum(row["label"] for row in rows)
    negatives = len(rows) - positives
    return {
        "questions": len(audit), "question_passage_pairs": len(rows),
        "positive_labels": positives, "negative_labels": negatives,
        "positive_to_negative_ratio": positives / negatives if negatives else None,
        "force_included_positives": sum(row["force_included_positive"] for row in rows),
        "questions_missing_natural_positive": sum(not row["natural_candidate_hit"] for row in audit),
        "natural_union_hit_at_10": sum(row["natural_candidate_hit"] for row in audit) / len(audit) if audit else 0.0,
        "bm25_hit_at_10": sum(row["bm25_hit"] for row in audit) / len(audit) if audit else 0.0,
        "semantic_hit_at_10": sum(row["semantic_hit"] for row in audit) / len(audit) if audit else 0.0,
        "candidate_sources": dict(Counter(row["candidate_source"] for row in rows)),
    }
