"""Prepare supervised prototype reranker features; never fit a model or open test labels.

Run: python -m experiments.prepare_reranker_data --local-files-only
"""

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch

from experiments.compare_retrievers import MODEL_REVISION, SEED, file_hash, validate_inputs
from src.bm25 import BM25Retriever, CORPUS_PATH, read_jsonl
from src.reranker_features import (
    FEATURE_COLUMNS, ROW_COLUMNS, RerankerFeatureExtractor, build_feature_rows,
    dataset_statistics, validate_feature_rows, validate_split_isolation,
)
from src.semantic import MODEL_NAME, SemanticRetriever


PROCESSED_DIR = CORPUS_PATH.parent
RESULTS_DIR = PROCESSED_DIR.parents[1] / "results"


def write_and_verify_csv(path, rows, questions, passages, split, audit):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames) != ROW_COLUMNS:
            raise ValueError("Saved feature columns do not match expected order.")
        saved = list(reader)
    for row in saved:
        for column in FEATURE_COLUMNS:
            row[column] = int(row[column]) if column == "passage_length" else float(row[column])
        for column in ("label", "force_included_positive"):
            row[column] = int(row[column])
    validate_feature_rows(saved, questions, passages, split, audit)
    if saved != rows:
        raise ValueError("Feature records changed during CSV serialization.")


def prepare_reranker_data(processed_dir=PROCESSED_DIR, results_dir=RESULTS_DIR, local_files_only=False):
    processed_dir, results_dir = Path(processed_dir), Path(results_dir)
    paths = {name: processed_dir / f"{name}.jsonl" for name in ("train", "val", "corpus")}
    hashes = {name: file_hash(path) for name, path in paths.items()}
    data = {name: read_jsonl(path) for name, path in paths.items()}
    # These checks inspect annotations only for data quality, never for features.
    passages = validate_inputs(data["corpus"], data["train"], expected_count=500)
    validate_inputs(data["corpus"], data["val"], expected_count=50)
    validate_split_isolation(data["train"], data["val"])
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    bm25 = BM25Retriever(paths["corpus"])
    semantic = SemanticRetriever(paths["corpus"], revision=MODEL_REVISION, local_files_only=local_files_only)
    extractor = RerankerFeatureExtractor(bm25, semantic)
    all_rows, audits, stats = {}, {}, {}
    for split in ("train", "val"):
        print(f"Preparing {split}: {len(data[split])} questions", flush=True)
        all_rows[split], audits[split] = build_feature_rows(data[split], extractor, split)
        stats[split] = dataset_statistics(all_rows[split], audits[split])
    if hashes != {name: file_hash(path) for name, path in paths.items()}:
        raise ValueError("Source data changed during preparation.")
    for split in ("train", "val"):
        write_and_verify_csv(processed_dir / f"reranker_{split}.csv", all_rows[split], data[split], passages, split, audits[split])
    report = {
        "seed": SEED, "top_k_per_retriever": 10, "feature_columns": list(FEATURE_COLUMNS),
        "csv_columns": list(ROW_COLUMNS), "statistics": stats,
        "corpus_size": len(passages), "model": MODEL_NAME, "model_revision": MODEL_REVISION,
        "input_sha256": hashes,
        "output_sha256": {split: file_hash(processed_dir / f"reranker_{split}.csv") for split in ("train", "val")},
        "protocol": {
            "positive": "The single annotated source passage.",
            "negative": "Label 0 means not annotated as relevant, not guaranteed irrelevant.",
            "features": "Only feature_columns enter the future classifier. IDs, label, candidate_source and force_included_positive are excluded.",
            "answers": "Reference answer text is never used to generate input features.",
            "candidate_source": "BM25, semantic, both, or forced_positive (training positive absent from both lists).",
            "force_inclusion": "Training only. Natural candidate hits are measured before force-inclusion; validation candidates are never augmented.",
            "embeddings": "Corpus vectors reconstructed once from the existing FAISS index; passages are not re-encoded for feature rows. Only queries are encoded.",
            "word_features": "Lowercase Unicode word tokens from the existing BM25 tokenizer. Overlap is intersection / unique query token count; empty query token sets yield 0.",
            "scope": "Fixed answer-bearing prototype with incomplete relevance annotations. No test labels loaded, no model fitted.",
            "truncation": "Semantic similarity retains the embedding model token limit; lexical features use full passages.",
        },
        "candidate_audit": audits,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "reranker_dataset_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\nDataset statistics:")
    print(json.dumps(stats, indent=2))
    for split in ("train", "val"):
        print(f"\nSample {split} feature rows (first 3):")
        for row in all_rows[split][:3]:
            print(json.dumps(row, ensure_ascii=False, allow_nan=False))
    print(f"\nFeatures saved to {processed_dir}/reranker_{{train,val}}.csv")
    print(f"Statistics and natural candidate audit saved to {results_dir / 'reranker_dataset_statistics.json'}")
    return report, all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    try:
        prepare_reranker_data(args.processed_dir, args.results_dir, args.local_files_only)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Reranker data preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
