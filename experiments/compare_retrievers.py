"""Compare fixed BM25 and semantic retrieval on the 50 validation questions.

Run from the project root: python -m experiments.compare_retrievers
No final test questions are loaded. Rank-based wins refer to the single annotated
source passage; a miss outside the top 10 is worse than any recovered rank.
"""

import argparse
import csv
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from src.bm25 import BM25Retriever, CORPUS_PATH, read_jsonl
from src.dataset import normalize_question, prepare_answers
from src.evaluation import DEFAULT_KS, evaluate_retrieval, reciprocal_rank_at_k
from src.semantic import MODEL_NAME, SemanticRetriever


VAL_PATH = CORPUS_PATH.with_name("val.jsonl")
RESULTS_DIR = CORPUS_PATH.parents[2] / "results"
SEED = 42
TOP_K = 10
# Exact snapshot inspected in the existing model cache, for repeatable experiments.
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def validate_inputs(corpus, questions, expected_count=50):
    """Fail explicitly on unexpected schemas, duplicate passages, or bad labels."""
    if not corpus:
        raise ValueError("Corpus is empty.")
    passages, texts = {}, set()
    for record in corpus:
        passage_id, text = record.get("passage_id"), record.get("text")
        if not isinstance(passage_id, str) or not passage_id.strip():
            raise ValueError("Corpus records need a nonempty passage_id.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty passage text: {passage_id}")
        if passage_id in passages or text in texts:
            raise ValueError(f"Duplicate passage ID or text: {passage_id}")
        passages[passage_id] = text
        texts.add(text)
    if len(questions) != expected_count:
        raise ValueError(f"Expected {expected_count} validation questions, found {len(questions)}.")
    seen_ids, seen_questions = set(), set()
    for record in questions:
        if not {"question_id", "question", "answers", "relevant_passage_id"}.issubset(record):
            raise ValueError("Unexpected question schema: missing required fields.")
        question_id, question = record["question_id"], record["question"]
        if not isinstance(question_id, str) or not question_id.strip() or question_id in seen_ids:
            raise ValueError("Invalid or duplicate validation question ID.")
        if not isinstance(question, str) or not normalize_question(question):
            raise ValueError(f"Empty validation question: {question_id}")
        normalized = normalize_question(question)
        if normalized in seen_questions:
            raise ValueError(f"Duplicate validation question: {question_id}")
        passage_id = record["relevant_passage_id"]
        if not isinstance(passage_id, str) or passage_id not in passages:
            raise ValueError(f"Invalid relevant passage mapping: {question_id}")
        if prepare_answers(record["answers"], passages[passage_id]) is None:
            raise ValueError(f"Invalid answer annotations: {question_id}")
        seen_ids.add(question_id)
        seen_questions.add(normalized)
    return passages


def check_rankings(results, passages, top_k):
    if not isinstance(results, list) or len(results) != min(top_k, len(passages)):
        raise ValueError("Retriever returned an unexpected number of passages.")
    seen, previous = set(), math.inf
    for result in results:
        if not isinstance(result, dict) or not {"passage_id", "score", "text"}.issubset(result):
            raise ValueError("Unexpected retrieval result schema.")
        passage_id, score = result["passage_id"], result["score"]
        if not isinstance(passage_id, str) or passage_id not in passages or passage_id in seen:
            raise ValueError("Retriever returned an unknown or duplicate passage ID.")
        if result["text"] != passages[passage_id]:
            raise ValueError("Retrieved passage text does not match its corpus ID.")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score > previous:
            raise ValueError("Retrieval scores must be finite and in descending order.")
        previous = score
        seen.add(passage_id)


def evaluate_method(questions, retriever, passages):
    """Reuse Hit@K evaluation with a timed query-only adapter; supports future rerankers."""
    latencies = []

    def timed_retrieve(question, top_k):
        started = perf_counter()
        results = retriever.retrieve(question=question, top_k=top_k)
        elapsed_ms = (perf_counter() - started) * 1000
        check_rankings(results, passages, top_k)
        latencies.append(elapsed_ms)
        return results

    table, details = evaluate_retrieval(questions, timed_retrieve, ks=DEFAULT_KS)
    metrics = {f"Hit@{row['k']}": row["hit_at_k"] for row in table}
    metrics["MRR@10"] = mean(reciprocal_rank_at_k(row["retrieved"], row["relevant_passage_id"], TOP_K)
                             for row in details)
    metrics["avg_query_latency_ms"] = mean(latencies)
    for detail, latency in zip(details, latencies):
        detail["latency_ms"] = latency
        detail["relevant_rank"] = next((rank for rank, result in enumerate(detail["retrieved"], 1)
                                        if result["passage_id"] == detail["relevant_passage_id"]), None)
    return metrics, details


def combine_results(questions, method_details):
    combined = []
    by_method = {name: {row["question_id"]: row for row in details}
                 for name, details in method_details.items()}
    for question in questions:
        record = {key: question[key] for key in ("question_id", "question", "relevant_passage_id")}
        for name, details in by_method.items():
            detail = details[question["question_id"]]
            record[name] = {"ranked_passages": detail["retrieved"], "latency_ms": detail["latency_ms"],
                            "relevant_rank": detail["relevant_rank"]}
        combined.append(record)
    return combined


def print_wins(records, winner, loser, limit=5):
    # A lower recovered rank wins; None means the source was not in the top 10.
    wins = [row for row in records
            if (row[winner]["relevant_rank"] or math.inf) < (row[loser]["relevant_rank"] or math.inf)]
    print(f"\n{winner.title()} outperforms {loser.title()}: {len(wins)} questions; showing up to {limit}")
    for row in wins[:limit]:
        ranks = ", ".join(f"{name}: {row[name]['relevant_rank'] or 'not in top 10'}" for name in (winner, loser))
        print(f"- {row['question_id']} | {row['question']} | {ranks}")
    if not wins:
        print("No such questions.")


def save_chart(method_metrics, path):
    # Keep Matplotlib's font cache in the writable, ignored results directory.
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DEFAULT_KS))
    for offset, name, label, color in [(-0.19, "bm25", "BM25", "#3366AA"),
                                       (0.19, "semantic", "Semantic", "#D87525")]:
        values = [method_metrics[name][f"Hit@{k}"] for k in DEFAULT_KS]
        bars = ax.bar(x + offset, values, width=0.38, label=label, color=color)
        ax.bar_label(bars, labels=[f"{value:.0%}" for value in values], padding=3)
    ax.set_xticks(x, [f"Hit@{k}" for k in DEFAULT_KS])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Fraction of validation questions")
    ax.set_title("Annotated-source recovery on 50 validation questions")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.015, "Fixed answer-bearing prototype corpus; one annotated source per question.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_results(output_dir, records, report):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval_comparison_val.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records), encoding="utf-8")
    fieldnames = ["question_id", "question", "relevant_passage_id"]
    for name in ("bm25", "semantic"):
        fieldnames.extend(f"{name}_{suffix}" for suffix in ("ranked_passage_ids", "scores", "latency_ms", "relevant_rank"))
    with (output_dir / "retrieval_comparison_val.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            flat = {key: record[key] for key in ("question_id", "question", "relevant_passage_id")}
            for name in ("bm25", "semantic"):
                results = record[name]["ranked_passages"]
                flat[f"{name}_ranked_passage_ids"] = json.dumps([row["passage_id"] for row in results])
                flat[f"{name}_scores"] = json.dumps([row["score"] for row in results], allow_nan=False)
                flat[f"{name}_latency_ms"] = record[name]["latency_ms"]
                flat[f"{name}_relevant_rank"] = record[name]["relevant_rank"]
            writer.writerow(flat)
    (output_dir / "retrieval_metrics_val.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_chart(report["metrics"], output_dir / "retrieval_comparison.png")


def run_comparison(corpus_path=CORPUS_PATH, output_dir=RESULTS_DIR, seed=SEED, local_files_only=False):
    # Validation path is fixed deliberately; this experiment never opens test.jsonl.
    input_hashes = {"corpus": file_hash(corpus_path), "validation": file_hash(VAL_PATH)}
    corpus, questions = read_jsonl(corpus_path), read_jsonl(VAL_PATH)
    passages = validate_inputs(corpus, questions)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    setup_ms, retrievers = {}, {}
    started = perf_counter()
    retrievers["bm25"] = BM25Retriever(corpus_path)
    setup_ms["bm25"] = {"total": (perf_counter() - started) * 1000}
    started = perf_counter()
    retrievers["semantic"] = SemanticRetriever(corpus_path, local_files_only=local_files_only,
                                               revision=MODEL_REVISION)
    setup_ms["semantic"] = {"total": (perf_counter() - started) * 1000,
                            **retrievers["semantic"].setup_timings_ms}
    for name, retriever in retrievers.items():
        actual = {row["passage_id"]: row["text"] for row in retriever.passages}
        if actual != passages or len(retriever.passages) != len(passages):
            raise ValueError(f"{name} is not using the exact shared corpus.")
    # The same label-free warm-up query is excluded from average query latency.
    warmup_ms = {}
    for name, retriever in retrievers.items():
        started = perf_counter()
        results = retriever.retrieve(question="What is the history of science?", top_k=TOP_K)
        warmup_ms[name] = (perf_counter() - started) * 1000
        check_rankings(results, passages, TOP_K)
    metrics, details = {}, {}
    for name, retriever in retrievers.items():
        metrics[name], details[name] = evaluate_method(questions, retriever, passages)
    records = combine_results(questions, details)
    if input_hashes != {"corpus": file_hash(corpus_path), "validation": file_hash(VAL_PATH)}:
        raise ValueError("The corpus or validation questions changed during evaluation.")
    report = {
        "split": "validation", "question_count": len(questions), "corpus_size": len(passages),
        "seed": seed, "top_k": TOP_K, "metrics": metrics, "setup_ms": setup_ms,
        "warmup_ms": warmup_ms, "input_sha256": input_hashes,
        "corpus_path": str(Path(corpus_path).resolve()), "questions_path": str(VAL_PATH.resolve()),
        "model": MODEL_NAME, "model_commit": MODEL_REVISION,
        "model_max_sequence_length": retrievers["semantic"].model.max_seq_length,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "device": "cpu", "torch_threads": torch.get_num_threads(),
                    "versions": {name: version(name) for name in
                                 ("numpy", "torch", "rank-bm25", "faiss-cpu", "sentence-transformers")}},
        "protocol": {
            "relevance": "Single annotated source passage; Hit@K is not exhaustive relevance recall.",
            "corpus": "Controlled answer-bearing prototype; identical fixed corpus for both methods.",
            "latency": "perf_counter around retrieve only; includes query encoding/scoring and result assembly. Setup, warm-up and metric checks excluded. Warm-cache single sequential run: BM25 then semantic.",
            "setup": "Constructor costs; imports excluded. Semantic model loading, corpus embedding, and index construction are broken out.",
            "scores": "BM25 scores and cosine similarities use different scales and are not compared.",
            "wins": "Lower annotated passage rank within top 10 wins; both misses tie. null rank means not retrieved.",
            "reproducibility": "Seeded CPU inference with deterministic algorithms; rankings and metrics should repeat in the recorded environment. Wall-clock latencies vary between runs.",
            "truncation": "Semantic model truncates contexts to its token limit; BM25 indexes complete context text.",
        },
    }
    save_results(Path(output_dir), records, report)
    print(f"Validation only: {len(questions)} questions, {len(passages)} shared corpus passages")
    print("| Metric | BM25 | Semantic |\n|---|---:|---:|")
    for label in ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10", "avg_query_latency_ms"]:
        values = [metrics[name][label] for name in ("bm25", "semantic")]
        formatted = [f"{value:.3f}" if label == "avg_query_latency_ms" else f"{value:.4f}" for value in values]
        title = "Avg. query latency (ms)" if label == "avg_query_latency_ms" else label
        print(f"| {title} | {' | '.join(formatted)} |")
    print("\nSetup costs (ms), excluded from query latency:", json.dumps(setup_ms))
    print("Hit@K measures annotated-source recovery, not exhaustive relevance recall.")
    for winner, loser in [("semantic", "bm25"), ("bm25", "semantic")]:
        print_wins(records, winner, loser)
    print(f"\nSaved comparison artifacts to {Path(output_dir).resolve()}")
    return report, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--local-files-only", action="store_true", help="Use the previously downloaded model")
    args = parser.parse_args()
    try:
        run_comparison(args.corpus, args.output_dir, args.seed, args.local_files_only)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Retrieval comparison failed: {exc}\n")


if __name__ == "__main__":
    main()
