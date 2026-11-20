"""Fit on training pairs only; choose C by validation MRR@10.

Run: python -m experiments.train_reranker
Uses verified saved natural retrieval rankings/features. No model download,
new candidate mining, final test access, or LLM is needed.
"""

import argparse
from collections import Counter, defaultdict
import csv
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
from statistics import mean
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from experiments.compare_retrievers import file_hash, validate_inputs
from src.bm25 import CORPUS_PATH, read_jsonl
from src.evaluation import evaluate_retrieval, reciprocal_rank_at_k
from src.reranker import MODEL_PATH, LogisticReranker, feature_matrix, load_feature_rows, make_pipeline, rank_feature_rows
from src.reranker_features import FEATURE_COLUMNS, candidate_union, validate_feature_rows, validate_split_isolation


PROCESSED_DIR = CORPUS_PATH.parent
RESULTS_DIR = PROCESSED_DIR.parents[1] / "results"
C_VALUES = (0.01, 0.1, 1.0, 10.0)


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["question_id"]].append(row)
    return dict(grouped)


def ranking_metrics(questions, rankings):
    """Reuse existing Hit@K and reciprocal-rank implementations, averaging over all questions."""
    question_ids = {row["question"]: row["question_id"] for row in questions}
    if len(question_ids) != len(questions) or set(rankings) != {q["question_id"] for q in questions}:
        raise ValueError("Question rankings are missing, duplicated, or misaligned.")

    def retrieve(question, top_k):
        return rankings[question_ids[question]][:top_k]

    table, details = evaluate_retrieval(questions, retrieve)
    metrics = {f"Hit@{row['k']}": row["hit_at_k"] for row in table}
    metrics["MRR@10"] = mean(reciprocal_rank_at_k(row["retrieved"], row["relevant_passage_id"], 10)
                             for row in details)
    return metrics, details


def tune_pipeline(train_rows, val_rows, val_questions):
    """Each scaler and classifier is fitted on train only; validation is prediction-only."""
    X = feature_matrix(train_rows)
    y = np.array([row["label"] for row in train_rows], dtype=int)
    distribution = Counter(y.tolist())
    if set(distribution) != {0, 1}:
        raise ValueError("Training requires both binary classes.")
    # sklearn's balanced weighting uses training count n/(2*n_class).
    weights = {label: len(y) / (2 * count) for label, count in sorted(distribution.items())}
    print(f"Training label counts: {dict(sorted(distribution.items()))}; negatives per positive: {distribution[0] / distribution[1]:.3f}")
    print(f"Using class_weight='balanced'; effective training weights: {weights}")
    grouped = group_rows(val_rows)
    best_model, best_metrics, best_rankings, best_c = None, None, None, None
    tuning = []
    for C in C_VALUES:
        pipeline = make_pipeline(C=C, class_weight="balanced")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                pipeline.fit(X, y)
            except ConvergenceWarning as exc:
                raise ValueError(f"Logistic regression failed to converge for C={C}.") from exc
        rankings = {qid: rank_feature_rows(pipeline, records) for qid, records in grouped.items()}
        metrics, _ = ranking_metrics(val_questions, rankings)
        tuning.append({"C": C, **metrics})
        print(f"C={C:g}: validation MRR@10={metrics['MRR@10']:.6f}, Hit@1={metrics['Hit@1']:.4f}")
        # C_VALUES is ascending: ties retain the smaller, more regularized C.
        if best_metrics is None or metrics["MRR@10"] > best_metrics["MRR@10"]:
            best_model, best_metrics, best_rankings, best_c = pipeline, metrics, rankings, C
    # Deliberately retain the train-only fitted model; never refit on train+val.
    return best_model, best_c, best_metrics, best_rankings, tuning, weights


def natural_baselines(questions, val_rows, audits):
    grouped = {qid: {row["passage_id"]: row for row in rows} for qid, rows in group_rows(val_rows).items()}
    rankings = {name: {} for name in ("bm25", "semantic", "candidate_union_order")}
    for entry in audits:
        qid = entry["question_id"]
        for name, feature in (("bm25", "bm25_score"), ("semantic", "semantic_similarity")):
            rankings[name][qid] = [{"passage_id": pid, "score": grouped[qid][pid][feature]}
                                   for pid in entry[f"{name}_passage_ids"]]
        union = candidate_union(rankings["bm25"][qid], rankings["semantic"][qid])
        rankings["candidate_union_order"][qid] = [
            {"passage_id": pid, "score": float(len(union) - i)} for i, pid in enumerate(union)]
    metrics = {name: ranking_metrics(questions, values)[0] for name, values in rankings.items()}
    hits = sum(question["relevant_passage_id"] in {row["passage_id"] for row in rankings["candidate_union_order"][question["question_id"]]}
               for question in questions)
    pool = {
        "bm25_top10_hit": metrics["bm25"]["Hit@10"],
        "semantic_top10_hit": metrics["semantic"]["Hit@10"],
        "union_annotated_source_coverage": hits / len(questions),
        "questions_missing_annotated_source": len(questions) - hits,
        "mean_candidate_count": mean(len(rows) for rows in grouped.values()),
        "min_candidate_count": min(len(rows) for rows in grouped.values()),
        "max_candidate_count": max(len(rows) for rows in grouped.values()),
        "before_reranking_hit_at_k": metrics["candidate_union_order"],
        "union_order": "BM25 top10 in order, then unseen semantic top10 IDs; no positive insertion.",
        "interpretation": "Whole-pool coverage is the ranking ceiling for this single annotated source. It is not exhaustive relevance recall, and union coverage is not the same as union Hit@10.",
    }
    return {name: metrics[name] for name in ("bm25", "semantic")}, pool


def coefficient_rows(pipeline):
    scaler, logistic = pipeline.named_steps["scaler"], pipeline.named_steps["logistic"]
    return [{"feature": feature, "coefficient_standardized": float(beta),
             "odds_multiplier_per_training_sd": math.exp(float(beta)),
             "training_mean": float(center), "training_scale": float(scale)}
            for feature, beta, center, scale in zip(FEATURE_COLUMNS, logistic.coef_[0], scaler.mean_, scaler.scale_)]


def save_comparison_chart(metrics, path):
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    labels = ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    for offset, name, label, color in [(-0.25, "bm25", "BM25", "#3366aa"),
                                       (0, "semantic", "Semantic", "#d87525"),
                                       (0.25, "reranker", "Logistic reranker", "#33865b")]:
        values = [metrics[name][key] for key in labels]
        bars = ax.bar(x + offset, values, width=0.25, label=label, color=color)
        ax.bar_label(bars, labels=[f"{value:.0%}" if key.startswith("Hit") else f"{value:.1%}"
                                  for key, value in zip(labels, values)], padding=3, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Validation score")
    ax.set_title("Validation retrieval and reranking comparison (50 questions)")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.015, "Single annotated source; natural candidate unions only. C selected on this validation set.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train_reranker(processed_dir=PROCESSED_DIR, results_dir=RESULTS_DIR, model_path=MODEL_PATH):
    processed_dir, results_dir = Path(processed_dir), Path(results_dir)
    paths = {name: processed_dir / name for name in
             ("train.jsonl", "val.jsonl", "corpus.jsonl", "reranker_train.csv", "reranker_val.csv")}
    statistics_path = results_dir / "reranker_dataset_statistics.json"
    paths["preparation_statistics"] = statistics_path
    hashes = {name: file_hash(path) for name, path in paths.items()}
    preparation = json.loads(statistics_path.read_text(encoding="utf-8"))
    if tuple(preparation.get("feature_columns", ())) != FEATURE_COLUMNS:
        raise ValueError("Preparation feature schema is incompatible.")
    for name in ("train", "val", "corpus"):
        if hashes[f"{name}.jsonl"] != preparation["input_sha256"].get(name):
            raise ValueError(f"Source {name} differs from feature-preparation inputs.")
    for name in ("train", "val"):
        if hashes[f"reranker_{name}.csv"] != preparation["output_sha256"].get(name):
            raise ValueError(f"{name} feature CSV differs from audited preparation output.")
    questions = {name: read_jsonl(paths[f"{name}.jsonl"]) for name in ("train", "val")}
    corpus = read_jsonl(paths["corpus.jsonl"])
    passages = validate_inputs(corpus, questions["train"], expected_count=500)
    validate_inputs(corpus, questions["val"], expected_count=50)
    validate_split_isolation(questions["train"], questions["val"])
    rows = {name: load_feature_rows(paths[f"reranker_{name}.csv"]) for name in ("train", "val")}
    audits = preparation["candidate_audit"]
    for name in ("train", "val"):
        validate_feature_rows(rows[name], questions[name], passages, name, audits[name])
        if name == "val" and any(row["force_included_positive"] for row in rows[name]):
            raise ValueError("Validation contains force-inserted positives.")
    np.random.seed(42)
    pipeline, C, reranker_metrics, rankings, tuning, weights = tune_pipeline(rows["train"], rows["val"], questions["val"])
    baseline_metrics, pool = natural_baselines(questions["val"], rows["val"], audits["val"])
    metrics = {**baseline_metrics, "reranker": reranker_metrics}
    if hashes != {name: file_hash(path) for name, path in paths.items()}:
        raise ValueError("Input artifacts changed during training/evaluation.")
    coefficients = coefficient_rows(pipeline)
    metadata = {"selected_C": C, "seed": 42, "class_weight": "balanced",
                "corpus_path": str(paths["corpus.jsonl"].resolve()), "corpus_sha256": hashes["corpus.jsonl"],
                "model_revision": preparation["model_revision"], "input_sha256": hashes}
    reranker = LogisticReranker(pipeline, metadata=metadata)
    reranker.save(model_path)
    # Predictions must remain identical after serialization.
    restored = LogisticReranker.load(model_path)
    first_group = next(iter(group_rows(rows["val"]).values()))
    if rank_feature_rows(pipeline, first_group) != rank_feature_rows(restored.pipeline, first_group):
        raise ValueError("Model serialization changed the ranking.")
    _, details = ranking_metrics(questions["val"], rankings)
    report = {
        "split": "validation", "training_questions": len(questions["train"]), "validation_questions": len(questions["val"]),
        "training_pairs": len(rows["train"]), "validation_pairs": len(rows["val"]),
        "training_class_counts": dict(sorted(Counter(row["label"] for row in rows["train"]).items())),
        "effective_class_weights": weights, "class_weight": "balanced", "selected_C": C,
        "selection_metric": "validation MRR@10; ties choose smaller C", "tuning": tuning,
        "metrics": metrics, "candidate_pool": pool, "coefficients": coefficients,
        "intercept_standardized": float(pipeline.named_steps["logistic"].intercept_[0]),
        "validation_rankings": details, "model_metadata": metadata,
        "runtime_versions": {name: version(name) for name in ("numpy", "scikit-learn", "joblib")},
        "protocol": {
            "fit": "Each scaler and logistic regression fit uses reranker_train.csv only. No train+validation refit and no test access.",
            "candidate_provenance": "Reused checksum-verified feature-preparation audit: BM25 and semantic top10 on identical validation questions and corpus. Validation pool is their natural union; no positive insertion.",
            "training_forced_positives": sum(row["force_included_positive"] for row in rows["train"]),
            "features": list(FEATURE_COLUMNS),
            "labels": "Natural Questions single-source annotations can be incomplete. Label 0 means not annotated as relevant, not guaranteed irrelevant.",
            "probabilities": "Class-weighted label-1 probabilities are ranking scores; not calibrated factual relevance probabilities.",
            "coefficients": "One training-standard-deviation increase changes weighted label-1 log odds by beta, holding other standardized inputs fixed. exp(beta) is the odds multiplier, not a causal effect.",
            "scope": "Controlled answer-bearing prototype. Validation metrics were used to choose C and are not a held-out final performance estimate.",
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "reranker_val_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (results_dir / "reranker_feature_coefficients.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coefficients[0]))
        writer.writeheader()
        writer.writerows(coefficients)
    save_comparison_chart(metrics, results_dir / "reranker_comparison.png")
    print(f"\nSelected C={C:g} by validation MRR@10. Pipeline remains fitted on training pairs only.")
    print("| Metric | BM25 | Semantic | Logistic reranker |\n|---|---:|---:|---:|")
    for metric in ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10"):
        print(f"| {metric} | " + " | ".join(f"{metrics[name][metric]:.4f}" for name in ("bm25", "semantic", "reranker")) + " |")
    print("\nNatural candidate pool:", json.dumps(pool, indent=2))
    print("\nCoefficients on standardized features:")
    for row in coefficients:
        print(f"{row['feature']}: beta={row['coefficient_standardized']:.6f}; odds multiplier per training SD={row['odds_multiplier_per_training_sd']:.4f}")
    print("Positive beta increases predicted label-1 log odds; negative beta decreases it, holding other inputs fixed.")
    print("Labels are incomplete annotations; class-weighted probabilities are not calibrated factual relevance.")
    print(f"Saved model: {model_path}\nSaved metrics, coefficients and comparison figure: {results_dir}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    try:
        train_reranker(args.processed_dir, args.results_dir, args.model_path)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Reranker training failed: {exc}\n")


if __name__ == "__main__":
    main()
