"""Controlled validation-pilot and frozen test RAG evaluation.

Pilot: python -m experiments.evaluate_rag --phase pilot --local-files-only
Final: python -m experiments.evaluate_rag --phase test --local-files-only
Reports: python -m experiments.evaluate_rag --rescore --phase test
Plots: python -m experiments.evaluate_rag --plots-only
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import shutil
from statistics import mean
import csv
import os
import time

from experiments.generate_answers import RetrievalConditions
from src.answer_evaluation import SCORING_VERSION, normalize_answer, score_answer
from src.bm25 import read_jsonl
from src.dataset import prepare_answers, normalize_question
from src.evaluation import hit_at_k, reciprocal_rank_at_k
from src.generation import (AnswerGenerator, GenerationConfig, GENERATION_DIR, PROMPT_VERSION,
                            SYSTEM_PROMPT, build_messages, json_hash)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODEL = ROOT / "models" / "logistic_reranker.joblib"
KS = (3, 5, 10)
METHODS = ("bm25", "semantic", "reranker")
CONFIGURATIONS = [("llm_only", 0)] + [(method, k) for k in KS for method in METHODS]
EXPERIMENT_VERSION = "cs391-controlled-rag-v1"


def file_hash(path):
    from hashlib import sha256
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records), encoding="utf-8")
    temporary.replace(path)


def frozen_spec(config, test_size, seed):
    if type(test_size) is not int or not 1 <= test_size <= 150:
        raise ValueError("test_size must be between 1 and 150, fixed before test performance is seen.")
    code_paths = ["src/generation.py", "src/answer_evaluation.py", "src/bm25.py", "src/semantic.py",
                  "src/reranker.py", "src/reranker_features.py", "experiments/generate_answers.py",
                  "experiments/evaluate_rag.py", "src/evaluation.py"]
    return {
        "experiment_version": EXPERIMENT_VERSION, "settings": config.public_settings(),
        "prompt_version": PROMPT_VERSION, "system_prompt_sha256": json_hash(SYSTEM_PROMPT),
        "scoring_version": SCORING_VERSION, "seed": seed, "test_size": test_size,
        "pilot_size": 5, "ks": list(KS), "configurations": [list(pair) for pair in CONFIGURATIONS],
        "corpus_sha256": file_hash(DATA / "corpus.jsonl"), "reranker_sha256": file_hash(MODEL),
        "validation_sha256": file_hash(DATA / "val.jsonl"),
        "code_sha256": {str(path): file_hash(ROOT / path) for path in code_paths},
        "pricing_usd_per_million": {"input": config.input_usd_per_million, "output": config.output_usd_per_million},
    }


def load_questions(phase, test_size=150, seed=42):
    path = DATA / ("val.jsonl" if phase == "pilot" else "test.jsonl")
    rows = read_jsonl(path)
    expected = 50 if phase == "pilot" else 150
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} {phase} questions, found {len(rows)}.")
    corpus = read_jsonl(DATA / "corpus.jsonl")
    passages = {}
    for row in corpus:
        pid, text = row.get("passage_id"), row.get("text")
        if not isinstance(pid, str) or not isinstance(text, str) or not pid.strip() or not text.strip() or pid in passages:
            raise ValueError("Invalid/duplicate corpus passage.")
        passages[pid] = text
    if len(set(passages.values())) != len(passages):
        raise ValueError("Duplicate context text in shared corpus.")
    ids, normalized = set(), set()
    for row in rows:
        qid, question, pid = row.get("question_id"), row.get("question"), row.get("relevant_passage_id")
        if not isinstance(qid, str) or not qid.strip() or qid in ids or not isinstance(question, str) or not normalize_question(question):
            raise ValueError("Invalid or duplicate question schema.")
        if normalize_question(question) in normalized or not isinstance(pid, str) or pid not in passages:
            raise ValueError("Duplicate question or invalid relevant passage mapping.")
        if prepare_answers(row.get("answers"), passages[pid]) is None:
            raise ValueError("Invalid reference answer annotations.")
        ids.add(qid)
        normalized.add(normalize_question(question))
    if phase == "pilot":
        selected = rows[:5]
    elif test_size == 150:
        selected = rows
    else:
        # Sorting before sampling makes selection independent of file row order.
        selected = random.Random(seed).sample(sorted(rows, key=lambda r: r["question_id"]), test_size)
    return selected, passages, path


def validate_prompt(record, question, passages):
    ids = record["retrieved_passage_ids"]
    if len(ids) != len(set(ids)) or any(pid not in passages for pid in ids):
        raise ValueError("Prompt has duplicate or unknown passage IDs.")
    if record["retrieval_method"] == "llm_only" and ids:
        raise ValueError("The LLM-only baseline must not receive any passages.")
    ranking = record.get("retrieval_ranking")
    if ranking is not None:
        ranking_ids = [row["passage_id"] for row in ranking]
        if (len(ranking_ids) != len(set(ranking_ids)) or any(pid not in passages for pid in ranking_ids)
                or ids != ranking_ids[:record["top_k"]]):
            raise ValueError("Supplied sources do not match the ranked passage prefix.")
        scores = [row["score"] for row in ranking]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in scores):
            raise ValueError("Retrieval scores must be numeric and finite.")
        if scores != sorted(scores, reverse=True):
            raise ValueError("Retrieval ranking scores are not descending.")
        if record["retrieval_method"] == "reranker" and not set(ranking_ids).issubset(record.get("candidate_passage_ids", [])):
            raise ValueError("Reranker introduced a passage outside its natural candidate union.")
    expected = None if record["retrieval_method"] == "llm_only" else [
        {"passage_id": pid, "text": passages[pid]} for pid in ids]
    if record["messages"] != build_messages(question["question"], expected):
        raise ValueError("Prompt contains unexpected fields, reordered text, or leaked references.")
    if record["question_id"] != question["question_id"] or record["question"] != question["question"]:
        raise ValueError("Generated answer is associated with a different question.")


def score_records(records, questions, passages):
    question_map = {q["question_id"]: q for q in questions}
    expected = {(q["question_id"], method, k) for q in questions for method, k in CONFIGURATIONS}
    seen, scored = set(), []
    for raw in records:
        key = (raw["question_id"], raw["retrieval_method"], raw["top_k"])
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected question/configuration result.")
        seen.add(key)
        if raw["status"] not in {"completed", "error", "not_attempted", "dry_run"}:
            raise ValueError("Unexpected generation result status.")
        question = question_map[raw["question_id"]]
        if raw.get("messages") is not None:
            validate_prompt(raw, question, passages)
        refs = question["answers"]["text"]
        is_dry = raw["status"] == "dry_run"
        answer = raw.get("generated_answer") if raw["status"] == "completed" else None
        scores = score_answer(answer, refs)
        if is_dry:
            scores["exact_match"], scores["f1"] = None, None
        ranking = raw.get("retrieval_ranking")
        relevant_id = question["relevant_passage_id"]
        retrieval_metrics = ({f"Hit@{k}": hit_at_k(ranking, relevant_id, k) for k in (1, 3, 5, 10)}
                             if ranking is not None else {})
        if ranking is not None:
            retrieval_metrics["MRR@10"] = reciprocal_rank_at_k(ranking, relevant_id, 10)
        rank = next((i for i, row in enumerate(ranking or [], 1) if row["passage_id"] == relevant_id), None)
        scored.append({**raw, **scores, "references": refs, "relevant_passage_id": relevant_id,
                       "annotated_source_rank": rank, "retrieval_metrics": retrieval_metrics,
                       "context_source_hit": int(relevant_id in raw["retrieved_passage_ids"]) if ranking is not None else None,
                       "candidate_pool_source_hit": int(relevant_id in raw.get("candidate_passage_ids", []))
                       if raw["retrieval_method"] == "reranker" and ranking is not None else None})
    if seen != expected:
        raise ValueError("Missing question/configuration records; failures must be explicit, not omitted.")
    return scored


def average_known(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def aggregate_results(records, phase):
    groups = defaultdict(list)
    for row in records:
        groups[(row["retrieval_method"], row["top_k"])].append(row)
    table = []
    for method, k in CONFIGURATIONS:
        rows = groups[(method, k)]
        completed = [r for r in rows if r["status"] == "completed"]
        dry = all(r["status"] == "dry_run" for r in rows)
        new = [r for r in rows if not r.get("cache_hit")]
        known_cost = [r["estimated_cost_usd"] for r in completed if r.get("estimated_cost_usd") is not None]
        item = {"phase": phase, "method": method, "k": k, "questions": len(rows),
                "completed": len(completed), "api_failures": sum(r["status"] == "error" for r in rows),
                "not_attempted": sum(r["status"] == "not_attempted" for r in rows),
                "exact_match": None if dry else sum(r["exact_match"] or 0 for r in rows) / len(rows),
                "f1": None if dry else sum(r["f1"] or 0 for r in rows) / len(rows),
                "exact_match_completed_only": average_known(completed, "exact_match"),
                "f1_completed_only": average_known(completed, "f1"),
                "abstentions": sum(r["abstained"] for r in completed),
                "empty_completed_answers": sum(r["empty_answer"] for r in completed),
                "avg_generation_latency_ms": average_known(completed, "generation_latency_ms"),
                "avg_retrieval_latency_ms": average_known(rows, "retrieval_latency_ms"),
                "avg_input_tokens": average_known(completed, "input_tokens"),
                "avg_output_tokens": average_known(completed, "output_tokens"),
                "usage_missing": sum(r.get("input_tokens") is None or r.get("output_tokens") is None for r in completed),
                "llm_calls_this_run": sum(r.get("request_attempts", 0) for r in new),
                "original_llm_calls_including_cache": sum(r.get("request_attempts", 0) for r in rows),
                "cache_hits": sum(bool(r.get("cache_hit")) for r in rows),
                "api_error_attempts_this_run": sum(len(r.get("attempt_errors", [])) for r in new),
                "estimated_cost_usd": sum(known_cost) if len(known_cost) == len(completed) and completed else None,
                "estimated_new_cost_usd": sum(r["estimated_cost_usd"] for r in new if r.get("estimated_cost_usd") is not None)
                if known_cost else None,
                "context_source_hit": average_known(rows, "context_source_hit"),
                "candidate_pool_source_hit": average_known(rows, "candidate_pool_source_hit")}
        for metric in ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10"):
            values = [r["retrieval_metrics"][metric] for r in rows if metric in r["retrieval_metrics"]]
            item[metric] = mean(values) if values else None
        table.append(item)
    return table


def representative_examples(records, limit=3):
    by_key = {(r["question_id"], r["retrieval_method"], r["top_k"]): r for r in records}
    ids = list(dict.fromkeys(r["question_id"] for r in records))
    examples = {name: [] for name in ("bm25_correct_semantic_wrong", "semantic_correct_bm25_wrong",
                                     "reranker_improves_rank", "reranker_worsens_rank",
                                     "source_present_answer_wrong", "increasing_k_improves", "increasing_k_harms")}
    def add(name, *rows):
        if len(examples[name]) < limit:
            examples[name].append([{key: row.get(key) for key in
                                   ("question_id", "question", "retrieval_method", "top_k", "generated_answer", "references",
                                    "exact_match", "f1", "annotated_source_rank", "context_source_hit")} for row in rows])
    for qid in ids:
        b, s, r = (by_key[(qid, method, 3)] for method in METHODS)
        if b["status"] == s["status"] == "completed":
            if b["exact_match"] == 1 and s["exact_match"] == 0:
                add("bm25_correct_semantic_wrong", b, s)
            if s["exact_match"] == 1 and b["exact_match"] == 0:
                add("semantic_correct_bm25_wrong", s, b)
        if s.get("retrieval_ranking") is not None and r.get("retrieval_ranking") is not None:
            sr, rr = s["annotated_source_rank"] or float("inf"), r["annotated_source_rank"] or float("inf")
            if rr < sr:
                add("reranker_improves_rank", s, r)
            if rr > sr:
                add("reranker_worsens_rank", s, r)
        for method in METHODS:
            for k in KS:
                row = by_key[(qid, method, k)]
                if row["status"] == "completed" and row["context_source_hit"] == 1 and row["exact_match"] == 0:
                    add("source_present_answer_wrong", row)
            for left, right in ((3, 5), (5, 10), (3, 10)):
                a, b = by_key[(qid, method, left)], by_key[(qid, method, right)]
                if a["status"] == b["status"] == "completed":
                    if b["exact_match"] > a["exact_match"]:
                        add("increasing_k_improves", a, b)
                    if b["exact_match"] < a["exact_match"]:
                        add("increasing_k_harms", a, b)
    return examples


def plots(report, output_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    rows = report["aggregate"]
    by_key = {(r["method"], r["k"]): r for r in rows}
    phase = report["phase"]
    n = report["question_count"]
    partial = not report["all_completed"]
    suffix = " — INCOMPLETE / DRY RUN" if partial else ""
    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=180)
        plt.close(fig)
    names = ["LLM only"] + [f"{r['method']}\nK={r['k']}" for r in rows[1:]]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(rows)))
    ax.bar([i - 0.18 for i in x], [r["exact_match"] or 0 for r in rows], width=0.36, label="Exact match")
    ax.bar([i + 0.18 for i in x], [r["f1"] or 0 for r in rows], width=0.36, label="Token F1")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(f"{phase.title()} answer accuracy (n={n}){suffix}")
    ax.legend()
    finish(fig, "cs391_accuracy_comparison.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        ax.plot(KS, [by_key[(method, k)]["f1"] for k in KS], marker="o", label=method)
    baseline = by_key[("llm_only", 0)]["f1"]
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", label="LLM only", color="gray")
    ax.set_xticks(KS)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Passages supplied to the LLM (K)")
    ax.set_ylabel("Token F1")
    ax.set_title(f"{phase.title()} F1 by K (n={n}){suffix}")
    ax.legend()
    finish(fig, "cs391_f1_by_k.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, [(r["avg_generation_latency_ms"] or 0) / 1000 for r in rows], label="Generation")
    ax.bar(x, [(r["avg_retrieval_latency_ms"] or 0) / 1000 for r in rows],
           bottom=[(r["avg_generation_latency_ms"] or 0) / 1000 for r in rows], label="Retrieval")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("Seconds (setup and deliberate pacing excluded)")
    ax.set_title(f"{phase.title()} measured latency{suffix}")
    ax.legend()
    finish(fig, "cs391_latency_by_method.png")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in METHODS:
        group = [by_key[(method, k)] for k in KS]
        pairs = [(r["context_source_hit"], r["exact_match"], r["k"]) for r in group if r["exact_match"] is not None]
        ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], label=method, s=60)
        for hit, accuracy, k in pairs:
            ax.annotate(f"K={k}", (hit, accuracy), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set_xlim(-0.03, 1.07)
    ax.set_ylim(-0.03, 1.07)
    ax.set_xlabel("Annotated source present in supplied context")
    ax.set_ylabel("Normalized exact match")
    ax.set_title(f"{phase.title()} source recovery vs answer quality{suffix}")
    ax.legend()
    finish(fig, "cs391_retrieval_vs_answer_quality.png")


def save_reports(records, questions, passages, phase, results_dir, manifest):
    scored = score_records(records, questions, passages)
    table = aggregate_results(scored, phase)
    report = {"phase": phase, "question_count": len(questions), "all_completed": all(r["status"] == "completed" for r in scored),
              "aggregate": table, "representative_examples": representative_examples(scored),
              "manifest_path": str(results_dir / "cs391_experiment_manifest.json"),
              "scoring_version": SCORING_VERSION,
              "interpretation": ["Primary EM/F1 use every assigned question; API errors and unattempted answers contribute zero. Completed-only scores are separate.",
                                 "Dry-run answer metrics are null. Incomplete runs cannot support method-comparison conclusions.",
                                 "Answer references can be incomplete; string scores are not semantic factual verification.",
                                 "Hit@1/3/5/10 and MRR@10 use the original top10 rankings, independent of delivered K. context_source_hit measures the actual K passages.",
                                 "Reranker rank improvement examples compare with semantic retrieval; misses are treated as ranks beyond ten.",
                                 "Cached answers retain original timings/usage. llm_calls_this_run excludes cache hits; original_llm_calls_including_cache includes historical attempts.",
                                 "Costs require explicit prices, exclude cache discounts and unreported retry usage, and are not invoices.",
                                 "An earlier BM25-only experiment inspected test retrieval performance; this limits claims that every aspect of the test set was historically untouched."]}
    archive = results_dir / "cs391" / phase
    archive.mkdir(parents=True, exist_ok=True)
    write_json(archive / "cs391_answer_metrics.json", report)
    write_jsonl(archive / "cs391_per_question_results.jsonl", scored)
    with (archive / "cs391_answer_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for name in ("cs391_answer_metrics.json", "cs391_answer_metrics.csv", "cs391_per_question_results.jsonl"):
        shutil.copyfile(archive / name, results_dir / name)
    plots(report, results_dir)
    for name in ("cs391_accuracy_comparison.png", "cs391_f1_by_k.png", "cs391_latency_by_method.png", "cs391_retrieval_vs_answer_quality.png"):
        shutil.copyfile(results_dir / name, archive / name)
    print(f"\n{phase.upper()} — {len(questions)} questions; completed={sum(r['status']=='completed' for r in scored)}/{len(scored)}")
    print("| Method | K | EM | F1 | Completed | API failures | Calls this run |\n|---|---:|---:|---:|---:|---:|---:|")
    for row in table:
        em = 'n/a' if row['exact_match'] is None else f"{row['exact_match']:.3f}"
        f1 = 'n/a' if row['f1'] is None else f"{row['f1']:.3f}"
        print(f"| {row['method']} | {row['k']} | {em} | {f1} | {row['completed']} | {row['api_failures']} | {row['llm_calls_this_run']} |")
    for category, examples in report["representative_examples"].items():
        print(f"\n{category} ({len(examples)} shown):")
        if not examples:
            print("No qualifying measured examples.")
        for example in examples:
            print(json.dumps(example, ensure_ascii=False))
    return report


def placeholder(question, method, k):
    return {"question_id": question["question_id"], "question": question["question"],
            "retrieval_method": method, "top_k": k, "status": "not_attempted",
            "generated_answer": None, "retrieved_passage_ids": [], "retrieval_ranking": None,
            "request_attempts": 0, "cache_hit": False, "api_error": None}


def pilot_checks(records, config):
    return {
        "all_50_completed": len(records) == 50 and all(r["status"] == "completed" for r in records),
        "same_settings": all(r.get("model_settings") == config.public_settings() for r in records),
        "one_response_model": len({r.get("response_model") for r in records if r.get("response_model")}) <= 1,
        "usage_recorded": all(type(r.get("input_tokens")) is int and type(r.get("output_tokens")) is int for r in records)
                          if config.provider in {"openai", "anthropic"} else True,
        "all_prompts_validated": all(r.get("messages") is not None for r in records),
    }


def run_experiment(phase="pilot", test_size=150, seed=42, results_dir=RESULTS, dry_run=False,
                   local_files_only=False, rescore=False, config=None, generator=None, retrieval=None):
    if phase not in {"pilot", "test"}:
        raise ValueError("phase must be pilot or test")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    config = config or GenerationConfig.from_env()
    spec = frozen_spec(config, test_size, seed)
    manifest_path = results_dir / "cs391_experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"phases": {}}
    if manifest.get("test_started") and manifest.get("frozen_spec") != spec:
        raise ValueError("Final settings are frozen; refuse changes after test evaluation began.")
    if phase == "test":
        if not manifest.get("pilot_passed") or manifest.get("frozen_spec") != spec:
            raise ValueError("A successful five-question, ten-configuration pilot with matching frozen settings is required first.")
    elif manifest.get("frozen_spec") != spec:
        manifest = {"phases": {}, "pilot_passed": False}
    manifest["frozen_spec"] = spec
    manifest["experiment_id"] = json_hash(spec)
    prep_path = DATA / "metadata.json"
    if prep_path.exists():
        preparation = json.loads(prep_path.read_text(encoding="utf-8"))
        manifest["data_preparation"] = {key: preparation.get(key) for key in
                                         ("dataset", "revision", "split_sizes", "corpus_size", "random_state")}
    manifest["protocol"] = {
        "candidate_depth": 10, "passage_ks": list(KS), "test_size_predeclared": test_size,
        "subset_limitations": "Any seeded subset is less precise and less representative than the full 150 questions.",
        "no_tuning_on_test": True, "model_refit": False,
        "pricing": "Only explicit configured rates are used; no hardcoded prices.",
    }
    manifest["runtime"] = {"python": platform.python_version(), "platform": platform.platform(),
                           "packages": {name: version(name) for name in ("numpy", "torch", "matplotlib", "scikit-learn", "sentence-transformers", "faiss-cpu", "joblib", "datasets")}}
    # Test data is opened only after the matching pilot gate above succeeds.
    questions, passages, input_path = load_questions(phase, test_size, seed)
    selected_ids = [q["question_id"] for q in questions]
    dataset_hash = file_hash(input_path)
    phase_info = manifest["phases"].get(phase, {})
    if phase == "test" and phase_info and (phase_info["dataset_sha256"] != dataset_hash or phase_info["question_ids"] != selected_ids):
        raise ValueError("The final question set is frozen and cannot be changed on resume.")
    raw_path = results_dir / "cs391" / phase / "generation_raw.jsonl"
    phase_info.update({"dataset_sha256": dataset_hash, "question_ids": selected_ids, "sample_size": len(questions),
                       "full_split_size": 50 if phase == "pilot" else 150, "raw_path": str(raw_path),
                       "sampling": "first five validation questions" if phase == "pilot" else "entire saved test split" if test_size == 150 else "random.Random(seed).sample over sorted question IDs; selected before measuring performance"})
    manifest["phases"][phase] = phase_info
    if rescore:
        if not raw_path.exists():
            raise ValueError("No raw generation records exist for this phase.")
        records = read_jsonl(raw_path)
        existing = {(r["question_id"], r["retrieval_method"], r["top_k"]): r for r in records}
        if len(existing) != len(records):
            raise ValueError("Duplicate saved generation records.")
        expected_keys = {(q["question_id"], method, k) for q in questions for method, k in CONFIGURATIONS}
        if not set(existing).issubset(expected_keys):
            raise ValueError("Unexpected saved question/configuration records.")
        records = [existing.get((q["question_id"], method, k), placeholder(q, method, k))
                   for q in questions for method, k in CONFIGURATIONS]
    else:
        if not dry_run:
            config.require_key()
        if phase == "test" and not dry_run:
            manifest["test_started"] = True
        phase_info["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        retrieval = retrieval or RetrievalConditions(local_files_only=local_files_only)
        phase_info["retrieval_setup_latency_ms"] = getattr(retrieval, "setup_latency_ms", None)
        if hasattr(retrieval, "reranker"):
            phase_info["reranker_metadata"] = {key: retrieval.reranker.metadata.get(key) for key in
                                               ("selected_C", "model_revision", "corpus_sha256")}
        phase_info["corpus_size"] = len(passages)
        generator = generator or AnswerGenerator(config, cache_dir=GENERATION_DIR / "cache")
        if generator.config.public_settings() != config.public_settings():
            raise ValueError("Experiment generator settings do not match the frozen settings.")
        provenance = {"validation_sha256" if phase == "pilot" else "test_sha256": dataset_hash,
                      "corpus_sha256": spec["corpus_sha256"], "reranker_sha256": spec["reranker_sha256"]}
        if phase == "test":
            provenance["selected_question_ids_sha256"] = json_hash(selected_ids)
        records = [placeholder(q, method, k) for q in questions for method, k in CONFIGURATIONS]
        stop = False
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        progress_handle = raw_path.open("w", encoding="utf-8")
        for qi, question in enumerate(questions):
            # Only question text enters retrieval; annotations stay in the scorer.
            retrieved = retrieval.retrieve(question=question["question"], top_k=10)
            for ci, (method, k) in enumerate(CONFIGURATIONS):
                item = retrieved[method]
                evidence = None if method == "llm_only" else item["passages"][:k]
                result = generator.generate_answer(question=question["question"], passages=evidence,
                           question_id=question["question_id"], retrieval_method=method, top_k=k,
                           retrieval_latency_ms=item["retrieval_latency_ms"], dry_run=dry_run, provenance=provenance)
                result["retrieval_ranking"] = None if method == "llm_only" else [
                    {"passage_id": row["passage_id"], "score": row["score"]} for row in item["passages"]]
                result["candidate_passage_ids"] = item.get("candidate_passage_ids", [])
                validate_prompt(result, question, passages)
                records[qi * len(CONFIGURATIONS) + ci] = result
                progress_handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                progress_handle.flush()
                if not dry_run:
                    completed = sum(r["status"] == "completed" for r in records)
                    print(f"{phase} {qi*10+ci+1}/{len(records)} | {method} K={k} | {result['status']} | completed={completed} | cache={result['cache_hit']}", flush=True)
                error = result.get("api_error")
                if error and (not error["retryable"] or error.get("retry_deferred")):
                    stop = True
                    break
            if stop:
                break
        progress_handle.close()
        write_jsonl(raw_path, records)
        resolved = {r.get("response_model") for r in records if r.get("response_model")}
        if len(resolved) > 1:
            raise ValueError("Provider returned inconsistent response model identities.")
    if spec != frozen_spec(config, test_size, seed) or dataset_hash != file_hash(input_path):
        raise ValueError("Frozen code/data/settings changed during execution.")
    # Scoring validates every completed prompt before a pilot can pass.
    report = save_reports(records, questions, passages, phase, results_dir, manifest)
    phase_info["response_models"] = sorted({r["response_model"] for r in records if r.get("response_model")})
    phase_info["system_fingerprints"] = sorted({r["system_fingerprint"] for r in records if r.get("system_fingerprint")})
    if phase == "test":
        pilot_models = manifest["phases"]["pilot"].get("response_models", [])
        if pilot_models and phase_info["response_models"] and phase_info["response_models"] != pilot_models:
            raise ValueError("Returned test model identities differ from the successful pilot.")
    phase_info.update({"completed_answers": sum(r["status"] == "completed" for r in records),
                       "expected_answers": len(records), "dry_run": dry_run,
                       "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    if phase == "pilot":
        checks = pilot_checks(records, config)
        phase_info["checks"] = checks
        manifest["pilot_passed"] = all(checks.values()) and not dry_run
        if manifest["pilot_passed"]:
            manifest["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    print(f"\nPilot passed: {manifest.get('pilot_passed', False)}. Files saved to {results_dir}")
    return report, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "test"), default="pilot")
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.plots_only:
            report = json.loads((args.results_dir / "cs391_answer_metrics.json").read_text())
            plots(report, args.results_dir)
            print("Regenerated four figures without API calls.")
            return
        report, manifest = run_experiment(args.phase, args.test_size, args.seed, args.results_dir,
                                          args.dry_run, args.local_files_only, args.rescore)
        if not args.dry_run and not report["all_completed"]:
            parser.exit(1, "Experiment incomplete: failures/unattempted answers are recorded; successful answers remain cached.\n")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Controlled RAG evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
