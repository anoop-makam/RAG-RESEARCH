"""Read-only, paired statistical analysis of the audited CS391 test experiment.

Run: python -B -m experiments.analyze_rag_results
No inference modules are imported; only saved local artifacts are used.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

from experiments.validate_rag_results import independent_scores


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
ITERATIONS = 10_000
CONFIGS = [("llm_only", 0)] + [(m, k) for m in ("bm25", "semantic", "reranker") for k in (3, 5, 10)]
NUMERIC = ("exact_match", "f1", "input_tokens", "output_tokens",
           "generation_latency_ms", "retrieval_latency_ms")
LIMITATIONS = [
    "The fixed 696-passage corpus was assembled from selected Natural Questions answer-bearing contexts; this is a controlled prototype, not unrestricted Wikipedia retrieval.",
    "Natural Questions source annotations are incomplete relevance labels. An annotated-source miss does not establish that no useful evidence was retrieved. Hit@K measures annotated-source recovery, not exhaustive relevance recall.",
    "EM and token F1 measure lexical agreement with accepted references, not verified factual correctness. A nonmatching answer can be factually plausible; no factual verification or answer revision is performed here.",
    "Earlier BM25-only test retrieval evaluation means the test set was not historically untouched. These results must not be used to retune prompts or hyperparameters.",
    "The API reported 26 backend fingerprints despite fixed requested model/settings; one generation per condition does not measure generation variability.",
    "Two transient API errors were recovered; latency outliers and seven unusual wall-clock gaps are retained. Their causes are not independently established. Recorded client latency is not pure inference time and supports no strong speed claim.",
    "There are 150 distinct question resampling units, not 1,500 independent questions. Question-level bootstrap intervals assume exchangeable questions and do not separately model dependence from shared contexts. Pointwise intervals are not multiplicity-adjusted, do not remove dataset-selection bias, and do not establish causation or universal superiority.",
    "Training provenance is supported by source files, hashes, scaler statistics and metadata, rather than independently signed historical fit logs. Local logs also cannot independently attest API server handling or billing.",
    "Pricing was not configured, so monetary cost remains unavailable. Token usage is recorded; unreported retry usage may be missing.",
]


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def artifact_snapshot(root):
    """Protect data, saved models, all original results, and the Step 9 audit."""
    return {str(p.relative_to(root)): digest(p)
            for folder in ("data/processed", "models", "results")
            for p in sorted((root / folder).rglob("*"))
            if p.is_file() and not p.is_relative_to(root / "results/analysis")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(row, name):
    value = row.get(name)
    require(type(value) in (float, int) and math.isfinite(value) and value >= 0,
            f"Missing/invalid {name} for {row.get('question_id')}: {value!r}")
    if name in ("exact_match", "f1"):
        require(value <= 1, f"{name} outside [0,1]")
    if name == "exact_match":
        require(value in (0, 1), "Exact Match must be binary")
    return value


def align_records(records, test, expected_n=150):
    """Reject discrepancies; never drop, impute, or independently align conditions."""
    ids = sorted(q["question_id"] for q in test)
    require(len(ids) == expected_n and len(set(ids)) == expected_n,
            f"Expected exactly {expected_n} unique test questions")
    questions = {q["question_id"]: q for q in test}
    groups = {config: {} for config in CONFIGS}
    for row in records:
        config = (row.get("retrieval_method"), row.get("top_k"))
        qid = row.get("question_id")
        require(config in groups and qid in questions, "Unexpected question/configuration")
        require(qid not in groups[config], f"Duplicate record: {qid}, {config}")
        require(row.get("status") == "completed" and row.get("api_error") is None,
                f"Incomplete or failed response: {qid}, {config}")
        require(isinstance(row.get("generated_answer"), str) and row["generated_answer"].strip(),
                f"Empty/malformed response: {qid}")
        for field in NUMERIC:
            numeric(row, field)
        require(type(row.get("abstained")) is bool, "Missing/invalid abstention flag")
        question = questions[qid]
        require(row.get("question") == question["question"] and
                row.get("relevant_passage_id") == question["relevant_passage_id"] and
                row.get("references") == question["answers"]["text"],
                f"Original question/reference/source mismatch: {qid}")
        scores = independent_scores(row["generated_answer"], row["references"])
        require(all(math.isclose(row[k], scores[k], abs_tol=1e-9, rel_tol=0)
                    for k in ("exact_match", "f1")) and row["abstained"] == scores["abstained"],
                f"Independent scoring mismatch: {qid}, {config}")
        groups[config][qid] = row
    require(len(records) == expected_n * len(CONFIGS), "Wrong total record count")
    for config, rows in groups.items():
        require(set(rows) == set(ids), f"Question IDs differ for {config}")
    return ids, {config: [rows[qid] for qid in ids] for config, rows in groups.items()}


def mean(rows, field):
    return float(np.mean([numeric(row, field) for row in rows]))


def summary(config, rows):
    method, k = config
    n = len(rows)
    em = mean(rows, "exact_match")
    zeros = sum(row["f1"] == 0 for row in rows)
    abstentions = sum(row["abstained"] for row in rows)
    gen, retrieval = mean(rows, "generation_latency_ms"), mean(rows, "retrieval_latency_ms")
    return dict(method=method, k=k, questions=n, mean_em=em, mean_f1=mean(rows, "f1"),
                f1_std_sample=float(np.std([r["f1"] for r in rows], ddof=1)),
                f1_median=float(np.median([r["f1"] for r in rows])),
                exact_match_count=sum(r["exact_match"] == 1 for r in rows),
                exact_match_percent=100 * em, zero_f1_count=zeros, zero_f1_percent=100 * zeros / n,
                abstention_count=abstentions, abstention_rate=abstentions / n,
                avg_input_tokens=mean(rows, "input_tokens"), avg_output_tokens=mean(rows, "output_tokens"),
                avg_generation_latency_ms=gen, avg_retrieval_latency_ms=retrieval,
                avg_recorded_end_to_end_latency_ms=gen + retrieval)


def bootstrap_indices(n, iterations=ITERATIONS, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, size=(iterations, n))


def paired_comparison(left, right, indices, family, left_config, right_config):
    """Left minus right, subtracting per question BEFORE the sole resampling step."""
    left_ids = [r["question_id"] for r in left]
    require(left_ids == [r["question_id"] for r in right] and len(set(left_ids)) == len(left_ids),
            "Paired question IDs/order must match exactly")
    require(indices.ndim == 2 and indices.shape[1] == len(left) and
            np.issubdtype(indices.dtype, np.integer) and indices.min() >= 0 and indices.max() < len(left),
            "Invalid shared question resampling matrix")
    result = dict(family=family, comparison=f"{label(left_config)} minus {label(right_config)}",
                  left_method=left_config[0], left_k=left_config[1],
                  right_method=right_config[0], right_k=right_config[1], questions=len(left))
    intervals = []
    for metric, unit, scale in (("exact_match", "percentage_points", 100), ("f1", "fraction", 1)):
        diff = np.array([numeric(a, metric) - numeric(b, metric) for a, b in zip(left, right)])
        low, high = np.percentile(diff[indices].mean(axis=1), [2.5, 97.5], method="linear")
        prefix = "em_pp" if metric == "exact_match" else "f1"
        result.update({f"delta_{prefix}": float(diff.mean() * scale),
                       f"{prefix}_ci_low": float(low * scale), f"{prefix}_ci_high": float(high * scale),
                       f"{metric}_wins": int((diff > 1e-12).sum()),
                       f"{metric}_ties": int((abs(diff) <= 1e-12).sum()),
                       f"{metric}_losses": int((diff < -1e-12).sum())})
        intervals.append(dict(family=family, comparison=result["comparison"], metric=metric, units=unit,
                              questions=len(left), estimate=float(diff.mean() * scale),
                              absolute_effect_magnitude=float(abs(diff.mean()) * scale),
                              ci_low=float(low * scale), ci_high=float(high * scale),
                              iterations=len(indices), seed=SEED, multiplicity_adjusted=False))
    for field in ("retrieval_latency_ms", "generation_latency_ms", "input_tokens", "output_tokens"):
        result[f"delta_{field}"] = mean(left, field) - mean(right, field)
    result["delta_recorded_end_to_end_latency_ms"] = (
        result["delta_retrieval_latency_ms"] + result["delta_generation_latency_ms"])
    return result, intervals


def label(config):
    return f"{config[0]} K={config[1]}"


def retrieval_relationship(config, rows, corpus_ids):
    if config[0] == "llm_only":
        return None
    ranks, hits = [], []
    for row in rows:
        ranking = row.get("retrieval_ranking")
        require(isinstance(ranking, list) and len(ranking) == 10, "Missing full top-10 ranking")
        ids = [item["passage_id"] for item in ranking]
        require(len(set(ids)) == 10 and set(ids) <= corpus_ids, "Invalid/duplicate ranking IDs")
        delivered = row.get("retrieved_passage_ids")
        require(delivered == ids[:config[1]], "Delivered passages differ from top-K ranking")
        source = row["relevant_passage_id"]
        require(source in corpus_ids, "Invalid source mapping")
        rank = ids.index(source) + 1 if source in ids else None
        require(rank == row.get("annotated_source_rank"), "Saved annotated rank mismatch")
        ranks.append(rank)
        hits.append(source in delivered)
    out = dict(method=config[0], k=config[1], questions=len(rows))
    for k in (1, 3, 5, 10):
        out[f"Hit@{k}"] = sum(r is not None and r <= k for r in ranks) / len(rows)
    out["MRR@10"] = sum(1 / r if r else 0 for r in ranks) / len(rows)
    out["context_source_hit"] = sum(hits) / len(rows)
    for present, prefix in ((True, "source_present"), (False, "source_absent")):
        subset = [row for row, hit in zip(rows, hits) if hit == present]
        out[f"{prefix}_n"] = len(subset)
        out[f"{prefix}_em"] = mean(subset, "exact_match") if subset else None
        out[f"{prefix}_f1"] = mean(subset, "f1") if subset else None
    out["source_present_em_fail_count"] = sum(h and r["exact_match"] == 0 for r, h in zip(rows, hits))
    out["source_absent_em_success_count"] = sum(not h and r["exact_match"] == 1 for r, h in zip(rows, hits))
    out["candidate_pool_hit"] = (sum(r["relevant_passage_id"] in r["candidate_passage_ids"]
                                      for r in rows) / len(rows) if config[0] == "reranker" else None)
    return out


def question_type(question):
    """Fixed lexical rule, declared before inspection of subgroup performance."""
    words = question.lower().strip().split()
    if words[:2] in (["how", "many"], ["how", "much"]):
        return "how_quantity"
    return words[0] if words and words[0] in {"who", "when", "where", "what", "which", "how", "why"} else "other"


def disagreements(groups):
    """Save ALL qualifying cases in stable question order, not handpicked examples."""
    output = []

    def add(category, config, row, other_config=None, other=None):
        output.append(dict(category=category, question_id=row["question_id"], question=row["question"],
                           references=row["references"], relevant_passage_id=row["relevant_passage_id"],
                           configuration=label(config), generated_answer=row["generated_answer"],
                           em=row["exact_match"], f1=row["f1"],
                           retrieved_passage_ids=row["retrieved_passage_ids"],
                           annotated_source_rank=row["annotated_source_rank"],
                           other_configuration=label(other_config) if other_config else None,
                           other_answer=other["generated_answer"] if other else None,
                           other_em=other["exact_match"] if other else None,
                           other_f1=other["f1"] if other else None,
                           other_retrieved_passage_ids=other["retrieved_passage_ids"] if other else None,
                           other_annotated_source_rank=other["annotated_source_rank"] if other else None))

    for k in (3, 5, 10):
        pairs = [(("bm25", k), ("semantic", k), "bm25_vs_semantic"),
                 (("reranker", k), ("bm25", k), "reranker_vs_bm25"),
                 (("reranker", k), ("semantic", k), "reranker_vs_semantic")]
        for left, right, category in pairs:
            for a, b in zip(groups[left], groups[right]):
                if a["exact_match"] != b["exact_match"]:
                    add(category + ("_em_improves" if a["exact_match"] > b["exact_match"] else "_em_worsens"), left, a, right, b)
    for method in ("bm25", "semantic", "reranker"):
        for low, high in ((3, 5), (5, 10), (3, 10)):
            for a, b in zip(groups[(method, high)], groups[(method, low)]):
                if a["f1"] != b["f1"] or a["exact_match"] != b["exact_match"]:
                    direction = "improves" if a["f1"] > b["f1"] else "degrades" if a["f1"] < b["f1"] else "ties_f1"
                    add("higher_k_" + direction, (method, high), a, (method, low), b)
    for config in CONFIGS[1:]:
        for row, baseline in zip(groups[config], groups[("llm_only", 0)]):
            present = row["relevant_passage_id"] in row["retrieved_passage_ids"]
            if present and row["exact_match"] == 0:
                add("source_present_em_fail", config, row)
            if not present and row["exact_match"] == 1:
                add("source_absent_em_success", config, row)
            if row["exact_match"] == 1 and baseline["exact_match"] == 0:
                add("rag_succeeds_baseline_fails", config, row, ("llm_only", 0), baseline)
    for i, row in enumerate(groups[("llm_only", 0)]):
        if row["exact_match"] == 1 and all(groups[c][i]["exact_match"] == 0 for c in CONFIGS[1:]):
            add("baseline_succeeds_all_rag_fail", ("llm_only", 0), row)
    return sorted(output, key=lambda r: (r["category"], r["question_id"], r["configuration"], r["other_configuration"] or ""))


def write_csv(path, rows):
    require(bool(rows), f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def table(rows, columns):
    def cell(value):
        if value is None:
            return "unavailable"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(title for _, title in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def comparison_table(rows):
    display = [dict(comparison=r["comparison"],
                    em=f"{r['delta_em_pp']:+.2f} [{r['em_pp_ci_low']:+.2f}, {r['em_pp_ci_high']:+.2f}]",
                    f1=f"{r['delta_f1']:+.4f} [{r['f1_ci_low']:+.4f}, {r['f1_ci_high']:+.4f}]",
                    tokens=f"{r['delta_input_tokens']:+.1f}",
                    latency=f"{r['delta_retrieval_latency_ms']:+.2f}",
                    total=f"{r['delta_recorded_end_to_end_latency_ms']:+.2f}",
                    wins=f"{r['f1_wins']}/{r['f1_ties']}/{r['f1_losses']}") for r in rows]
    return table(display, [("comparison", "Left minus right"), ("em", "EM pp [95% CI]"),
                           ("f1", "F1 fraction [95% CI]"), ("tokens", "Input token delta"),
                           ("latency", "Retrieval ms delta"), ("total", "Recorded total ms delta"),
                           ("wins", "F1 W/T/L")])


def k_findings(rows):
    directions = {"positive": sum(r["delta_em_pp"] > 1e-12 for r in rows),
                  "negative": sum(r["delta_em_pp"] < -1e-12 for r in rows)}
    f1_up = sum(r["delta_f1"] > 1e-12 for r in rows)
    tokens_up = sum(r["delta_input_tokens"] > 0 for r in rows)
    latency_up = sum(r["delta_recorded_end_to_end_latency_ms"] > 0 for r in rows)
    return (f"Across {len(rows)} correlated within-method K comparisons, EM increased in "
            f"{directions['positive']} and decreased in {directions['negative']}; F1 increased in "
            f"{f1_up} and decreased in {sum(r['delta_f1'] < -1e-12 for r in rows)}. "
            f"Input tokens increased in {tokens_up}/{len(rows)}; recorded total latency increased in "
            f"{latency_up}/{len(rows)}. These comparison counts are descriptive, not independent evidence. "
            "Increasing K has mixed outcomes across methods and questions; inspect the paired intervals rather than assuming that more passages are better.")


def make_report(summaries, pairs, sensitivity, relationships, cases, subgroups, training, coefficients, audit):
    compact = [dict(configuration=label((r["method"], r["k"])), n=r["questions"],
                    em=f"{r['mean_em'] * 100:.2f}%", f1=f"{r['mean_f1']:.4f}",
                    sd=f"{r['f1_std_sample']:.4f}", median=f"{r['f1_median']:.4f}",
                    tokens=f"{r['avg_input_tokens']:.1f}",
                    latency=f"{r['avg_recorded_end_to_end_latency_ms']:.1f}") for r in summaries]
    quality = table(compact, [("configuration", "Configuration"), ("n", "N"), ("em", "EM"),
                              ("f1", "F1"), ("sd", "F1 sample SD"), ("median", "Median F1"),
                              ("tokens", "Input tokens"), ("latency", "Recorded total ms")])
    best_em = max(r["mean_em"] for r in summaries)
    best_f1 = max(r["mean_f1"] for r in summaries)
    winners_em = ", ".join(label((r["method"], r["k"])) for r in summaries if r["mean_em"] == best_em)
    winners_f1 = ", ".join(label((r["method"], r["k"])) for r in summaries if r["mean_f1"] == best_f1)
    lines = ["# CS391 statistical analysis", "", "## 1. Research questions", "",
             "How do keyword, semantic and trained reranking retrieval affect lexical answer accuracy? Does semantic retrieval outperform BM25 on these questions? How does K affect accuracy, tokens and recorded latency? Does the trained reranker improve retrieval and downstream answers?", "",
             "## 2. Dataset and experimental configuration", "",
             f"Readiness: **{audit['summary']['overall_readiness']}**. All 1,500 valid records cover the same 150 test question IDs across ten configurations. No records or outliers were removed, scores were not repaired, and no API call, generation or model fitting was performed. The Step 9 artifact hashes and independently recomputed scores were checked again before analysis.", "",
             "Training/validation/test sizes are 500/50/150; the unchanged shared corpus contains 696 passages. All conditions requested gpt-4.1-mini-2025-04-14, temperature=0, maximum output tokens=64, seed=42 and prompt nq-short-answer-v1. Reranking used the natural union of BM25 top-10 and semantic top-10, with no forced validation/test positives. The saved logistic regression was trained previously; LLM inference is a separate fixed-model process. No test-based tuning occurs here.", "",
             "## 3. Answer-quality summary", "", quality, "",
             "The CSV/JSON summary also reports exact-match and zero-F1 counts/percentages, abstentions, output tokens and component latencies. F1 SD uses ddof=1. Means use all 150 questions; abstentions score zero. EM/F1 use maximum score over original accepted references after the frozen lowercase/ASCII-punctuation/article/whitespace normalization.", "",
             "Recorded end-to-end latency is retrieval_latency_ms + generation_latency_ms. Model/index setup and deliberate rate-limit pacing are excluded; retries/backoff within generation remain. This is recorded client time, not full experiment wall time or pure model inference.", "",
             "## 4. Retrieval-method comparisons", "", comparison_table(pairs), "",
             "EM effects are percentage points (100 times the paired difference of binary scores); F1 effects are fractions on [0,1]. Effects have a direction: left minus right. W/T/L counts refer to question-level F1, not independent trials. Retrieval scores from different methods are not on comparable scales.", "",
             "## 5. Statistical uncertainty", "",
             "A single shared NumPy default_rng(42) matrix of 10,000 resamples of 150 sorted question IDs is reused for every comparison. For each metric, per-question left-minus-right differences are computed first, then sampled using that matrix. Confidence limits are the 2.5th/97.5th percentiles of resampled mean differences (linear interpolation). The bootstrap CSV includes signed effects and absolute magnitudes. All comparisons use identical question units; the 1,500 answers are never treated as independent questions.", "",
             "These are pointwise percentile intervals with no multiple-comparison adjustment. Exclusion of zero describes a particular interval, not a family-wide significance claim. Intervals address question sampling uncertainty within this selected dataset, not selection bias, alternate corpora, model-generation variability, causality, or universal superiority.", "",
             "## 6. Effect of K", "", comparison_table(sensitivity), "",
             k_findings(sensitivity), "",
             "The table reports question-level wins as well as losses. Token and component latency changes are also recorded in k_sensitivity.csv. All latency outliers are retained.", "",
             "## 7. Retrieval versus answer quality", "",
             "Full top-10 rankings supply Hit@1/3/5/10 and MRR@10; these metrics do not change with delivered K. Conditional answer metrics below use the source in the actual delivered top-K context. Empty conditional groups are unavailable (JSON null/CSV blank), never zero-filled.", "",
             table(relationships, [("method", "Method"), ("k", "K"), ("Hit@1", "Hit@1"),
                                   ("Hit@3", "Hit@3"), ("Hit@5", "Hit@5"), ("Hit@10", "Hit@10"),
                                   ("MRR@10", "MRR@10"), ("source_present_n", "Source present N"),
                                   ("source_present_em", "Present EM"), ("source_present_f1", "Present F1"),
                                   ("source_absent_n", "Absent N"), ("source_absent_em", "Absent EM"),
                                   ("source_absent_f1", "Absent F1")]), "",
             "These conditional associations do not isolate a causal retrieval effect. Source-present EM failures are lexical failures, not confirmed factual errors. Source-absent successes may reflect other useful evidence or model knowledge.", "",
             "## 8. Reranker findings", "",
             f"Saved training report: C={training['selected_C']}, class weights={training['class_weight']}; training class counts={training['training_class_counts']}. The scaler and classifier were fitted on training pairs only, with C selected on validation MRR@10. No model was loaded or refitted for this analysis.", "",
             table(coefficients, [("feature", "Feature"), ("coefficient_standardized", "Standardized coefficient"),
                                  ("odds_multiplier_per_training_sd", "Odds multiplier per training SD")]), "",
             "Holding the other features fixed, each coefficient is the log-odds change per one training-standard-deviation increase in that feature after StandardScaler. BM25 score, semantic similarity and passage length have positive weights; word overlap has a negative conditional weight. Correlated features, training pair selection and class weights prevent interpreting coefficient magnitudes as causal importance. These are relevance-label scores, not calibrated factual-confidence probabilities.", "",
             "Reranker MRR@10 is compared with both baselines in the retrieval table; higher source recovery does not imply proportionally higher answer accuracy. Same-K paired answer differences and their intervals are in section 4. The natural candidate-pool hit rate is recorded in retrieval_answer_relationship.csv.", "",
             "Exploratory question-type rule: lowercase first word in who/when/where/what/which/how/why; how many/how much form how_quantity; all others are other. This fixed rule uses no answers or performance values. Every subgroup's size and scores are saved in configuration_summary.json. Small subgroups support no robust type-specific superiority claim.", "",
             table(subgroups, [("question_type", "Lexical type"), ("questions", "N"), ("k", "K"),
                               ("reranker_minus_bm25_em_pp", "Reranker−BM25 EM pp"),
                               ("reranker_minus_semantic_em_pp", "Reranker−semantic EM pp"),
                               ("reranker_minus_bm25_f1", "Reranker−BM25 F1"),
                               ("reranker_minus_semantic_f1", "Reranker−semantic F1")]), "",
             "## 9. Representative disagreement cases", "",
             "question_disagreements.csv contains ALL qualifying cases, including both directions, original question IDs/text, reference answers, generated answers, source IDs, delivered IDs, scores and comparator fields. Categories use EM unless named higher_k, which compares F1 and retains EM too. Cases may appear in multiple categories/comparisons; they are not independent observations. The first two cases per category in stable question-ID/configuration order are shown below, without selecting by effect size. No category is evidence of a confirmed factual error.", ""]
    worse_types = [r for r in subgroups if r["reranker_minus_bm25_f1"] < -1e-12 or
                   r["reranker_minus_semantic_f1"] < -1e-12]
    # Insert the explicit exploratory negative findings before the cases section.
    subgroup_note = "Descriptive subgroup F1 deficits for the reranker: " + "; ".join(
        f"{r['question_type']} (N={r['questions']}), K={r['k']}: "
        f"vs BM25 {r['reranker_minus_bm25_f1']:+.4f}, "
        f"vs semantic {r['reranker_minus_semantic_f1']:+.4f}" for r in worse_types)
    index = lines.index("## 9. Representative disagreement cases")
    lines[index:index] = [subgroup_note + ". These small, exploratory groups do not establish a stable type-specific effect.", ""]
    categories = sorted({r["category"] for r in cases} | {"baseline_succeeds_all_rag_fail",
                                                                    "source_absent_em_success"})
    for category in categories:
        selected = [r for r in cases if r["category"] == category]
        lines.extend([f"### {category} ({len(selected)} case-comparison records)", ""])
        for row in selected[:2]:
            lines.extend([f"- `{row['question_id']}` — {row['question']}",
                          f"  References: {json.dumps(row['references'], ensure_ascii=False)}. Annotated source: `{row['relevant_passage_id']}`.",
                          f"  {row['configuration']}: {json.dumps(row['generated_answer'], ensure_ascii=False)}; EM={row['em']}, F1={row['f1']:.4f}; delivered IDs: {json.dumps(row['retrieved_passage_ids'])}."])
            if row["other_configuration"]:
                lines.append(f"  {row['other_configuration']}: {json.dumps(row['other_answer'], ensure_ascii=False)}; EM={row['other_em']}, F1={row['other_f1']:.4f}; delivered IDs: {json.dumps(row['other_retrieved_passage_ids'])}.")
        if not selected:
            lines.append("No qualifying cases.")
        lines.append("")
    lines.extend(["## 10. Limitations", "", *[f"- {value}" for value in LIMITATIONS], "",
                  "Step 9 reports 47 passing checks, 8 warnings and 2 unverified checks. Readiness with limitations is not proof of absolute correctness.", "",
                  "## 11. Evidence-supported preliminary conclusions", "",
                  f"The highest observed EM is {best_em * 100:.2f}% ({winners_em}); highest observed mean F1 is {best_f1:.4f} ({winners_f1}). These are descriptive rankings on this fixed test sample, not a model-selection decision or a universal superiority claim.", "",
                  "Semantic and reranker conditions have higher observed mean EM/F1 than corresponding BM25 conditions and the LLM-only baseline in this experiment. The magnitude and uncertainty of each difference are reported above; closely ranked conditions must be interpreted using their paired intervals. K effects are mixed, and higher retrieval success alone does not guarantee lexical answer correctness. No settings were changed in response to these test results.", "",
                  "Reproduce with `python -B -m experiments.analyze_rag_results`; verify saved files without writing with `python -B -m experiments.analyze_rag_results --verify-only`; test with `python -B -m unittest discover -s tests -p test_analyze_rag_results.py -v`. All analysis outputs are under results/analysis/. The manifest records input/output hashes, software versions, exact pairing order, shared bootstrap matrix hash and original-artifact preservation."])
    return "\n".join(lines) + "\n", quality


def run(root=ROOT):
    root = Path(root).resolve()
    before = artifact_snapshot(root)
    audit = read_json(root / "results/validation/experiment_audit.json")
    audit_md = (root / "results/validation/experiment_audit.md").read_text(encoding="utf-8")
    require(audit["summary"]["overall_readiness"] == "READY_WITH_DOCUMENTED_LIMITATIONS" and
            "READY_WITH_DOCUMENTED_LIMITATIONS" in audit_md and audit["summary"]["valid_records"] == 1500 and
            audit["summary"]["failed_checks"] == 0, "Critical audit readiness inconsistency; stopping")
    for path, expected in {**audit["protected_snapshot_after"], **audit["input_sha256"]}.items():
        require((root / path).is_file() and digest(root / path) == expected,
                f"Artifact changed since Step 9 audit: {path}; stopping")
    test = read_jsonl(root / "data/processed/test.jsonl")
    records = read_jsonl(root / "results/cs391_per_question_results.jsonl")
    ids, groups = align_records(records, test)
    corpus = read_jsonl(root / "data/processed/corpus.jsonl")
    corpus_ids = {r["passage_id"] for r in corpus}
    require(len(corpus) == len(corpus_ids) == 696, "Unexpected corpus size/duplicate IDs")
    original = read_json(root / "results/cs391_answer_metrics.json")
    experiment = read_json(root / "results/cs391_experiment_manifest.json")
    retrieval_validation = read_json(root / "results/retrieval_metrics_val.json")
    training = read_json(root / "results/reranker_val_metrics.json")
    with (root / "results/reranker_feature_coefficients.csv").open(encoding="utf-8") as handle:
        coefficients = list(csv.DictReader(handle))
    summaries = [summary(c, groups[c]) for c in CONFIGS]
    with (root / "results/validation/recomputed_metrics.csv").open(encoding="utf-8") as handle:
        audited_csv = list(csv.DictReader(handle))
    for row in summaries:
        config = (row["method"], row["k"])
        for source, em_field, f1_field in ((audit["recomputed_metrics"], "recomputed_exact_match", "recomputed_f1"),
                                           (audited_csv, "recomputed_exact_match", "recomputed_f1"),
                                           (original["aggregate"], "exact_match", "f1")):
            matches = [r for r in source if (r["method"], int(r["k"])) == config]
            require(len(matches) == 1, f"Missing/duplicate saved aggregate: {config}")
            require(math.isclose(row["mean_em"], float(matches[0][em_field]), abs_tol=1e-9, rel_tol=0) and
                    math.isclose(row["mean_f1"], float(matches[0][f1_field]), abs_tol=1e-9, rel_tol=0),
                    f"Aggregate differs from audited metrics: {config}")
    indices = bootstrap_indices(len(ids))
    pairs, sensitivity, intervals = [], [], []

    def compare(left, right, family, target):
        result, cis = paired_comparison(groups[left], groups[right], indices, family, left, right)
        target.append(result)
        intervals.extend(cis)

    for k in (3, 5, 10):
        for a, b in (("semantic", "bm25"), ("reranker", "bm25"), ("reranker", "semantic")):
            compare((a, k), (b, k), "same_k_method", pairs)
    for config in CONFIGS[1:]:
        compare(config, ("llm_only", 0), "rag_vs_baseline", pairs)
    for method in ("bm25", "semantic", "reranker"):
        for high, low in ((5, 3), (10, 5), (10, 3)):
            compare((method, high), (method, low), "within_method_k", sensitivity)
    relationships = [retrieval_relationship(c, groups[c], corpus_ids) for c in CONFIGS[1:]]
    cases = disagreements(groups)
    types = [question_type(r["question"]) for r in groups[("llm_only", 0)]]
    subgroups = []
    for qtype in sorted(set(types)):
        selected = [i for i, t in enumerate(types) if t == qtype]
        for k in (3, 5, 10):
            row = dict(question_type=qtype, questions=len(selected), k=k)
            for metric in ("exact_match", "f1"):
                values = {m: float(np.mean([groups[(m, k)][i][metric] for i in selected]))
                          for m in ("bm25", "semantic", "reranker")}
                for m, value in values.items():
                    row[f"{m}_{metric}"] = value
                for m in ("bm25", "semantic"):
                    row[f"reranker_minus_{m}_" + ("em_pp" if metric == "exact_match" else "f1")] = (
                        (values["reranker"] - values[m]) * (100 if metric == "exact_match" else 1))
            subgroups.append(row)
    report, terminal_table = make_report(summaries, pairs, sensitivity, relationships, cases,
                                          subgroups, training, coefficients, audit)
    output = root / "results/analysis"
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("configuration_summary", summaries), ("paired_comparisons", pairs),
                       ("bootstrap_confidence_intervals", intervals), ("k_sensitivity", sensitivity),
                       ("retrieval_answer_relationship", relationships), ("question_disagreements", cases)):
        write_csv(output / f"{name}.csv", rows)
    write_json(output / "configuration_summary.json", dict(configurations=summaries,
               question_type_analysis=subgroups, retrieval_answer_relationship=relationships,
               units={"mean_em": "fraction", "mean_f1": "fraction", "em_difference": "percentage points",
                      "f1_difference": "fraction", "latency": "milliseconds", "f1_sd_ddof": 1}))
    (output / "statistical_analysis.md").write_text(report, encoding="utf-8")
    after = artifact_snapshot(root)
    require(before == after, "Original artifacts changed during analysis; outputs must not be trusted")
    manifest = dict(analysis_version="cs391-paired-bootstrap-v1", seed=SEED,
                    bootstrap_iterations=ITERATIONS, resampling_unit="question_id",
                    shared_question_ids=ids, question_count=len(ids), record_count=len(records),
                    configurations=[dict(method=m, k=k) for m, k in CONFIGS],
                    bootstrap_matrix_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                    bootstrap_method="paired mean differences; same shared matrix for all 27 comparisons",
                    ci="pointwise percentile 2.5/97.5, NumPy linear interpolation; no multiplicity correction",
                    no_outliers_removed=True, api_calls=0, models_fitted=0,
                    source_files=before, protected_snapshot_before=before, protected_snapshot_after=after,
                    original_artifacts_unchanged=before == after,
                    software_versions=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform()),
                    analysis_source_sha256=digest(Path(__file__)),
                    experiment_manifest=experiment, audit_summary=audit["summary"],
                    saved_validation_retrieval_metrics=retrieval_validation,
                    coefficient_source="results/reranker_feature_coefficients.csv",
                    disagreement_selection="all qualifying cases; stable category/question_id/configuration order; first 2 per category in report",
                    question_type_rule="first lexical word; how many/much => how_quantity; other otherwise; exploratory",
                    limitations=LIMITATIONS,
                    output_sha256={p.name: digest(p) for p in sorted(output.iterdir())
                                   if p.is_file() and p.name != "analysis_manifest.json"})
    write_json(output / "analysis_manifest.json", manifest)
    print(terminal_table)
    for metric in ("mean_em", "mean_f1"):
        best = max(r[metric] for r in summaries)
        print(f"Best observed {metric}: {best:.6f}: " + ", ".join(
            label((r["method"], r["k"])) for r in summaries if r[metric] == best))
    print("\nPaired comparisons and pointwise 95% intervals:\n" + comparison_table(pairs))
    print("\nK effects:\n" + comparison_table(sensitivity) + "\n" + k_findings(sensitivity))
    print("\nLimitations:\n" + "\n".join("- " + item for item in LIMITATIONS))
    print("\nOriginal artifacts unchanged; API calls=0; model fits=0. Saved:")
    for path in sorted(output.iterdir()):
        if path.is_file():
            print(path)
    return manifest


def verify_outputs(root=ROOT):
    """Read-only verification of protected originals and all saved analysis tables."""
    root = Path(root).resolve()
    output = root / "results/analysis"
    manifest = read_json(output / "analysis_manifest.json")
    require(artifact_snapshot(root) == manifest["protected_snapshot_before"] ==
            manifest["protected_snapshot_after"], "Original artifacts differ from the analysis manifest")
    require(manifest["analysis_source_sha256"] == digest(Path(__file__)),
            "Analysis source changed; rerun analysis to regenerate outputs")
    for name, expected in manifest["output_sha256"].items():
        require(digest(output / name) == expected, f"Analysis output hash mismatch: {name}")
    require(manifest["question_count"] == 150 and manifest["record_count"] == 1500 and
            manifest["api_calls"] == manifest["models_fitted"] == 0, "Manifest protocol mismatch")
    print("Verified: original artifacts unchanged; all saved analysis hashes match; 150 questions, 1,500 records; zero API calls/model fits recorded.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify hashes without rewriting any files")
    args = parser.parse_args()
    try:
        if args.verify_only:
            verify_outputs()
        else:
            run()
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Statistical analysis stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
