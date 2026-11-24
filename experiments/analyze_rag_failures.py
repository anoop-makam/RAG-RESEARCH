"""Step 12: deterministic failure signals and an unfilled human-review worksheet.

No inference, web access, model loading, fitting, or changes to saved scores.
Run again after editing manual_review.csv to refresh derived reports. Existing
review bytes are never overwritten. Step 12 remains pending until human review.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd

from experiments.analyze_rag_results import align_records
from src.answer_evaluation import normalize_answer


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TARGET = 24
VERSION = "cs391-qualitative-signals-v1"
METHODS = ("bm25", "semantic", "reranker")
KS = (3, 5, 10)
CONFIGS = [("llm_only", 0)] + [(m, k) for m in METHODS for k in KS]
TAXONOMY = {
    "A": ("Retrieval miss", "RAG: annotated source is absent from actual top-K. Other evidence may still be useful."),
    "B": ("Retrieval success, answer failure", "RAG: annotated source is in actual top-K and saved EM=0. This is a lexical failure, not a verified factual error."),
    "C": ("Semantic retrieval advantage", "Semantic minus BM25 at the same K: saved F1 or EM increases."),
    "D": ("BM25 advantage", "Semantic minus BM25 at the same K: saved F1 or EM decreases."),
    "E": ("Reranker improvement", "Reranker minus BM25 or semantic at the same K: saved F1 or EM increases."),
    "F": ("Reranker regression", "Reranker minus BM25 or semantic at the same K: saved F1 or EM decreases."),
    "G": ("Higher-K degradation", "Same method: K=5 minus 3, 10 minus 5, or 10 minus 3 decreases saved F1 or EM."),
    "H": ("Higher-K improvement", "Same method and K contrasts as G: saved F1 or EM increases."),
    "I": ("LLM-only versus RAG disagreement", "A RAG condition and LLM-only have different binary saved EM values. This is lexical success/failure, not factual adjudication."),
}
MANUAL_CHOICES = {
    "manual_factual_correctness": {"correct", "incorrect", "partially_correct", "uncertain"},
    "manual_evidence_support": {"supported", "contradicted", "insufficient", "uncertain"},
    "manual_failure_cause": {"retrieval_miss", "evidence_distraction", "generation_error", "abstention",
                             "lexical_mismatch", "ambiguous_reference", "other", "uncertain"},
}
ANCHORS = [
    ("who did rachel end up with on friends", ("semantic", 3), ("bm25", 3), "C"),
    ("who won the battle of bull run manassas", ("reranker", 3), ("bm25", 3), "F"),
    ("when did the first witcher book come out", ("reranker", 5), ("semantic", 5), "E"),
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def snapshot(root):
    return {str(p.relative_to(root)): digest(p) for folder in ("data/processed", "models", "results")
            for p in sorted((root / folder).rglob("*"))
            if p.is_file() and not p.is_relative_to(root / "results/failure_analysis")}


def verify_hashes(root, values):
    for name, expected in values.items():
        require((root / name).is_file() and digest(root / name) == expected,
                f"Missing or changed frozen input: {name}")


def load_inputs(root):
    audit = read_json(root / "results/validation/experiment_audit.json")
    require(audit["summary"]["overall_readiness"] == "READY_WITH_DOCUMENTED_LIMITATIONS" and
            audit["summary"]["failed_checks"] == 0 and audit["summary"]["valid_records"] == 1500,
            "Step 9 readiness inconsistency")
    verify_hashes(root, audit["input_sha256"])
    analysis = read_json(root / "results/analysis/analysis_manifest.json")
    verify_hashes(root, analysis["protected_snapshot_after"])
    verify_hashes(root, analysis["experiment_manifest"]["frozen_spec"]["code_sha256"])
    verify_hashes(root, {"results/analysis/" + k: v for k, v in analysis["output_sha256"].items()})
    figures = read_json(root / "results/figures/figure_manifest.json")
    verify_hashes(root, figures["source_files"])
    verify_hashes(root, {"results/figures/" + k: v for k, v in figures["output_sha256"].items()})
    test = read_jsonl(root / "data/processed/test.jsonl")
    records = read_jsonl(root / "results/cs391_per_question_results.jsonl")
    ids, aligned = align_records(records, test)
    by_question = {qid: {config: aligned[config][i] for config in CONFIGS} for i, qid in enumerate(ids)}
    corpus_rows = read_jsonl(root / "data/processed/corpus.jsonl")
    corpus = {p["passage_id"]: p["text"] for p in corpus_rows}
    require(len(corpus_rows) == len(corpus) == 696 and all(isinstance(t, str) and t.strip() for t in corpus.values()),
            "Missing/duplicate/empty corpus passages")
    validate_evidence(by_question, corpus)
    summary = pd.read_csv(root / "results/analysis/configuration_summary.csv")
    for config, rows in aligned.items():
        saved = summary[(summary.method == config[0]) & (summary.k == config[1])]
        require(len(saved) == 1 and int(saved.iloc[0].questions) == 150, "Step 10 configuration mismatch")
        require(all(np.isclose(np.mean([r[field] for r in rows]), saved.iloc[0][column], atol=1e-9, rtol=0)
                    for field, column in (("f1", "mean_f1"), ("exact_match", "mean_em"))),
                "Saved scores disagree with Step 10 means")
    # Locate/read the previous disagreement evidence, but do not treat its overlapping rows as independent questions.
    previous = pd.read_csv(root / "results/analysis/question_disagreements.csv")
    require(set(previous.question_id) <= set(ids), "Step 10 disagreement IDs outside the test split")
    return by_question, corpus, summary, audit["summary"]


def validate_evidence(by_question, corpus):
    for qid, configs in by_question.items():
        for (method, k), row in configs.items():
            require(row["relevant_passage_id"] in corpus, f"Missing annotated evidence: {qid}")
            delivered = row["retrieved_passage_ids"]
            require(len(delivered) == k and len(set(delivered)) == k and set(delivered) <= set(corpus),
                    f"Missing/duplicate/invalid delivered evidence: {qid}/{method}/{k}")
            content = json.loads(row["messages"][1]["content"])
            sources = content.get("source_passages", [])
            require(content.get("question") == row["question"] and
                    [p["passage_id"] for p in sources] == delivered, "Saved prompt/evidence ID mismatch")
            require(all(p["text"] == corpus[p["passage_id"]] and p["source_number"] == i
                        for i, p in enumerate(sources, 1)), "Missing or changed saved prompt passage text")
            ranking = row["retrieval_ranking"]
            if method == "llm_only":
                require(ranking is None and not sources, "LLM-only must have no retrieved sources")
                continue
            require(isinstance(ranking, list) and len(ranking) == 10, "Missing full saved top-10 ranking")
            ranked = [p["passage_id"] for p in ranking]
            require(len(set(ranked)) == 10 and set(ranked) <= set(corpus) and ranked[:k] == delivered,
                    "Invalid saved ranking/mapping")
            expected_rank = ranked.index(row["relevant_passage_id"]) + 1 if row["relevant_passage_id"] in ranked else None
            require(row["annotated_source_rank"] == expected_rank, "Annotated top-10 source rank mismatch")
            require(ranking == configs[(method, 3)]["retrieval_ranking"], "Shared rankings differ across K")
            if method == "reranker":
                union = list(dict.fromkeys([p["passage_id"] for m in ("bm25", "semantic")
                                            for p in configs[(m, 3)]["retrieval_ranking"]]))
                require(row["candidate_passage_ids"] == union and set(ranked) <= set(union),
                        "Saved natural reranker candidate pool mismatch")


def label(config):
    return "LLM-only" if config == ("llm_only", 0) else f"{config[0]} K={config[1]}"


def key(row):
    return row["retrieval_method"], row["top_k"]


def sequence_contains(haystack, needle):
    return bool(needle) and any(haystack[i:i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def format_signals(row):
    """Literal token/length signals, never a semantic-equivalence judgment."""
    if row["exact_match"] == 1 or row["abstained"]:
        return []
    predicted = normalize_answer(row["generated_answer"]).split()
    found = []
    for reference in row["references"]:
        gold = normalize_answer(reference).split()
        if len(predicted) > len(gold) and sequence_contains(predicted, gold):
            found.append("accepted_reference_token_sequence_in_longer_prediction")
        if len(gold) > len(predicted) and sequence_contains(gold, predicted):
            found.append("prediction_token_sequence_in_longer_reference")
    if re.match(r"^(?:dr\.?|sir|mr\.?|mrs\.?|prof\.?)\s", row["generated_answer"], flags=re.I):
        found.append("prediction_has_title_or_honorific")
    if predicted and re.search(r"\b(?:years?|meters?|metres?|kilometers?|kilometres?|kg|miles?|percent)\b",
                               row["generated_answer"], flags=re.I) and any(c.isdigit() for c in row["generated_answer"]):
        found.append("numeric_prediction_contains_unit_word")
    return sorted(set(found))


def literal_reference_mentions(row, corpus):
    """A literal mention is not verified useful evidence or entailment."""
    return [pid for pid in row["retrieved_passage_ids"] if pid != row["relevant_passage_id"] and
            any(sequence_contains(normalize_answer(corpus[pid]).split(), normalize_answer(ref).split())
                for ref in row["references"])]


def make_signal(a, b, family, categories, corpus, hints):
    config_a, config_b = key(a), key(b) if b else None
    identity = [a["question_id"], family, list(config_a), list(config_b) if config_b else None]
    signal_id = "signal_" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    ids_a, ids_b = a["retrieved_passage_ids"], b["retrieved_passage_ids"] if b else []
    source = a["relevant_passage_id"]
    rank_a, rank_b = a["annotated_source_rank"], b["annotated_source_rank"] if b else None
    return dict(signal_id=signal_id, family=family, question_id=a["question_id"], question=a["question"],
                references=a["references"], relevant_passage_id=source, categories=categories,
                method=config_a[0], k=config_a[1], generated_answer=a["generated_answer"],
                em=a["exact_match"], f1=a["f1"], abstained=a["abstained"],
                source_present=source in ids_a if config_a[0] != "llm_only" else None,
                annotated_source_rank_top10=rank_a, retrieved_passage_ids=ids_a,
                ranking=a["retrieval_ranking"],
                candidate_pool_source_present=(source in a["candidate_passage_ids"] if config_a[0] == "reranker" else None),
                candidate_passage_ids=a["candidate_passage_ids"],
                comparison_method=config_b[0] if b else None, comparison_k=config_b[1] if b else None,
                comparison_answer=b["generated_answer"] if b else None,
                comparison_em=b["exact_match"] if b else None, comparison_f1=b["f1"] if b else None,
                comparison_abstained=b["abstained"] if b else None,
                comparison_source_present=(source in ids_b if config_b[0] != "llm_only" else None) if b else None,
                comparison_source_rank_top10=rank_b, comparison_retrieved_passage_ids=ids_b if b else None,
                delta_em=a["exact_match"] - b["exact_match"] if b else None,
                delta_f1=a["f1"] - b["f1"] if b else None,
                source_rank_delta=(rank_a - rank_b if rank_a is not None and rank_b is not None else None),
                same_evidence_membership=(set(ids_a) == set(ids_b)) if b else None,
                same_evidence_order=(ids_a == ids_b) if b else None,
                passages_only_in_a=[pid for pid in ids_a if pid not in ids_b] if b else [],
                passages_only_in_b=[pid for pid in ids_b if pid not in ids_a] if b else [],
                additional_passages_at_higher_k=[pid for pid in ids_a if pid not in ids_b] if family == "higher_k" else [],
                possible_format_mismatch=bool(hints[a["question_id"], config_a]["format_flags"]),
                format_flags=hints[a["question_id"], config_a]["format_flags"],
                comparison_format_flags=hints[a["question_id"], config_b]["format_flags"] if b else [],
                alternative_literal_reference_mentions=hints[a["question_id"], config_a]["alternative_mentions"],
                comparison_alternative_literal_reference_mentions=hints[a["question_id"], config_b]["alternative_mentions"] if b else [],
                explanation_status="automated observations; causal/factual interpretation unverified")


def build_signals(by_question, corpus):
    hints = {(qid, config): dict(format_flags=format_signals(row), alternative_mentions=literal_reference_mentions(row, corpus))
             for qid, configs in by_question.items() for config, row in configs.items()}
    rows = []
    for qid in sorted(by_question):
        configs = by_question[qid]
        for config in CONFIGS:
            row = configs[config]
            present = row["relevant_passage_id"] in row["retrieved_passage_ids"]
            categories = (["A"] if not present else ["B"] if row["exact_match"] == 0 else []) if config[0] != "llm_only" else []
            rows.append(make_signal(row, None, "configuration", categories, corpus, hints))
        comparisons = [(a, k, b, k, "method", win, loss) for k in KS for a, b, win, loss in
                       (("semantic", "bm25", "C", "D"), ("reranker", "bm25", "E", "F"),
                        ("reranker", "semantic", "E", "F"))]
        comparisons += [(m, high, m, low, "higher_k", "H", "G") for m in METHODS for high, low in ((5, 3), (10, 5), (10, 3))]
        comparisons += [(m, k, "llm_only", 0, "baseline", "I", "I") for m in METHODS for k in KS]
        for m, k, other, j, family, win, loss in comparisons:
            a, b = configs[(m, k)], configs[(other, j)]
            de, df = a["exact_match"] - b["exact_match"], a["f1"] - b["f1"]
            if family == "baseline":
                categories = ["I"] if de != 0 else []
            else:
                categories = sorted(set(([win] if de > 1e-12 or df > 1e-12 else []) +
                                        ([loss] if de < -1e-12 or df < -1e-12 else [])))
            rows.append(make_signal(a, b, family, categories, corpus, hints))
    require(len(rows) == 5550 and len({r["signal_id"] for r in rows}) == 5550, "Signal coverage/identity mismatch")
    return rows


def priority(signal):
    return (0 if signal["k"] == 3 and signal["comparison_k"] == 3 else 1,
            0 if signal["family"] != "configuration" else 1,
            signal["k"], signal["method"], signal["comparison_method"] or "", signal["signal_id"])


def select_cases(signals, target=TARGET):
    rng = np.random.default_rng(SEED)
    eligible = {cat: {s["question_id"] for s in signals if cat in s["categories"]} for cat in TAXONOMY}
    chosen, reasons, focus = [], {}, {}

    def choose(qid, reason, category=None, event=None):
        if qid in chosen:
            return
        chosen.append(qid)
        reasons[qid] = reason
        options = [s for s in signals if s["question_id"] == qid and s["categories"] and
                   (category is None or category in s["categories"])]
        require(bool(options), f"No eligible signal for selected question {qid}")
        focus[qid] = event or sorted(options, key=priority)[0]

    anchors = []
    for question, left, right, category in ANCHORS:
        matches = [s for s in signals if normalize_answer(s["question"]) == normalize_answer(question) and
                   (s["method"], s["k"]) == left and (s["comparison_method"], s["comparison_k"]) == right]
        if matches and category in matches[0]["categories"]:
            choose(matches[0]["question_id"], "user-suggested starting case", event=matches[0])
            anchors.append(dict(question=question, found=True, question_id=matches[0]["question_id"]))
        else:
            anchors.append(dict(question=question, found=False, reason="No qualifying saved comparison; no case fabricated"))
    # Rare categories first, with random draws from sorted IDs within each stratum.
    order = sorted(TAXONOMY, key=lambda cat: (len(eligible[cat]), cat))
    for category in order:
        while len(set(chosen) & eligible[category]) < min(2, len(eligible[category])):
            candidates = sorted(eligible[category] - set(chosen))
            require(bool(candidates), f"Unable to cover category {category}")
            choose(candidates[int(rng.integers(len(candidates)))], f"seeded category coverage: {category}", category)
    ambiguous = {s["question_id"] for s in signals if s["family"] == "configuration" and s["possible_format_mismatch"]}
    while len(set(chosen) & ambiguous) < min(2, len(ambiguous)):
        candidates = sorted(ambiguous - set(chosen))
        choose(candidates[int(rng.integers(len(candidates)))], "seeded potential-format-mismatch stratum")
    pool = sorted(set().union(*eligible.values()) - set(chosen))
    if pool and len(chosen) < target:
        for qid in rng.permutation(pool)[:target - len(chosen)]:
            choose(str(qid), "seeded fill from remaining eligible question IDs")
    selection = dict(seed=SEED, target=target, actual_unique_questions=len(chosen),
                     policy="User-suggested anchors, then rare-category-first coverage of two distinct IDs per category using seeded draws from sorted IDs; at least two possible-format-mismatch IDs; seeded fill to target. K=3 comparison priority, no effect-size ranking.",
                     category_order=order, eligibility_unique_questions={c: len(ids) for c, ids in eligible.items()},
                     selected_category_unique_questions={c: len(set(chosen) & ids) for c, ids in eligible.items()},
                     shortfalls={c: max(0, 2 - len(ids)) for c, ids in eligible.items() if len(ids) < 2},
                     anchors=anchors, selection_order=chosen,
                     representative=False, overlap="A question can belong to multiple categories; it remains one question, not independent replicated cases.")
    return sorted(chosen), reasons, focus, selection


def excerpt(text, needles=(), limit=900):
    """Return a contiguous, exact original-text slice with explicit offset/omissions."""
    starts = [m.start() for needle in needles if needle for m in [re.search(re.escape(needle), text, flags=re.I)] if m]
    start = max(0, min(starts) - limit // 3) if starts and len(text) > limit else 0
    stop = min(len(text), start + limit)
    return dict(text=text[start:stop], start=start, stop=stop, original_characters=len(text),
                omitted_before=start > 0, omitted_after=stop < len(text))


def complete_case(qid, focus, reason, by_question, signals, corpus):
    configs = by_question[qid]
    a = configs[(focus["method"], focus["k"])]
    b_config = (focus["comparison_method"], focus["comparison_k"]) if focus["comparison_method"] else (
        "semantic" if focus["method"] == "bm25" else "bm25", focus["k"])
    b = configs[b_config]
    category_signals = [s for s in signals if s["question_id"] == qid and s["categories"]]
    categories = sorted({c for s in category_signals for c in s["categories"]})
    source = a["relevant_passage_id"]
    evidence_ids = list(dict.fromkeys([source] + a["retrieved_passage_ids"] + b["retrieved_passage_ids"]))
    evidence = [dict(passage_id=pid, annotated_source=pid == source,
                     delivered_to_a=pid in a["retrieved_passage_ids"], delivered_to_b=pid in b["retrieved_passage_ids"],
                     excerpt=excerpt(corpus[pid], a["references"] + [a["generated_answer"], b["generated_answer"]]))
                for pid in evidence_ids]
    raw = json.dumps([qid, list(key(a)), list(key(b)), VERSION])
    return dict(case_id="case_" + hashlib.sha256(raw.encode()).hexdigest()[:20], question_id=qid,
                question=a["question"], references=a["references"], relevant_passage_id=source,
                selection_reason=reason, primary_signal_id=focus["signal_id"], primary_categories=focus["categories"],
                overlapping_candidate_categories=categories, configuration=label(key(a)), comparison_configuration=label(key(b)),
                generated_answer=a["generated_answer"], exact_match=a["exact_match"], f1=a["f1"],
                retrieved_passage_ids=a["retrieved_passage_ids"], annotated_source_rank_top10=a["annotated_source_rank"],
                source_present=source in a["retrieved_passage_ids"], candidate_passage_ids=a["candidate_passage_ids"],
                candidate_pool_source_present=(source in a["candidate_passage_ids"] if key(a)[0] == "reranker" else None),
                comparison_generated_answer=b["generated_answer"], comparison_exact_match=b["exact_match"], comparison_f1=b["f1"],
                comparison_retrieved_passage_ids=b["retrieved_passage_ids"], comparison_source_rank_top10=b["annotated_source_rank"],
                comparison_source_present=source in b["retrieved_passage_ids"], delta_em=a["exact_match"] - b["exact_match"],
                delta_f1=a["f1"] - b["f1"], passages_only_in_a=[p for p in a["retrieved_passage_ids"] if p not in b["retrieved_passage_ids"]],
                passages_only_in_b=[p for p in b["retrieved_passage_ids"] if p not in a["retrieved_passage_ids"]],
                same_evidence_order=a["retrieved_passage_ids"] == b["retrieved_passage_ids"],
                format_flags=format_signals(a), comparison_format_flags=format_signals(b),
                alternative_literal_reference_mentions=literal_reference_mentions(a, corpus),
                comparison_alternative_literal_reference_mentions=literal_reference_mentions(b, corpus),
                evidence_excerpts=evidence, hypothesis_status="unverified; human review required",
                all_configuration_results=[dict(configuration=label(config), answer=r["generated_answer"], em=r["exact_match"],
                                                f1=r["f1"], abstained=r["abstained"], source_present=source in r["retrieved_passage_ids"] if config[0] != "llm_only" else None,
                                                source_rank_top10=r["annotated_source_rank"], retrieved_passage_ids=r["retrieved_passage_ids"])
                                           for config, r in configs.items()],
                all_candidate_signals=[dict(signal_id=s["signal_id"], categories=s["categories"],
                                           configuration=label((s["method"], s["k"])),
                                           comparison=label((s["comparison_method"], s["comparison_k"])) if s["comparison_method"] else None,
                                           delta_em=s["delta_em"], delta_f1=s["delta_f1"]) for s in category_signals])


def write_csv(path, rows):
    require(bool(rows), f"No output rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()}
                         for row in rows)


def review_template(cases):
    return [dict(case_id=c["case_id"], question_id=c["question_id"], question=c["question"],
                 configuration=c["configuration"], generated_answer=c["generated_answer"],
                 comparison_configuration=c["comparison_configuration"], comparison_generated_answer=c["comparison_generated_answer"],
                 references=c["references"], candidate_categories=c["overlapping_candidate_categories"],
                 **{name: "" for name in MANUAL_CHOICES}, manual_notes="", reviewed="false") for c in cases]


def load_review(path, cases):
    """Read existing human annotations without rewriting a single byte."""
    if not path.exists():
        return review_template(cases), True
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected = {c["case_id"]: c for c in cases}
    require(len(rows) == len(expected) and {r.get("case_id") for r in rows} == set(expected),
            "Existing manual review IDs differ from selection; preserved untouched. Resolve selection/provenance mismatch before continuing.")
    for row in rows:
        case = expected[row["case_id"]]
        require(all(row.get(field) == case[field] for field in
                    ("question_id", "question", "configuration", "comparison_configuration", "generated_answer", "comparison_generated_answer")),
                "Existing review identity/answer fields differ; preserved untouched")
        require(json.loads(row.get("references", "null")) == case["references"],
                "Existing review references differ; preserved untouched")
        require(all(name in row for name in (*MANUAL_CHOICES, "manual_notes", "reviewed")), "Missing manual review columns")
        require(row["reviewed"].strip().lower() in {"true", "false"}, "reviewed must be true or false")
        for name, choices in MANUAL_CHOICES.items():
            require(row[name] in choices | {""}, f"Invalid manual label in {name}: {row[name]!r}; review file preserved")
        if row["reviewed"].strip().lower() == "true":
            require(all(row[name] for name in MANUAL_CHOICES), "Fill all three judgment fields before marking reviewed=true")
    return rows, False


def reviewed(row):
    return str(row["reviewed"]).strip().lower() == "true"


def descriptive_counts(signals, selected, reviews):
    counts = []
    ids = set(selected)

    def add(scope, metric, count, denominator, unit, category="", comparison="", note=""):
        counts.append(dict(scope=scope, metric=metric, category=category, comparison=comparison,
                           unit=unit, count=int(count), denominator=int(denominator), note=note))

    for cat in TAXONOMY:
        eligible = [s for s in signals if cat in s["categories"]]
        unique = {s["question_id"] for s in eligible}
        denominator = 1350 if cat in {"A", "B", "G", "H", "I"} else 450 if cat in {"C", "D"} else 900
        add("all_test_questions", "taxonomy_eligibility", len(unique), 150, "unique_questions", cat,
            note="Overlaps other categories; automated eligibility, not confirmed causes")
        add("question_configuration_or_pair_events", "taxonomy_eligibility", len(eligible), denominator,
            "question_configuration_records" if cat in {"A", "B"} else "paired_question_comparisons", cat)
        add("selected_sample", "taxonomy_eligibility", len(unique & ids), len(ids), "unique_questions", cat,
            note="Purposeful overlapping sample, not statistically representative")
    for config in CONFIGS:
        rows = [s for s in signals if s["family"] == "configuration" and (s["method"], s["k"]) == config]
        add("all_test_questions_by_configuration", "abstentions", sum(s["abstained"] for s in rows), 150,
            "question_configuration_records", comparison=label(config))
        if config[0] != "llm_only":
            add("all_test_questions_by_configuration", "source_present_em_failure", sum("B" in s["categories"] for s in rows),
                150, "question_configuration_records", "B", label(config))
            add("all_test_questions_by_configuration", "source_miss", sum("A" in s["categories"] for s in rows),
                150, "question_configuration_records", "A", label(config))
    comparisons = sorted({(s["family"], s["method"], s["k"], s["comparison_method"], s["comparison_k"])
                          for s in signals if s["family"] in {"method", "higher_k"}})
    for family, m, k, other, j in comparisons:
        rows = [s for s in signals if (s["family"], s["method"], s["k"], s["comparison_method"], s["comparison_k"]) ==
                (family, m, k, other, j)]
        for metric in ("em", "f1"):
            for outcome, predicate in (("wins", lambda x: x > 1e-12), ("ties", lambda x: abs(x) <= 1e-12),
                                       ("losses", lambda x: x < -1e-12)):
                add("all_test_questions_per_pair", f"{metric}_{outcome}", sum(predicate(s[f"delta_{metric}"]) for s in rows),
                    150, "paired_question_comparisons", comparison=label((m, k)) + " minus " + label((other, j)))
    formatting = [s for s in signals if s["family"] == "configuration" and s["possible_format_mismatch"]]
    add("all_configuration_records", "possible_lexical_mismatch", len(formatting), 1500, "question_configuration_records",
        note="Literal substring/title/unit heuristic only; semantic/factual status unverified")
    mismatch_ids = {s["question_id"] for s in formatting}
    add("all_test_questions", "possible_lexical_mismatch", len(mismatch_ids), 150, "unique_questions")
    add("selected_sample", "possible_lexical_mismatch", len(mismatch_ids & ids), len(ids), "unique_questions")
    add("selected_sample", "reviewed_cases", sum(reviewed(r) for r in reviews), len(ids), "unique_questions")
    add("selected_sample", "unreviewed_cases", sum(not reviewed(r) for r in reviews), len(ids), "unique_questions")
    done = [r for r in reviews if reviewed(r)]
    if done:
        for field in MANUAL_CHOICES:
            for value, number in sorted(Counter(r[field] for r in done).items()):
                add("manually_reviewed_sample", field, number, len(done), "human_reviewed_primary_comparisons",
                    comparison=value, note="Reviewer-entered judgment about primary configuration A only; no extrapolation to all 150 questions")
    return counts


def md_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(rows, columns):
    return "\n".join(["| " + " | ".join(title for _, title in columns) + " |",
                      "| " + " | ".join("---" for _ in columns) + " |"] +
                     ["| " + " | ".join(md_cell(row.get(key, "")) for key, _ in columns) + " |" for row in rows])


def evidence_observations(case):
    lines = [f"Annotated source is {'present' if case['source_present'] else 'absent'} in A and "
             f"{'present' if case['comparison_source_present'] else 'absent'} in B. "
             f"Saved source ranks within the full top-10 are A={case['annotated_source_rank_top10']}, B={case['comparison_source_rank_top10']}. "
             "None means not found in saved top-10 (or no retrieval); it is not an estimated rank.",
             f"A minus B: EM {case['delta_em']:+.0f}, token F1 {case['delta_f1']:+.4f}. "
             f"Evidence order identical: {case['same_evidence_order']}; added/replaced IDs: A-only "
             f"{len(case['passages_only_in_a'])}, B-only {len(case['passages_only_in_b'])}."]
    if case["candidate_pool_source_present"] is not None:
        lines.append(f"The annotated source is {'present' if case['candidate_pool_source_present'] else 'absent'} in A's natural reranker candidate union.")
    if case["format_flags"] or case["comparison_format_flags"]:
        lines.append("Unverified formatting hypotheses: A=" + json.dumps(case["format_flags"]) + "; B=" + json.dumps(case["comparison_format_flags"]) + ". Literal token containment is not a semantic-equivalence judgment.")
    if case["source_present"] and case["comparison_source_present"]:
        lines.append("Both prompts contained the same annotated source text. A source-availability difference cannot explain this pair; context/order, interpretation or generation variability are hypotheses, not established causes.")
    if case["alternative_literal_reference_mentions"] or case["comparison_alternative_literal_reference_mentions"]:
        lines.append("Other delivered passages with literal accepted-reference token mentions: A=" +
                     json.dumps(case["alternative_literal_reference_mentions"]) + "; B=" +
                     json.dumps(case["comparison_alternative_literal_reference_mentions"]) +
                     ". A mention alone does not establish usefulness, support, or contradiction.")
    return lines


def taxonomy_document():
    lines = ["# Failure taxonomy and review guide", "", "## Automated categories", "",
             "Categories describe saved lexical scores and source IDs. They are overlapping eligibility signals, not verified mechanisms or factual labels. Advantages/regressions use either metric: if EM and F1 move in opposite directions, both categories apply. Ties use tolerance 1e-12.", ""]
    lines.append(table([dict(code=c, name=v[0], rule=v[1]) for c, v in TAXONOMY.items()],
                       [("code", "Code"), ("name", "Category"), ("rule", "Detection rule")]))
    lines.extend(["", "Source presence refers to actual delivered top-K, while source rank refers to saved top-10. No source hit is imputed beyond rank 10. LLM-only is never a retrieval miss. A missing annotation does not mean no relevant evidence. Reranker source eligibility uses the saved natural BM25/semantic top-10 union.", "",
                  "## Unverified formatting signals", "", "Normalized contiguous reference-token containment in a longer prediction (or reverse containment), leading titles/honorifics, and numerical predictions with unit words flag possible formatting issues. These rules do not alter EM/F1 or certify an answer as correct. Alternative-source literal reference mentions do not establish entailment. No automatic distraction, ignored-evidence, factual-error or semantic-equivalence labels are assigned.", "",
                  "## How to review", "", "Open manual_review.csv alongside case_studies.md. Match rows using case_id/question_id. Each worksheet row concerns primary configuration A; B is the comparison. Inspect both answers, original references and exact passage excerpts. All ten configurations are listed for context; the row's judgment does not classify every other answer for that question.", ""])
    for name, choices in MANUAL_CHOICES.items():
        lines.append(f"- `{name}`: " + " / ".join(sorted(choices)))
    lines.extend(["- `manual_notes`: explain evidence and uncertainty, including whether a reference is ambiguous, and distinguish A from B.",
                  "- `reviewed`: set to `true` only after filling all three judgment fields. Otherwise retain `false`. Use `uncertain` where appropriate; do not force a factual judgment.", "",
                  "Judgment fields start blank. Do not modify identity, question, answer or comparison columns. Save CSV using UTF-8 with ordinary CSV quoting, then rerun `python -B -m experiments.analyze_rag_failures`. Existing worksheet bytes, including completed/partial judgments and extra columns, are preserved. Selection/identity mismatches or invalid labels stop execution rather than replacing annotations.", "",
                  "Manual failure-cause labels are reviewer judgments, not causal experimental identification. No factual-correctness or failure-cause statistics are reported before reviewed=true. Step 12 is not fully complete until every selected question is reviewed."])
    return "\n".join(lines) + "\n"


def case_studies(cases, reviews):
    lookup = {r["case_id"]: r for r in reviews}
    lines = ["# Selected qualitative cases", "", "Purposeful, overlapping 24-question review sample; not representative of all 150 questions. All automated explanations remain unverified. Excerpts are contiguous exact original text with zero-based character offsets and explicit omissions. Every displayed delivered passage was verified byte-for-byte against the saved prompt and corpus. The annotated source is also included when absent from both prompts, clearly marked as not delivered. Consult the complete saved corpus when an excerpt omits relevant text.", ""]
    for case in cases:
        review = lookup[case["case_id"]]
        lines.extend([f"## {case['case_id']}: {case['question']}", "", f"Question ID: `{case['question_id']}`.",
                      f"References: {json.dumps(case['references'], ensure_ascii=False)}. Annotated source: `{case['relevant_passage_id']}`.",
                      f"Selection: {case['selection_reason']}. Primary signal categories: {', '.join(case['primary_categories'])}. "
                      f"Overlapping eligibility anywhere across this question's configurations: {', '.join(case['overlapping_candidate_categories'])}.", "",
                      f"A: **{case['configuration']}** — {json.dumps(case['generated_answer'], ensure_ascii=False)}; EM={case['exact_match']}, F1={case['f1']:.4f}.",
                      f"B: **{case['comparison_configuration']}** — {json.dumps(case['comparison_generated_answer'], ensure_ascii=False)}; EM={case['comparison_exact_match']}, F1={case['comparison_f1']:.4f}.", "",
                      *["- " + text for text in evidence_observations(case)], "",
                      f"A ordered evidence: {json.dumps(case['retrieved_passage_ids'])}",
                      f"B ordered evidence: {json.dumps(case['comparison_retrieved_passage_ids'])}",
                      f"A-only IDs: {json.dumps(case['passages_only_in_a'])}; B-only IDs: {json.dumps(case['passages_only_in_b'])}.", "",
                      "### Exact evidence excerpts", ""])
        for item in case["evidence_excerpts"]:
            ex = item["excerpt"]
            lines.extend([f"Passage `{item['passage_id']}` — annotated source={item['annotated_source']}; delivered to A={item['delivered_to_a']}, to B={item['delivered_to_b']}; characters [{ex['start']}:{ex['stop']}] of {ex['original_characters']}.", ""])
            if ex["omitted_before"]:
                lines.append("[Earlier text omitted]")
            lines.extend(["> " + line for line in ex["text"].splitlines()])
            if ex["omitted_after"]:
                lines.append("[Later text omitted]")
            lines.append("")
        lines.extend(["### All ten saved responses", "",
                      table([dict(configuration=r["configuration"], answer=r["answer"], em=r["em"], f1=f"{r['f1']:.4f}",
                                  source=r["source_present"], rank=r["source_rank_top10"], ids=json.dumps(r["retrieved_passage_ids"]))
                             for r in case["all_configuration_results"]],
                            [("configuration", "Configuration"), ("answer", "Saved answer"), ("em", "EM"), ("f1", "F1"),
                             ("source", "Source in K"), ("rank", "Rank in top-10"), ("ids", "Ordered passage IDs")]), "",
                      "### Overlapping automated signals", "",
                      table([dict(categories=",".join(s["categories"]), configuration=s["configuration"],
                                  comparison=s["comparison"], delta_em=s["delta_em"], delta_f1=s["delta_f1"])
                             for s in case["all_candidate_signals"]],
                            [("categories", "Categories"), ("configuration", "A"), ("comparison", "B"),
                             ("delta_em", "EM A−B"), ("delta_f1", "F1 A−B")]), "",
                      "### Human review", "",
                      f"Status: {'reviewed (human-entered labels)' if reviewed(review) else 'UNREVIEWED; all causal/factual interpretations provisional'}.",
                      *[f"- {name}: {review[name] or '[blank]'}" for name in (*MANUAL_CHOICES, "manual_notes")], ""])
    return "\n".join(lines) + "\n"


def paper_cases(cases, reviews):
    lookup = {r["case_id"]: r for r in reviews}
    used, selected = set(), []
    requirements = [("Semantic improvement", "C"), ("BM25 improvement", "D"), ("Reranker improvement", "E"),
                    ("Reranker regression", "F"), ("Higher-K degradation", "G"), ("Possible lexical mismatch", None)]
    for name, category in requirements:
        # Use focused comparisons, not an unrelated overlapping category on the same question.
        eligible = [c for c in cases if c["question_id"] not in used and
                    ((category in c["primary_categories"]) if category else (c["format_flags"] or c["comparison_format_flags"]))]
        # Overlap can cover a category without making it the worksheet's primary
        # pair. Retain the selected question and show its actual alternate pair;
        # never transfer human labels between configurations automatically.
        if not eligible and category:
            for case in cases:
                if case["question_id"] in used:
                    continue
                options = [s for s in case["all_candidate_signals"] if category in s["categories"] and s["comparison"]]
                options.sort(key=lambda s: ("K=3" not in s["configuration"], s["configuration"], s["comparison"]))
                if not options:
                    continue
                signal = options[0]
                a = next(r for r in case["all_configuration_results"] if r["configuration"] == signal["configuration"])
                b = next(r for r in case["all_configuration_results"] if r["configuration"] == signal["comparison"])
                variant = dict(case, configuration=a["configuration"], comparison_configuration=b["configuration"],
                               generated_answer=a["answer"], comparison_generated_answer=b["answer"],
                               exact_match=a["em"], comparison_exact_match=b["em"], f1=a["f1"], comparison_f1=b["f1"],
                               source_present=a["source_present"], comparison_source_present=b["source_present"],
                               annotated_source_rank_top10=a["source_rank_top10"], comparison_source_rank_top10=b["source_rank_top10"],
                               passages_only_in_a=[p for p in a["retrieved_passage_ids"] if p not in b["retrieved_passage_ids"]],
                               passages_only_in_b=[p for p in b["retrieved_passage_ids"] if p not in a["retrieved_passage_ids"]])
                eligible.append(variant)
        if not eligible:
            selected.append(dict(question=f"{name}: no unused focused case", a="unavailable", b="unavailable",
                                 evidence="No case invented", outcome="unavailable", interpretation="Provisional; choose another reviewed comparison if needed"))
            continue
        def review_applies(c):
            row = lookup[c["case_id"]]
            return reviewed(row) and row["configuration"] == c["configuration"] and row["comparison_configuration"] == c["comparison_configuration"]
        eligible.sort(key=lambda c: (not review_applies(c), c["case_id"]))
        case = eligible[0]
        used.add(case["question_id"])
        review = lookup[case["case_id"]]
        selected.append(dict(question=f"{name}: {case['question']} ({case['case_id']})",
                             a=f"{case['configuration']}: {case['generated_answer']}",
                             b=f"{case['comparison_configuration']}: {case['comparison_generated_answer']}",
                             evidence=f"Source in A/B: {case['source_present']}/{case['comparison_source_present']}; top-10 source ranks {case['annotated_source_rank_top10']}/{case['comparison_source_rank_top10']}; A-only/B-only passages {len(case['passages_only_in_a'])}/{len(case['passages_only_in_b'])}",
                             outcome=f"EM A/B {case['exact_match']}/{case['comparison_exact_match']}; F1 A/B {case['f1']:.4f}/{case['comparison_f1']:.4f}",
                             interpretation=("Human-reviewed primary A: " + "; ".join(review[n] for n in MANUAL_CHOICES) + "; " + review["manual_notes"]
                                             if review_applies(case) else "PROVISIONAL — this pair has no completed human judgment; consult all ten responses and source IDs in case_studies.md. Primary worksheet labels are not transferred to alternate pairs.")))
    return "\n".join(["# Paper case-study candidates", "", "Six distinct-question illustrations, selected deterministically from saved comparisons. Primary reviewed pairs are preferred. Alternate pairs from the same selected question remain provisional unless reviewed separately; no primary labels are transferred. No nonmatching response is automatically called factually wrong.", "",
                      table(selected, [("question", "Question"), ("a", "Method A"), ("b", "Method B"), ("evidence", "Evidence difference"),
                                       ("outcome", "Observed outcome"), ("interpretation", "Reviewed interpretation")]), "",
                      "See case_studies.md and manual_review.csv for exact passages and review status. These examples are not independent quantitative proof or a representative sample.", ""])


def discussion(summary, reviews, signals, selected):
    done = [r for r in reviews if reviewed(r)]
    observed = summary.loc[summary.mean_f1.idxmax()]
    retained = [s for s in signals if s["question_id"] in selected and s["family"] == "higher_k" and
                s["k"] == 10 and s["comparison_k"] == 3 and "G" in s["categories"] and
                s["source_present"] and s["comparison_source_present"]]
    retained.sort(key=lambda s: (not bool(s["format_flags"]), s["question_id"], s["method"]))
    if retained:
        example = retained[0]
        higher_k_text = (f"For “{example['question']}”, {example['method']} K=3 returned "
                         f"{json.dumps(example['comparison_answer'], ensure_ascii=False)} (F1={example['comparison_f1']:.4f}), "
                         f"while K=10 returned {json.dumps(example['generated_answer'], ensure_ascii=False)} "
                         f"(F1={example['f1']:.4f}). The annotated source remained in both prompts; "
                         f"{len(example['additional_passages_at_higher_k'])} extra passages were added. "
                         "This is an observed decrease with retained source availability, not proof that any added passage caused distraction.")
    else:
        higher_k_text = "No selected K=10-minus-3 degradation with the annotated source in both prompts was found; no such example is invented."
    return f"""# Draft qualitative discussion — {'human review incomplete' if len(done) < TARGET else 'selected-case human review entered'}

## Observed quantitative evidence

The ten conditions were evaluated on the same 150 Natural Questions queries using one fixed 696-passage corpus. The highest observed mean token F1 was {observed.mean_f1:.4f} for {observed.method} K={int(observed.k)} (EM {observed.mean_em * 100:.2f}%). This is a descriptive result, not universal superiority. Step 10 contains the paired uncertainty; these selected examples add no independent statistical significance test. Semantic retrieval and reranking recovered the annotated source more often than BM25, but the answer-level differences are smaller and sensitive to K. Source recovery and lexical answer accuracy measure different stages of the system.

The Rachel/Friends comparison illustrates an observed source-availability change: semantic and reranker K=3 include the finale passage at rank 1 and produce the accepted name, while BM25 K=3 omits that source and abstains. The source literally states that Ross and Rachel resume their relationship. The question's phrase “end up with” and the passage's “resume their relationship” differ lexically. This is consistent with a semantic-match explanation; it does not establish that a particular term caused BM25's ranking or that none of BM25's other evidence was useful.

The Bull Run comparison illustrates a measured answer-length effect: BM25 K=3 returns the accepted reference “Confederate forces”, whereas reranker K=3 returns a longer sentence containing those reference tokens. Both retrieve the same annotated source first. Saved EM/F1 penalize the longer response. Literal token containment makes this a candidate lexical-format mismatch, rather than automatically a factual error; equivalence and evidential support still require human judgment.

For the Witcher question at K=5, semantic and reranker prompts share the first four passages, including the annotated source at rank 1; their fifth passages differ. The annotated passage mentions a 1986 short story and a 1990 collection. Semantic answers “1986” and the reranker answers the accepted “1990”. This pair does not support a claim that reranking raised the annotated source rank or newly supplied it. Different secondary context or interpretation of “book” versus “short story” are plausible hypotheses, not isolated causes.

{higher_k_text}

## Manually reviewed qualitative evidence

{len(done)} of {len(reviews)} selected questions have reviewed=true. {('No factual-correctness or confirmed failure-cause statistics are available until manual judgments are entered.' if not done else 'Reviewer-entered labels are summarized separately in failure_counts.csv; they concern primary configuration A in this purposeful sample and must not be extrapolated to all answers.')}

## Unverified mechanisms to examine

Semantic retrieval may bridge wording differences that keyword weighting handles less well. Reranking can change membership or order within the natural candidate union, but a missing source outside that union cannot be recovered by reordering it. Cases with unchanged source availability and changed scores require examining the remaining evidence, answer wording and generation behavior. Different contexts, order effects, ambiguity and single-generation variability remain competing explanations.

Higher K can add an annotated source that was previously absent, but gains can also occur while the source was already present. Decreases with the source still available do not by themselves demonstrate distraction. Compare exact added passages, whether they contain alternative dates/entities, and whether answers become longer. Increasing context has diminishing observed accuracy returns and consumes more tokens; these observations do not establish a causal effect of any particular added passage. Reranker regressions are retained in the sample and can involve worse source order, different secondary evidence, or lexical answer formatting; none is preassigned as the confirmed cause.

## Scope and limitations

EM and token F1 reward lexical agreement with accepted references, not independently verified factual correctness. Partial token overlap or literal answer mentions do not certify semantic equivalence or evidential entailment. An annotated-source miss does not imply the absence of every useful passage, because Natural Questions annotations need not exhaust relevance. The corpus consists of selected answer-bearing contexts rather than unrestricted Wikipedia, and multiple questions can share passages. The 24-question sample is deliberately stratified and overlapping, not statistically representative. Earlier BM25-only test exposure, API backend variation, one response per condition, and the audit's unverified historical provenance limits remain relevant. No prompt, model, scoring rule or experimental setting is changed based on these test outcomes. No model-based grading, factual verification pipeline or answer revision was implemented.
"""


def run(root=ROOT):
    root = Path(root).resolve()
    before = snapshot(root)
    by_question, corpus, summary, audit = load_inputs(root)
    signals = build_signals(by_question, corpus)
    selected, reasons, focus, selection = select_cases(signals)
    cases = [complete_case(qid, focus[qid], reasons[qid], by_question, signals, corpus) for qid in selected]
    output = root / "results/failure_analysis"
    review_path = output / "manual_review.csv"
    review_hash_before = digest(review_path) if review_path.exists() else None
    reviews, create_review = load_review(review_path, cases)
    counts = descriptive_counts(signals, selected, reviews)
    output.mkdir(parents=True, exist_ok=True)
    # Never open an existing review worksheet for writing, including partial annotations.
    if create_review:
        write_csv(review_path, reviews)
    if review_hash_before:
        require(digest(review_path) == review_hash_before, "Manual review changed unexpectedly")
    write_csv(output / "all_failure_signals.csv", signals)
    write_csv(output / "selected_cases.csv", cases)
    write_csv(output / "failure_counts.csv", counts)
    documents = {"failure_taxonomy.md": taxonomy_document(), "case_studies.md": case_studies(cases, reviews),
                 "paper_ready_cases.md": paper_cases(cases, reviews),
                 "qualitative_discussion_draft.md": discussion(summary, reviews, signals, set(selected))}
    category_counts = [r for r in counts if r["metric"] == "taxonomy_eligibility" and r["scope"] in
                       {"all_test_questions", "selected_sample", "question_configuration_or_pair_events"}]
    pairs = [r for r in counts if r["scope"] == "all_test_questions_per_pair"]
    done = sum(reviewed(r) for r in reviews)
    report = ["# Step 12 qualitative failure analysis", "", f"Status: {'MANUAL REVIEW PENDING — Step 12 is not fully complete' if done < len(cases) else 'All selected primary comparisons have human review labels; interpret their scope carefully'}.", "",
              f"Analyzed 150 distinct test questions and 1,500 question-configuration records. Selected {len(cases)} unique questions; reviewed {done}, unreviewed {len(cases) - done}. All saved prompts, passage IDs, source mappings and frozen hashes were validated. No API calls, generation, model loading, fitting, scoring changes, protocol changes or outlier removal occurred.", "",
              "## Counts and denominators", "", "The automated export contains 1,500 configuration records and 4,050 paired question-comparison events (5,550 rows total), including ties. It is not a dataset of 5,550 independent questions. Taxonomy categories overlap; do not sum them. Advantages/decreases use either EM or F1; explicit win/tie/loss tables keep these metrics separate.", "",
              table(category_counts, [("scope", "Scope"), ("category", "Category"), ("count", "Count"), ("denominator", "Denominator"), ("unit", "Unit")]), "",
              "## Paired win/tie/loss counts", "", table(pairs, [("comparison", "A minus B"), ("metric", "Metric/outcome"), ("count", "Count"), ("denominator", "Questions")]), "",
              "## Sampling and overlap", "", json.dumps(selection, ensure_ascii=False, indent=2), "",
              "## Review workflow", "", "Open manual_review.csv first and case_studies.md alongside it. Judgments concern A, the primary configuration named in the same worksheet row. B and all ten outputs provide comparison context. Fill the three manual judgment columns, add evidence-based notes, and mark reviewed=true when finished. Leave uncertain judgments explicit. Do not edit identity or saved-answer fields. Rerun the script to refresh reports; existing annotations are never overwritten. A completed worksheet is necessary before calling Step 12 fully complete.", "",
              "## Limitations", "", "All unreviewed explanations are provisional. Source-present/EM-failure is not a verified factual failure. Token containment/title/unit heuristics are not semantic judging; source mentions are not entailment. No distraction or ignored-evidence mechanism is automatically verified. The selected overlapping sample is not representative of all 150 questions. Source annotations may omit other useful passages; the fixed 696-passage answer-bearing corpus is a controlled prototype. All Step 9 limitations remain in force. Manual reviewed labels, when present, are reviewer judgments about primary A, not automatic factual verification or population estimates.", "",
              "## Reproduce", "", "`python -B -m experiments.analyze_rag_failures`", "", "`python -B -m unittest discover -s tests -p test_analyze_rag_failures.py -v`", ""]
    documents["failure_analysis_report.md"] = "\n".join(report)
    for name, text in documents.items():
        (output / name).write_text(text, encoding="utf-8")
    after = snapshot(root)
    require(before == after, "Original Step 9/10/11 or experiment artifacts changed")
    manifest = dict(version=VERSION, seed=SEED, selection=selection, test_questions=150, configuration_records=1500,
                    signal_rows=len(signals), selected_unique_questions=len(cases), reviewed_cases=done,
                    unreviewed_cases=len(cases) - done,
                    status="MANUAL_REVIEW_PENDING" if done < len(cases) else "SELECTED_CASES_REVIEWED",
                    input_sha256=before, protected_snapshot_after=after, original_artifacts_unchanged=before == after,
                    manual_review_created=create_review, existing_manual_annotations_preserved=True,
                    review_sha256_before=review_hash_before, review_sha256_after=digest(review_path),
                    source_code_sha256=digest(Path(__file__)),
                    code_dependency_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                                            (ROOT / "src/answer_evaluation.py", ROOT / "experiments/analyze_rag_results.py",
                                             ROOT / "experiments/validate_rag_results.py")},
                    software_versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__),
                    audit_summary=audit, taxonomy=TAXONOMY, scoring="Original saved EM/F1 unchanged; validation reuses Step 10's frozen scoring checks",
                    exact_excerpts="Contiguous original corpus/prompt text, zero-based offsets, explicit omission markers; limit 900 characters per primary evidence passage",
                    missing_evidence=[], api_calls=0, models_fitted=0, network_access=False,
                    judgments="Manual fields initially blank; automated suggestions unverified. Reviewer labels apply only to primary A.",
                    output_sha256={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "failure_analysis_manifest.json"})
    (output / "failure_analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"150 test questions; 1,500 configuration records; {len(cases)} unique review questions; {done} reviewed / {len(cases) - done} unreviewed.")
    print(table(category_counts, [("scope", "Scope"), ("category", "Category"), ("count", "Count"), ("denominator", "Denominator")]))
    print("\nReranker and higher-K win/tie/loss counts (saved lexical metrics):")
    print(table([r for r in pairs if r["comparison"].startswith("reranker") or r["comparison"].split(" minus ")[0].split(" K=")[0] == r["comparison"].split(" minus ")[1].split(" K=")[0]],
                [("comparison", "Comparison"), ("metric", "Metric"), ("count", "Count"), ("denominator", "N")]))
    print("\nSource-present/EM-failure and abstention counts:")
    print(table([r for r in counts if r["metric"] in {"source_present_em_failure", "abstentions", "possible_lexical_mismatch"}],
                [("scope", "Scope"), ("comparison", "Configuration"), ("metric", "Signal"), ("count", "Count"), ("denominator", "N")]))
    print("\n" + manifest["status"] + ": causal/factual interpretations remain provisional until human review. Existing annotations preserved; original artifacts unchanged.")
    for path in sorted(output.iterdir()):
        print(path)
    print("Open manual_review.csv first; consult case_studies.md, fill judgment columns, then set reviewed=true and rerun.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        run()
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Failure analysis stopped without repairing inputs: {error}\n")


if __name__ == "__main__":
    main()
