"""Read-only CS391 result audit. No LLM requests, generation, fitting, or repairs.

Run: python -m experiments.validate_rag_results
Only the five audit reports under results/validation are written. Offline replay
uses the cached embedding model and predict_proba, never a training operation.
"""

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
import string
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("bm25_score", "semantic_similarity", "word_overlap", "passage_length")
INTENDED_CONFIGS = [("llm_only", 0)] + [(method, k) for k in (3, 5, 10) for method in ("bm25", "semantic", "reranker")]
STATUSES = ("PASS", "WARNING", "FAIL", "UNVERIFIED")
TOLERANCE = 1e-9


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def independent_normalize(text):
    """Separate implementation of the declared lowercase/ASCII punctuation/article policy."""
    if not isinstance(text, str):
        raise ValueError("Expected a string answer.")
    lowered = text.lower()
    unpunctuated = lowered.translate(str.maketrans("", "", string.punctuation))
    without_articles = re.sub(r"(?<!\w)(?:a|an|the)(?!\w)", " ", unpunctuated)
    return " ".join(without_articles.split())


def independent_scores(answer, references):
    """Recompute from references, not saved scores or the production metric function."""
    if not isinstance(references, list) or not references or any(not isinstance(r, str) or not r.strip() for r in references):
        raise ValueError("Missing or invalid accepted reference answers.")
    if answer is not None and not isinstance(answer, str):
        raise ValueError("Model answer must be a string or None.")
    text = answer or ""
    normalized = independent_normalize(text)
    abstained = normalized in {"insufficient evidence", "i dont know", "i do not know"}
    if not text.strip() or abstained:
        return {"exact_match": 0.0, "f1": 0.0, "empty_answer": not text.strip(), "abstained": abstained}
    exact, f1 = 0.0, 0.0
    predicted = normalized.split()
    for reference in references:
        gold = independent_normalize(reference)
        target = gold.split()
        exact = max(exact, float(normalized == gold))
        # Independently count the multiset intersection, with no production imports.
        shared = sum(min(predicted.count(token), target.count(token)) for token in set(predicted))
        if predicted and target and shared:
            precision, recall = shared / len(predicted), shared / len(target)
            f1 = max(f1, 2.0 * precision * recall / (precision + recall))
        elif not predicted and not target:
            # Explicit convention for nonempty article-only text; raw empties are zero above.
            f1 = max(f1, 1.0)
    return {"exact_match": exact, "f1": f1, "empty_answer": False, "abstained": abstained}


def finite_nonnegative(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def valid_count(value):
    return type(value) is int and value >= 0


def close(left, right, tolerance=TOLERANCE):
    return (not isinstance(left, bool) and not isinstance(right, bool)
            and isinstance(left, (int, float)) and isinstance(right, (int, float))
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance))


def question_key(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def record_key(row):
    qid, method, k = row.get("question_id"), row.get("retrieval_method"), row.get("top_k")
    return (qid, method, k) if isinstance(qid, str) and isinstance(method, str) and type(k) is int else None


def expected_request_key(row):
    """Reconstruct the logged request identity, excluding credentials and pacing."""
    settings, messages = row["model_settings"], row["messages"]
    if settings["provider"] == "anthropic":
        payload = {"model": settings["model"], "system": messages[0]["content"],
                   "messages": messages[1:], "max_tokens": settings["max_output_tokens"]}
    else:
        field = settings["max_tokens_field"]
        if field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Unknown output-token field")
        payload = {"model": settings["model"], "messages": messages, field: settings["max_output_tokens"]}
        if settings.get("seed") is not None:
            payload["seed"] = settings["seed"]
        if settings["provider"] == "openai":
            payload["store"] = False
    if settings.get("temperature") is not None:
        payload["temperature"] = settings["temperature"]
    return canonical_hash({"settings": settings, "payload": payload, "prompt_version": row["prompt_version"],
                           "question_id": row["question_id"], "retrieval_method": row["retrieval_method"],
                           "top_k": row["top_k"], "provenance": row.get("provenance", {})})


def infer_expected(manifest, test_questions):
    spec = manifest.get("frozen_spec", {})
    declared = spec.get("configurations")
    configs = [tuple(pair) for pair in declared] if (isinstance(declared, list) and all(
        isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str) and type(pair[1]) is int for pair in declared)) else list(INTENDED_CONFIGS)
    selected = manifest.get("phases", {}).get("test", {}).get("question_ids")
    ids = selected if isinstance(selected, list) and all(isinstance(qid, str) for qid in selected) else [r["question_id"] for r in test_questions]
    return configs, ids, "manifest" if declared and selected else "intended protocol fallback (manifest incomplete)"


def combination_audit(records, question_ids, configs):
    expected = {(qid, method, k) for qid in question_ids for method, k in configs}
    counts = Counter(key for row in records if (key := record_key(row)) is not None)
    return {"expected": expected, "counts": counts, "missing": sorted(expected - set(counts)),
            "unexpected": sorted(set(counts) - expected),
            "duplicates": {key: count for key, count in counts.items() if count > 1},
            "duplicate_excess": sum(count - 1 for count in counts.values() if count > 1)}


def protected_snapshot(root):
    """Hash original experimental artifacts, excluding the new audit-report directory."""
    snapshot = {}
    excluded = (root / "results" / "validation").resolve()
    for directory in (root / "data" / "processed", root / "models", root / "results"):
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.resolve().is_relative_to(excluded):
                snapshot[str(path.relative_to(root))] = digest(path)
    return snapshot


class Audit:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.output = self.root / "results" / "validation"
        self.checks, self.issues, self.inputs = [], [], {}
        self.failed_rows = set()

    def check(self, name, section, status, detail, evidence=None, critical=False):
        if status not in STATUSES:
            raise ValueError("Unknown audit status.")
        self.checks.append({"id": name, "section": section, "status": status,
                            "detail": detail, "evidence": evidence, "critical": critical})

    def issue(self, code, detail, row=None, severity="FAIL", source=None):
        line = row.get("_audit_line", "") if row else ""
        if row and severity == "FAIL" and source == "scored":
            self.failed_rows.add(line)
        self.issues.append({"severity": severity, "source": source or "experiment", "record_line": line,
                            "question_id": row.get("question_id", "") if row else "",
                            "method": row.get("retrieval_method", "") if row else "", "k": row.get("top_k", "") if row else "",
                            "issue_code": code, "detail": detail})

    def read_json(self, relative, required=True):
        path = self.root / relative
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            self.inputs[relative] = digest(path)
            return value
        except (OSError, ValueError) as exc:
            self.check(f"read:{relative}", "Inputs", "FAIL" if required else "UNVERIFIED",
                       f"Cannot read JSON object: {type(exc).__name__}", relative, critical=required)
            return {}

    def read_rows(self, relative, required=True):
        path = self.root / relative
        rows = []
        try:
            with path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("Expected object")
                    except ValueError:
                        row = {"_audit_malformed": True}
                        self.issue("malformed_jsonl", "Invalid JSON object; original line was not changed.",
                                   {"_audit_line": number}, source="scored" if relative == "results/cs391_per_question_results.jsonl" else relative)
                    rows.append({**row, "_audit_line": number})
            self.inputs[relative] = digest(path)
        except OSError as exc:
            self.check(f"read:{relative}", "Inputs", "FAIL" if required else "UNVERIFIED",
                       f"Cannot read JSONL: {type(exc).__name__}", relative, critical=required)
        return rows

    def read_csv(self, relative, required=True):
        path = self.root / relative
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fields, rows = list(reader.fieldnames or []), list(reader)
            self.inputs[relative] = digest(path)
            return fields, rows
        except OSError as exc:
            self.check(f"read:{relative}", "Inputs", "FAIL" if required else "UNVERIFIED",
                       f"Cannot read CSV: {type(exc).__name__}", relative, critical=required)
            return [], []

    def source(self, relative):
        try:
            return (self.root / relative).read_text(encoding="utf-8")
        except OSError:
            return ""

    def grouped_record_check(self, name, section, codes, description, severity="FAIL", critical=True):
        matching = [row for row in self.issues if row["source"] == "scored" and row["issue_code"] in codes]
        self.check(name, section, severity if matching else "PASS", description,
                   {"issue_count": len(matching), "issue_codes": sorted(codes)}, critical=critical)

    def dataset_checks(self, splits, corpus):
        maps, valid = {}, True
        for split, rows in splits.items():
            ids, normalized = [], []
            for row in rows:
                qid, text, pid = row.get("question_id"), row.get("question"), row.get("relevant_passage_id")
                answers = row.get("answers")
                usable = isinstance(qid, str) and bool(qid.strip()) and isinstance(text, str) and bool(text.strip())
                usable = usable and isinstance(pid, str) and pid in corpus and isinstance(answers, dict)
                if usable:
                    texts, starts = answers.get("text"), answers.get("answer_start")
                    usable = isinstance(texts, list) and isinstance(starts, list) and bool(texts) and len(texts) == len(starts)
                    if usable:
                        usable = all(isinstance(a, str) and a.strip() and type(start) is int and start >= 0
                                     and corpus[pid][start:start+len(a)] == a for a, start in zip(texts, starts))
                if not usable:
                    valid = False
                    self.issue("invalid_dataset_row", f"Invalid {split} question/reference mapping.", row, source=split)
                    continue
                ids.append(qid)
                normalized.append(question_key(text))
            if len(ids) != len(set(ids)) or len(normalized) != len(set(normalized)):
                valid = False
                self.issue("duplicate_dataset_question", f"Duplicate ID or normalized question in {split}.")
            maps[split] = {r["question_id"]: r for r in rows if isinstance(r.get("question_id"), str)}
        self.check("dataset_schema", "Datasets", "PASS" if valid else "FAIL", "Question text, source IDs, answer strings and character offsets checked.",
                   {split: len(rows) for split, rows in splits.items()}, critical=True)
        sizes = {name: len(rows) for name, rows in splits.items()}
        self.check("split_sizes", "Datasets", "PASS" if sizes == {"train": 500, "val": 50, "test": 150} else "FAIL",
                   "Original prepared split sizes.", sizes, critical=True)
        overlaps = {}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            ids = set(maps[left]) & set(maps[right])
            texts = {question_key(r["question"]) for r in maps[left].values() if isinstance(r.get("question"), str)} & {
                question_key(r["question"]) for r in maps[right].values() if isinstance(r.get("question"), str)}
            overlaps[f"{left}/{right}"] = {"ids": len(ids), "normalized_questions": len(texts)}
        self.check("split_isolation", "Leakage", "PASS" if all(v == {"ids": 0, "normalized_questions": 0} for v in overlaps.values()) else "FAIL",
                   "Question IDs and normalized question text must be disjoint across all three splits.", overlaps, critical=True)
        return maps

    def record_checks(self, records, question_map, corpus, manifest):
        settings = manifest.get("frozen_spec", {}).get("settings", {})
        prompt_version = manifest.get("frozen_spec", {}).get("prompt_version")
        system_hash = manifest.get("frozen_spec", {}).get("system_prompt_sha256")
        semantic_reference, bm25_reference, reference_reranker = {}, {}, {}
        recomputed = []
        for row in records:
            key = record_key(row)
            if key is None or row.get("question_id") not in question_map:
                self.issue("unknown_question_or_schema", "Record has an unexpected ID or schema.", row, source="scored")
                continue
            q = question_map[row["question_id"]]
            annotation = q.get("answers")
            if not isinstance(q.get("question"), str) or not isinstance(annotation, dict):
                self.issue("invalid_references", "Original question/reference schema is malformed; scores cannot be established.", row, source="scored")
                continue
            method, k = key[1:]
            if row.get("question") != q["question"]:
                self.issue("question_text_mismatch", "Question text differs from original test question.", row, source="scored")
            answer = row.get("generated_answer")
            if row.get("status") != "completed" or row.get("api_error"):
                self.issue("failed_response", "Response is failed, incomplete, or has a terminal API error.", row, source="scored")
            if not isinstance(answer, str) or not answer.strip():
                self.issue("empty_or_malformed_answer", "Generated answer is null, empty or not a string.", row, source="scored")
            elif not independent_normalize(answer) or any(ord(c) < 32 and c not in "\n\r\t" for c in answer):
                self.issue("empty_or_malformed_answer", "Answer has no normalized content or contains control characters.", row, source="scored")
            if row.get("finish_reason") in {"length", "max_tokens"}:
                self.issue("truncated_answer", "Provider explicitly reported the output-token cap.", row, source="scored")
            elif row.get("finish_reason") not in {"stop", "end_turn", "stop_sequence"}:
                self.issue("unknown_finish_reason", "Cannot establish normal response completion from finish_reason.", row, severity="WARNING", source="scored")
            elif valid_count(row.get("output_tokens")) and valid_count(settings.get("max_output_tokens")) and row["output_tokens"] >= settings["max_output_tokens"]:
                self.issue("at_output_cap", "Output reaches configured cap despite a normal stop; review, do not assume truncation.", row, severity="WARNING", source="scored")
            if isinstance(answer, str) and re.search(r'["\']?(?:references|answer_start|relevant_passage_id|question_id|answers)["\']?\s*[:=]', answer, re.I):
                self.issue("answer_metadata", "Answer appears to echo reference/result metadata; manual inspection required.", row, severity="WARNING", source="scored")
            if row.get("model_settings") != settings or row.get("model_name") != settings.get("model") or row.get("provider") != settings.get("provider"):
                self.issue("settings_mismatch", "Requested model/provider/generation settings differ from frozen manifest.", row, source="scored")
            if row.get("prompt_version") != prompt_version:
                self.issue("prompt_version_mismatch", "Prompt version differs from the frozen manifest.", row, source="scored")
            for field in ("retrieval_latency_ms", "generation_latency_ms"):
                if not finite_nonnegative(row.get(field)):
                    self.issue("invalid_latency", f"{field} is missing, negative or non-finite.", row, source="scored")
            for field in ("rate_limit_wait_ms", "generation_total_latency_ms"):
                if field in row and not finite_nonnegative(row[field]):
                    self.issue("invalid_latency", f"{field} is negative or non-finite.", row, source="scored")
            if all(finite_nonnegative(row.get(name)) for name in ("generation_total_latency_ms", "generation_latency_ms", "rate_limit_wait_ms")):
                if not close(row["generation_total_latency_ms"], row["generation_latency_ms"] + row["rate_limit_wait_ms"], 1e-6):
                    self.issue("timing_arithmetic", "Total generation time does not equal generation plus pacing time.", row, source="scored")
            if finite_nonnegative(row.get("generation_latency_ms")) and row["generation_latency_ms"] > 10000:
                self.issue("latency_outlier", "Generation exceeded 10 seconds, including retry/backoff or scheduling effects.", row, severity="WARNING", source="scored")
            for field in ("input_tokens", "output_tokens"):
                if row.get(field) is None:
                    self.issue("missing_usage", f"{field} was not recorded; no value invented.", row, severity="WARNING", source="scored")
                elif not valid_count(row[field]):
                    self.issue("invalid_usage", f"{field} must be a nonnegative integer.", row, source="scored")
            usage = row.get("usage")
            if isinstance(usage, dict):
                for saved, provider in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
                    if provider in usage and usage[provider] != row.get(saved):
                        self.issue("usage_mismatch", "Usage summary differs from provider token counts.", row, source="scored")
                if all(valid_count(usage.get(name)) for name in ("prompt_tokens", "completion_tokens", "total_tokens")):
                    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
                        self.issue("usage_mismatch", "Provider total_tokens is inconsistent.", row, source="scored")
            attempts, errors = row.get("request_attempts"), row.get("attempt_errors")
            if not valid_count(attempts) or not isinstance(errors, list) or (row.get("status") == "completed" and attempts != len(errors) + 1):
                self.issue("call_count_mismatch", "Attempt counter and retry history are inconsistent.", row, source="scored")
            if isinstance(errors, list) and errors:
                types = [{name: error.get(name) for name in ("type", "http_status", "provider_code")}
                         for error in errors if isinstance(error, dict)]
                self.issue("recovered_retry" if row.get("status") == "completed" else "retry_history",
                           f"Recorded retry history: {json.dumps(types)}. Failed-attempt token usage may be unavailable.",
                           row, severity="WARNING", source="scored")
            if row.get("cache_hit") not in (True, False):
                self.issue("invalid_cache_flag", "cache_hit must be boolean.", row, source="scored")
            pricing = row.get("pricing_usd_per_million", {})
            cost = row.get("estimated_cost_usd")
            if cost is not None:
                rates = [pricing.get(name) for name in ("input", "output")] if isinstance(pricing, dict) else []
                if len(rates) != 2 or not all(finite_nonnegative(v) for v in rates):
                    self.issue("unsupported_cost", "Cost was recorded without valid explicit pricing.", row, source="scored")
                elif all(valid_count(row.get(name)) for name in ("input_tokens", "output_tokens")):
                    expected_cost = (row["input_tokens"] * rates[0] + row["output_tokens"] * rates[1]) / 1e6
                    if not close(cost, expected_cost):
                        self.issue("cost_mismatch", "Cost does not match the configured base rates and recorded usage.", row, source="scored")
            ids = row.get("retrieved_passage_ids")
            ranking = row.get("retrieval_ranking")
            prompt_ok = True
            if not isinstance(ids, list) or any(not isinstance(pid, str) or pid not in corpus for pid in ids):
                self.issue("invalid_passage_id", "Prompt references an invalid corpus passage ID.", row, source="scored")
                ids, prompt_ok = [], False
            if len(ids) != len(set(ids)):
                self.issue("duplicate_passage_id", "Prompt passage IDs are duplicated.", row, source="scored")
            if len(ids) > k or row.get("passage_count") != len(ids):
                self.issue("wrong_k", "Passage count exceeds K or conflicts with saved passage_count.", row, source="scored")
            if method == "llm_only":
                if k != 0 or ids or ranking is not None:
                    self.issue("baseline_evidence", "No-retrieval baseline contains retrieval evidence.", row, source="scored")
            else:
                if len(ids) != min(k, len(corpus)):
                    self.issue("wrong_k", "Fewer than the requested passages despite an adequate corpus.", row, source="scored")
                if not isinstance(ranking, list) or len(ranking) != min(10, len(corpus)):
                    self.issue("invalid_ranking", "Full top-10 retrieval ranking is absent or wrong-sized.", row, source="scored")
                else:
                    ranked_ids = [r.get("passage_id") for r in ranking if isinstance(r, dict)]
                    scores = [r.get("score") for r in ranking if isinstance(r, dict)]
                    if len(ranked_ids) != len(ranking) or any(not isinstance(pid, str) or pid not in corpus for pid in ranked_ids) or len(ranked_ids) != len(set(ranked_ids)):
                        self.issue("invalid_ranking", "Ranking has unknown, duplicate or malformed IDs.", row, source="scored")
                    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores):
                        self.issue("invalid_ranking", "Ranking scores are not numeric and finite.", row, source="scored")
                    elif scores != sorted(scores, reverse=True):
                        self.issue("invalid_ranking", "Ranking scores are not descending.", row, source="scored")
                    if ids != ranked_ids[:k]:
                        self.issue("ranking_prefix", "Prompt passage IDs do not preserve the top-K ranked prefix.", row, source="scored")
                    reference = {"bm25": bm25_reference, "semantic": semantic_reference, "reranker": reference_reranker}.get(method)
                    if reference is not None:
                        previous = reference.setdefault(row["question_id"], ranking)
                        if previous != ranking:
                            self.issue("ranking_changes_across_k", "Top-10 ranking changed between K configurations.", row, source="scored")
            messages = row.get("messages")
            try:
                if not isinstance(messages, list) or len(messages) != 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
                    raise ValueError("Unexpected message roles/structure")
                if canonical_hash(messages[0]["content"]) != system_hash:
                    raise ValueError("System instruction differs from frozen hash")
                content = json.loads(messages[1]["content"])
                allowed = {"question", "condition", "evidence_policy"} | ({"source_passages"} if method != "llm_only" else set())
                if set(content) != allowed or content.get("question") != q["question"]:
                    raise ValueError("Question/reference/label or other unexpected fields in prompt")
                if method == "llm_only":
                    if content.get("condition") != "no-retrieval" or "source_passages" in content:
                        raise ValueError("Baseline is not the no-retrieval prompt")
                    expected_content = {"question": q["question"], "condition": "no-retrieval",
                                        "evidence_policy": "No external sources are supplied. Answer from existing knowledge, or respond 'I don't know.'"}
                else:
                    expected_sources = [{"source_number": i, "passage_id": pid, "text": corpus[pid]} for i, pid in enumerate(ids, 1)]
                    if not prompt_ok or content.get("condition") != "evidence-based" or content.get("source_passages") != expected_sources:
                        raise ValueError("Source numbers, order, IDs or corpus text differ from prompt")
                    expected_content = {"question": q["question"], "condition": "evidence-based", "source_passages": expected_sources,
                                        "evidence_policy": "Use only the source passages. If no answer is supported, respond 'Insufficient evidence.'"}
                if messages[1]["content"] != json.dumps(expected_content, ensure_ascii=False):
                    raise ValueError("Unexpected policy wording, duplicate JSON keys or additional prompt text")
            except (KeyError, ValueError, TypeError, AttributeError):
                self.issue("prompt_integrity", "Prompt structure/content differs from the audited source protocol; inspect potential leakage.", row, source="scored")
            provenance = row.get("provenance", {})
            expected_provenance = {"test_sha256": manifest.get("phases", {}).get("test", {}).get("dataset_sha256"),
                                   "corpus_sha256": manifest.get("frozen_spec", {}).get("corpus_sha256"),
                                   "reranker_sha256": manifest.get("frozen_spec", {}).get("reranker_sha256"),
                                   "selected_question_ids_sha256": canonical_hash(manifest.get("phases", {}).get("test", {}).get("question_ids", []))}
            if provenance != expected_provenance:
                self.issue("provenance_mismatch", "Record data/model provenance differs from frozen manifest.", row, source="scored")
            try:
                if row.get("cache_key") != expected_request_key(row):
                    raise ValueError("Request hash differs")
            except (KeyError, ValueError, TypeError, IndexError):
                self.issue("request_identity_mismatch", "Stored request hash is inconsistent with the exact prompt, generation settings and provenance.", row, source="scored")
            if row.get("references") != annotation.get("text") or row.get("relevant_passage_id") != q.get("relevant_passage_id"):
                self.issue("reference_mismatch", "Saved scoring reference differs from original test annotations.", row, source="scored")
            try:
                independent = independent_scores(answer if row.get("status") == "completed" else None, annotation.get("text"))
                for metric in ("exact_match", "f1"):
                    if not close(row.get(metric), independent[metric]):
                        self.issue("metric_mismatch", f"Saved {metric}={row.get(metric)!r}; independent value={independent[metric]!r}.", row, source="scored")
                if row.get("empty_answer") != independent["empty_answer"] or row.get("abstained") != independent["abstained"]:
                    self.issue("metric_mismatch", "Empty/abstention flags disagree with the declared metric policy.", row, source="scored")
                recomputed.append({"row": row, **independent})
            except (KeyError, ValueError):
                self.issue("invalid_references", "Cannot recompute scores from original reference answers.", row, source="scored")
        self.grouped_record_check("response_validity", "Completeness", {"failed_response", "empty_or_malformed_answer", "truncated_answer"}, "API success and valid text checked separately from correctness.")
        self.grouped_record_check("generation_consistency", "Consistency", {"settings_mismatch", "prompt_version_mismatch"}, "All records use frozen requested model and generation parameters.")
        self.grouped_record_check("questions_are_test", "Leakage", {"unknown_question_or_schema", "question_text_mismatch", "reference_mismatch"}, "Results refer to original held-out test IDs, text and annotations.")
        self.grouped_record_check("prompt_evidence_integrity", "Retrieval", {"prompt_integrity", "invalid_passage_id", "duplicate_passage_id", "wrong_k", "baseline_evidence"}, "Every saved prompt checked against original corpus; reference fields excluded structurally.")
        self.grouped_record_check("ranking_integrity", "Retrieval", {"invalid_ranking", "ranking_prefix", "ranking_changes_across_k"}, "Top-K prefixes, unique IDs, descending scores and cross-K order checked.")
        self.grouped_record_check("per_record_answer_metrics", "Metrics", {"metric_mismatch", "invalid_references"}, "Independent EM/F1 and abstention/empty flags compared against each saved row.")
        self.grouped_record_check("latency_integrity", "Usage and cost", {"invalid_latency", "timing_arithmetic"}, "Recorded timings are finite, nonnegative and arithmetically consistent.")
        self.grouped_record_check("token_usage_integrity", "Usage and cost", {"invalid_usage", "usage_mismatch"}, "Recorded token counts agree with provider usage, wherever present.")
        self.grouped_record_check("call_accounting_integrity", "Usage and cost", {"call_count_mismatch", "invalid_cache_flag"}, "Attempt/retry counters and local cache flags checked.")
        self.grouped_record_check("cost_arithmetic", "Usage and cost", {"unsupported_cost", "cost_mismatch"}, "Costs cannot be inferred without explicit rates; configured estimates checked.")
        self.grouped_record_check("record_provenance", "Consistency", {"provenance_mismatch"}, "Each record's test/corpus/reranker hashes and selected-ID hash checked.")
        self.grouped_record_check("request_identity", "Consistency", {"request_identity_mismatch"}, "Reconstructed request hashes bind the recorded prompts, settings, condition and input provenance.")
        return recomputed, {"bm25": bm25_reference, "semantic": semantic_reference, "reranker": reference_reranker}

    def recompute_aggregates(self, recomputed, configs, question_ids, saved_json, saved_csv):
        grouped = defaultdict(list)
        for record in recomputed:
            row = record["row"]
            grouped[(row["retrieval_method"], row["top_k"])].append(record)
        json_map = {(r.get("method"), r.get("k")): r for r in saved_json.get("aggregate", []) if isinstance(r, dict)}
        csv_map = {}
        for row in saved_csv:
            try:
                csv_map[(row["method"], int(row["k"]))] = row
            except (KeyError, TypeError, ValueError):
                self.issue("malformed_metrics_csv", "Malformed original aggregate CSV row.")
        table = []
        differences = 0
        for method, k in configs:
            values = grouped[(method, k)]
            # Never choose a duplicate's prediction or invent missing responses.
            unambiguous = len(values) == len(question_ids) and len({v["row"]["question_id"] for v in values}) == len(question_ids)
            em = math.fsum(v["exact_match"] for v in values) / len(question_ids) if unambiguous and question_ids else None
            f1 = math.fsum(v["f1"] for v in values) / len(question_ids) if unambiguous and question_ids else None
            saved = json_map.get((method, k), {})
            csv_saved = csv_map.get((method, k), {})
            comparisons = {}
            for metric, value in (("exact_match", em), ("f1", f1)):
                try:
                    csv_value = float(csv_saved[metric])
                except (KeyError, TypeError, ValueError):
                    csv_value = None
                comparisons[metric] = value is not None and close(saved.get(metric), value) and close(csv_value, value)
            if not all(comparisons.values()) or saved.get("questions") != len(question_ids) or saved.get("completed") != sum(v["row"].get("status") == "completed" for v in values):
                differences += 1
                self.issue("aggregate_metric_mismatch", f"{method} K={k}: original JSON/CSV score or denominator differs, or data is incomplete.")
            table.append({"method": method, "k": k, "expected_questions": len(question_ids), "observed_scored_records": len(values),
                          "recomputed_exact_match": em, "recomputed_f1": f1,
                          "original_exact_match": saved.get("exact_match"), "original_f1": saved.get("f1"),
                          "exact_match_difference": em - saved["exact_match"] if em is not None and isinstance(saved.get("exact_match"), (int, float)) else None,
                          "f1_difference": f1 - saved["f1"] if f1 is not None and isinstance(saved.get("f1"), (int, float)) else None,
                          "json_and_csv_match": all(comparisons.values()),
                          "status": "PASS" if all(comparisons.values()) and unambiguous else "FAIL"})
            # Verify usage, timing, cache and call aggregate accounting independently.
            rows = [v["row"] for v in values]
            fields = {"llm_calls_this_run": sum(r["request_attempts"] for r in rows if valid_count(r.get("request_attempts")) and not r.get("cache_hit")),
                      "original_llm_calls_including_cache": sum(r["request_attempts"] for r in rows if valid_count(r.get("request_attempts"))),
                      "cache_hits": sum(bool(r.get("cache_hit")) for r in rows),
                      "api_failures": sum(r.get("status") == "error" for r in rows),
                      "api_error_attempts_this_run": sum(len(r["attempt_errors"]) for r in rows if isinstance(r.get("attempt_errors"), list) and not r.get("cache_hit"))}
            for field, count in fields.items():
                if saved.get(field) != count:
                    self.issue("aggregate_usage_mismatch", f"{method} K={k}: {field} disagrees with record-level accounting.")
            for saved_field, record_field in (("avg_generation_latency_ms", "generation_latency_ms"), ("avg_retrieval_latency_ms", "retrieval_latency_ms"),
                                               ("avg_input_tokens", "input_tokens"), ("avg_output_tokens", "output_tokens")):
                known = [r[record_field] for r in rows if finite_nonnegative(r.get(record_field))]
                actual = math.fsum(known) / len(known) if known else None
                if actual is not None and not close(saved.get(saved_field), actual, 1e-7):
                    self.issue("aggregate_usage_mismatch", f"{method} K={k}: {saved_field} disagrees with record-level measurements.")
        self.check("aggregate_answer_metrics", "Metrics", "PASS" if differences == 0 and len(json_map) == len(configs) == len(csv_map) else "FAIL",
                   "Recomputed all configurations from original answers/references and compared original JSON and CSV; originals untouched.",
                   {"differing_configurations": differences, "tolerance": TOLERANCE}, critical=True)
        accounting_issues = [i for i in self.issues if i["issue_code"] == "aggregate_usage_mismatch"]
        self.check("aggregate_usage_accounting", "Usage and cost", "PASS" if not accounting_issues else "FAIL",
                   "Average timings/tokens, retries, calls and cached-response accounting cross-checked.", {"issues": len(accounting_issues)}, critical=True)
        return table

    def model_audit(self, maps, manifest, training_report):
        """Inspect fitted state and input provenance; never call fit or save."""
        model_path = self.root / "models/logistic_reranker.joblib"
        try:
            import joblib
            import numpy as np
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.utils.validation import check_is_fitted
            bundle = joblib.load(model_path)
            pipeline = bundle["pipeline"]
            check_is_fitted(pipeline)
            scaler, classifier = pipeline.named_steps["scaler"], pipeline.named_steps["logistic"]
            fitted = (isinstance(pipeline, Pipeline) and isinstance(scaler, StandardScaler)
                      and isinstance(classifier, LogisticRegression)
                      and np.isfinite(classifier.coef_).all() and classifier.coef_.shape == (1, 4)
                      and np.isfinite(classifier.intercept_).all() and list(classifier.classes_) == [0, 1])
            ordering = tuple(bundle.get("feature_columns", ())) == FEATURES and pipeline.n_features_in_ == 4
            self.check("fitted_model", "Reranker", "PASS" if fitted else "FAIL", "Saved pipeline loads and contains fitted scaler/logistic parameters.",
                       {"coefficient_shape": list(classifier.coef_.shape), "classes": classifier.classes_.tolist()}, critical=True)
            self.check("model_feature_order", "Reranker", "PASS" if ordering else "FAIL", "Artifact feature ordering and dimensionality match the four intended inputs.", list(bundle.get("feature_columns", ())), critical=True)
            metadata = bundle.get("metadata", {})
            model_hash = digest(model_path)
            self.inputs["models/logistic_reranker.joblib"] = model_hash
            self.check("frozen_model_hash", "Reranker", "PASS" if model_hash == manifest.get("frozen_spec", {}).get("reranker_sha256") else "FAIL",
                       "Current saved model equals the model frozen before the final test.", model_hash, critical=True)
            train_fields, train_rows = self.read_csv("data/processed/reranker_train.csv")
            val_fields, val_rows = self.read_csv("data/processed/reranker_val.csv")
            feature_valid = all(name in train_fields and name in val_fields for name in FEATURES)
            train_ids, val_ids = {row.get("question_id") for row in train_rows}, {row.get("question_id") for row in val_rows}
            exclusive = (train_ids == set(maps["train"]) and val_ids == set(maps["val"])
                         and not train_ids & (set(maps["val"]) | set(maps["test"])))
            self.check("training_question_membership", "Leakage", "PASS" if exclusive else "FAIL",
                       "Training CSV has exactly training-question IDs and no validation/test-question IDs.",
                       {"training_pairs": len(train_rows), "train_ids": len(train_ids), "val_ids": len(val_ids)}, critical=True)
            try:
                X = np.array([[float(row[name]) for name in FEATURES] for row in train_rows])
                feature_valid = feature_valid and X.shape == (len(train_rows), 4) and np.isfinite(X).all()
                train_labels = [int(row["label"]) for row in train_rows]
                feature_valid = feature_valid and set(train_labels) == {0, 1}
                means, variances = X.mean(axis=0), X.var(axis=0)
                train_only_scaler = (int(scaler.n_samples_seen_) == len(train_rows)
                                     and np.allclose(scaler.mean_, means, rtol=1e-10, atol=1e-10)
                                     and np.allclose(scaler.var_, variances, rtol=1e-10, atol=1e-10))
            except (TypeError, ValueError, KeyError):
                train_only_scaler, feature_valid, train_labels = False, False, []
            self.check("training_feature_schema", "Reranker", "PASS" if feature_valid else "FAIL", "Training/validation columns contain the intended numeric finite features and binary training labels.",
                       {"features": list(FEATURES), "training_class_counts": dict(Counter(train_labels))}, critical=True)
            self.check("scaler_training_population", "Reranker", "PASS" if train_only_scaler else "FAIL",
                       "Fitted scaler sample count, mean and variance match training CSV only.",
                       {"n_samples_seen": int(scaler.n_samples_seen_), "expected_training_pairs": len(train_rows)}, critical=True)
            trained_hashes = metadata.get("input_sha256", {})
            mismatches = []
            for relative, expected in trained_hashes.items():
                path = self.root / "results/reranker_dataset_statistics.json" if relative == "preparation_statistics" else self.root / "data/processed" / relative
                if not path.exists() or digest(path) != expected:
                    mismatches.append(relative)
            reported_metadata = training_report.get("model_metadata", {})
            self.check("training_input_provenance", "Reranker", "PASS" if trained_hashes and not mismatches and metadata == reported_metadata else "FAIL",
                       "Training input hashes and serialized model metadata match the saved training report.",
                       {"mismatched_inputs": mismatches, "metadata_matches_training_report": metadata == reported_metadata}, critical=True)
            params = {name: classifier.get_params().get(name) for name in ("C", "class_weight", "max_iter", "random_state", "solver", "fit_intercept")}
            recovered = params["C"] == metadata.get("selected_C") == training_report.get("selected_C") and params["max_iter"] == 1000 and params["random_state"] == 42
            self.check("training_hyperparameters", "Reranker", "PASS" if recovered else "FAIL", "Model hyperparameters and recorded validation-only C selection are recoverable.",
                       {"parameters": params, "selection_metric": training_report.get("selection_metric"), "iterations": classifier.n_iter_.tolist()}, critical=True)
            fit_source = self.source("experiments/train_reranker.py")
            train_code_support = all(text in fit_source for text in ("X = feature_matrix(train_rows)", "pipeline.fit(X, y)", "for row in train_rows"))
            self.check("training_code_uses_train_only", "Leakage", "PASS" if train_code_support else "UNVERIFIED",
                       "Current training code fits X/y constructed from training rows; validation is prediction-only. Scaler statistics provide an independent data-population check.",
                       "experiments/train_reranker.py", critical=not train_code_support)
            self.check("historical_classifier_fit_attestation", "Limitations", "UNVERIFIED",
                       "A joblib artifact does not independently reveal every row used in the classifier's historical fit. Code, metadata, input hashes and scaler statistics support the declared train-only workflow; no signed fit-event log was saved.")
            return {"bundle": bundle, "pipeline": pipeline, "metadata": metadata, "hyperparameters": params,
                    "training_pairs": len(train_rows), "class_counts": dict(Counter(train_labels))}
        except (OSError, ValueError, KeyError, AttributeError, TypeError, ImportError) as exc:
            self.check("fitted_model", "Reranker", "FAIL", f"Cannot establish fitted model state: {type(exc).__name__}: {exc}", critical=True)
            return None

    def replay(self, questions, model_info, rankings, records, enabled=True):
        """Replay local retrieval and classifier inference only; no answer generation."""
        if not enabled or model_info is None:
            self.check("offline_prediction_replay", "Reranker", "UNVERIFIED", "Offline replay disabled or the saved model could not be inspected.", critical=True)
            return
        prior = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TOKENIZERS_PARALLELISM")}
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        replay_errors, lexical_errors, cosine_errors = [], 0, 0
        selected = {(r.get("question_id"), r.get("retrieval_method")): r for r in records if r.get("top_k") == 3}
        try:
            import numpy as np
            from src.bm25 import BM25Retriever
            from src.semantic import SemanticRetriever
            from src.reranker_features import RerankerFeatureExtractor
            corpus_path = self.root / "data/processed/corpus.jsonl"
            bm25 = BM25Retriever(corpus_path)
            semantic = SemanticRetriever(corpus_path, revision=model_info["metadata"].get("model_revision"), local_files_only=True)
            extractor = RerankerFeatureExtractor(bm25, semantic)
            classifier = model_info["pipeline"]
            for index, q in enumerate(questions, 1):
                question_id, text = q["question_id"], q["question"]
                left, right = bm25.retrieve(text, top_k=10), semantic.retrieve(text, top_k=10)
                for name, actual in (("bm25", left), ("semantic", right)):
                    saved = rankings[name].get(question_id)
                    if not saved or [r["passage_id"] for r in actual] != [r["passage_id"] for r in saved] or not np.allclose(
                            [r["score"] for r in actual], [r["score"] for r in saved], rtol=1e-5, atol=1e-5):
                        replay_errors.append({"question_id": question_id, "method": name, "reason": "Saved retrieval does not match offline top10 replay."})
                candidate_ids = list(dict.fromkeys([r["passage_id"] for r in left + right]))
                for k in (3, 5, 10):
                    saved_rows = [r for r in records if r.get("question_id") == question_id and r.get("retrieval_method") == "reranker" and r.get("top_k") == k]
                    for saved_row in saved_rows:
                        if saved_row.get("candidate_passage_ids") != candidate_ids:
                            self.issue("non_natural_candidates", "Saved candidate union differs from BM25-top10/semantic-top10 union; no positives may be inserted.", saved_row, source="scored")
                features = extractor.extract(text, candidate_ids)
                query_tokens = re.findall(r"\b\w+\b", text.lower())
                query_set = set(query_tokens)
                query_vector = semantic._encode([text])[0]
                # Check each feature semantically rather than trusting its name.
                bm25_scores = bm25.index.get_scores(query_tokens)
                bm25_pos = {r["passage_id"]: i for i, r in enumerate(bm25.passages)}
                for feature in features:
                    pid = feature["passage_id"]
                    passage_tokens = re.findall(r"\b\w+\b", extractor.passages[pid].lower())
                    overlap = len(query_set & set(passage_tokens)) / len(query_set) if query_set else 0.0
                    if not close(feature["word_overlap"], overlap) or feature["passage_length"] != len(passage_tokens) or not close(feature["bm25_score"], float(bm25_scores[bm25_pos[pid]]), 1e-7):
                        lexical_errors += 1
                    vector = semantic.index.reconstruct(extractor.semantic_positions[pid])
                    cosine = float(np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector)))
                    if not close(feature["semantic_similarity"], cosine, 1e-5):
                        cosine_errors += 1
                X = np.array([[row[name] for name in FEATURES] for row in features], dtype=float)
                probabilities = classifier.predict_proba(X)[:, list(classifier.classes_).index(1)]
                actual = sorted([{"passage_id": pid, "score": float(p)} for pid, p in zip(candidate_ids, probabilities)],
                                key=lambda row: (-row["score"], row["passage_id"]))[:10]
                saved = rankings["reranker"].get(question_id)
                if not saved or [r["passage_id"] for r in actual] != [r["passage_id"] for r in saved] or not np.allclose(
                        [r["score"] for r in actual], [r["score"] for r in saved], rtol=1e-5, atol=1e-5):
                    replay_errors.append({"question_id": question_id, "method": "reranker", "reason": "Saved IDs/probabilities do not match the fitted classifier's predictions."})
                if index % 25 == 0:
                    print(f"Offline audit replay: {index}/{len(questions)} questions", flush=True)
            for error in replay_errors:
                row = selected.get((error["question_id"], error["method"]))
                self.issue("offline_replay_mismatch", error["reason"], row, source="scored")
            self.check("offline_prediction_replay", "Reranker", "PASS" if not replay_errors else "FAIL",
                       "Replayed BM25, semantic and fitted logistic predictions for every question using cached embeddings and no LLM API.",
                       {"questions": len(questions), "method_rankings": len(questions) * 3, "mismatches": replay_errors, "score_tolerance": 1e-5}, critical=True)
            self.check("feature_meaning", "Reranker", "PASS" if lexical_errors == 0 and cosine_errors == 0 else "FAIL",
                       "Checked BM25-index scores, cosine similarities, unique-query-token overlap and full-passage token counts on natural candidates.",
                       {"lexical_or_bm25_errors": lexical_errors, "cosine_errors": cosine_errors}, critical=True)
            self.grouped_record_check("natural_candidates_no_forced_positives", "Reranker", {"non_natural_candidates"},
                                      "Every saved reranker candidate list equals its replayed natural retrieval union; source labels never enter candidate mining.")
        except Exception as exc:
            # Missing cached model/dependency is not proof of incorrect predictions.
            self.check("offline_prediction_replay", "Reranker", "UNVERIFIED",
                       f"Local-only replay could not finish: {type(exc).__name__}: {exc}. No download or retraining was attempted.", critical=True)
        finally:
            for name, value in prior.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def run(self, replay=True):
        before = protected_snapshot(self.root)
        manifest = self.read_json("results/cs391_experiment_manifest.json")
        metrics = self.read_json("results/cs391_answer_metrics.json")
        training_report = self.read_json("results/reranker_val_metrics.json", required=False)
        preparation = self.read_json("data/processed/metadata.json", required=False)
        records = self.read_rows("results/cs391_per_question_results.jsonl")
        raw = self.read_rows("results/cs391/test/generation_raw.jsonl", required=False)
        _, metric_csv = self.read_csv("results/cs391_answer_metrics.csv")
        splits = {name: self.read_rows(f"data/processed/{name}.jsonl") for name in ("train", "val", "test")}
        corpus_rows = self.read_rows("data/processed/corpus.jsonl")
        corpus = {r["passage_id"]: r["text"] for r in corpus_rows
                  if isinstance(r.get("passage_id"), str) and isinstance(r.get("text"), str) and r["text"].strip()}
        valid_corpus = len(corpus) == len(corpus_rows) and len(set(corpus.values())) == len(corpus) and bool(corpus)
        self.check("corpus_unique", "Datasets", "PASS" if valid_corpus else "FAIL", "Corpus IDs/text are valid and exact contexts are deduplicated.",
                   {"rows": len(corpus_rows), "unique_ids": len(corpus), "unique_texts": len(set(corpus.values()))}, critical=True)
        maps = self.dataset_checks(splits, corpus)
        configs, ids, expected_basis = infer_expected(manifest, splits["test"])
        protocol_matches = configs == INTENDED_CONFIGS and len(ids) == 150 and set(ids) == set(maps["test"])
        self.check("intended_protocol", "Completeness", "PASS" if protocol_matches else "FAIL", "Manifest describes the intended 150 test questions and all ten configurations.",
                   {"configuration_count": len(configs), "question_count": len(ids), "basis": expected_basis}, critical=True)
        combinations = combination_audit(records, ids, configs)
        expected_count = len(ids) * len(configs)
        missing_count = len(combinations["missing"])
        complete = (len(records) == expected_count and not missing_count and not combinations["unexpected"] and not combinations["duplicates"])
        self.check("all_combinations_exactly_once", "Completeness", "PASS" if complete else "FAIL",
                   "Every manifest question/configuration pair exists exactly once, with no missing or unexpected pairs.",
                   {"expected": expected_count, "actual": len(records), "missing": missing_count,
                    "duplicate_excess": combinations["duplicate_excess"], "unexpected": len(combinations["unexpected"])}, critical=True)
        for row in records:
            key = record_key(row)
            if key in combinations["duplicates"]:
                self.issue("duplicate_combination", "Question/configuration occurs more than once; neither row was deleted or chosen automatically.", row, source="scored")
            if key is not None and key not in combinations["expected"]:
                self.issue("unexpected_combination", "Question/configuration was not part of the declared final experiment.", row, source="scored")
        for qid, method, k in combinations["missing"]:
            self.issue("missing_combination", "Expected question/configuration is absent; no answer was fabricated.",
                       {"question_id": qid, "retrieval_method": method, "top_k": k}, source="experiment")
        configuration_counts = []
        for method, k in configs:
            subset = [r for r in records if r.get("retrieval_method") == method and r.get("top_k") == k]
            distinct = {r.get("question_id") for r in subset if isinstance(r.get("question_id"), str)}
            configuration_counts.append({"method": method, "k": k, "expected_records": len(ids), "actual_records": len(subset),
                                         "distinct_question_ids": len(distinct), "completed": sum(r.get("status") == "completed" for r in subset),
                                         "missing_question_ids": len(set(ids) - distinct), "unexpected_question_ids": len(distinct - set(ids)),
                                         "status": "PASS" if len(subset) == len(ids) and distinct == set(ids) else "FAIL"})
        self.check("configuration_coverage", "Completeness", "PASS" if all(r["status"] == "PASS" for r in configuration_counts) else "FAIL",
                   "Each configuration contains exactly the same 150 distinct test IDs.", configuration_counts, critical=True)
        frozen = manifest.get("frozen_spec", {})
        frozen_checks = {}
        for path, expected in frozen.get("code_sha256", {}).items():
            actual = self.root / path
            frozen_checks[path] = actual.exists() and digest(actual) == expected
        self.check("frozen_source_code", "Consistency", "PASS" if frozen_checks and all(frozen_checks.values()) else "FAIL",
                   "Current inference/scoring source files match their pre-test frozen hashes.", frozen_checks, critical=True)
        self.check("manifest_identity", "Consistency", "PASS" if frozen and manifest.get("experiment_id") == canonical_hash(frozen) else "FAIL",
                   "The manifest experiment identifier equals the canonical hash of its frozen specification.", critical=True)
        current_hashes = {"corpus": self.inputs.get("data/processed/corpus.jsonl"), "test": self.inputs.get("data/processed/test.jsonl"),
                          "val": self.inputs.get("data/processed/val.jsonl")}
        matched_inputs = (current_hashes["corpus"] == frozen.get("corpus_sha256")
                          and current_hashes["test"] == manifest.get("phases", {}).get("test", {}).get("dataset_sha256")
                          and current_hashes["val"] == frozen.get("validation_sha256"))
        self.check("frozen_data_hashes", "Consistency", "PASS" if matched_inputs else "FAIL", "Original test/validation/corpus files match frozen experiment provenance.", current_hashes, critical=True)
        prep_matches = {}
        for name, expected in preparation.get("sha256", {}).items():
            path = self.root / "data/processed" / name
            prep_matches[name] = path.exists() and digest(path) == expected
        self.check("original_prepared_data", "Consistency", "PASS" if prep_matches and all(prep_matches.values()) else "FAIL" if prep_matches else "UNVERIFIED",
                   "Prepared data still match the initial preparation checksums.", prep_matches, critical=not bool(prep_matches))
        phase = manifest.get("phases", {}).get("test", {})
        self.check("test_not_validation", "Leakage", "PASS" if metrics.get("phase") == "test" and phase.get("sample_size") == 150 and set(ids) == set(maps["test"]) and not set(ids) & set(maps["val"]) else "FAIL",
                   "Final aggregate and selected question IDs describe test data rather than validation examples.", critical=True)
        try:
            frozen_time = datetime.fromisoformat(manifest["frozen_at_utc"])
            start_time = datetime.fromisoformat(phase["started_at_utc"])
            pilot_finish = datetime.fromisoformat(manifest["phases"]["pilot"]["finished_at_utc"])
            timing_ok = manifest.get("pilot_passed") is True and pilot_finish <= frozen_time <= start_time
        except (ValueError, TypeError, KeyError):
            timing_ok = False
        self.check("pilot_precedes_frozen_test", "Consistency", "PASS" if timing_ok else "UNVERIFIED",
                   "Saved pilot/freeze/test timestamps establish the intended order; timestamps are local audit evidence, not signed external attestations.", critical=not timing_ok)
        recomputed, rankings = self.record_checks(records, maps["test"], corpus, manifest)
        metric_table = self.recompute_aggregates(recomputed, configs, ids, metrics, metric_csv)
        raw_keys = {record_key(r): r for r in raw if record_key(r) is not None}
        mirror_ok = bool(raw) and len(raw_keys) == len(raw) == len(records)
        if mirror_ok:
            for row in records:
                original = raw_keys.get(record_key(row))
                if not original or any(row.get(key) != value for key, value in original.items() if not key.startswith("_audit_")):
                    mirror_ok = False
                    self.issue("raw_scored_mismatch", "Scored result changed a raw generation field.", row, source="scored")
        self.check("raw_scored_consistency", "Consistency", "PASS" if mirror_ok else "FAIL" if raw else "UNVERIFIED",
                   "Answers, prompts, settings, IDs, usage and timings in scored data match the independent raw-generation artifact.", critical=True)
        archive_path = self.root / "results/cs391/test/cs391_per_question_results.jsonl"
        self.check("test_archive_matches", "Consistency", "PASS" if archive_path.exists() and digest(archive_path) == self.inputs.get("results/cs391_per_question_results.jsonl") else "FAIL" if archive_path.exists() else "UNVERIFIED",
                   "Root-level results match the archived test-phase scored data.")
        missing_cache, mismatched_cache = 0, 0
        for row in records:
            key = row.get("cache_key")
            if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
                continue
            cache_path = self.root / "results/generation/cache" / f"{key}.json"
            if not cache_path.exists():
                missing_cache += 1
                continue
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                fields = ("question_id", "question", "retrieval_method", "top_k", "retrieved_passage_ids", "generated_answer",
                          "model_name", "model_settings", "prompt_version", "messages", "input_tokens", "output_tokens",
                          "response_id", "request_attempts", "cache_key", "status")
                if any(cached.get(field) != row.get(field) for field in fields):
                    raise ValueError("Cache differs")
            except (OSError, ValueError, AttributeError):
                mismatched_cache += 1
                self.issue("cache_artifact_mismatch", "Completed-answer cache conflicts with the saved generation record.", row, source="scored")
        self.check("completed_cache_consistency", "Consistency", "FAIL" if mismatched_cache else "UNVERIFIED" if missing_cache else "PASS",
                   "Completed-answer cache artifacts agree with saved raw/scored request and response fields; absent caches are not invented.",
                   {"missing": missing_cache, "mismatched": mismatched_cache})
        models = Counter(str(r.get("response_model")) for r in records)
        requested = frozen.get("settings", {}).get("model")
        self.check("returned_model_identity", "Consistency", "PASS" if set(models) == {requested} else "FAIL" if all(r.get("response_model") for r in records) else "UNVERIFIED",
                   "All recorded provider-returned model IDs match the configured fixed model.", dict(models), critical=True)
        fingerprints = Counter(r.get("system_fingerprint") for r in records if r.get("system_fingerprint"))
        self.check("provider_backend_variation", "Limitations", "WARNING" if len(fingerprints) > 1 else "PASS",
                   "Backend fingerprints can vary despite a fixed model, seed and temperature. Local logs do not establish bit-for-bit reproducibility.", {"distinct_fingerprints": len(fingerprints)})
        model_info = self.model_audit(maps, manifest, training_report)
        inference_files = ("experiments/evaluate_rag.py", "experiments/generate_answers.py", "src/reranker.py", "src/reranker_features.py", "src/bm25.py", "src/semantic.py")
        fit_calls = []
        for relative in inference_files:
            source = self.source(relative)
            try:
                tree = ast.parse(source)
                fit_calls.extend(f"{relative}:{node.lineno}" for node in ast.walk(tree) if isinstance(node, ast.Call)
                                 and isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform", "partial_fit"})
            except SyntaxError:
                fit_calls.append(f"{relative}:unparseable")
        self.check("inference_no_training_calls", "Reranker", "PASS" if not fit_calls and all(self.source(p) for p in inference_files) else "FAIL",
                   "Pinned final inference source uses model load/predict, with no fit/fit_transform/partial_fit call.", fit_calls, critical=True)
        self.replay(splits["test"], model_info, rankings, records, enabled=replay)
        summary = self.summarize(records, expected_count, combinations, fingerprints)
        self.check("incomplete_relevance_labels", "Limitations", "WARNING",
                   "Natural Questions provides one annotated source per question here. Other passages may also be relevant; a source miss is not proof of no useful evidence.")
        self.check("strict_string_scoring", "Limitations", "WARNING",
                   "Normalized EM/F1 are lexical reference scores, not factual adjudication. Descriptive spans, aliases and temporal ambiguity can penalize otherwise plausible answers.")
        previous_test = self.root / "results/bm25_evaluation.json"
        previous_report = self.source("results/cs391_answer_metrics.json")
        if previous_test.exists() or "earlier BM25-only" in previous_report:
            self.check("prior_test_exposure", "Limitations", "WARNING",
                       "Existing reports disclose an earlier BM25-only test retrieval evaluation. Do not claim the test set was historically untouched throughout project development.")
        self.check("external_api_attestation", "Limitations", "UNVERIFIED",
                   "Local response IDs, settings and usage are internally auditable, but no independently signed provider request log or billing ledger was supplied. The audit cannot prove server-side parameter handling or an exact bill.")
        after = protected_snapshot(self.root)
        changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        self.check("audit_read_only", "Protection", "PASS" if not changed else "FAIL", "All pre-existing prepared data, models, generation caches, metrics, results and figures hashed before/after the audit.",
                   {"protected_files": len(before), "changed_files": changed}, critical=True)
        summary["valid_records"] = len(records) - len(self.failed_rows)
        counts = Counter(check["status"] for check in self.checks)
        summary.update({"passing_checks": counts["PASS"], "warnings": counts["WARNING"], "failed_checks": counts["FAIL"], "unverified_checks": counts["UNVERIFIED"]})
        critical_unknown = any(c["status"] == "UNVERIFIED" and c["critical"] for c in self.checks)
        readiness = "CORRECTIONS_REQUIRED" if counts["FAIL"] else "ADDITIONAL_EVIDENCE_REQUIRED" if critical_unknown else "READY_WITH_DOCUMENTED_LIMITATIONS" if counts["WARNING"] or counts["UNVERIFIED"] else "READY"
        summary["overall_readiness"] = readiness
        report = {"audit_version": "cs391-integrity-audit-v1", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
                  "summary": summary, "checks": self.checks, "input_sha256": self.inputs,
                  "expected_configurations": [list(pair) for pair in configs], "expected_basis": expected_basis,
                  "protected_snapshot_before": before, "protected_snapshot_after": after,
                  "configuration_counts": configuration_counts, "recomputed_metrics": metric_table,
                  "issues": self.issues,
                  "repair_plan": self.repair_plan(),
                  "metric_policy": "Independent lowercase/ASCII-punctuation/English-article/whitespace normalization; multiset F1; maximum over original accepted references. Raw empty/contract abstentions are zero. No production score function called.",
                  "constraints": {"new_llm_api_requests": 0, "model_fits": 0, "original_records_deleted_or_modified": False,
                                  "writes_limited_to": "results/validation (audit reports only)"}}
        self.write_reports(report)
        self.print_summary(summary)
        return report

    def summarize(self, records, expected_count, combinations, fingerprints):
        completed = [r for r in records if r.get("status") == "completed"]
        empty = sum(not isinstance(r.get("generated_answer"), str) or not r.get("generated_answer", "").strip() for r in records)
        truncated = sum(r.get("finish_reason") in {"length", "max_tokens"} for r in records)
        retry_errors = sum(len(r.get("attempt_errors", [])) for r in records if isinstance(r.get("attempt_errors"), list))
        known_input = [r["input_tokens"] for r in records if valid_count(r.get("input_tokens"))]
        known_output = [r["output_tokens"] for r in records if valid_count(r.get("output_tokens"))]
        attempts = sum(r["request_attempts"] for r in records if valid_count(r.get("request_attempts")))
        new_attempts = sum(r["request_attempts"] for r in records if valid_count(r.get("request_attempts")) and not r.get("cache_hit"))
        known_cost = [r["estimated_cost_usd"] for r in records if finite_nonnegative(r.get("estimated_cost_usd"))]
        self.check("recovered_api_errors", "Usage and cost", "WARNING" if retry_errors else "PASS",
                   "Transient errors during retries are distinct from failed final responses; failed-attempt usage may be unreported.",
                   {"recovered_or_failed_attempts": retry_errors, "recorded_attempts": attempts})
        missing_usage = len(records) - min(len(known_input), len(known_output))
        self.check("usage_availability", "Usage and cost", "WARNING" if missing_usage else "PASS", "Missing token usage is reported, never inferred.", {"records_missing_usage": missing_usage})
        self.check("pricing_availability", "Usage and cost", "WARNING" if len(known_cost) != len(records) else "PASS",
                   "No monetary total is invented when explicit prices are absent. Recorded base-rate estimates are not provider invoices.",
                   {"records_with_estimated_cost": len(known_cost), "unknown_cost_records": len(records) - len(known_cost)})
        latency_outliers = [i for i in self.issues if i["issue_code"] == "latency_outlier"]
        self.check("latency_outliers", "Usage and cost", "WARNING" if latency_outliers else "PASS",
                   "Generation timings above 10 seconds should be reviewed and retained, not silently trimmed; they can include retries or scheduling effects.", {"records_above_10_seconds": len(latency_outliers)})
        timeline, missing_timestamps = [], 0
        for row in records:
            if row.get("status") != "completed" or row.get("cache_hit"):
                continue
            try:
                stamp = datetime.fromisoformat(row["created_at_utc"])
                if stamp.tzinfo is None:
                    raise ValueError("Timestamp lacks timezone")
                timeline.append((stamp.timestamp(), row))
            except (KeyError, ValueError, TypeError):
                missing_timestamps += 1
        chronological = all(right[0] >= left[0] for left, right in zip(timeline, timeline[1:]))
        self.check("completion_timestamps", "Usage and cost", "UNVERIFIED" if missing_timestamps else "PASS" if chronological else "FAIL",
                   "Fresh-response timestamps are timezone-qualified and chronologically ordered; historical cache hits are excluded.",
                   {"missing_or_invalid": missing_timestamps, "fresh_timestamp_count": len(timeline)})
        gaps = []
        for (previous, _), (current, row) in zip(timeline, timeline[1:]):
            gap = current - previous
            total_ms = row.get("generation_total_latency_ms")
            if finite_nonnegative(total_ms) and gap - total_ms / 1000 > 30:
                gaps.append({"record_line": row.get("_audit_line"), "inter_completion_seconds": gap,
                             "current_recorded_generation_total_seconds": total_ms / 1000,
                             "interval_outside_recorded_generation_seconds": gap - total_ms / 1000})
                self.issue("wall_clock_gap", f"Inter-completion interval {gap:.3f}s exceeds the current record's total generation timer {total_ms/1000:.3f}s by over 30s. Cause is unverified; inspect host/network scheduling before latency claims.",
                           row, severity="WARNING", source="scored")
        self.check("wall_clock_gaps", "Usage and cost", "WARNING" if gaps else "PASS",
                   "Long wall-clock intervals are not silently equated with LLM inference latency. The logs alone do not identify host suspension, scheduling, network or other causes; treat timing conclusions cautiously.", gaps)
        return {"expected_records": expected_count, "actual_records": len(records), "valid_records": None,
                "unique_test_question_ids": len({r.get("question_id") for r in records if isinstance(r.get("question_id"), str)}),
                "missing_records": len(combinations["missing"]), "duplicate_records": combinations["duplicate_excess"],
                "duplicate_combinations": len(combinations["duplicates"]), "unexpected_combinations": len(combinations["unexpected"]),
                "completed_responses": len(completed), "failed_responses": sum(r.get("status") != "completed" or bool(r.get("api_error")) for r in records),
                "empty_or_malformed_responses": empty, "truncated_responses": truncated,
                "input_tokens": sum(known_input), "output_tokens": sum(known_output),
                "reported_total_tokens": sum(known_input) + sum(known_output), "usage_missing_records": missing_usage,
                "recorded_api_attempts": attempts, "recorded_new_api_attempts": new_attempts,
                "local_cache_hits": sum(bool(r.get("cache_hit")) for r in records), "api_retry_errors": retry_errors,
                "api_calls_made_by_audit": 0,
                "estimated_cost_usd": sum(known_cost) if len(known_cost) == len(records) and records else None,
                "unknown_cost_records": len(records) - len(known_cost), "backend_fingerprint_count": len(fingerprints),
                "long_wall_clock_gaps": len(gaps),
                "fresh_first_to_last_completion_seconds": timeline[-1][0] - timeline[0][0] if timeline else None}

    def repair_plan(self):
        plan = []
        codes = {row["issue_code"] for row in self.issues}
        if codes & {"missing_combination", "duplicate_combination", "unexpected_combination", "malformed_jsonl"}:
            plan.append("Reconcile combination discrepancies against raw generation/cache artifacts into a NEW versioned dataset only after review. Preserve original files; do not choose duplicates by answer correctness.")
        if codes & {"metric_mismatch", "aggregate_metric_mismatch", "invalid_references"}:
            plan.append("Review normalization/reference-policy discrepancies with the independent recomputation. If correction is approved, produce versioned corrected metric reports for all conditions uniformly, without new LLM calls or overwriting originals.")
        if codes & {"prompt_integrity", "non_natural_candidates", "offline_replay_mismatch", "settings_mismatch", "provenance_mismatch"}:
            plan.append("Investigate affected prompts/rankings/settings with the frozen code, source checksums and provider records. Exclude or rerun nothing automatically; disclose any necessary new experiment as a separate version.")
        if codes & {"empty_or_malformed_answer", "failed_response", "truncated_answer"}:
            plan.append("Review failed/truncated outputs separately from answer errors. Any future regeneration requires explicit approval and a versioned protocol; preserve original failure counts and answers.")
        plan.append("Document incomplete relevance/answer annotations, strict lexical scoring, prior test exposure, backend fingerprint variation, and retry-related timing/usage limitations in the research paper.")
        plan.append("For stronger historical verification, obtain provider request/billing logs and a signed training-row/code provenance record if available. Do not fabricate missing evidence.")
        if any(check["id"] == "wall_clock_gaps" and check["status"] == "WARNING" for check in self.checks):
            plan.append("Before making strong computational-speed claims, review the reported long wall-clock gaps with host/network/provider telemetry if available. Retain all recorded timings and report a transparent sensitivity analysis; do not automatically trim or replace observations.")
        return plan

    def write_reports(self, report):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "experiment_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for filename, rows, fallback in (
            ("record_issues.csv", self.issues, ["severity", "source", "record_line", "question_id", "method", "k", "issue_code", "detail"]),
            ("configuration_counts.csv", report["configuration_counts"], []),
            ("recomputed_metrics.csv", report["recomputed_metrics"], []),
        ):
            with (self.output / filename).open("w", encoding="utf-8", newline="") as handle:
                fields = list(rows[0]) if rows else fallback
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        summary = report["summary"]
        lines = ["# CS391 final experiment integrity audit", "", f"**Readiness: {summary['overall_readiness']}**", "",
                 "This audit is read-only with respect to original experimental artifacts. It made no LLM API calls and fitted no model.", "",
                 "## Experiment overview", "", "The intended study contains 150 Natural Questions test questions and ten conditions: one LLM-only baseline and BM25, semantic and saved logistic reranker RAG at K=3,5,10.", "",
                 "## Expected versus actual records", "", "| Item | Value |", "|---|---:|"]
        for key in ("expected_records", "actual_records", "valid_records", "unique_test_question_ids", "missing_records", "duplicate_records", "unexpected_combinations", "failed_responses", "empty_or_malformed_responses", "truncated_responses"):
            lines.append(f"| {key} | {summary[key]} |")
        lines.extend(["", "API success is distinct from answer correctness. Abstentions are valid completed responses but score zero on this answer-bearing dataset. A record is valid here only if it has no confirmed record-level FAIL issue.", "",
                      "## Configuration completeness", "", "| Method | K | Expected | Actual | Distinct IDs | Status |", "|---|---:|---:|---:|---:|---|"])
        for row in report["configuration_counts"]:
            lines.append(f"| {row['method']} | {row['k']} | {row['expected_records']} | {row['actual_records']} | {row['distinct_question_ids']} | {row['status']} |")
        sections = ["Completeness", "Consistency", "Metrics", "Reranker", "Leakage", "Datasets", "Retrieval", "Usage and cost", "Protection", "Limitations", "Inputs"]
        for section in sections:
            checks = [c for c in self.checks if c["section"] == section]
            if not checks:
                continue
            lines.extend(["", f"## {section}", ""])
            for check in checks:
                lines.append(f"- **{check['status']} — {check['id']}**: {check['detail']}")
                if check["evidence"] is not None:
                    evidence = json.dumps(check["evidence"], ensure_ascii=False, sort_keys=True)
                    lines.append(f"  Evidence: `{evidence.replace('`', '')}`")
        lines.extend(["", "## Independent metric recomputation", "", report["metric_policy"], "",
                      "`recomputed_metrics.csv` compares each independently recalculated aggregate with the original JSON/CSV. No saved answer or metric was replaced. Ambiguous duplicate/missing inputs do not receive invented replacement scores.", "",
                      "## Token, latency and cost summary", "", "| Quantity | Value |", "|---|---:|"])
        for key in ("input_tokens", "output_tokens", "reported_total_tokens", "recorded_api_attempts", "recorded_new_api_attempts", "api_retry_errors", "local_cache_hits", "api_calls_made_by_audit", "estimated_cost_usd", "unknown_cost_records"):
            value = summary[key] if summary[key] is not None else "Not available: no complete explicit pricing"
            lines.append(f"| {key} | {value} |")
        lines.extend(["", "Recorded attempts describe the original generation run, not this audit. Token totals cover reported successful-response usage; retry-time usage can be unreported. No provider bill was inferred.", "",
                      "## Check totals", "", f"PASS={summary['passing_checks']}; WARNING={summary['warnings']}; FAIL={summary['failed_checks']}; UNVERIFIED={summary['unverified_checks']}.", "",
                      "## Readiness and proposed actions", ""])
        if summary["overall_readiness"] in {"READY", "READY_WITH_DOCUMENTED_LIMITATIONS"}:
            lines.append("The saved records are ready for controlled lexical-metric answer-quality statistical analysis with the limitations above. Latency conclusions require the documented retry and wall-clock-gap caveats. This does not establish causal superiority, exhaustive relevance, factual correctness of every reference, or perfect provider-side reproducibility.")
        else:
            lines.append("Resolve the confirmed failures or obtain the missing critical evidence before treating the experiment as ready for final analysis. Preserve the original data and report all discrepancies.")
        lines.extend(["", *[f"{index}. {action}" for index, action in enumerate(report["repair_plan"], 1)], ""])
        (self.output / "experiment_audit.md").write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def print_summary(summary):
        print("\nCS391 FINAL EXPERIMENT AUDIT")
        for key in ("expected_records", "actual_records", "valid_records", "missing_records", "duplicate_records", "failed_responses",
                    "empty_or_malformed_responses", "truncated_responses", "passing_checks", "warnings", "failed_checks", "unverified_checks", "overall_readiness"):
            print(f"{key}: {summary[key]}")
        print("Audit reports: results/validation/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-replay", action="store_true", help="Skip local encoder/predict replay and mark that check UNVERIFIED")
    args = parser.parse_args()
    report = Audit(args.root).run(replay=not args.skip_replay)
    if report["summary"]["overall_readiness"] in {"CORRECTIONS_REQUIRED", "ADDITIONAL_EVIDENCE_REQUIRED"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
