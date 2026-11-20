"""Generate answers for validation questions under four fixed retrieval conditions.

Default: first five validation questions, top three passages, 20 API requests
before caching. --dry-run prints/saves exact prompts and never calls an LLM.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

from src.generation import AnswerGenerator, GENERATION_DIR, GenerationConfig, PROMPT_VERSION, json_hash


ROOT = Path(__file__).resolve().parents[1]
VAL_PATH = ROOT / "data" / "processed" / "val.jsonl"
CORPUS_PATH = VAL_PATH.with_name("corpus.jsonl")
MODEL_PATH = ROOT / "models" / "logistic_reranker.joblib"
CONDITIONS = ("llm_only", "bm25", "semantic", "reranker")


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_validation_questions(limit=5):
    """Read only IDs and question text for generation; never expose annotations."""
    if type(limit) is not int or limit < 1:
        raise ValueError("--limit must be a positive integer.")
    rows, seen = [], set()
    with VAL_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Validation rows must be JSON objects.")
            question_id, question = row.get("question_id"), row.get("question")
            if not isinstance(question_id, str) or not question_id.strip() or question_id in seen:
                raise ValueError("Validation question IDs must be nonempty and unique.")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Validation question text must be nonempty.")
            seen.add(question_id)
            rows.append({"question_id": question_id, "question": question})
            if len(rows) == limit:
                break
    if len(rows) != limit:
        raise ValueError(f"Requested {limit} validation questions but found {len(rows)}.")
    return rows


class RetrievalConditions:
    """Load each retriever and the saved reranker once; never fit a model."""

    def __init__(self, local_files_only=False):
        # Lazy imports let a missing API key fail fast before loading embeddings.
        from src.bm25 import BM25Retriever
        from src.reranker import LogisticReranker
        from src.reranker_features import RerankerFeatureExtractor
        from src.semantic import SemanticRetriever

        started = time.perf_counter()
        self.reranker = LogisticReranker.load(MODEL_PATH)
        expected_hash = self.reranker.metadata.get("corpus_sha256")
        if not expected_hash or file_hash(CORPUS_PATH) != expected_hash:
            raise ValueError("Saved reranker does not match the fixed passage corpus.")
        self.bm25 = BM25Retriever(CORPUS_PATH)
        self.semantic = SemanticRetriever(CORPUS_PATH, local_files_only=local_files_only,
                                           revision=self.reranker.metadata.get("model_revision"))
        self.extractor = RerankerFeatureExtractor(self.bm25, self.semantic)
        if len(set(self.extractor.passages.values())) != len(self.extractor.passages):
            raise ValueError("Corpus contains duplicate passage text.")
        self.reranker.extractor = self.extractor
        self.setup_latency_ms = (time.perf_counter() - started) * 1000
        self.setup_details_ms = self.semantic.setup_timings_ms

    def _checked(self, results):
        seen = set()
        for row in results:
            passage_id = row["passage_id"]
            if passage_id in seen or passage_id not in self.extractor.passages or row["text"] != self.extractor.passages[passage_id]:
                raise ValueError("Retrieved passage IDs or text do not match the corpus.")
            seen.add(passage_id)
        return results

    def retrieve(self, question, top_k=3):
        from src.reranker_features import candidate_union

        started = time.perf_counter()
        bm25 = self._checked(self.bm25.retrieve(question=question, top_k=10))
        bm25_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        semantic = self._checked(self.semantic.retrieve(question=question, top_k=10))
        semantic_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        union = list(candidate_union(bm25, semantic))
        reranked = self._checked(self.reranker.rerank(question=question, candidate_passages=union))
        rerank_ms = (time.perf_counter() - started) * 1000
        return {
            "llm_only": {"passages": None, "retrieval_latency_ms": 0.0},
            "bm25": {"passages": bm25[:top_k], "retrieval_latency_ms": bm25_ms},
            "semantic": {"passages": semantic[:top_k], "retrieval_latency_ms": semantic_ms},
            "reranker": {"passages": reranked[:top_k], "candidate_passage_ids": union,
                         "retrieval_latency_ms": bm25_ms + semantic_ms + rerank_ms,
                         "reranking_latency_ms": rerank_ms},
        }


def run_generation(limit=5, top_k=3, dry_run=False, local_files_only=False,
                   output_dir=GENERATION_DIR, config=None, generator=None, retrieval=None):
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("--top-k must be between 1 and 10 (candidate retrieval uses top10 per method).")
    config = config or GenerationConfig.from_env()
    if not dry_run:
        config.require_key()
    questions = load_validation_questions(limit)
    provenance = {"validation_sha256": file_hash(VAL_PATH), "corpus_sha256": file_hash(CORPUS_PATH),
                  "reranker_sha256": file_hash(MODEL_PATH)}
    output_dir = Path(output_dir)
    generator = generator or AnswerGenerator(config, cache_dir=output_dir / "cache")
    if generator.config.public_settings() != config.public_settings():
        raise ValueError("Generator configuration differs from the shared experiment settings.")
    retrieval = retrieval or RetrievalConditions(local_files_only=local_files_only)
    identity = {"settings": config.public_settings(), "provenance": provenance,
                "prompt_version": PROMPT_VERSION, "question_ids": [q["question_id"] for q in questions],
                "top_k": top_k, "conditions": CONDITIONS, "dry_run": dry_run}
    run_id = json_hash(identity)[:16]
    prefix = f"validation_first{limit}_top{top_k}_{'dry_run_' if dry_run else ''}{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}.jsonl"
    records = []
    fatal_error = False
    # Each completed answer is cached atomically before the progress file is flushed.
    with output_path.open("w", encoding="utf-8") as handle:
        for question in questions:
            conditions = retrieval.retrieve(question=question["question"], top_k=top_k)
            if set(conditions) != set(CONDITIONS):
                raise ValueError("Expected all four retrieval conditions.")
            for condition in CONDITIONS:
                item = conditions[condition]
                record = generator.generate_answer(
                    question=question["question"], passages=item["passages"],
                    question_id=question["question_id"], retrieval_method=condition,
                    top_k=0 if condition == "llm_only" else top_k,
                    retrieval_latency_ms=item["retrieval_latency_ms"],
                    dry_run=dry_run, provenance=provenance)
                if "candidate_passage_ids" in item:
                    record["candidate_passage_ids"] = item["candidate_passage_ids"]
                    record["current_reranking_latency_ms"] = item["reranking_latency_ms"]
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                records.append(record)
                if not dry_run:
                    print(f"{question['question_id']} | {condition} | {record['status']} | cache={record['cache_hit']}", flush=True)
                if record.get("api_error") and (not record["api_error"]["retryable"] or record["api_error"].get("retry_deferred")):
                    fatal_error = True
                    break
            if fatal_error:
                break
    if provenance != {"validation_sha256": file_hash(VAL_PATH), "corpus_sha256": file_hash(CORPUS_PATH),
                       "reranker_sha256": file_hash(MODEL_PATH)}:
        raise ValueError("Validation data, corpus or saved reranker changed during generation.")
    resolved_models = {r.get("response_model") for r in records if r.get("response_model")}
    if len(resolved_models) > 1:
        raise ValueError("Provider returned different model identities within the comparison; inspect records before evaluating.")
    summary = {"run_id": run_id, "split": "validation", "question_count": len(questions),
               "conditions": list(CONDITIONS), "expected_records": len(questions) * len(CONDITIONS),
               "written_records": len(records), "completed": sum(r["status"] == "completed" for r in records),
               "errors": sum(r["status"] == "error" for r in records),
               "cache_hits": sum(r["cache_hit"] for r in records), "dry_run": dry_run,
               "requests_per_minute": config.requests_per_minute,
               "completed_without_reported_usage": sum(r["status"] == "completed" and
                                                        (r["input_tokens"] is None or r["output_tokens"] is None)
                                                        for r in records),
               "settings": config.public_settings(), "prompt_version": PROMPT_VERSION, "provenance": provenance,
               "retrieval_setup_latency_ms": getattr(retrieval, "setup_latency_ms", None),
               "semantic_setup_details_ms": getattr(retrieval, "setup_details_ms", None),
               "new_input_tokens": sum((r.get("input_tokens") or 0) for r in records if not r["cache_hit"]),
               "new_output_tokens": sum((r.get("output_tokens") or 0) for r in records if not r["cache_hit"]),
               "notes": ["Setup excluded from query retrieval latency. Reranker latency includes both candidate retrievers and reranking.",
                         "Request attempts are paced, including retries. generation_latency_ms excludes local pacing; generation_total_latency_ms includes it. Retry backoff remains in both.",
                         "Completed local cache hits retain original generation/usage/timing metadata; current retrieval timing is recorded separately. Cache hits make no LLM requests.",
                         "API retries can incur unreported usage; totals include only reported token counts, not an exact bill.",
                         "Temperature zero and an optional seed are best-effort controls, not guarantees of identical provider outputs.",
                         "Only validation question IDs/text are used. No reference answers are passed to retrieval or generation.",
                         "Prompts use complete source text without hidden context truncation. No factual verification or revision is performed."]}
    (output_dir / f"{prefix}.metadata.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\n{'Dry run' if dry_run else 'Generation'}: {len(records)} records saved to {output_path}")
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached embedding model; does not disable LLM API requests")
    parser.add_argument("--output-dir", type=Path, default=GENERATION_DIR)
    args = parser.parse_args()
    try:
        summary, _ = run_generation(args.limit, args.top_k, args.dry_run, args.local_files_only, args.output_dir)
        if summary["errors"]:
            parser.exit(1, "Generation recorded API errors; inspect the JSONL. Successful answers are cached for retry.\n")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Answer generation failed: {exc}\n")


if __name__ == "__main__":
    main()
