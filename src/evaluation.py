"""Evaluate source-passage Hit@K on the controlled answer-bearing prototype.

This measures recovery of the single annotated source passage, not exhaustive
passage relevance. All retrieval approaches must use the same fixed corpus.
"""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

from src.bm25 import BM25Retriever, CORPUS_PATH, read_jsonl


DEFAULT_KS = (1, 3, 5, 10)
TEST_PATH = CORPUS_PATH.with_name("test.jsonl")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def hit_at_k(results, relevant_passage_id, k):
    """Return 1 if the single annotated passage occurs in the first k results."""
    if type(k) is not int or k < 1:
        raise ValueError("K must be a positive integer.")
    return int(any(result["passage_id"] == relevant_passage_id for result in results[:k]))


def evaluate_retrieval(questions, retrieve, ks=DEFAULT_KS):
    """Evaluate any compatible retrieval callable, retrieving max(K) only once per question."""
    ks = tuple(ks)
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("Provide at least one positive integer K.")
    ks = sorted(set(ks))
    questions = list(questions)
    if not questions:
        raise ValueError("Cannot evaluate an empty question set.")
    seen = set()
    details = []
    hits = {k: 0 for k in ks}
    for question in questions:
        question_id = question.get("question_id")
        text = question.get("question")
        relevant_id = question.get("relevant_passage_id")
        if not isinstance(question_id, str) or not question_id.strip() or question_id in seen:
            raise ValueError("Evaluation question IDs must be nonempty and unique.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty question: {question_id}")
        if not isinstance(relevant_id, str) or not relevant_id.strip():
            raise ValueError(f"Missing relevant_passage_id: {question_id}")
        seen.add(question_id)
        results = retrieve(question=text, top_k=max(ks))
        question_hits = {f"Hit@{k}": hit_at_k(results, relevant_id, k) for k in ks}
        for k in ks:
            hits[k] += question_hits[f"Hit@{k}"]
        details.append({"question_id": question_id, "question": text,
                        "relevant_passage_id": relevant_id,
                        "retrieved": [{"passage_id": result["passage_id"], "score": result["score"]}
                                      for result in results],
                        "hits": question_hits})
    table = [{"k": k, "hits": hits[k], "questions": len(questions),
              "hit_at_k": hits[k] / len(questions)} for k in ks]
    return table, details


def run_evaluation(corpus_path=CORPUS_PATH, questions_path=TEST_PATH, output_dir=RESULTS_DIR):
    retriever = BM25Retriever(corpus_path)
    questions = read_jsonl(questions_path)
    for question in questions:
        if question.get("relevant_passage_id") not in retriever.passage_ids:
            raise ValueError(f"Question {question.get('question_id')!r} references a passage outside this corpus.")
    table, details = evaluate_retrieval(questions, retriever.retrieve)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "bm25_hit_at_k.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k", "hits", "questions", "hit_at_k"])
        writer.writeheader()
        writer.writerows(table)
    details_path = output_dir / "bm25_predictions.jsonl"
    details_path.write_text("".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                                    for record in details), encoding="utf-8")
    metadata = {
        "retriever": "BM25Okapi", "tokenization": "lowercase Unicode word tokens",
        "tie_break": "passage_id ascending", "ks": list(DEFAULT_KS),
        "corpus_path": str(Path(corpus_path).resolve()),
        "questions_path": str(Path(questions_path).resolve()),
        "corpus_sha256": sha256(Path(corpus_path).read_bytes()).hexdigest(),
        "questions_sha256": sha256(Path(questions_path).read_bytes()).hexdigest(),
        "corpus_size": len(retriever.passages), "question_count": len(questions),
        "metric": "Hit@K: recovery of the single annotated source passage",
        "scope": "Controlled answer-bearing prototype; not exhaustive passage relevance or full Wikipedia retrieval.",
    }
    (output_dir / "bm25_evaluation.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"BM25 | {len(questions)} questions | {len(retriever.passages)} fixed corpus passages")
    print("| K | Hits | Questions | Hit@K |")
    print("|---|---:|---:|---:|")
    for row in table:
        print(f"| {row['k']} | {row['hits']} | {row['questions']} | {row['hit_at_k']:.2%} |")
    print("Hit@K measures the single annotated source passage in this controlled prototype.")
    print(f"Saved table: {table_path}")
    print(f"Saved per-question rankings: {details_path}")
    return table, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--questions", type=Path, default=TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    try:
        run_evaluation(args.corpus, args.questions, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Retrieval evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
