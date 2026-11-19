"""BM25 retrieval over the fixed corpus created by ``python -m src.dataset``."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import re

from rank_bm25 import BM25Okapi


CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "corpus.jsonl"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def read_jsonl(path):
    """Read JSON objects with useful file/line errors."""
    path = Path(path)
    try:
        records = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}, line {line_number}: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object in {path}, line {line_number}.")
                records.append(record)
        return records
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing {path}. Prepare the dataset with: python -m src.dataset") from exc


class BM25Retriever:
    """Build one reusable index. Equal scores are ordered by passage ID."""

    def __init__(self, corpus_path=CORPUS_PATH):
        self.passages = sorted(read_jsonl(corpus_path), key=lambda row: row.get("passage_id", ""))
        if not self.passages:
            raise ValueError("The retrieval corpus is empty.")
        seen = set()
        tokens = []
        for passage in self.passages:
            passage_id, text = passage.get("passage_id"), passage.get("text")
            if not isinstance(passage_id, str) or not passage_id.strip() or passage_id in seen:
                raise ValueError("Corpus passage IDs must be nonempty and unique.")
            if not isinstance(text, str) or not tokenize(text):
                raise ValueError(f"Passage {passage_id!r} has no searchable text.")
            seen.add(passage_id)
            tokens.append(tokenize(text))
        self.passage_ids = frozenset(seen)
        self.index = BM25Okapi(tokens)

    def retrieve(self, question, top_k=5):
        """Return up to top_k dictionaries containing passage_id, score, and text."""
        if not isinstance(question, str) or not tokenize(question):
            raise ValueError("Question must contain searchable words.")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        scores = self.index.get_scores(tokenize(question))
        # Python's stable sort preserves the passage-ID order when scores tie.
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"passage_id": self.passages[i]["passage_id"],
             "score": float(scores[i]), "text": self.passages[i]["text"]}
            for i in top_indices
        ]


@lru_cache(maxsize=1)
def _default_retriever():
    return BM25Retriever()


def retrieve(question, top_k=5):
    """Search the prepared shared corpus; lazily build and reuse its BM25 index."""
    return _default_retriever().retrieve(question=question, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    args = parser.parse_args()
    try:
        results = BM25Retriever(args.corpus).retrieve(args.question, args.top_k)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"BM25 retrieval failed: {exc}\n")
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
