"""Sentence Transformer + FAISS retrieval over the same fixed corpus as BM25.

Run: python -m src.semantic "Who discovered penicillin?" --top-k 5
The first run downloads the model into data/raw/models. No labels are used.
"""

import argparse
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.bm25 import CORPUS_PATH, read_jsonl


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = CORPUS_PATH.parents[1] / "raw" / "models"


class SemanticRetriever:
    """Encode the fixed corpus once, preserving JSONL row order in the index.

    Normalized vectors and IndexFlatIP give cosine-similarity scores. Original
    contexts are returned intact; the model truncates inputs to its token limit.
    """

    def __init__(self, corpus_path=CORPUS_PATH, model_name=MODEL_NAME,
                 cache_folder=MODEL_CACHE, local_files_only=False, model=None, revision=None):
        self.passages = read_jsonl(corpus_path)
        if not self.passages:
            raise ValueError("The retrieval corpus is empty.")
        seen = set()
        for passage in self.passages:
            passage_id, text = passage.get("passage_id"), passage.get("text")
            if not isinstance(passage_id, str) or not passage_id.strip() or passage_id in seen:
                raise ValueError("Corpus passage IDs must be nonempty and unique.")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Passage {passage_id!r} has empty text.")
            seen.add(passage_id)
        # Row i in all three structures always refers to the same source passage.
        self.passage_ids = tuple(passage["passage_id"] for passage in self.passages)
        texts = [passage["text"] for passage in self.passages]
        # Conservative CPU limits avoid native threading crashes seen on this Mac.
        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)
        started = perf_counter()
        self.model = model if model is not None else SentenceTransformer(
            model_name, cache_folder=str(cache_folder), device="cpu",
            local_files_only=local_files_only, revision=revision,
        )
        model_loading_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        embeddings = self._encode(texts)
        embedding_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.setup_timings_ms = {
            "model_loading": model_loading_ms,
            "corpus_embedding": embedding_ms,
            "index_construction": (perf_counter() - started) * 1000,
        }
        if self.index.ntotal != len(self.passage_ids):
            raise ValueError("FAISS index and passage IDs are misaligned.")

    def _encode(self, texts):
        embeddings = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if (embeddings.ndim != 2 or embeddings.shape[0] != len(texts)
                or embeddings.shape[1] == 0 or not np.isfinite(embeddings).all()):
            raise ValueError("Model returned invalid embeddings.")
        if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
            raise ValueError("Model embeddings must have unit length for cosine similarity.")
        return embeddings

    def retrieve(self, question, top_k=5):
        """Return ranked passage_id/score/text records, using cosine similarity."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a nonempty string.")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        query = self._encode([question])
        if query.shape[1] != self.index.d:
            raise ValueError("Query and corpus embedding dimensions differ.")
        # Clamp K so FAISS never returns its -1 sentinel for missing neighbors.
        scores, indices = self.index.search(query, min(top_k, len(self.passages)))
        return [
            {"passage_id": self.passage_ids[int(idx)], "score": float(score),
             "text": self.passages[int(idx)]["text"]}
            for idx, score in zip(indices[0], scores[0])
        ]


@lru_cache(maxsize=1)
def _default_retriever():
    return SemanticRetriever()


def retrieve(question, top_k=5):
    """Search the shared saved corpus, reusing the model and in-memory index."""
    return _default_retriever().retrieve(question=question, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--local-files-only", action="store_true", help="Use the downloaded model without network access")
    args = parser.parse_args()
    try:
        retriever = SemanticRetriever(args.corpus, local_files_only=args.local_files_only)
        results = retriever.retrieve(question=args.question, top_k=args.top_k)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Semantic retrieval failed: {exc}\n")
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
