import re

from datasets import load_dataset
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


dataset = load_dataset("LLukas22/nq-simplified")
split_name = next(iter(dataset.keys()))

# Small prototype corpus, not our final evaluation dataset
examples = dataset[split_name].select(range(1000))
passages = [
    text
    for text in examples["context"]
    if isinstance(text, str) and text.strip()
]

bm25 = BM25Okapi([tokenize(p) for p in passages])

question = "Who discovered penicillin?"
scores = bm25.get_scores(tokenize(question))

top_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True,
)[:5]

for rank, idx in enumerate(top_indices, start=1):
    print(f"\nRank {rank} | Score: {scores[idx]:.3f}")
    print(passages[idx][:500])
