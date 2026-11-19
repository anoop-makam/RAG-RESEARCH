"""Prepare a fixed, answer-bearing retrieval prototype (not a full Wikipedia benchmark).

Run: python -m src.dataset
All retrievers must share corpus.jsonl. Train rerankers using train.jsonl labels
only; val.jsonl and test.jsonl labels are held out.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import unicodedata

from datasets import load_dataset


DATASET_NAME = "LLukas22/nq-simplified"
# Revision inspected during implementation; pin the source for reproducibility.
DATASET_REVISION = "d0b4aa2c6b142fda852d4f4f371c1134560b69fa"
RANDOM_STATE = 42
SPLIT_SIZES = {"train": 500, "val": 50, "test": 150}
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def normalize_question(text):
    """Ignore Unicode compatibility, case, punctuation, and spacing for deduplication."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def stable_id(prefix, text):
    return f"{prefix}_{sha256(text.encode('utf-8')).hexdigest()}"


def inspect_schema(dataset):
    """Inspect every available split before assuming the supported answer structure."""
    if not dataset:
        raise ValueError("The dataset has no splits.")
    for name in sorted(dataset):
        split = dataset[name]
        print(f"Source split {name}: {len(split):,} rows; schema: {split.features}")
        if not {"question", "context", "answers"}.issubset(split.column_names):
            raise ValueError(f"Split {name!r} lacks required columns.")
        answer_feature = split.features["answers"]
        if not isinstance(answer_feature, dict) or set(answer_feature) != {"text", "answer_start"}:
            raise ValueError(f"Unsupported answer schema in {name!r}: {answer_feature}")
        if len(split):
            print(f"  Sample answer annotation: {split[0]['answers']!r}")


def prepare_answers(annotation, context):
    """Preserve complete valid text/offset pairs in a consistent JSON object.

    Malformed annotations are rejected rather than silently dropping answers.
    Contexts must remain unmodified so character offsets retain their meaning.
    """
    if not isinstance(annotation, dict) or set(annotation) != {"text", "answer_start"}:
        return None
    texts, starts = annotation["text"], annotation["answer_start"]
    if not isinstance(texts, list) or not isinstance(starts, list):
        return None
    if not texts or len(texts) != len(starts):
        return None
    for text, start in zip(texts, starts):
        if not isinstance(text, str) or not text.strip():
            return None
        if type(start) is not int or start < 0 or context[start:start + len(text)] != text:
            return None
    return {"text": list(texts), "answer_start": list(starts)}


def collect_candidates(dataset):
    """Pool source splits in stable order; retain the first usable duplicate."""
    candidates = {}
    stats = Counter()
    for split_name in sorted(dataset):
        for row_index, row in enumerate(dataset[split_name]):
            stats["source_rows"] += 1
            question, context = row["question"], row["context"]
            if not isinstance(question, str) or not question.strip():
                stats["invalid_question"] += 1
                continue
            normalized = normalize_question(question)
            if not normalized:
                stats["invalid_question"] += 1
                continue
            if not isinstance(context, str) or not context.strip():
                stats["invalid_context"] += 1
                continue
            answers = prepare_answers(row["answers"], context)
            if answers is None:
                stats["invalid_answers"] += 1
                continue
            if normalized in candidates:
                stats["duplicate_questions"] += 1
                continue
            candidates[normalized] = {
                "question": question,
                "context": context,
                "answers": answers,
                "source": {"split": split_name, "row_index": row_index},
            }
    stats["usable_unique_questions"] = len(candidates)
    return candidates, dict(stats)


def build_outputs(candidates, random_state=RANDOM_STATE):
    """Sample without replacement, then partition into disjoint question splits."""
    required = sum(SPLIT_SIZES.values())
    if len(candidates) < required:
        raise ValueError(f"Need {required} usable unique questions; found {len(candidates)}.")
    selected = random.Random(random_state).sample(sorted(candidates), required)
    corpus = {}
    splits = {}
    sources = {}
    offset = 0
    for split_name, size in SPLIT_SIZES.items():
        records = []
        for normalized in selected[offset:offset + size]:
            example = candidates[normalized]
            question_id = stable_id("q", normalized)
            passage_id = stable_id("p", example["context"])
            corpus.setdefault(passage_id, {"passage_id": passage_id, "text": example["context"]})
            records.append({
                "question_id": question_id,
                "question": example["question"],
                "answers": example["answers"],
                "relevant_passage_id": passage_id,
            })
            sources[question_id] = {**example["source"], "relevant_passage_id": passage_id}
        splits[split_name] = records
        offset += size
    return splits, [corpus[key] for key in sorted(corpus)], sources


def validate_outputs(splits, corpus):
    """Check sizes, question isolation, exact context deduplication, and answers."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(set(splits) == set(SPLIT_SIZES), "Unexpected output splits.")
    passages = {record["passage_id"]: record["text"] for record in corpus}
    require(len(passages) == len(corpus), "Duplicate passage IDs.")
    require(len(set(passages.values())) == len(corpus), "Duplicate contexts.")
    for passage_id, text in passages.items():
        require(isinstance(text, str) and bool(text.strip()), "Empty passage.")
        require(passage_id == stable_id("p", text), "Unstable passage ID.")
    seen_questions, seen_ids = set(), set()
    referenced = set()
    for name, records in splits.items():
        require(len(records) == SPLIT_SIZES[name], f"Incorrect {name} size.")
        for record in records:
            question = record["question"]
            require(isinstance(question, str) and bool(question.strip()), "Empty question.")
            normalized = normalize_question(question)
            require(bool(normalized) and normalized not in seen_questions, "Duplicate question across/within splits.")
            question_id = record["question_id"]
            require(question_id == stable_id("q", normalized) and question_id not in seen_ids, "Invalid question ID.")
            passage_id = record["relevant_passage_id"]
            require(passage_id in passages, "Relevant passage missing from corpus.")
            answers = record["answers"]
            checked_answers = prepare_answers(answers, passages[passage_id])
            require(checked_answers is not None and checked_answers == answers, "Invalid answer annotation.")
            require(json.loads(json.dumps(answers, allow_nan=False)) == answers, "Answers do not round-trip as JSON.")
            seen_questions.add(normalized)
            seen_ids.add(question_id)
            referenced.add(passage_id)
    require(referenced == set(passages), "Unexpected unreferenced prototype passages.")


def jsonl_text(records):
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for record in records)


def prepare_dataset(output_dir=OUTPUT_DIR, cache_dir=None):
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, cache_dir=cache_dir)
    inspect_schema(dataset)
    candidates, stats = collect_candidates(dataset)
    splits, corpus, sources = build_outputs(candidates)
    validate_outputs(splits, corpus)

    # Rebuild from a reversed mapping to verify both RNG and ordering independence.
    repeated = build_outputs(dict(reversed(list(candidates.items()))))
    if (splits, corpus, sources) != repeated:
        raise ValueError("Preparation is not deterministic.")
    payloads = {f"{name}.jsonl": jsonl_text(records) for name, records in splits.items()}
    payloads["corpus.jsonl"] = jsonl_text(corpus)
    manifest = {
        "dataset": DATASET_NAME,
        "revision": DATASET_REVISION,
        "random_state": RANDOM_STATE,
        "sampling": "stdlib random.Random.sample over sorted normalized unique questions",
        "question_normalization": "NFKC, casefold, Unicode word tokens joined with spaces",
        "source_splits": {name: {"rows": len(split), "fingerprint": split._fingerprint}
                          for name, split in sorted(dataset.items())},
        "filter_counts": stats,
        "split_sizes": SPLIT_SIZES,
        "corpus_size": len(corpus),
        "question_sources": sources,
        "sha256": {name: sha256(text.encode("utf-8")).hexdigest() for name, text in payloads.items()},
        "scope": "Controlled answer-bearing prototype; not an unbiased full Wikipedia benchmark.",
        "usage": "All retrievers share corpus.jsonl; train rerankers on train labels only.",
        "extension": "Keep existing passage IDs when adding distractor contexts. Source split/row provenance is recorded per question.",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in payloads.items():
        (output_dir / name).write_text(text, encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Validate the serialized files, not only the in-memory objects.
    saved = {name: [json.loads(line) for line in (output_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
             for name in SPLIT_SIZES}
    saved_corpus = [json.loads(line) for line in (output_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    validate_outputs(saved, saved_corpus)
    print(f"Filtering: {stats}")
    for name, records in splits.items():
        print(f"{name}: {len(records)} questions")
    print(f"Shared corpus: {len(corpus)} passages")
    print("Validation passed: split isolation, references, nonempty text, answers, determinism, JSON round-trip.")
    sample = splits["train"][0]
    passage = next(record for record in corpus if record["passage_id"] == sample["relevant_passage_id"])
    print("Sample question:", json.dumps(sample, ensure_ascii=False))
    print("Source passage:", json.dumps(passage, ensure_ascii=False))
    print(f"Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        prepare_dataset(args.output_dir, str(args.cache_dir) if args.cache_dir else None)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Dataset preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
