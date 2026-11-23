"""Normalized exact match and multiset token F1 for short annotated answers.

Normalization follows the standard SQuAD evaluator: lowercase, remove ASCII
punctuation and English articles, then collapse whitespace. No semantic grading.
"""

from collections import Counter
import re
import string


SCORING_VERSION = "nq-squad-em-f1-v1"
ABSTENTIONS = {"insufficient evidence", "i dont know", "i do not know"}


def normalize_answer(text):
    if not isinstance(text, str):
        raise ValueError("Answers must be strings.")
    text = "".join(char for char in text.lower() if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score_answer(prediction, references):
    if not isinstance(references, list) or not references or any(not isinstance(r, str) or not r.strip() for r in references):
        raise ValueError("References must be a nonempty list of nonempty answer strings.")
    if prediction is not None and not isinstance(prediction, str):
        raise ValueError("Prediction must be a string or None.")
    prediction = prediction or ""
    normalized = normalize_answer(prediction)
    empty = not prediction.strip()
    abstained = normalized in ABSTENTIONS
    if empty or abstained:
        return {"exact_match": 0.0, "f1": 0.0, "empty_answer": empty, "abstained": abstained}
    exact, f1 = 0.0, 0.0
    tokens = normalized.split()
    for reference in references:
        gold = normalize_answer(reference)
        gold_tokens = gold.split()
        exact = max(exact, float(normalized == gold))
        overlap = sum((Counter(tokens) & Counter(gold_tokens)).values())
        if not tokens or not gold_tokens:
            score = float(tokens == gold_tokens)
        elif not overlap:
            score = 0.0
        else:
            precision, recall = overlap / len(tokens), overlap / len(gold_tokens)
            score = 2 * precision * recall / (precision + recall)
        f1 = max(f1, score)
    return {"exact_match": exact, "f1": f1, "empty_answer": False, "abstained": False}
