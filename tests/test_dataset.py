"""Offline checks for dataset filtering, leakage prevention, and stable artifacts."""

import copy
import unittest

from datasets import Dataset, DatasetDict

from src.dataset import (
    build_outputs,
    collect_candidates,
    inspect_schema,
    jsonl_text,
    normalize_question,
    prepare_answers,
    validate_outputs,
)


def row(question, context="The answer is here."):
    return {"question": question, "context": context,
            "answers": {"text": ["answer"], "answer_start": [4]}}


class PreparationTests(unittest.TestCase):
    def test_normalization_keeps_equivalent_questions_together(self):
        self.assertEqual(normalize_question("  WHO discovered Penicillin? "),
                         normalize_question("who DISCOVERED penicillin"))
        self.assertEqual(normalize_question("Ｗｈｏ?"), "who")

    def test_filters_before_deduplication(self):
        invalid_answer = row("valid later")
        invalid_answer["answers"]["answer_start"] = [0]
        data = DatasetDict({
            "a": Dataset.from_list([
                row(""), row("   "), row("???"), row("missing context", ""),
                {**row("missing answers"), "answers": None}, invalid_answer,
                row("Duplicate?"), row("good"),
            ]),
            "b": Dataset.from_list([row("duplicate"), row("valid later")]),
        })
        candidates, stats = collect_candidates(data)
        self.assertEqual(set(candidates), {"duplicate", "good", "valid later"})
        self.assertEqual(stats["duplicate_questions"], 1)
        self.assertEqual(candidates["valid later"]["source"], {"split": "b", "row_index": 1})

    def test_answers_preserve_multiple_annotations_and_offsets(self):
        annotation = {"text": ["answer", "here"], "answer_start": [4, 14]}
        self.assertEqual(prepare_answers(annotation, "The answer is here."), annotation)
        for invalid in [None, {}, {"text": [], "answer_start": []},
                        {"text": ["answer"], "answer_start": []},
                        {"text": ["answer"], "answer_start": [True]},
                        {"text": ["missing"], "answer_start": [4]}]:
            self.assertIsNone(prepare_answers(invalid, "The answer is here."))

    def test_sampling_deduplicates_contexts_and_is_deterministic(self):
        data = DatasetDict({"source": Dataset.from_list([row(f"Question {i}?") for i in range(720)])})
        candidates, _ = collect_candidates(data)
        splits, corpus, sources = build_outputs(candidates)
        validate_outputs(splits, corpus)
        self.assertEqual([len(splits[name]) for name in ("train", "val", "test")], [500, 50, 150])
        self.assertEqual(len(corpus), 1)
        self.assertEqual(len(sources), 700)
        repeated = build_outputs(dict(reversed(list(candidates.items()))))
        self.assertEqual((splits, corpus, sources), repeated)
        for name in splits:
            self.assertEqual(jsonl_text(splits[name]), jsonl_text(repeated[0][name]))
        question_id = splits["train"][0]["question_id"]
        altered = build_outputs(candidates, random_state=43)
        all_ids = {record["question_id"] for records in altered[0].values() for record in records}
        if question_id in all_ids:
            self.assertEqual(sources[question_id], altered[2][question_id])

        broken = copy.deepcopy(splits)
        broken["test"][0] = broken["train"][0]
        with self.assertRaisesRegex(ValueError, "Duplicate question"):
            validate_outputs(broken, corpus)
        broken = copy.deepcopy(splits)
        broken["train"][0]["relevant_passage_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_outputs(broken, corpus)
        broken = copy.deepcopy(splits)
        broken["train"][0]["answers"] = None
        with self.assertRaisesRegex(ValueError, "Invalid answer"):
            validate_outputs(broken, corpus)
        with self.assertRaisesRegex(ValueError, "Empty passage"):
            validate_outputs(splits, [{"passage_id": corpus[0]["passage_id"], "text": " "}])
        with self.assertRaisesRegex(ValueError, "Need 700"):
            build_outputs({})

    def test_unknown_schema_fails_explicitly(self):
        data = DatasetDict({"other": Dataset.from_list([{"question": "q", "context": "c", "answers": "a"}])})
        with self.assertRaisesRegex(ValueError, "Unsupported answer schema"):
            inspect_schema(data)


if __name__ == "__main__":
    unittest.main()
