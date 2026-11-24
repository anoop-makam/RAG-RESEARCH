"""Frozen-data signals, reproducible sampling, exact evidence and human-label safety."""

import ast
from contextlib import redirect_stdout
import copy
import csv
import io
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from experiments import analyze_rag_failures as failures


class QualitativeFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (failures.ROOT / "results/analysis/analysis_manifest.json").exists():
            raise unittest.SkipTest("Frozen experiment unavailable")
        cls.by_question, cls.corpus, cls.summary, cls.audit = failures.load_inputs(failures.ROOT)
        cls.signals = failures.build_signals(cls.by_question, cls.corpus)
        cls.selected, cls.reasons, cls.focus, cls.selection = failures.select_cases(cls.signals)
        cls.cases = [failures.complete_case(qid, cls.focus[qid], cls.reasons[qid], cls.by_question,
                                           cls.signals, cls.corpus) for qid in cls.selected]

    def test_all_records_test_ids_and_comparison_denominators(self):
        self.assertEqual(len(self.by_question), 150)
        self.assertEqual(sum(len(rows) for rows in self.by_question.values()), 1500)
        self.assertEqual(len(self.signals), 5550)
        self.assertEqual(len({s["signal_id"] for s in self.signals}), 5550)
        self.assertEqual({s["question_id"] for s in self.signals}, set(self.by_question))
        self.assertEqual(sum(s["family"] == "configuration" for s in self.signals), 1500)

    def test_taxonomy_exact_source_presence_and_no_baseline_miss(self):
        for signal in self.signals:
            if signal["family"] != "configuration":
                continue
            if signal["method"] == "llm_only":
                self.assertEqual(signal["categories"], [])
                self.assertIsNone(signal["source_present"])
            else:
                self.assertEqual("A" in signal["categories"], not signal["source_present"])
                self.assertEqual("B" in signal["categories"], signal["source_present"] and signal["em"] == 0)
            self.assertIn("unverified", signal["explanation_status"])
            self.assertNotIn("manual_factual_correctness", signal)

    def test_sampling_is_reproducible_unique_and_covers_all_categories(self):
        again, reasons, focus, selection = failures.select_cases(list(reversed(self.signals)))
        # Every randomized stratum starts with sorted IDs, independent of input row order.
        self.assertEqual(again, self.selected)
        self.assertEqual(reasons, self.reasons)
        self.assertEqual({q: s["signal_id"] for q, s in focus.items()},
                         {q: s["signal_id"] for q, s in self.focus.items()})
        self.assertEqual(len(self.selected), len(set(self.selected)))
        self.assertEqual(len(self.selected), 24)
        self.assertTrue(set(self.selected) <= set(self.by_question))
        for cat in failures.TAXONOMY:
            self.assertGreaterEqual(selection["selected_category_unique_questions"][cat],
                                    min(2, selection["eligibility_unique_questions"][cat]))
        self.assertFalse(selection["representative"])
        self.assertEqual(selection["seed"], 42)
        self.assertTrue(any(s["k"] == 3 and "E" in s["categories"] for s in focus.values()))
        self.assertTrue(any("F" in s["categories"] for s in focus.values()))

    def test_requested_examples_are_saved_observations(self):
        self.assertTrue(all(a["found"] for a in self.selection["anchors"]))
        witcher = next(c for c in self.cases if "witcher" in c["question"])
        self.assertTrue(witcher["source_present"] and witcher["comparison_source_present"])
        self.assertEqual(witcher["annotated_source_rank_top10"], 1)
        self.assertEqual(witcher["comparison_source_rank_top10"], 1)
        self.assertEqual(witcher["retrieved_passage_ids"][:4], witcher["comparison_retrieved_passage_ids"][:4])
        source = self.corpus[witcher["relevant_passage_id"]]
        self.assertIn("1986", source)
        self.assertIn("1990", source)
        bull = next(c for c in self.cases if "bull run" in c["question"])
        self.assertIn("accepted_reference_token_sequence_in_longer_prediction", bull["format_flags"])

    def test_exact_excerpts_offsets_and_omissions(self):
        text = "prefix " * 400 + "needle words" + " suffix" * 400
        ex = failures.excerpt(text, ["needle words"], limit=160)
        self.assertEqual(ex["text"], text[ex["start"]:ex["stop"]])
        self.assertTrue(ex["omitted_before"])
        self.assertTrue(ex["omitted_after"])
        self.assertIn("needle words", ex["text"])
        for case in self.cases:
            for item in case["evidence_excerpts"]:
                ex = item["excerpt"]
                self.assertEqual(ex["text"], self.corpus[item["passage_id"]][ex["start"]:ex["stop"]])
                self.assertLessEqual(len(ex["text"]), 900)

    def test_missing_or_fabricated_evidence_fails(self):
        qid = self.selected[0]
        by_question = {qid: copy.deepcopy(self.by_question[qid])}
        by_question[qid][("bm25", 3)]["retrieved_passage_ids"][0] = "unknown"
        with self.assertRaisesRegex(ValueError, "evidence"):
            failures.validate_evidence(by_question, self.corpus)
        by_question = {qid: copy.deepcopy(self.by_question[qid])}
        by_question[qid][("semantic", 3)]["relevant_passage_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Missing annotated"):
            failures.validate_evidence(by_question, self.corpus)

    def test_abstentions_not_marked_as_verified_wrong_or_format_equivalent(self):
        row = dict(generated_answer="Insufficient evidence.", references=["evidence"], exact_match=0, abstained=True)
        self.assertEqual(failures.format_signals(row), [])
        row = dict(generated_answer="Sir Alan", references=["Alan"], exact_match=0, abstained=False)
        self.assertIn("prediction_has_title_or_honorific", failures.format_signals(row))
        self.assertIn("accepted_reference_token_sequence_in_longer_prediction", failures.format_signals(row))

    def test_initial_judgments_empty_and_no_manual_factual_statistics(self):
        reviews = failures.review_template(self.cases)
        for row in reviews:
            self.assertFalse(failures.reviewed(row))
            for name in (*failures.MANUAL_CHOICES, "manual_notes"):
                self.assertEqual(row[name], "")
        counts = failures.descriptive_counts(self.signals, self.selected, reviews)
        self.assertFalse(any(r["scope"] == "manually_reviewed_sample" for r in counts))
        totals = {(r["category"], r["scope"]): r["count"] for r in counts if r["metric"] == "taxonomy_eligibility"}
        self.assertEqual(totals["A", "question_configuration_or_pair_events"], 106)
        self.assertEqual(totals["B", "question_configuration_or_pair_events"], 639)
        self.assertEqual(totals["B", "all_test_questions"], 83)

    def test_manual_notes_and_judgments_preserved_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manual_review.csv"
            rows = failures.review_template(self.cases)
            rows[0].update(manual_factual_correctness="uncertain", manual_evidence_support="uncertain",
                           manual_failure_cause="ambiguous_reference", manual_notes='Human note, with "quotes"\nand café.', reviewed="true")
            rows[1]["manual_notes"] = "Partial notes remain unreviewed."
            failures.write_csv(path, rows)
            before = path.read_bytes()
            loaded, create = failures.load_review(path, self.cases)
            self.assertFalse(create)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(sum(failures.reviewed(r) for r in loaded), 1)
            counts = failures.descriptive_counts(self.signals, self.selected, loaded)
            manual = [r for r in counts if r["scope"] == "manually_reviewed_sample"]
            self.assertTrue(manual)
            self.assertTrue(all(r["denominator"] == 1 for r in manual))

    def test_invalid_manual_labels_or_changed_identity_stop_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manual_review.csv"
            for field, bad in (("manual_failure_cause", "automatically_confirmed"), ("reviewed", "maybe"),
                               ("question_id", "unknown"), ("reviewed", "true")):
                rows = failures.review_template(self.cases)
                rows[0][field] = bad
                failures.write_csv(path, rows)
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    failures.load_review(path, self.cases)
                self.assertEqual(before, path.read_bytes())

    def test_paper_cases_have_both_directions_and_no_transferred_labels(self):
        reviews = failures.review_template(self.cases)
        paper = failures.paper_cases(self.cases, reviews)
        for name in ("Semantic improvement", "BM25 improvement", "Reranker improvement", "Reranker regression",
                     "Higher-K degradation", "Possible lexical mismatch"):
            self.assertIn(name, paper)
        self.assertNotIn("no unused focused case", paper)
        self.assertEqual(paper.count("PROVISIONAL"), 6)

    def test_no_generation_or_training_imports(self):
        tree = ast.parse(Path(failures.__file__).read_text())
        forbidden = {"openai", "requests", "httpx", "sklearn", "joblib", "torch", "faiss", "sentence_transformers"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(n.name.split(".")[0] not in forbidden for n in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, {"src.generation", "src.reranker", "src.semantic"})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"fit", "fit_transform", "generate_answer"})

    def test_rerun_reproducibility_without_network_or_source_changes(self):
        before = failures.snapshot(failures.ROOT)
        review_path = failures.ROOT / "results/failure_analysis/manual_review.csv"
        review_before = review_path.read_bytes() if review_path.exists() else None
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")):
            with redirect_stdout(io.StringIO()):
                first = failures.run()
                second = failures.run()
        self.assertEqual(before, failures.snapshot(failures.ROOT))
        if review_before is not None:
            self.assertEqual(review_path.read_bytes(), review_before)
        self.assertEqual(first["output_sha256"], second["output_sha256"])
        self.assertEqual(first["selection"], second["selection"])
        self.assertEqual(first["status"], "MANUAL_REVIEW_PENDING")
        self.assertEqual(first["api_calls"], 0)
        self.assertEqual(first["models_fitted"], 0)

    def test_full_rerun_consumes_completed_and_partial_reviews_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "results/failure_analysis"
            output.mkdir(parents=True)
            path = output / "manual_review.csv"
            rows = failures.review_template(self.cases)
            rows[0].update(manual_factual_correctness="uncertain", manual_evidence_support="uncertain",
                           manual_failure_cause="uncertain", manual_notes="Test human judgment; not a factual conclusion.", reviewed="true")
            rows[1]["manual_notes"] = "Partial annotation should not be lost."
            failures.write_csv(path, rows)
            before = path.read_bytes()
            with patch.object(failures, "load_inputs", return_value=(self.by_question, self.corpus, self.summary, self.audit)):
                with redirect_stdout(io.StringIO()):
                    manifest = failures.run(root)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(manifest["reviewed_cases"], 1)
            self.assertEqual(manifest["unreviewed_cases"], 23)
            self.assertEqual(manifest["status"], "MANUAL_REVIEW_PENDING")
            with (output / "failure_counts.csv").open() as handle:
                counts = list(csv.DictReader(handle))
            self.assertTrue(any(r["scope"] == "manually_reviewed_sample" for r in counts))


if __name__ == "__main__":
    unittest.main()
