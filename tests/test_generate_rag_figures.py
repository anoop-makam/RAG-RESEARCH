"""Checks of frozen-data plots, tables, pairing, units and reproducibility."""

import ast
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from experiments import generate_rag_figures as figures
import matplotlib.pyplot as plt


class ResearchFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (figures.ROOT / "results/analysis/analysis_manifest.json").exists():
            raise unittest.SkipTest("Frozen Step 10 analysis files unavailable")
        cls.data, cls.frames, cls.analysis_manifest = figures.load_inputs(figures.ROOT)

    def validate(self, frames):
        return figures.validate_frames(*frames)

    def test_ten_conditions_and_baseline_k_zero(self):
        self.assertEqual(set(self.data.index), set(figures.CONFIGS))
        self.assertEqual(len(self.data), 10)
        self.assertIn(("llm_only", 0), self.data.index)
        self.assertNotIn(("llm_only", 3), self.data.index)
        frames = copy.deepcopy(self.frames)
        frames[0].loc[frames[0].method == "llm_only", "k"] = 3
        with self.assertRaisesRegex(ValueError, "ten unique configurations"):
            self.validate(frames)

    def test_150_questions_required_no_configuration_omitted(self):
        frames = copy.deepcopy(self.frames)
        frames[0].loc[0, "questions"] = 149
        with self.assertRaisesRegex(ValueError, "150"):
            self.validate(frames)
        frames[0] = self.frames[0].iloc[:-1].copy()
        with self.assertRaises(ValueError):
            self.validate(frames)

    def test_em_and_f1_fraction_units(self):
        self.assertTrue(self.data.mean_em.between(0, 1).all())
        self.assertTrue(self.data.mean_f1.between(0, 1).all())
        for field in ("mean_em", "mean_f1"):
            frames = copy.deepcopy(self.frames)
            frames[0].loc[0, field] = 20.0
            with self.assertRaisesRegex(ValueError, "fraction"):
                self.validate(frames)

    def test_nonfinite_or_missing_values_fail(self):
        for bad in (float("nan"), float("inf")):
            frames = copy.deepcopy(self.frames)
            frames[0].loc[0, "avg_input_tokens"] = bad
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                self.validate(frames)
        frames = copy.deepcopy(self.frames)
        frames[0] = frames[0].drop(columns=["mean_f1"])
        with self.assertRaises(ValueError):
            self.validate(frames)

    def test_hit_metrics_in_range_and_deduplicated_by_method(self):
        table = figures.paper_tables(self.data, self.frames[1], self.frames[4])[1]
        self.assertEqual(len(table), 3)
        for c in ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10"):
            self.assertTrue(table[c].between(0, 1).all())
        frames = copy.deepcopy(self.frames)
        frames[4].loc[0, "Hit@1"] = 1.2
        with self.assertRaisesRegex(ValueError, "fractions"):
            self.validate(frames)
        frames = copy.deepcopy(self.frames)
        frames[4].loc[0, "Hit@1"] -= .1
        with self.assertRaisesRegex(ValueError, "Shared top-10"):
            self.validate(frames)

    def test_interval_key_units_and_value_correspondence(self):
        rows = figures.selected_pairs(self.frames[1])
        self.assertEqual([figures.comparison_key(pd.Series(row)) for row in rows], figures.KEY_PAIRS)
        for key, row in zip(figures.KEY_PAIRS, rows):
            ci = self.frames[2][(self.frames[2].comparison == figures.pair_id(key)) &
                                (self.frames[2].metric == "f1")].iloc[0]
            self.assertAlmostEqual(ci.ci_low, row["f1_ci_low"])
            self.assertAlmostEqual(ci.ci_high, row["f1_ci_high"])
            self.assertEqual(ci.units, "fraction")
        frames = copy.deepcopy(self.frames)
        target = frames[2].metric == "f1"
        frames[2].loc[target, "ci_low"] += .01
        with self.assertRaisesRegex(ValueError, "does not correspond"):
            self.validate(frames)
        frames = copy.deepcopy(self.frames)
        frames[2].loc[target, "units"] = "percentage_points"
        with self.assertRaisesRegex(ValueError, "units/family"):
            self.validate(frames)

    def test_main_table_matches_step_10_metrics_tokens_and_latency(self):
        main, _, _ = figures.paper_tables(self.data, self.frames[1], self.frames[4])
        for row, config in zip(main.itertuples(), figures.CONFIGS):
            source = self.data.loc[config]
            self.assertEqual(row.n, 150)
            self.assertAlmostEqual(row.exact_match_percent, source.mean_em * 100)
            self.assertAlmostEqual(row.token_f1, source.mean_f1)
            self.assertAlmostEqual(row.avg_input_tokens, source.avg_input_tokens)
            self.assertAlmostEqual(row.recorded_client_latency_ms,
                                   source.avg_generation_latency_ms + source.avg_retrieval_latency_ms)

    def test_latency_conversion_is_explicit_and_consistent(self):
        frames = copy.deepcopy(self.frames)
        frames[0].loc[0, "avg_recorded_end_to_end_latency_ms"] /= 1000
        with self.assertRaisesRegex(ValueError, "Latency"):
            self.validate(frames)

    def test_source_hash_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "results/analysis"
            destination.parent.mkdir()
            shutil.copytree(figures.ROOT / "results/analysis", destination)
            with (destination / "configuration_summary.csv").open("a") as handle:
                handle.write("changed\n")
            with self.assertRaisesRegex(ValueError, "output hash mismatch"):
                figures.load_inputs(root)
            self.assertFalse((root / "results/figures").exists())

    def test_latex_escaping(self):
        value = figures.tex_escape(r"a_b & 10% $x# {z} ~ ^ \end")
        for expected in (r"\_", r"\&", r"\%", r"\$", r"\#", r"\{", r"\}",
                         r"\textasciitilde{}", r"\textasciicircum{}", r"\textbackslash{}"):
            self.assertIn(expected, value)

    def test_no_generation_training_or_api_imports(self):
        tree = ast.parse(Path(figures.__file__).read_text())
        forbidden = {"openai", "requests", "httpx", "sklearn", "joblib", "torch", "faiss", "sentence_transformers"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(n.name.split(".")[0] not in forbidden for n in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith("src."))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"fit", "fit_transform", "generate_answer"})

    def test_outputs_and_repeatability_with_no_network_or_source_changes(self):
        before = figures.snapshot(figures.ROOT)
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")):
            with redirect_stdout(io.StringIO()):
                first = figures.run()
                second = figures.run()
                figures.verify()
        self.assertEqual(first["plotted_values"], second["plotted_values"])
        self.assertEqual(before, figures.snapshot(figures.ROOT))
        self.assertEqual(first["source_files"], second["source_files"])
        self.assertEqual(plt.get_fignums(), [])
        output = figures.ROOT / "results/figures"
        for stem, _, inches in figures.FIGURES:
            png, pdf = output / f"{stem}.png", output / f"{stem}.pdf"
            self.assertGreater(png.stat().st_size, 10_000)
            self.assertGreater(pdf.stat().st_size, 1_000)
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
            with Image.open(png) as image:
                self.assertEqual(image.size, tuple(round(n * 300) for n in inches))
                self.assertAlmostEqual(image.info["dpi"][0], 300, places=1)
        for stem in ("table_1_main_results", "table_2_retrieval_metrics", "table_3_paired_comparisons"):
            for suffix in ("csv", "md", "tex"):
                self.assertGreater((output / f"{stem}.{suffix}").stat().st_size, 100)
            self.assertEqual(first["output_sha256"][f"{stem}.csv"], second["output_sha256"][f"{stem}.csv"])
        self.assertTrue((output / "figure_captions.md").is_file())
        self.assertTrue((output / "figure_manifest.json").is_file())
        saved_main = pd.read_csv(output / "table_1_main_results.csv")
        expected_main = figures.paper_tables(self.data, self.frames[1], self.frames[4])[0]
        pd.testing.assert_frame_equal(saved_main, expected_main, check_exact=False, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
