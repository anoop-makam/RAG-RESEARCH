"""Publication figures from frozen Step 10 outputs, with no inference or rescoring.

    python -B -m experiments.generate_rag_figures
    python -B -m experiments.generate_rag_figures --verify-only
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

# Keep font caches outside the source/results tree and support headless execution.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cs391-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("bm25", "semantic", "reranker")
KS = (3, 5, 10)
CONFIGS = [("llm_only", 0)] + [(m, k) for m in METHODS for k in KS]
NAMES = {"llm_only": "LLM only", "bm25": "BM25", "semantic": "Semantic", "reranker": "Trained reranker"}
COLORS = {"llm_only": "#707070", "bm25": "#0072B2", "semantic": "#D55E00", "reranker": "#009E73"}
MARKERS = {"bm25": "o", "semantic": "s", "reranker": "^"}
STYLES = {"bm25": "-", "semantic": "--", "reranker": "-."}
HATCHES = ("", "//", "xx")
KEY_PAIRS = [("semantic", 3, "bm25", 3), ("reranker", 3, "bm25", 3),
             ("reranker", 3, "semantic", 3), ("reranker", 5, "semantic", 5),
             ("reranker", 10, "semantic", 10), ("reranker", 3, "llm_only", 0)]
FIGURES = [
    ("figure_1_exact_match", "Exact Match Accuracy Across Retrieval Configurations", (8.4, 4.9)),
    ("figure_2_f1_vs_k", "Effect of Retrieved Passage Count on Answer Quality", (8.4, 4.9)),
    ("figure_3_retrieval_hit_at_k", "Annotated-Source Retrieval Performance", (8.4, 4.9)),
    ("figure_4_f1_vs_input_tokens", "Answer Quality versus Input Token Consumption", (8.4, 5.2)),
    ("figure_5_paired_f1_differences", "Paired Differences in Answer F1 (95% Bootstrap Intervals)", (9.4, 5.4)),
    ("figure_6_latency", "Recorded Client Latency Across RAG Configurations", (8.4, 5.3)),
]
INPUTS = ("configuration_summary.csv", "paired_comparisons.csv", "bootstrap_confidence_intervals.csv",
          "k_sensitivity.csv", "retrieval_answer_relationship.csv", "statistical_analysis.md", "analysis_manifest.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def snapshot(root):
    """Protect all existing data/models/results, excluding only this step's output."""
    return {str(p.relative_to(root)): sha(p) for folder in ("data/processed", "models", "results")
            for p in sorted((root / folder).rglob("*"))
            if p.is_file() and not p.is_relative_to(root / "results/figures")}


def numeric_frame(frame, columns, name):
    require(set(columns) <= set(frame.columns), f"Missing columns in {name}: {set(columns) - set(frame.columns)}")
    for column in columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid numeric {name}.{column}") from error
        require(np.isfinite(frame[column].to_numpy(dtype=float)).all(), f"Missing/nonfinite {name}.{column}")


def close(a, b, message):
    require(np.allclose(a, b, atol=1e-9, rtol=0), message)


def comparison_key(row):
    return row.left_method, int(row.left_k), row.right_method, int(row.right_k)


def pair_id(key):
    a, k, b, j = key
    return f"{a} K={k} minus {b} K={j}"


def display_config(method, k):
    return NAMES[method] if method == "llm_only" else f"{NAMES[method]} K={k}"


def display_pair(key):
    a, k, b, j = key
    return display_config(a, k) + " minus " + display_config(b, j)


def validate_frames(summary, pairs, bootstrap, sensitivity, retrieval):
    """Validate real Step 10 schema and cross-file correspondence, with no imputation."""
    numeric_frame(summary, [c for c in summary if c != "method"], "summary")
    required = {"method", "k", "questions", "mean_em", "mean_f1", "exact_match_percent",
                "avg_input_tokens", "avg_output_tokens", "avg_generation_latency_ms",
                "avg_retrieval_latency_ms", "avg_recorded_end_to_end_latency_ms"}
    require(required <= set(summary), f"Missing summary fields: {required - set(summary)}")
    require(len(summary) == 10 and set(zip(summary.method, summary.k)) == set(CONFIGS),
            "Expected exactly ten unique configurations, including LLM-only K=0")
    require((summary.questions == 150).all(), "Each mean must use 150 questions")
    for column in ("mean_em", "mean_f1", "abstention_rate"):
        require(column in summary and summary[column].between(0, 1).all(), f"{column} must be a fraction in [0,1]")
    require((summary.select_dtypes(include="number") >= 0).all().all(), "Negative summary value")
    close(summary.mean_em * 100, summary.exact_match_percent, "EM fraction/percentage mismatch")
    close(summary.mean_em * 150, summary.exact_match_count, "EM count mismatch")
    close(summary.avg_generation_latency_ms + summary.avg_retrieval_latency_ms,
          summary.avg_recorded_end_to_end_latency_ms, "Latency must be retrieval + generation milliseconds")
    require((summary.avg_input_tokens > 0).all(), "Missing input token usage")
    indexed = summary.set_index(["method", "k"])
    expected_pairs = {(a, k, b, k) for k in KS for a, b in
                      (("semantic", "bm25"), ("reranker", "bm25"), ("reranker", "semantic"))}
    expected_pairs |= {(m, k, "llm_only", 0) for m in METHODS for k in KS}
    expected_k = {(m, high, m, low) for m in METHODS for high, low in ((5, 3), (10, 5), (10, 3))}
    for name, frame, expected in (("pairs", pairs, expected_pairs), ("sensitivity", sensitivity, expected_k)):
        numeric_frame(frame, [c for c in frame if c not in
                             {"family", "comparison", "left_method", "right_method"}], name)
        needed = {"comparison", "family", "left_method", "left_k", "right_method", "right_k", "questions",
                  "delta_em_pp", "delta_f1", "em_pp_ci_low", "em_pp_ci_high", "f1_ci_low", "f1_ci_high"}
        require(needed <= set(frame), f"Missing {name} fields")
        keys = [comparison_key(r) for r in frame.itertuples()]
        require(len(keys) == len(expected) and set(keys) == expected, f"Wrong {name} comparison coverage")
        require((frame.questions == 150).all(), f"{name} pairing N must be 150")
        for row in frame.itertuples():
            key = comparison_key(row)
            require(row.comparison == pair_id(key), f"Mislabeled comparison: {row.comparison}")
            a, k, b, j = key
            left, right = indexed.loc[(a, k)], indexed.loc[(b, j)]
            close(row.delta_em_pp, 100 * (left.mean_em - right.mean_em), "EM difference units/value mismatch")
            close(row.delta_f1, left.mean_f1 - right.mean_f1, "F1 difference value mismatch")
            require(-1 <= row.f1_ci_low <= row.f1_ci_high <= 1, "Invalid F1 confidence interval")
            require(-100 <= row.em_pp_ci_low <= row.em_pp_ci_high <= 100, "Invalid EM confidence interval")
    numeric_frame(bootstrap, ["questions", "estimate", "absolute_effect_magnitude", "ci_low", "ci_high", "iterations", "seed"], "bootstrap")
    require(len(bootstrap) == 54 and not bootstrap.duplicated(["comparison", "metric"]).any(),
            "Expected exactly 54 unique metric-specific paired intervals")
    require((bootstrap.questions == 150).all() and (bootstrap.iterations == 10000).all() and
            (bootstrap.seed == 42).all(), "Unexpected bootstrap protocol")
    require(bootstrap.multiplicity_adjusted.astype(str).str.lower().eq("false").all(),
            "Expected pointwise, unadjusted intervals")
    for row in pd.concat([pairs, sensitivity]).itertuples():
        for metric, units, prefix in (("exact_match", "percentage_points", "em_pp"), ("f1", "fraction", "f1")):
            selected = bootstrap[(bootstrap.comparison == row.comparison) & (bootstrap.metric == metric)]
            require(len(selected) == 1, f"Missing interval for {row.comparison}/{metric}")
            ci = selected.iloc[0]
            require(ci.units == units and ci.family == row.family, "Bootstrap units/family mismatch")
            close([ci.estimate, ci.ci_low, ci.ci_high],
                  [getattr(row, f"delta_{prefix}"), getattr(row, f"{prefix}_ci_low"), getattr(row, f"{prefix}_ci_high")],
                  f"Interval does not correspond to paired comparison: {row.comparison}")
            close(ci.absolute_effect_magnitude, abs(ci.estimate), "Absolute effect magnitude mismatch")
    hits = ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10"]
    numeric_frame(retrieval, ["k", "questions", *hits, "context_source_hit"], "retrieval")
    require(len(retrieval) == 9 and set(zip(retrieval.method, retrieval.k)) == set(CONFIGS[1:]),
            "Expected nine retrieval rows and no LLM-only retrieval condition")
    require((retrieval.questions == 150).all(), "Retrieval N must be 150")
    require(retrieval[hits + ["context_source_hit"]].apply(lambda col: col.between(0, 1)).all().all(),
            "Hit/MRR must be fractions in [0,1]")
    for method in METHODS:
        subset = retrieval[retrieval.method == method]
        for column in hits:
            close(subset[column], subset.iloc[0][column], "Shared top-10 rankings differ by delivered K")
        values = subset.iloc[0][hits[:4]].to_numpy(dtype=float)
        require((np.diff(values) >= -1e-12).all(), "Hit@K must be nondecreasing")
        for row in subset.to_dict("records"):
            close(row["context_source_hit"], row[f"Hit@{int(row['k'])}"], "Actual-context source hit mismatch")
    return indexed


def load_inputs(root):
    directory = root / "results/analysis"
    for name in INPUTS:
        require((directory / name).is_file(), f"Required Step 10 input missing: {name}")
    manifest = json.loads((directory / "analysis_manifest.json").read_text())
    require(manifest.get("question_count") == 150 and manifest.get("record_count") == 1500 and
            manifest.get("original_artifacts_unchanged") is True, "Step 10 readiness/size mismatch")
    for name, expected in manifest["output_sha256"].items():
        require((directory / name).is_file() and sha(directory / name) == expected,
                f"Step 10 output hash mismatch: {name}")
    require(set(INPUTS) - {"analysis_manifest.json"} <= set(manifest["output_sha256"]),
            "Step 10 manifest does not hash all expected analysis inputs")
    # Check previously recorded originals, allowing new Step 11 files to coexist.
    for name, expected in manifest["protected_snapshot_after"].items():
        require((root / name).is_file() and sha(root / name) == expected,
                f"Protected experiment artifact changed: {name}")
    frames = [pd.read_csv(directory / name) for name in INPUTS[:5]]
    indexed = validate_frames(*frames)
    return indexed, frames, manifest


def selected_pairs(pairs):
    lookup = {comparison_key(r): r._asdict() for r in pairs.itertuples(index=False)}
    require(set(KEY_PAIRS) <= set(lookup), "Missing selected comparisons")
    return [lookup[key] for key in KEY_PAIRS]


def paper_tables(summary, pairs, retrieval):
    main = pd.DataFrame([dict(configuration=display_config(m, k), n=int(summary.loc[(m, k), "questions"]),
                              exact_match_percent=float(summary.loc[(m, k), "mean_em"] * 100),
                              token_f1=float(summary.loc[(m, k), "mean_f1"]),
                              avg_input_tokens=float(summary.loc[(m, k), "avg_input_tokens"]),
                              recorded_client_latency_ms=float(summary.loc[(m, k), "avg_recorded_end_to_end_latency_ms"]))
                         for m, k in CONFIGS])
    retrieval_table = pd.DataFrame([dict(method=NAMES[m], **{
        field: float(retrieval[retrieval.method == m].iloc[0][field])
        for field in ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10")}) for m in METHODS])
    selected = []
    for key, row in zip(KEY_PAIRS, selected_pairs(pairs)):
        positive = row["f1_ci_low"] > 0
        negative = row["f1_ci_high"] < 0
        interpretation = ("Positive pointwise interval" if positive else "Negative pointwise interval" if negative
                          else "Interval includes zero")
        selected.append(dict(comparison=display_pair(key), delta_em_pp=row["delta_em_pp"],
                             delta_f1=row["delta_f1"], f1_ci_low=row["f1_ci_low"], f1_ci_high=row["f1_ci_high"],
                             em_ci_low_pp=row["em_pp_ci_low"], em_ci_high_pp=row["em_pp_ci_high"],
                             interpretation=interpretation))
    return main, retrieval_table, pd.DataFrame(selected)


def tex_escape(value):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(c, c) for c in str(value))


def export_table(output, stem, data, headers, formatted, caption, note, column_spec):
    data.to_csv(output / f"{stem}.csv", index=False, float_format="%.17g")
    markdown = [f"# {caption}", "", "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |"]
    markdown.extend("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in formatted)
    markdown.extend(["", note, ""])
    (output / f"{stem}.md").write_text("\n".join(markdown), encoding="utf-8")
    latex = ["% Requires \\usepackage{booktabs}; use table* for a two-column paper.",
             r"\begin{table*}[t]", r"\centering", r"\small", r"\caption{" + tex_escape(caption) + "}",
             r"\label{tab:" + stem.replace("_", "-") + "}",
             r"\begin{tabular}{" + column_spec + "}", r"\toprule",
             " & ".join(map(tex_escape, headers)) + r" \\", r"\midrule"]
    latex.extend(" & ".join(map(tex_escape, row)) + r" \\" for row in formatted)
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
                  r"\parbox{0.97\textwidth}{\footnotesize " + tex_escape(note) + "}", r"\end{table*}", ""])
    (output / f"{stem}.tex").write_text("\n".join(latex), encoding="utf-8")


def export_tables(output, tables):
    main, retrieval, paired = tables
    export_table(output, "table_1_main_results", main,
                 ["Configuration", "N", "EM (%)", "Token F1", "Input tokens", "Client latency (ms)"],
                 [[r.configuration, r.n, f"{r.exact_match_percent:.2f}", f"{r.token_f1:.4f}",
                   f"{r.avg_input_tokens:.1f}", f"{r.recorded_client_latency_ms:.1f}"] for r in main.itertuples()],
                 "Main test-set answer results", "All conditions use the same 150 questions. EM/F1 are lexical scores. "
                 "Latency is recorded retrieval plus generation client time, excluding deliberate pacing and model/index setup; retries and network effects remain.", "@{}lrrrrr@{}")
    export_table(output, "table_2_retrieval_metrics", retrieval,
                 ["Method", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10"],
                 [[r["method"], *[f"{r[c]:.4f}" for c in ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10")]]
                  for r in retrieval.to_dict("records")], "Annotated-source retrieval results",
                 "One row per shared top-10 ranking method, not per delivered-generation K. Hit@K is annotated-source recovery, not exhaustive relevance recall; all metrics are fractions.", "@{}lrrrrr@{}")
    export_table(output, "table_3_paired_comparisons", paired,
                 ["Comparison", "EM change (pp)", "F1 change", "95% CI (F1)", "Interpretation"],
                 [[r.comparison, f"{r.delta_em_pp:+.2f}", f"{r.delta_f1:+.4f}",
                   f"[{r.f1_ci_low:+.4f}, {r.f1_ci_high:+.4f}]", r.interpretation] for r in paired.itertuples()],
                 "Selected paired answer-quality comparisons",
                 "Left minus right over the same 150 questions. F1 effects and intervals are fractions; EM changes are percentage points. "
                 "Intervals use 10,000 paired question bootstrap resamples, seed 42, and are pointwise, without multiplicity adjustment. "
                 "An interval excluding zero is not conclusive proof of general superiority. CSV also includes EM interval endpoints.",
                 "@{}p{0.27\\textwidth}rrp{0.17\\textwidth}p{0.23\\textwidth}@{}")


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.titlepad": 14, "axes.labelsize": 10, "legend.fontsize": 9,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.facecolor": "white", "figure.facecolor": "white",
                         "axes.axisbelow": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})


def save_figure(fig, output, spec):
    stem, title, _ = spec
    for extension in ("png", "pdf"):
        metadata = {"Title": title, "Creator": "CS391 saved-results visualization"} if extension == "pdf" else None
        fig.savefig(output / f"{stem}.{extension}", dpi=300, metadata=metadata)
    plt.close(fig)


def baseline_line(ax, value, label):
    ax.axhline(value, color=COLORS["llm_only"], linewidth=1.3, linestyle=":", label=label, zorder=2)


def plot_exact_match(output, data):
    fig, ax = plt.subplots(figsize=FIGURES[0][2], layout="constrained")
    width = .23
    best = float(data.mean_em.max())
    for mi, method in enumerate(METHODS):
        for ki, k in enumerate(KS):
            value = float(data.loc[(method, k), "mean_em"])
            x = mi + (ki - 1) * width
            ax.bar(x, value * 100, width * .92, color=COLORS[method], hatch=HATCHES[ki],
                   edgecolor="#222222", linewidth=1.4 if np.isclose(value, best) else .6)
            ax.text(x, value * 100 + 1.0, f"{value * 100:.1f}", ha="center", fontsize=9)
            if np.isclose(value, best):
                ax.annotate("Best observed", (x, value * 100 + 2.0), xytext=(0, 25),
                            textcoords="offset points", ha="center", fontsize=8,
                            arrowprops={"arrowstyle": "-", "color": "#555555", "lw": .7})
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="white", edgecolor="#333333", hatch=h, label=f"K={k}") for k, h in zip(KS, HATCHES)]
    base = float(data.loc[("llm_only", 0), "mean_em"]) * 100
    baseline_line(ax, base, f"LLM only ({base:.1f}%)")
    handles.append(ax.lines[-1])
    ax.legend(handles=handles, loc="upper left", ncols=4, frameon=False)
    ax.set(xticks=range(3), xticklabels=[NAMES[m] for m in METHODS], ylim=(0, 65),
           ylabel="Mean normalized Exact Match (%)", xlabel="Retrieval method", title=FIGURES[0][1])
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.grid(axis="y")
    save_figure(fig, output, FIGURES[0])


def plot_f1(output, data):
    fig, ax = plt.subplots(figsize=FIGURES[1][2], layout="constrained")
    for method in METHODS:
        values = [float(data.loc[(method, k), "mean_f1"]) for k in KS]
        ax.plot(KS, values, color=COLORS[method], marker=MARKERS[method], linestyle=STYLES[method],
                label=NAMES[method], linewidth=1.8, markersize=7)
    base = float(data.loc[("llm_only", 0), "mean_f1"])
    baseline_line(ax, base, f"LLM only (F1={base:.4f})")
    ax.set(xticks=KS, xlim=(2.5, 10.5), ylim=(0, 1), ylabel="Mean token-level F1 (fraction)",
           xlabel="Number of retrieved passages (K)", title=FIGURES[1][1])
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y")
    save_figure(fig, output, FIGURES[1])


def plot_retrieval(output, retrieval):
    fig, (ax, small) = plt.subplots(1, 2, figsize=FIGURES[2][2], width_ratios=[3.5, 1], layout="constrained")
    x = np.arange(4)
    for i, method in enumerate(METHODS):
        row = retrieval[retrieval.method == method].iloc[0]
        ax.plot(x, [row[f"Hit@{k}"] for k in (1, 3, 5, 10)], color=COLORS[method],
                marker=MARKERS[method], linestyle=STYLES[method], label=NAMES[method], linewidth=1.8)
        small.bar(i, row["MRR@10"], color=COLORS[method], edgecolor="#222222", linewidth=.5)
        small.text(i, row["MRR@10"] + .025, f"{row['MRR@10']:.3f}", ha="center", fontsize=8)
    ax.set(xticks=x, xticklabels=["1", "3", "5", "10"], ylim=(0, 1.05), xlabel="Retrieval cutoff K",
           ylabel="Hit@K: annotated-source recovery")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y")
    small.set(ylim=(0, 1.05), xticks=range(3), xticklabels=["BM25", "Semantic", "Reranker"], title="MRR@10")
    small.tick_params(axis="x", rotation=55, labelsize=8)
    small.spines["left"].set_visible(False)
    small.tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle(FIGURES[2][1], fontsize=12)
    save_figure(fig, output, FIGURES[2])


def plot_tokens(output, data):
    fig, ax = plt.subplots(figsize=FIGURES[3][2], layout="constrained")
    offsets = {("llm_only", 0): (10, -20), ("bm25", 3): (-10, -18), ("bm25", 5): (0, -18),
               ("bm25", 10): (-12, -18), ("semantic", 3): (-12, -22), ("semantic", 5): (-8, -23),
               ("semantic", 10): (-24, -22), ("reranker", 3): (25, 28), ("reranker", 5): (5, 19),
               ("reranker", 10): (16, 15)}
    best = float(data.mean_f1.max())
    for m, k in CONFIGS:
        row = data.loc[(m, k)]
        ax.scatter(row.avg_input_tokens, row.mean_f1, color=COLORS[m], s=65,
                   marker="D" if m == "llm_only" else MARKERS[m], edgecolors="#333333", linewidths=.5,
                   label=NAMES[m] if k in (0, 3) else None, zorder=3)
        text = "LLM only" if m == "llm_only" else f"K={k}"
        if np.isclose(row.mean_f1, best):
            text += "\nBest observed F1"
        ax.annotate(text, (row.avg_input_tokens, row.mean_f1), xytext=offsets[(m, k)],
                    textcoords="offset points", fontsize=8, ha="center", va="center",
                    arrowprops={"arrowstyle": "-", "lw": .5, "color": COLORS[m]})
    ax.set(xlim=(0, float(data.avg_input_tokens.max()) * 1.12), ylim=(0, 1),
           xlabel="Mean input tokens per generated answer", ylabel="Mean token-level F1 (fraction)", title=FIGURES[3][1])
    ax.legend(loc="lower right", frameon=False, ncols=2)
    ax.grid(axis="both")
    save_figure(fig, output, FIGURES[3])


def plot_forest(output, pairs):
    rows = selected_pairs(pairs)
    fig, ax = plt.subplots(figsize=FIGURES[4][2], layout="constrained")
    for i, (key, row) in enumerate(zip(KEY_PAIRS, rows)):
        # Draw endpoints directly: a percentile interval need not contain its point estimate.
        ax.plot([row["f1_ci_low"], row["f1_ci_high"]], [i, i], color=COLORS[key[0]], linewidth=2)
        ax.plot([row["f1_ci_low"], row["f1_ci_high"]], [i, i], "|", color=COLORS[key[0]], markersize=9)
        ax.scatter(row["delta_f1"], i, color=COLORS[key[0]], marker=MARKERS[key[0]], zorder=3, s=40)
        ax.text(.995, i, f"{row['delta_f1']:+.4f}  [{row['f1_ci_low']:+.4f}, {row['f1_ci_high']:+.4f}]",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    low = min(r["f1_ci_low"] for r in rows)
    high = max(r["f1_ci_high"] for r in rows)
    ax.set(xlim=(min(-.05, low - .02), high + .26), ylim=(-.8, len(rows) - .2),
           yticks=range(len(rows)), yticklabels=[display_pair(k) for k in KEY_PAIRS],
           xlabel="Paired mean F1 difference (fraction; left minus right)", title=FIGURES[4][1])
    ax.invert_yaxis()
    ax.grid(axis="x")
    fig.supxlabel("Same 150 questions; 10,000 resamples; seed 42; pointwise intervals, not multiplicity-adjusted", fontsize=8)
    save_figure(fig, output, FIGURES[4])


def plot_latency(output, data):
    fig, ax = plt.subplots(figsize=FIGURES[5][2], layout="constrained")
    labels = ["LLM only"] + [f"{NAMES[m]}\nK={k}" for m, k in CONFIGS[1:]]
    values = [float(data.loc[c, "avg_recorded_end_to_end_latency_ms"]) for c in CONFIGS]
    bars = ax.bar(range(10), values, color=[COLORS[m] for m, _ in CONFIGS],
                  edgecolor="#333333", linewidth=.6)
    for i, (_, k) in enumerate(CONFIGS):
        bars[i].set_hatch(HATCHES[KS.index(k)] if k else "")
    ax.bar_label(bars, labels=[f"{v:.0f}" for v in values], padding=4, fontsize=8)
    ax.set(xticks=range(10), xticklabels=labels, ylim=(0, max(values) * 1.18),
           ylabel="Mean recorded retrieval + generation client latency (ms)", title=FIGURES[5][1])
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(axis="y")
    fig.supxlabel("Excludes pacing and setup; retains outliers, retries and network effects. Observational timings only.", fontsize=8)
    save_figure(fig, output, FIGURES[5])


def captions(data):
    best_em = data.mean_em.idxmax()
    best_f1 = data.mean_f1.idxmax()
    notes = [
        f"Grouped bars show mean normalized EM for all nine RAG conditions; the separate dotted line shows the LLM-only baseline. K is encoded by hatch; colors identify retrieval methods. {display_config(*best_em)} has the highest observed EM ({data.loc[best_em, 'mean_em'] * 100:.2f}%). Bars start at zero. Lexical exact agreement is not direct factual verification.",
        f"Mean token F1 at K=3, 5 and 10, with a no-retrieval reference line. {display_config(*best_f1)} has the highest observed F1 ({data.loc[best_f1, 'mean_f1']:.4f}). Increasing K does not consistently improve the observed means. The axis spans 0-1. No individual-mean confidence intervals are supplied by the frozen summary; paired-difference intervals are not repurposed as mean intervals. This plot is descriptive, with uncertainty for selected differences shown in Figure 5.",
        "Hit@1/3/5/10 measures recovery of each question's single annotated Natural Questions source among the shared top-10 rankings; MRR@10 is shown separately. Semantic retrieval and reranking have higher observed source-recovery rates than BM25 on this dataset. Each method appears once, rather than being duplicated for generation K. Incomplete annotations prevent interpreting Hit@K as exhaustive relevance recall.",
        f"Each point shows mean input tokens and mean F1 for one of ten conditions. Colors and markers identify retrieval method; text identifies K. {display_config(*best_f1)} is annotated as the highest observed F1. Larger contexts consume more input tokens without consistently improving F1. Tokens are usage units, not monetary prices; no financial cost is inferred.",
        "Signed paired mean F1 differences and saved Step 10 percentile 95% intervals for six preselected comparisons. All comparisons pair the same 150 question IDs; the bootstrap uses 10,000 resamples and seed 42. Zero denotes no mean difference. Reranker-versus-semantic intervals at K=5 and K=10 span zero, whereas the selected K=3 intervals are positive. These are pointwise intervals without multiple-comparison adjustment, not proof of universal superiority.",
        "Mean recorded retrieval-plus-generation client latency for all ten conditions, in milliseconds. Deliberate rate-limit pacing, one-time setup and unrelated wall-clock gaps are excluded. All observed latency outliers remain; retries and network effects can affect timings. The repeated retrieval rankings and their recorded query times are shared across K conditions. These observations do not support strong claims about pure inference speed.",
    ]
    lines = ["# Research figure captions", "", "All figures use the same 150 test questions, 696-passage controlled prototype corpus and requested gpt-4.1-mini-2025-04-14 settings. Results are drawn from the frozen Step 10 analysis; no answers were regenerated or scores changed.", ""]
    for i, (spec, note) in enumerate(zip(FIGURES, notes), 1):
        lines.extend([f"## Figure {i}. {spec[1]}", "", note, ""])
    lines.extend(["## Shared limitations", "", "The corpus is not unrestricted Wikipedia. Source annotations may omit relevant evidence. EM/F1 are lexical measures. The 150-question sample, prior BM25-only test exposure, multiple API backend fingerprints, retries and timing gaps constrain interpretation. Bootstrap intervals do not remove selection bias or establish causality. See results/analysis/statistical_analysis.md and the Step 9 audit for the full limitations.", "",
                  "## Paper placement", "", "Prioritize Figures 1 (main EM result), 2 (effect of K) and 5 (paired uncertainty). Figure 3 separates source recovery from downstream answer quality; Figures 4 and 6 document token and recorded latency tradeoffs. Include the three tables using booktabs in LaTeX. Do not choose model settings based on these held-out results.", "",
                  "## Reproduction", "", "Run `python -B -m experiments.generate_rag_figures`; verify with `python -B -m experiments.generate_rag_figures --verify-only`. Tests: `python -B -m unittest discover -s tests -p test_generate_rag_figures.py -v`.", ""])
    return "\n".join(lines)


def montage(output):
    thumbs = []
    for stem, _, _ in FIGURES:
        with Image.open(output / f"{stem}.png") as img:
            thumbs.append(ImageOps.contain(img.convert("RGB"), (1000, 650)))
    canvas = Image.new("RGB", (2000, 3 * 680), "#eeeeee")
    for i, thumb in enumerate(thumbs):
        x, y = (i % 2) * 1000, (i // 2) * 680
        canvas.paste(thumb, (x + (1000 - thumb.width) // 2, y + 20))
        ImageDraw.Draw(canvas).text((x + 12, y + 3), f"Figure {i + 1}", fill="#222222")
    canvas.save(output / "figures_montage.png")


def json_records(data):
    return json.loads(data.to_json(orient="records", double_precision=15))


def run(root=ROOT):
    root = Path(root).resolve()
    before = snapshot(root)
    data, frames, analysis_manifest = load_inputs(root)
    summary, pairs, _, _, retrieval = frames
    tables = paper_tables(data, pairs, retrieval)
    output = root / "results/figures"
    output.mkdir(parents=True, exist_ok=True)
    style()
    plot_exact_match(output, data)
    plot_f1(output, data)
    plot_retrieval(output, retrieval)
    plot_tokens(output, data)
    plot_forest(output, pairs)
    plot_latency(output, data)
    export_tables(output, tables)
    (output / "figure_captions.md").write_text(captions(data), encoding="utf-8")
    montage(output)
    after = snapshot(root)
    require(before == after, "Source experiment/analysis artifacts changed during figure generation")
    manifest = dict(visualization_version="cs391-figures-v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                    source_files={f"results/analysis/{name}": sha(root / "results/analysis" / name) for name in INPUTS},
                    protected_snapshot_before=before, protected_snapshot_after=after,
                    sources_unchanged=before == after, analysis_seed=analysis_manifest["seed"],
                    analysis_bootstrap_iterations=analysis_manifest["bootstrap_iterations"],
                    versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                                  matplotlib=matplotlib.__version__, pillow=Image.__version__),
                    source_code_sha256=sha(Path(__file__)),
                    axis_units=dict(figure_1="EM percent", figure_2="F1 fraction [0,1]", figure_3="Hit fraction as percent; MRR fraction",
                                    figure_4="input token count vs F1 fraction", figure_5="paired F1 difference, fraction",
                                    figure_6="recorded client latency, milliseconds"),
                    colors=COLORS, method_markers=MARKERS, grayscale_line_styles=STYLES,
                    figures=[dict(stem=stem, description=title, inches=list(size), formats=["png", "pdf"], png_dpi=300,
                                  png_pixels=[round(300 * n) for n in size], vector_pdf=True) for stem, title, size in FIGURES],
                    plotted_values=dict(configuration_summary=json_records(summary), retrieval_unique=json_records(tables[1]),
                                        selected_paired_comparisons=json_records(tables[2])),
                    confidence_intervals="Only frozen paired differences from Step 10; no individual-mean error bars; pointwise, not multiplicity-adjusted",
                    missing_inputs=[], unavailable_data=["Individual-configuration mean confidence intervals were not supplied in Step 10 summaries and are not added here", "Monetary cost: no configured pricing"],
                    exclusions="No observations/outliers excluded; baseline is K=0, never K=3", api_calls=0, models_fitted=0,
                    output_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "figure_manifest.json"})
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print((output / "table_1_main_results.md").read_text())
    print("Saved six PNGs (300 DPI), six vector PDFs, three tables (CSV/Markdown/LaTeX), captions, manifest and montage:")
    for path in sorted(output.iterdir()):
        print(path)
    print("Source hashes verified; 10 configurations x 150 questions; source artifacts unchanged; no APIs or model fitting.")
    return manifest


def verify(root=ROOT):
    root = Path(root).resolve()
    output = root / "results/figures"
    manifest = json.loads((output / "figure_manifest.json").read_text())
    load_inputs(root)
    require(snapshot(root) == manifest["protected_snapshot_before"] == manifest["protected_snapshot_after"],
            "Source artifacts changed since figure generation")
    require(sha(Path(__file__)) == manifest["source_code_sha256"], "Plotting code changed; regenerate figures")
    for name, expected in manifest["source_files"].items():
        require(sha(root / name) == expected, f"Source hash changed: {name}")
    for name, expected in manifest["output_sha256"].items():
        require(sha(output / name) == expected, f"Figure/table output hash changed: {name}")
    print("Verified all figure/table hashes and unchanged frozen sources; all 10 configurations present, N=150 each.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Read-only source/output validation")
    args = parser.parse_args()
    try:
        verify() if args.verify_only else run()
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f"Figure generation stopped: {error}\n")


if __name__ == "__main__":
    main()
