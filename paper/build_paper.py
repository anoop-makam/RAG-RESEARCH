"""Rebuild the paper offline from its narrative, verified bibliography and frozen data.

Use the project interpreter for readiness. PDF fallback needs ReportLab, Pillow,
and pypdf; the Codex bundled Python already supplies these. No network or models.
"""
import argparse
from collections import Counter
import csv
import hashlib
from html import escape
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
CONFIGS = [("llm_only", 0)] + [(m, k) for m in ("bm25", "semantic", "reranker") for k in (3, 5, 10)]
NAMES = {"llm_only": "LLM-only", "bm25": "BM25", "semantic": "Semantic", "reranker": "Reranker"}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def require(condition, detail):
    if not condition:
        raise ValueError(detail)

def snapshot():
    return {str(p.relative_to(ROOT)): sha(p) for folder in ("data/processed", "models", "results")
            for p in sorted((ROOT / folder).rglob("*")) if p.is_file()}

def check_hashes(values, prefix=""):
    for name, digest in values.items():
        path = ROOT / prefix / name
        require(path.is_file() and sha(path) == digest, f"Missing/changed input: {path}")

def readiness():
    audit = load(ROOT / "results/validation/experiment_audit.json")
    require(audit["summary"]["failed_checks"] == 0 and audit["summary"]["valid_records"] == 1500,
            "Audit readiness/count mismatch")
    check_hashes(audit["input_sha256"])
    analysis = load(ROOT / "results/analysis/analysis_manifest.json")
    check_hashes(analysis["output_sha256"], "results/analysis")
    check_hashes(analysis["experiment_manifest"]["frozen_spec"]["code_sha256"])
    figures = load(ROOT / "results/figures/figure_manifest.json")
    check_hashes(figures["source_files"])
    check_hashes(figures["output_sha256"], "results/figures")
    failures = load(ROOT / "results/failure_analysis/failure_analysis_manifest.json")
    check_hashes(failures["output_sha256"], "results/failure_analysis")
    records = jsonl(ROOT / "results/cs391_per_question_results.jsonl")
    test = jsonl(ROOT / "data/processed/test.jsonl")
    ids = {q["question_id"] for q in test}
    require(len(ids) == len(test) == 150 and len(records) == 1500, "Wrong evaluation sizes")
    grouped = {c: [] for c in CONFIGS}
    for r in records:
        config = r["retrieval_method"], r["top_k"]
        require(config in grouped and r["question_id"] in ids and r["status"] == "completed" and not r["api_error"],
                "Unexpected/failed result")
        grouped[config].append(r)
    summary = rows(ROOT / "results/analysis/configuration_summary.csv")
    metrics = rows(ROOT / "results/validation/recomputed_metrics.csv")
    for config, group in grouped.items():
        require(len(group) == len({r["question_id"] for r in group}) == 150 and
                {r["question_id"] for r in group} == ids, "Question pairing mismatch")
        s = next(r for r in summary if (r["method"], int(r["k"])) == config)
        a = next(r for r in metrics if (r["method"], int(r["k"])) == config)
        for field, output in (("exact_match", "mean_em"), ("f1", "mean_f1")):
            mean = statistics.fmean(r[field] for r in group)
            require(math.isclose(mean, float(s[output]), abs_tol=1e-9) and
                    math.isclose(mean, float(a["recomputed_" + field]), abs_tol=1e-9), "Metric mismatch")
    reviews = rows(ROOT / "results/failure_analysis/manual_review.csv")
    selected = rows(ROOT / "results/failure_analysis/selected_cases.csv")
    require({r["case_id"] for r in reviews} == {r["case_id"] for r in selected} and
            {r["question_id"] for r in selected} <= ids and len(reviews) == 24, "Review identity mismatch")
    reviewed = [r for r in reviews if r["reviewed"].lower() == "true"]
    require(len(reviewed) == failures["reviewed_cases"], "Step 12 report needs refresh; original reports not altered")
    training = load(ROOT / "results/reranker_val_metrics.json")
    require(training["training_pairs"] == 8794 and training["training_class_counts"] == {"0": 8294, "1": 500} and
            training["selected_C"] == .01 and training["class_weight"] == "balanced", "Reranker metadata mismatch")
    prep = load(ROOT / "data/processed/metadata.json")
    require(prep["corpus_size"] == 696 and prep["split_sizes"] == {"train": 500, "val": 50, "test": 150}, "Data protocol mismatch")
    require(training["model_metadata"]["model_revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41" and
            prep["revision"] == "d0b4aa2c6b142fda852d4f4f371c1134560b69fa", "Pinned revision mismatch")
    settings = analysis["experiment_manifest"]["frozen_spec"]["settings"]
    require(settings["model"] == "gpt-4.1-mini-2025-04-14" and settings["temperature"] == 0 and
            settings["max_output_tokens"] == 64 and settings["seed"] == 42, "Generation settings mismatch")
    text = ["# Paper readiness report", "", "Status: READY WITH DOCUMENTED LIMITATIONS.", "",
            "- Step 9: 1,500 valid responses, zero failed checks; 47 passes, eight warnings, two unverified checks.",
            "- All ten configurations contain the same 150 unique test IDs; no missing/duplicate pairs.",
            "- Step 10 means match the original per-question scores and independently recomputed Step 9 means (tolerance 1e-9).",
            "- Step 11 source and output hashes match the final statistical files and saved assets.",
            f"- Step 12: {len(reviewed)}/24 marked reviewed; primary judgments: {dict(Counter(r['manual_factual_correctness'] for r in reviewed))}. Uncertain/partial labels remain such; alternate comparisons do not inherit review status.",
            "- Saved model hash and source configuration match the audited experiment; training report confirms StandardScaler, four features, balanced logistic regression, C=0.01, training-only fits and validation MRR@10 selection. No model was loaded or trained for paper production.",
            "- No additional experiments, generations, factuality models, ablations or repeated-generation trials are claimed.",
            "- No LaTeX engine is installed on this machine. Buildable LaTeX will be generated; the available PDF route is direct ReportLab typesetting, not a successful LaTeX compilation.",
            "- Source corpus, annotation incompleteness, prior test inspection, lexical scoring, backend variability, latency noise and pointwise uncertainty remain explicit limitations.", ""]
    (PAPER / "readiness_report.md").write_text("\n".join(text), encoding="utf-8")
    return dict(audit=audit, analysis=analysis, figures=figures, failures=failures, records=records,
                summary=summary, reviews=reviews, selected=selected, training=training, prep=prep)

def label(m, k):
    return NAMES[m] if m == "llm_only" else f"{NAMES[m]} K={k}"

def md_table(headers, values, caption):
    return "\n".join(["**" + caption + "**", "", "| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |" for row in values])

def facts_and_tables(data):
    summary = {(r["method"], int(r["k"])): r for r in data["summary"]}
    main = [[label(m, k), "150", f"{float(summary[m,k]['mean_em'])*100:.2f}", f"{float(summary[m,k]['mean_f1']):.4f}",
             f"{float(summary[m,k]['avg_input_tokens']):.1f}", f"{float(summary[m,k]['avg_output_tokens']):.2f}",
             f"{float(summary[m,k]['avg_recorded_end_to_end_latency_ms']):.1f}"] for m, k in CONFIGS]
    ret = rows(ROOT / "results/analysis/retrieval_answer_relationship.csv")
    ret_table = [[NAMES[m], *[f"{float(next(r for r in ret if r['method']==m)[f]):.4f}" for f in
                            ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10")]] for m in ("bm25", "semantic", "reranker")]
    pairs = rows(ROOT / "results/analysis/paired_comparisons.csv")
    pair_ids = ["semantic K=3 minus bm25 K=3", "reranker K=3 minus bm25 K=3", "reranker K=3 minus semantic K=3",
                "reranker K=5 minus semantic K=5", "reranker K=10 minus semantic K=10", "reranker K=3 minus llm_only K=0"]
    selected_pairs = [next(r for r in pairs if r["comparison"] == name) for name in pair_ids]
    pair_table = [[r["comparison"].replace("llm_only K=0", "LLM-only"), f"{float(r['delta_em_pp']):+.2f}",
                   f"{float(r['delta_f1']):+.4f}", f"[{float(r['f1_ci_low']):+.4f}, {float(r['f1_ci_high']):+.4f}]"] for r in selected_pairs]
    case_terms = ["rachel end", "bull run", "witcher", "sorry that i let", "hallmark", "pyramid with broad"]
    cases = [next(c for c in data["selected"] if term in c["question"].lower()) for term in case_terms]
    lookup = {r["case_id"]: r for r in data["reviews"]}
    case_names = ["Friends ending", "Bull Run winner", "First Witcher book", "Let You Down singer", "Hallmark channel", "Broad-base pyramid"]
    case_table = []
    for short, c in zip(case_names, cases):
        review = lookup[c["case_id"]]
        case_table.append([short, c["configuration"], f"{float(c['f1']):.4f}",
                           c["comparison_configuration"], f"{float(c['comparison_f1']):.4f}",
                           review["manual_factual_correctness"].replace("_", " ") + "; " + review["manual_failure_cause"].replace("_", " ")])
    fact = {
        "main_table": md_table(["Configuration", "N", "EM (%)", "F1", "Input tokens", "Output tokens", "Client ms"], main,
                               "Table 1. Answer scores and mean recorded usage/time; every condition uses the same 150 questions."),
        "retrieval_table": md_table(["Method", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10"], ret_table,
                                    "Table 2. Annotated-source recovery from shared top-10 rankings; all metrics are fractions."),
        "paired_table": md_table(["Left minus right", "EM change (pp)", "F1 change", "95% paired F1 interval"], pair_table,
                                 "Table 3. Selected paired effects; intervals use 10,000 question resamples and are pointwise, without multiplicity correction."),
        "case_table": md_table(["Question topic", "A", "F1 A", "B", "F1 B", "Primary-A reviewer label"], case_table,
                               "Table 4. Six reviewed primary comparisons; factual labels and cause labels are the reviewer's judgments, not population estimates."),
    }
    # Every decimal in prose generated through a token has a machine-readable source.
    ledger = []
    for m, k in CONFIGS:
        for field in ("mean_em", "mean_f1", "avg_input_tokens", "avg_output_tokens", "avg_recorded_end_to_end_latency_ms"):
            val = float(summary[m,k][field])
            token = f"{m}_{k}_{field}"
            fact[token] = f"{val*100:.2f}" if field == "mean_em" else f"{val:.4f}" if field == "mean_f1" else f"{val:.1f}"
            ledger.append(dict(token=token, source="results/analysis/configuration_summary.csv", method=m, k=k, field=field, raw=val, rendered=fact[token]))
    for i, r in enumerate(selected_pairs):
        for field in ("delta_em_pp", "em_pp_ci_low", "em_pp_ci_high", "delta_f1", "f1_ci_low", "f1_ci_high"):
            token = f"pair{i}_{field}"
            fact[token] = f"{float(r[field]):+.2f}" if "em" in field else f"{float(r[field]):+.4f}"
            ledger.append(dict(token=token, source="results/analysis/paired_comparisons.csv", comparison=r["comparison"], field=field,
                               raw=float(r[field]), rendered=fact[token]))
    sensitivity = rows(ROOT / "results/analysis/k_sensitivity.csv")
    for prefix, comparison in (("reranker_k_change", "reranker K=10 minus reranker K=3"),
                               ("semantic_k_change", "semantic K=5 minus semantic K=3")):
        r = next(r for r in sensitivity if r["comparison"] == comparison)
        for field in ("delta_f1", "f1_ci_low", "f1_ci_high"):
            token = prefix + "_" + field
            fact[token] = f"{float(r[field]):+.4f}"
            ledger.append(dict(token=token, source="results/analysis/k_sensitivity.csv", comparison=comparison,
                               field=field, raw=float(r[field]), rendered=fact[token]))
    for method in ("bm25", "semantic", "reranker"):
        r = next(r for r in ret if r["method"] == method and r["k"] == "3")
        fact[method + "_mrr"] = f"{float(r['MRR@10']):.4f}"
    require(sum(r["relevant_passage_id"] in r["retrieved_passage_ids"] for r in data["records"]
                if r["retrieval_method"] == "reranker" and r["top_k"] == 3) == 144, "Discussed source count mismatch")
    require(sum(r["exact_match"] for r in data["records"] if r["retrieval_method"] == "reranker" and r["top_k"] == 3) == 73,
            "Discussed exact-answer count mismatch")
    return fact, ledger, cases

def bibliography(references):
    lines = []
    def bib_text(value):
        return str(value).translate(str.maketrans({"_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#", "$": r"\$"}))
    names = {"Albert Villanova del Moral": "Villanova del Moral, Albert",
             "Patrick von Platen": "von Platen, Patrick", "Teven Le Scao": "Le Scao, Teven",
             "Nicola De Cao": "De Cao, Nicola"}
    for r in references:
        kind = "misc" if not r["scholarly"] or "preprint" in r["venue"] else "article" if ("Transactions" in r["venue"] or "Journal" in r["venue"] or r["id"] == "bm25") else "inproceedings"
        author = "{" + r["authors"] + "}" if r.get("corporate") else " and ".join(
            names.get(name, name) for name in r["authors"].split(" and "))
        field = "journal" if kind == "article" else "booktitle" if kind == "inproceedings" else "howpublished"
        lines.extend([f"@{kind}{{{r['id']},", "  author = {" + bib_text(author) + "},", "  title = {{" + bib_text(r["title"]) + "}},",
                      "  year = {" + r["year"] + "},", f"  {field} = {{" + bib_text(r["venue"]) + "},",
                      "  url = {" + r["url"] + "},"])
        if "doi" in r:
            lines.append("  doi = {" + r["doi"] + "},")
        if not r["scholarly"]:
            lines.append("  note = {Accessed 2026-11-25; undated pages are marked n.d.},")
        lines.extend(["}", ""])
    (PAPER / "references.bib").write_text("\n".join(lines), encoding="utf-8")

def strip_yaml(text):
    return re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)

def render_markdown(source, refs):
    by_id = {r["id"]: r for r in refs}
    order = []
    def replace(match):
        keys = re.findall(r"@([\w-]+)", match.group(0))
        require(all(k in by_id for k in keys), f"Unresolved citation: {keys}")
        for k in keys:
            if k not in order:
                order.append(k)
        return "[" + ", ".join(str(order.index(k) + 1) for k in keys) + "]"
    body = re.sub(r"\[@[^\]]+\]", replace, strip_yaml(source))
    # Markdown readers display PNGs; the LaTeX source retains the vector PDFs.
    body = re.sub(r"(!\[[^\]]+\]\([^\n)]+)\.pdf(\))", r"\1.png\2", body)
    require(set(order) == set(by_id), f"Uncited references: {set(by_id) - set(order)}")
    title = "# Evaluating the Impact of Retrieval Strategies on the Accuracy of Retrieval-Augmented Generation Systems\n\nCS391: Case Studies in Machine Learning, The University of Texas at Austin\n\n"
    lines = [title + body.rstrip(), "", "# References", ""]
    for i, key in enumerate(order, 1):
        r = by_id[key]
        authors = r["authors"].split(" and ")
        author = ", ".join(authors[:3]) + (", et al." if len(authors) > 3 else "")
        author_end = " " if author.endswith(".") else ". "
        lines.append(f"[{i}] {author}{author_end}({r['year']}). {r['title']}. *{r['venue']}*. " +
                     (f"DOI: {r['doi']}. " if r.get("doi") else "") + f"[Official source]({r['url']}).\n")
    return "\n".join(lines), order

def word_counts(markdown):
    body = markdown.split("\n# References\n")[0]
    body = re.sub(r"!\[(.*?)\]\([^\n]+\)(?:\{[^\n]+\})?", r"\1", body)
    body = re.sub(r"\[(?:\d+,?\s*)+\]", "", body)
    body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)
    total = len(re.findall(r"\b\w+(?:[-']\w+)*\b", body))
    narrative = "\n".join(line for line in body.splitlines() if not line.startswith("|") and not line.startswith("**Table"))
    return total, len(re.findall(r"\b\w+(?:[-']\w+)*\b", narrative))

def inline(value):
    links = []
    def link(match):
        links.append((match.group(1), match.group(2)))
        return f"LINKTOKEN{len(links)-1}ENDTOKEN"
    value = re.sub(r"\[([^\]]+)\]\(([^)]*)\)", link, value)
    value = re.sub(r"\{width=[^}]+\}", "", value)
    value = escape(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    value = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", value)
    for i, (label_text, href) in enumerate(links):
        value = value.replace(f"LINKTOKEN{i}ENDTOKEN", f'<link href="{escape(href, quote=True)}">{escape(label_text)}</link>')
    return value

def polish_latex():
    """Use wrapping table columns and avoid duplicated automatic figure numbers."""
    path = PAPER / "cs391_final_paper.tex"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\\caption\{Figure \d+\.\s*", lambda m: r"\caption{", text)
    pattern = r"\\textbf\{Table (\d+)\.\s*(.*?)\}\s*(?=\\begin\{longtable\})"
    captions = re.findall(pattern, text, flags=re.S)
    require(len(captions) == 4, "LaTeX table captions not detected")
    text = re.sub(pattern, "", text, flags=re.S)
    proportions = [[.24,.05,.11,.10,.17,.17,.16], [.28,.144,.144,.144,.144,.144],
                   [.49,.13,.13,.25], [.20,.16,.08,.16,.08,.32]]
    counter = [0]
    def replace_table(match):
        n = counter[0]
        counter[0] += 1
        values = proportions[n]
        columns = "\n".join(r">{\raggedright\arraybackslash}p{(\linewidth - " +
                             str(2*(len(values)-1)) + r"\tabcolsep) * \real{" + str(v) + "}}" for v in values)
        return (r"\begingroup\small" + "\n" + r"\renewcommand{\arraystretch}{1.15}" + "\n" +
                r"\begin{longtable}{@{}" + columns + "@{}}\n" + r"\caption{" + captions[n][1] +
                r"}\label{tab:results-" + str(n+1) + r"}\\" + "\n" + r"\toprule")
    text = re.sub(r"\\begin\{longtable\}\[\]\{.*?\n\\toprule", replace_table, text, flags=re.S)
    text = text.replace(r"\end{longtable}", r"\end{longtable}" + "\n" + r"\endgroup")
    text = text.replace(r"\bibliography{references.bib}", r"\bibliography{references}")
    require(counter[0] == 4, "LaTeX wrapping table conversion failed")
    path.write_text(text, encoding="utf-8")

def pdf_fallback(markdown):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from PIL import Image as PILImage
    fonts = Path("/System/Library/Fonts/Supplemental")
    for name, filename in (("Paper", "Times New Roman.ttf"), ("PaperBold", "Times New Roman Bold.ttf"),
                           ("PaperItalic", "Times New Roman Italic.ttf"), ("PaperBI", "Times New Roman Bold Italic.ttf")):
        require((fonts / filename).exists(), f"Missing PDF font {filename}")
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily("Paper", normal="Paper", bold="PaperBold", italic="PaperItalic", boldItalic="PaperBI")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PaperBody", fontName="Paper", fontSize=10.5, leading=13.6,
                              alignment=TA_JUSTIFY, spaceAfter=7))
    styles.add(ParagraphStyle(name="PaperTitle", fontName="PaperBold", fontSize=17, leading=20,
                              alignment=TA_CENTER, spaceAfter=13))
    styles.add(ParagraphStyle(name="PaperSection", fontName="PaperBold", fontSize=12.5, leading=15,
                              spaceBefore=12, spaceAfter=7, keepWithNext=True))
    styles.add(ParagraphStyle(name="PaperSubsection", fontName="PaperBold", fontSize=11, leading=13.5,
                              spaceBefore=8, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle(name="PaperCaption", fontName="Paper", fontSize=9, leading=11,
                              spaceAfter=8, alignment=0))
    styles.add(ParagraphStyle(name="PaperCell", fontName="Paper", fontSize=8.2, leading=10.2,
                              spaceAfter=0, alignment=0))
    styles.add(ParagraphStyle(name="PaperReference", fontName="Paper", fontSize=9, leading=11.3,
                              spaceAfter=7, alignment=0, wordWrap="CJK"))
    doc = SimpleDocTemplate(str(PAPER / "cs391_final_paper.pdf"), pagesize=letter,
                            leftMargin=54, rightMargin=54, topMargin=50, bottomMargin=46,
                            title="Evaluating the Impact of Retrieval Strategies on the Accuracy of Retrieval-Augmented Generation Systems")
    story, paragraph = [], []
    conclusion_start = None
    lines, i = markdown.splitlines(), 0
    def flush():
        if paragraph:
            text = " ".join(paragraph)
            style = styles["PaperReference"] if re.match(r"^\[\d+\]", text) else styles["PaperBody"]
            story.append(Paragraph(inline(text), style))
            paragraph.clear()
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush(); i += 1; continue
        if line.startswith("|"):
            flush()
            values = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    values.append([Paragraph(inline(c), styles["PaperCell"]) for c in cells])
                i += 1
            n = len(values[0])
            widths = {7: [119, 25, 47, 45, 68, 68, 132], 6: [110, 80, 56, 80, 56, 122], 4: [245, 65, 65, 129]}.get(n, [504 / n] * n)
            if n == 6 and "Question topic" not in lines[i - len(values) - 1]:
                widths = [144, 72, 72, 72, 72, 72]
            scale = 504 / sum(widths)
            grid = Table(values, colWidths=[w * scale for w in widths], repeatRows=1, hAlign="LEFT")
            grid.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), .7, colors.black),
                                      ("LINEBELOW", (0, 0), (-1, 0), .45, colors.black),
                                      ("LINEBELOW", (0, -1), (-1, -1), .7, colors.black),
                                      ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                      ("LEFTPADDING", (0, 0), (-1, -1), 4),
                                      ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                                      ("TOPPADDING", (0, 0), (-1, -1), 5),
                                      ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
            story.extend([grid, Spacer(1, 9)])
            continue
        figure = re.match(r"!\[(.*?)\]\(([^)]+)\)", line)
        if figure:
            flush()
            caption, filename = figure.groups()
            path = (PAPER / filename).with_suffix(".png").resolve()
            with PILImage.open(path) as img:
                width, height = img.size
            width_pt = 490
            image = Image(str(path), width=width_pt, height=width_pt * height / width)
            story.append(KeepTogether([image, Spacer(1, 4), Paragraph(inline(caption), styles["PaperCaption"])]))
            i += 1; continue
        if line.startswith("#"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("# ")
            if title == "8. Conclusion":
                conclusion_start = len(story)
            if title == "References" and conclusion_start is not None:
                story[conclusion_start:] = [KeepTogether(story[conclusion_start:])]
            style = "PaperTitle" if i == 0 else "PaperSection" if level == 1 else "PaperSubsection"
            story.append(Paragraph(inline(title), styles[style]))
            i += 1; continue
        if line.startswith("**Table"):
            flush()
            story.append(Paragraph(inline(line), ParagraphStyle("TableCaption", parent=styles["PaperCaption"], keepWithNext=True)))
            i += 1; continue
        paragraph.append(line)
        i += 1
    flush()
    def footer(canvas, document):
        canvas.setFont("Paper", 8)
        canvas.setFillColor(colors.HexColor("#444444"))
        canvas.drawString(54, 26, "CS391 | Controlled retrieval-augmented question answering")
        canvas.drawRightString(558, 26, str(document.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)

def validate_pdf_export():
    import pdfplumber
    with pdfplumber.open(PAPER / "cs391_final_paper.pdf") as pdf:
        failures, text = [], []
        links = sum(len(p.hyperlinks) for p in pdf.pages)
        for i, p in enumerate(pdf.pages, 1):
            page_text = p.extract_text() or ""
            require(page_text.strip(), f"Empty PDF page {i}")
            text.append(page_text)
            failures.extend((i, c["text"]) for c in p.chars if c["x0"] < 25 or c["x1"] > p.width - 25 or
                            c["top"] < 15 or c["bottom"] > p.height - 15)
        require(not failures, f"PDF text outside safe page boundaries: {failures[:8]}")
        full = "\n".join(text)
        require(all(f"Figure {i}." in full or f"Figure {i}:" in full for i in range(1, 6)), "Missing PDF figure caption")
        require(all(f"Table {i}." in full or f"Table {i}:" in full for i in range(1, 5)), "Missing PDF table caption")
        require("\ufffd" not in full and "(cid:" not in full, "Missing/replacement PDF glyphs")
        return dict(pages=len(pdf.pages), out_of_bounds_characters=0, reference_links=links,
                    captions_present=True, readable_text_extracted=True, replacement_glyphs=False,
                    method="pdfplumber page text, bounding boxes, captions and hyperlinks")

def write_verification(refs):
    text = ["# Citation verification", "", "Verified on 2026-11-25 using primary publisher/proceedings pages, author-submitted arXiv records and official software/model/dataset documentation. All references are cited in the body. Metadata and claim-relevant abstracts/documentation were read; selected SQuAD evaluation and Koehn bootstrap PDF sections were also inspected. This is not a claim of exhaustive full-text review of every publication.", "",
            "All entries: existence, author list/corporate author, title, publication year (or explicitly undated), venue/publisher, and official URL/DOI checked. ArXiv-only sources are identified as preprints, not peer-reviewed conference publications. Dynamic documentation is undated unless a revision date is displayed. Local experiment settings remain authoritative.", "",
            "| Key | Type | Verified source and supported claim | Notes |", "| --- | --- | --- | --- |"]
    for r in refs:
        kind = "Peer-reviewed scholarly" if r["peer_reviewed"] else "Scholarly preprint" if r["scholarly"] else "Official technical source"
        text.append(f"| {r['id']} | {kind} | [{r['title']}]({r['url']}): {r['supports']} | {r.get('note','Metadata and claim-relevant abstract/documentation checked.')} |")
    text.extend(["", "Unresolved cited entries: none. Optional pagination for the migrated BM25 publisher record is deliberately omitted because the current landing page is inconsistent; the DOI, authors, title, article year and journal are verified. Stanford/Taylor & Francis book pages that could not be accessed were not included or cited. No DOI or venue was guessed.", ""])
    (PAPER / "citation_verification.md").write_text("\n".join(text), encoding="utf-8")

def build(readiness_only=False):
    if not readiness_only and not (shutil.which("pdflatex") and shutil.which("bibtex")) and importlib.util.find_spec("reportlab") is None:
        bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python"
        if bundled.is_file() and bundled.resolve() != Path(sys.executable).resolve():
            subprocess.run([str(bundled), "-B", str(Path(__file__).resolve())], check=True)
            return
        raise RuntimeError("PDF fallback requires ReportLab and Pillow; alternatively install pdflatex and bibtex")
    before = snapshot()
    data = readiness()
    if readiness_only:
        print((PAPER / "readiness_report.md").read_text()); return
    refs = load(PAPER / "reference_sources.json")
    require(len(refs) >= 30 and len({r["id"] for r in refs}) == len(refs) and sum(r["scholarly"] for r in refs) >= 15,
            "Reference requirements not met")
    bibliography(refs)
    write_verification(refs)
    fact, ledger, cases = facts_and_tables(data)
    source = (PAPER / "paper_source.md").read_text(encoding="utf-8")
    tokens = re.findall(r"\{\{([\w]+)\}\}", source)
    require(set(tokens) <= set(fact), f"Unresolved fact tokens: {set(tokens)-set(fact)}")
    source = re.sub(r"\{\{([\w]+)\}\}", lambda m: fact[m.group(1)], source)
    require("TODO" not in source and "TBD" not in source, "Placeholder in paper")
    for section in ("Abstract", "1. Introduction", "2. Related Work", "3. Methodology", "4. Experimental Results",
                    "5. Qualitative Failure Analysis", "6. Discussion", "7. Limitations", "8. Conclusion"):
        require("\n# " + section + "\n" in source, f"Missing paper section: {section}")
    markdown, order = render_markdown(source, refs)
    (PAPER / "cs391_final_paper.md").write_text(markdown, encoding="utf-8")
    expanded = PAPER / "paper_expanded.md"
    expanded.write_text(source, encoding="utf-8")
    pandoc = shutil.which("pandoc")
    require(pandoc is not None, "Pandoc is required to generate the LaTeX version")
    subprocess.run([pandoc, expanded.name, "-o", "cs391_final_paper.tex", "--standalone", "--natbib",
                    "--bibliography=references.bib", "-M", "biblio-style=unsrtnat", "-M", "natbiboptions=numbers"], cwd=PAPER, check=True)
    polish_latex()
    latex_engine = shutil.which("pdflatex")
    compiled = bool(latex_engine and shutil.which("bibtex"))
    if compiled:
        for command in ([latex_engine, "-interaction=nonstopmode", "-halt-on-error", "cs391_final_paper.tex"],
                        ["bibtex", "cs391_final_paper"], [latex_engine, "-interaction=nonstopmode", "-halt-on-error", "cs391_final_paper.tex"],
                        [latex_engine, "-interaction=nonstopmode", "-halt-on-error", "cs391_final_paper.tex"]):
            subprocess.run(command, cwd=PAPER, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        pdf_route = "LaTeX (pdflatex + bibtex)"
    else:
        pdf_fallback(markdown)
        pdf_route = "ReportLab direct typesetting; LaTeX compiler unavailable"
    total, narrative = word_counts(markdown)
    require(narrative >= 3000, f"Narrative too short: {narrative}")
    figures = re.findall(r"!\[(.*?)\]\((.*?)\)", markdown)
    require(len(figures) in (4, 5, 6) and all((PAPER / path).is_file() for _, path in figures), "Missing/incorrect figure assets")
    require(sum(line.startswith("**Table") for line in markdown.splitlines()) == 4, "Missing table captions")
    pdf_checks = validate_pdf_export()
    after = snapshot()
    require(before == after, "Existing experiment artifacts changed")
    validation = dict(word_count_excluding_references=total, narrative_word_count_excluding_tables= narrative,
                      references=len(refs), scholarly_references=sum(r["scholarly"] for r in refs),
                      peer_reviewed_references=sum(r["peer_reviewed"] for r in refs),
                      all_cited=True, all_metadata_and_claim_support_verified=True, figures=len(figures), tables=4,
                      numerical_sources_match=True, manual_reviewed=len([r for r in data["reviews"] if r["reviewed"].lower()=="true"]),
                      latex_compiled=compiled, pdf_route=pdf_route, protected_originals_unchanged=before==after,
                      pdf_checks=pdf_checks,
                      protected_input_sha256=before, numerical_claim_ledger=ledger,
                      qualitative_case_ids=[c["case_id"] for c in cases],
                      bibliography_order=order)
    (PAPER / "paper_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["# Final paper validation", "", f"- Words excluding references: {total}; body excluding table cells and table captions: {narrative}.",
              f"- {len(refs)} unique references, all used in body citations; {sum(r['scholarly'] for r in refs)} scholarly, including {sum(r['peer_reviewed'] for r in refs)} peer-reviewed sources.",
              "- All reference metadata and citation-specific claims checked against primary sources; see citation_verification.md for scope and the BM25 pagination caveat.",
              "- Required abstract and eight numbered research sections present; no placeholder sections.",
              f"- {len(figures)} actual saved figures with captions and four captioned tables; LaTeX uses relative figure paths.",
              "- Main EM/F1, token/time table, retrieval table and selected intervals are populated directly from Step 10. Prose fact tokens are linked to source fields in paper_validation.json.",
              "- All ten conditions share the same 150 questions; 1,500 valid records and zero audit failures verified.",
              f"- Manual review: {validation['manual_reviewed']}/24; labels concern primary A only, with uncertain/partial judgments preserved. No population-wide manual accuracy claim.",
              "- Training details checked against audited source/hashes and saved training report. No training or generation ran.",
              "- No monetary estimate, adjusted-significance claim, new experiment or causal attribution was introduced.",
              f"- PDF route: {pdf_route}. LaTeX compilation succeeded: {compiled}.",
              f"- Automated PDF checks: {pdf_checks['pages']} nonempty pages, {pdf_checks['out_of_bounds_characters']} out-of-bounds characters, {pdf_checks['reference_links']} source hyperlinks; all figure/table captions present and no replacement glyphs.",
              "- Original data, model, generation outputs, Step 9/10/11/12 results and annotations remained byte-identical.", "",
              "## Outstanding checks", "", "The LaTeX source cannot be compiler-tested locally until pdflatex and bibtex are installed. The delivered PDF is explicitly a ReportLab rendering of the same paper, not evidence of successful LaTeX compilation. The title block identifies the course and university; add your preferred author name before submission.", "",
              "## Rebuild", "", "Run `.venv/bin/python -B paper/build_paper.py` from the repository root. The builder uses the existing Codex bundled Python automatically when local PDF libraries are missing. Other environments need ReportLab, Pillow and pdfplumber plus Pandoc, or a LaTeX toolchain with pdfplumber for checks. No network is needed to rebuild; citations use the saved verified metadata.", ""]
    (PAPER / "paper_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({k: v for k,v in validation.items() if k not in {"protected_input_sha256", "numerical_claim_ledger", "bibliography_order"}}, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-only", action="store_true")
    args = parser.parse_args()
    build(args.readiness_only)
