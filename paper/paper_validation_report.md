# Final paper validation

- Words excluding references: 4497; body excluding table cells and table captions: 4038.
- 32 unique references, all used in body citations; 25 scholarly, including 23 peer-reviewed sources.
- All reference metadata and citation-specific claims checked against primary sources; see citation_verification.md for scope and the BM25 pagination caveat.
- Required abstract and eight numbered research sections present; no placeholder sections.
- 5 actual saved figures with captions and four captioned tables; LaTeX uses relative figure paths.
- Main EM/F1, token/time table, retrieval table and selected intervals are populated directly from Step 10. Prose fact tokens are linked to source fields in paper_validation.json.
- All ten conditions share the same 150 questions; 1,500 valid records and zero audit failures verified.
- Manual review: 24/24; labels concern primary A only, with uncertain/partial judgments preserved. No population-wide manual accuracy claim.
- Training details checked against audited source/hashes and saved training report. No training or generation ran.
- No monetary estimate, adjusted-significance claim, new experiment or causal attribution was introduced.
- PDF route: ReportLab direct typesetting; LaTeX compiler unavailable. LaTeX compilation succeeded: False.
- Automated PDF checks: 13 nonempty pages, 0 out-of-bounds characters, 32 source hyperlinks; all figure/table captions present and no replacement glyphs.
- Original data, model, generation outputs, Step 9/10/11/12 results and annotations remained byte-identical.

## Outstanding checks

The LaTeX source cannot be compiler-tested locally until pdflatex and bibtex are installed. The delivered PDF is explicitly a ReportLab rendering of the same paper, not evidence of successful LaTeX compilation. The title block identifies the course and university; add your preferred author name before submission.

## Rebuild

Run `.venv/bin/python -B paper/build_paper.py` from the repository root. The builder uses the existing Codex bundled Python automatically when local PDF libraries are missing. Other environments need ReportLab, Pillow and pdfplumber plus Pandoc, or a LaTeX toolchain with pdfplumber for checks. No network is needed to rebuild; citations use the saved verified metadata.
