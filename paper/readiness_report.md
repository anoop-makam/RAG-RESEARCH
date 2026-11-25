# Paper readiness report

Status: READY WITH DOCUMENTED LIMITATIONS.

- Step 9: 1,500 valid responses, zero failed checks; 47 passes, eight warnings, two unverified checks.
- All ten configurations contain the same 150 unique test IDs; no missing/duplicate pairs.
- Step 10 means match the original per-question scores and independently recomputed Step 9 means (tolerance 1e-9).
- Step 11 source and output hashes match the final statistical files and saved assets.
- Step 12: 24/24 marked reviewed; primary judgments: {'correct': 18, 'uncertain': 5, 'partially_correct': 1}. Uncertain/partial labels remain such; alternate comparisons do not inherit review status.
- Saved model hash and source configuration match the audited experiment; training report confirms StandardScaler, four features, balanced logistic regression, C=0.01, training-only fits and validation MRR@10 selection. No model was loaded or trained for paper production.
- No additional experiments, generations, factuality models, ablations or repeated-generation trials are claimed.
- No LaTeX engine is installed on this machine. Buildable LaTeX will be generated; the available PDF route is direct ReportLab typesetting, not a successful LaTeX compilation.
- Source corpus, annotation incompleteness, prior test inspection, lexical scoring, backend variability, latency noise and pointwise uncertainty remain explicit limitations.
