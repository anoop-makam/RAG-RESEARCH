# Controlled CS391 RAG evaluation

The experiment uses the saved corpus, frozen logistic reranker, the existing
`AnswerGenerator`, and one shared API configuration. It never retrains a model or
revises answers. The ten conditions are LLM-only (once), and BM25, semantic, and
reranker RAG at K=3,5,10. Each question's top-10 retrieval/reranking is computed
once; K conditions use prefixes of the same order. The reranker uses only the
natural BM25-top10/semantic-top10 union.

## 1. Validation pilot

Configure `.env` as described in `generation.md`. To execute all ten conditions
on the first five validation questions:

```bash
.venv/bin/python -m experiments.evaluate_rag --phase pilot --local-files-only
```

This requires 50 answers, reusing completed matching cache entries (including the
previous 20 K=3 smoke-test answers). At 8 requests/minute, 30 uncached requests take
roughly four minutes, plus local model setup. A dry run is also available:

```bash
.venv/bin/python -m experiments.evaluate_rag --phase pilot --dry-run --local-files-only
```

A dry run does not pass the pilot gate. A successful live pilot requires every
configuration to complete, consistent settings/model identity, valid full prompts,
and usage counts from providers that support them. Accuracy is **not** a gate:
a low-scoring answer does not justify tuning the test run. API failures prevent the
pilot passing and are saved explicitly.

Scoring rules and edge cases are checked with:

```bash
.venv/bin/python -m unittest discover -s tests -p test_rag_evaluation.py -v
```

## 2. Freeze and evaluate the test set

After the pilot passes:

```bash
.venv/bin/python -m experiments.evaluate_rag --phase test --local-files-only
```

This runs all 150 saved test questions, or 1,500 answers. At the configured 8 RPM,
expect a little over three hours without existing cache entries. The pilot model,
prompt, scoring version, source code, corpus, reranker, generation settings, seed,
and declared sample size are frozen. Changing them after the test starts is
rejected. Do not revise choices based on test results.

For a smaller **predeclared** study, use the same size in both commands:

```bash
.venv/bin/python -m experiments.evaluate_rag --phase pilot --test-size 10 --local-files-only
.venv/bin/python -m experiments.evaluate_rag --phase test --test-size 10 --local-files-only
```

The subset is selected without inspecting performance, using seed 42 and sorted
question IDs. IDs, seed, size, and source checksum are recorded. Small subsets have
lower precision and weaker representativeness. The current requested research run
uses all 150 questions, not a subset.

Interrupted runs resume by repeating the exact command: successful cached answers
are reused, errors are retried, and pacing remains active. Only one runner should
use the shared cache at a time. Fatal provider quota/authentication errors and
long server retry deferrals stop the run and preserve completed answers. Missing
combinations are represented as `not_attempted`, never silently dropped.

## Outputs

The latest completed/partial phase writes:

- `results/cs391_answer_metrics.csv`
- `results/cs391_answer_metrics.json`
- `results/cs391_per_question_results.jsonl`
- `results/cs391_accuracy_comparison.png`
- `results/cs391_f1_by_k.png`
- `results/cs391_latency_by_method.png`
- `results/cs391_retrieval_vs_answer_quality.png`
- `results/cs391_experiment_manifest.json`

Pilot and test copies are kept separately in `results/cs391/pilot/` and
`results/cs391/test/`, including `generation_raw.jsonl`. During generation, this raw
file is flushed after every answer, so progress can be inspected immediately.
References are attached only during scoring, not in raw generation requests.
The root-level tables/figures always identify their phase. Partial runs are marked
incomplete. No dry-run values should be interpreted as measured answer accuracy.

Inspect the table:

```bash
.venv/bin/python - <<'PY'
import csv
from pathlib import Path
with Path('results/cs391_answer_metrics.csv').open() as handle:
    for row in csv.DictReader(handle):
        print(row['phase'], row['method'], row['k'], row['exact_match'], row['f1'],
              row['completed'], row['api_failures'])
PY
```

Rescore saved raw results without API calls or changing answers:

```bash
.venv/bin/python -m experiments.evaluate_rag --phase test --rescore
```

For pilot rescoring use `--phase pilot`. The matching original size/seed and
frozen environment settings are required. Regenerate plots from the latest report:

```bash
.venv/bin/python -m experiments.evaluate_rag --plots-only
```

## Metric interpretation

Normalized exact match and multiset token F1 use lowercase, ASCII punctuation
removal, English-article removal, and whitespace normalization, following the
[standard SQuAD evaluator](https://github.com/huggingface/evaluate/blob/main/metrics/squad/compute_score.py).
Each metric takes its best value over all annotated answers. No extra alias
expansion, LLM grading, or reference-based answer rewriting is applied. Empty
predictions and the instructed abstentions (`Insufficient evidence.`, `I don't
know.`) receive zero on this answer-bearing dataset.

Primary answer metrics include every assigned question: API failures and
unattempted outputs contribute zero, with failure counts and completed-only scores
reported separately. Comparisons should be interpreted only after all conditions
complete. Token averages exclude unknown usage and report the number missing;
unknown counts are never imputed as measured zeros.

Retrieval Hit@1/3/5/10 and MRR@10 refer to the full original top-10 ranking and are
therefore independent of the number of sources supplied to the LLM.
`context_source_hit` measures the annotated source within the actual K sources.
`candidate_pool_source_hit` reports the reranker's full natural union coverage.
These measure recovery of one annotated source, not exhaustive relevance recall.

Timing excludes setup and deliberate client pacing. API retry backoff remains in
generation latency. Cached answers retain their original measured timing/usage;
`llm_calls_this_run` counts only new HTTP attempts, while
`original_llm_calls_including_cache` includes historical attempts. Errors during
successful retries are separately counted. Candidate retrieval is reused across K,
so table rows are configuration-level timings, not a sum of independent retrieval
operations executed for every K.

Costs are null unless explicit provider prices were configured before freezing.
Estimates exclude provider cache discounts and unreported retry usage, and are
not exact invoices. Pricing does not replace logging raw provider usage.

Representative comparisons use exact match for answer success and semantic
retrieval as the reference when describing reranker rank improvements/worsening.
An annotated source appearing in context with EM=0 is not proof of a factual
hallucination: reference annotations can be incomplete or differently worded.
Results concern this controlled answer-bearing prototype corpus. An earlier
BM25-only experiment inspected test retrieval results; report that history rather
than claiming the test set was untouched throughout development.
