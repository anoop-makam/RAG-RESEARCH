# Answer generation experiments

The generator uses one API configuration for all four conditions: `llm_only`,
`bm25`, `semantic`, and `reranker`. It does not score answer correctness, verify
facts, revise answers, fit models, or open the final test set.

## Configure

An ignored `.env` template has been created with an empty key. Edit it locally:

```ini
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini-2025-04-14
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-provider-key
LLM_TEMPERATURE=0
LLM_MAX_OUTPUT_TOKENS=64
LLM_SEED=42
LLM_MAX_TOKENS_FIELD=max_completion_tokens
LLM_REQUESTS_PER_MINUTE=8
```

Use the actual credential in `.env`, never in Python files or command arguments.
`.env.example` is the shareable template. Exported environment variables take
precedence over `.env`. The loader accepts simple assignments and quoted strings;
it does not execute shell expressions, expand variables, or accept multiline values.
`OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are fallback key names if `LLM_API_KEY` is empty.
No additional Python packages are required for the API transport.

The default model is a fixed GPT-4.1 mini snapshot. Temperature zero and an optional
seed are best-effort reproducibility controls, not a guarantee. The provider's
returned model and system fingerprint, when supplied, are retained.

For another OpenAI-compatible endpoint, set `LLM_PROVIDER=openai_compatible`, the
base URL ending in its API prefix (often `/v1`), and the provider's model name.
Set `LLM_MAX_TOKENS_FIELD=max_tokens` if that endpoint requires it. Set
`LLM_SEED=none` if seeds are unsupported. For Anthropic, set
`LLM_PROVIDER=anthropic`, `LLM_BASE_URL=https://api.anthropic.com/v1`, a valid Claude
model name, and `LLM_SEED=none`. Set `LLM_TEMPERATURE=none` for models that reject
temperature. Settings are fixed across all four conditions; unsupported parameters
cause a logged error rather than silently changing a condition's configuration.

Official API references:
[OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini),
[Anthropic Messages](https://platform.claude.com/docs/en/api/messages).

## Inspect five validation questions without LLM requests

Run from the project root:

```bash
.venv/bin/python -m experiments.generate_answers \
  --dry-run --limit 5 --top-k 3 --local-files-only
```

This uses the first five validation questions in saved order and prints 20 exact
request payloads: five questions times four conditions. No API key is needed, no
LLM calls are made, and token counts/answers are left null. Real local retrieval
and reranking are still performed using the existing corpus and saved classifier.
`--local-files-only` applies only to the embedding model; it avoids Hugging Face
network access and requires the previously downloaded embedding model cache.

To capture printed prompts:

```bash
.venv/bin/python -m experiments.generate_answers \
  --dry-run --limit 5 --top-k 3 --local-files-only > results/generation/prompts.txt
```

The runner creates `results/generation/`; use the first command before shell
redirection if the directory does not yet exist.

## Execute the five-question API test

After configuring a real API key:

```bash
.venv/bin/python -m experiments.generate_answers \
  --limit 5 --top-k 3 --local-files-only
```

This makes up to 20 initial successful LLM calls, plus bounded retries for temporary
errors. `--local-files-only` does **not** disable LLM API requests. Each call has
up to 64 output tokens by default. Inspect the small run before increasing `--limit`.

Request attempts are paced at 8 RPM by default (at least 7.5 seconds between
starts), including retries. This leaves headroom under the shared 10 RPM limit
reported for this project. The 20-call smoke test takes roughly 2.5 minutes plus
setup/response overhead. Other applications share the provider budget; this
per-process limiter does not coordinate concurrent clients or enforce token limits.
Set `LLM_REQUESTS_PER_MINUTE` to another positive rate of at least 1, or 0 to disable
it. Cached answers and dry runs do not wait. Changing pacing does not invalidate
completed answers or change model/prompt settings.

All RAG conditions use the same K and full source texts. Sources have explicit
numbers and passage IDs in the JSON-formatted user prompt. The same system
instruction is used in every condition. Baseline and evidence-based requests differ
only in their evidence policy and supplied sources. An empty RAG source list asks
for abstention, rather than falling back to model knowledge. Reference answer and
relevance-label fields are discarded before generation.

Candidate retrieval uses BM25 top 10 and semantic top 10 once per question. BM25
and semantic conditions take their first K results. The reranker scores the natural
union with the **saved** logistic model, then supplies its top K. No positives are
inserted. One-time loading, corpus embedding and index construction are separate
from query retrieval timing. Reranker retrieval timing includes both candidate
retrieval times and the reranking operation. K is restricted to 1–10 for this protocol.

## Outputs and resumption

Files are saved under `results/generation/`:

- `validation_first5_top3_<run-id>.jsonl`: live output records.
- `validation_first5_top3_dry_run_<run-id>.jsonl`: prompt-only records; not answers.
- Matching `.metadata.json`: model settings, checksums, setup times and usage totals.
- `cache/<request-hash>.json`: completed answer cache.

Each record includes question ID/text, retrieval condition, requested K, actual
passage IDs/count, complete messages, generated answer, requested and returned
model, prompt version, effective generation settings, input/output token counts,
raw usage details, finish reason, generation and retrieval times, API error status,
retry history, cache status, and source/model artifact checksums. Missing provider
usage remains null, not a fabricated count. The reranker condition also records
its candidate union.

Completed answers are cached by provider, model, effective settings, exact prompt,
question/condition, K and input provenance. Repeating the command reuses successful
answers without LLM calls. Errors are logged but never cached as answers, so a later
run can retry them. A progress JSONL is flushed after each record. Cached answers
retain original usage/timing; current retrieval time is recorded separately. Summary
`new_*_tokens` excludes local cache hits. Concurrent runs sharing a cache are not
coordinated; use one runner per experiment.

`generation_latency_ms` excludes deliberate rate-limit pacing, while
`generation_total_latency_ms` includes it. `rate_limit_wait_ms` records pacing
separately so a later condition does not appear slower merely because it waited
for the request budget. Retry backoff remains part of generation timing.

Temporary HTTP/network errors use bounded exponential backoff (defaults: three
attempts, 1/2-second sleeps, maximum backoff 8 seconds). Pacing may require a longer
wait between request starts. Retry-After values are respected as minimums: if the
server requests a delay above the configured backoff budget, the run stops and
records a deferred retry rather than retrying too soon. Known quota/billing errors
are terminal even when their HTTP status is 429. Authentication, invalid parameter
and other nonretryable failures also stop the run, preserving completed caches.
Only recognized provider error codes/types are retained; exception messages,
response bodies, credentials and request headers are never written to logs.
API timeouts/retries can have unreported
billable usage, so token totals are not an exact bill.

Optionally set `LLM_INPUT_USD_PER_MILLION` and `LLM_OUTPUT_USD_PER_MILLION` using your
provider's current rates to populate `estimated_cost_usd`. Rates are not hardcoded;
estimates exclude cache-specific pricing, discounts and unreported retry usage.
Provider prompt-cache usage is retained separately in the raw usage object.

## Reusable Python API

```python
from src.generation import generate_answer

baseline = generate_answer("Who discovered penicillin?")
print(baseline["generated_answer"])

rag = generate_answer(
    "Who discovered penicillin?",
    passages=[{"passage_id": "p_example", "text": "Alexander Fleming discovered penicillin."}],
)
print(rag["generated_answer"])
```

The standalone function caches completed records. Use the experiment runner to
write the four-condition validation JSONL. Use passage IDs/text from the real fixed
corpus for research runs; the small Python example only illustrates the function.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -p test_generation.py -v
```

These tests use mocked API responses, not generated research measurements. They
check prompt ordering, settings consistency, label exclusion, all 20 five-question
conditions, cache resumption, bounded retries, terminal errors, provider response
parsing, missing usage, and configuration handling.
