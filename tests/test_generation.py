"""Network-free generation, prompt, provider, retry, cache, and five-question checks."""

from contextlib import redirect_stdout
from dataclasses import replace
from email.message import Message
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from experiments.generate_answers import CONDITIONS, run_generation
from src.generation import (
    AnswerGenerator, GenerationConfig, build_messages, load_env,
    parse_response, request_payload,
)


def completion(answer="short answer"):
    return ({"id": "response1", "model": "fixed-model", "choices": [
        {"message": {"content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 3,
                  "prompt_tokens_details": {"cached_tokens": 20}}}, {"x-request-id": "request1"})


class GenerationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = GenerationConfig(model="fixed-model", api_key="secret-key-never-log", requests_per_minute=0)
        self.passages = [{"passage_id": "p_z", "text": "First evidence."},
                         {"passage_id": "p_a", "text": "Second evidence."}]

    def generator(self, **kwargs):
        return AnswerGenerator(self.config, cache_dir=self.root / "cache", **kwargs)

    def test_prompt_order_numbering_and_no_reference_fields(self):
        supplied = [{**self.passages[0], "label": 1, "answers": "private reference"}, self.passages[1]]
        messages = build_messages("question", supplied)
        content = json.loads(messages[1]["content"])
        self.assertEqual([p["passage_id"] for p in content["source_passages"]], ["p_z", "p_a"])
        self.assertEqual([p["source_number"] for p in content["source_passages"]], [1, 2])
        self.assertNotIn("private reference", json.dumps(messages))
        self.assertNotIn("label", json.dumps(messages))
        baseline = build_messages("question")
        self.assertEqual(messages[0], baseline[0])
        self.assertNotIn("source_passages", json.loads(baseline[1]["content"]))
        self.assertIn("Insufficient evidence", json.dumps(build_messages("question", [])))
        with self.assertRaises(ValueError):
            build_messages("question", [self.passages[0], self.passages[0]])

    def test_settings_shared_and_provider_adapters(self):
        messages = build_messages("question", self.passages)
        baseline = build_messages("question")
        left = request_payload(self.config, messages)
        right = request_payload(self.config, baseline)
        for field in ("model", "temperature", "seed", "max_completion_tokens"):
            self.assertEqual(left[field], right[field])
        self.assertEqual(left["max_completion_tokens"], 64)
        anthropic = replace(self.config, provider="anthropic", seed=None, base_url="https://api.anthropic.com/v1")
        payload = request_payload(anthropic, messages)
        self.assertEqual(payload["max_tokens"], 64)
        self.assertNotIn("seed", payload)
        self.assertNotIn("max_completion_tokens", payload)
        parsed = parse_response("anthropic", {"content": [{"type": "text", "text": "answer"}],
                                "usage": {"input_tokens": 15, "output_tokens": 2}, "stop_reason": "end_turn"})
        self.assertEqual(parsed["generated_answer"], "answer")
        self.assertEqual(parsed["input_tokens"], 15)
        compatible = replace(self.config, provider="openai_compatible", seed=None, max_tokens_field="max_tokens")
        self.assertNotIn("store", request_payload(compatible, messages))
        self.assertIn("max_tokens", request_payload(compatible, messages))

    def test_dry_run_never_calls_api_or_requires_key(self):
        calls = []
        generator = AnswerGenerator(replace(self.config, api_key=""), self.root / "cache",
                                    transport=lambda *args: calls.append(args))
        with redirect_stdout(io.StringIO()):
            result = generator.generate_answer("question", self.passages, dry_run=True)
        self.assertEqual(result["status"], "dry_run")
        self.assertIsNone(result["generated_answer"])
        self.assertEqual(calls, [])
        self.assertFalse((self.root / "cache").exists())

    def test_success_usage_and_cache_do_not_repeat_requests(self):
        calls = []

        def transport(*args):
            calls.append(args)
            return completion()

        generator = self.generator(transport=transport)
        result = generator.generate_answer("question", self.passages, question_id="q1", retrieval_method="bm25")
        repeated = generator.generate_answer("question", self.passages, question_id="q1", retrieval_method="bm25", retrieval_latency_ms=17)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["input_tokens"], 120)
        self.assertEqual(result["output_tokens"], 3)
        self.assertEqual(result["usage"]["prompt_tokens_details"]["cached_tokens"], 20)
        self.assertTrue(repeated["cache_hit"])
        self.assertEqual(repeated["current_retrieval_latency_ms"], 17)
        self.assertEqual(result["generation_latency_ms"], repeated["generation_latency_ms"])
        self.assertNotIn(self.config.api_key, json.dumps(result))
        self.assertNotIn(self.config.api_key, repr(self.config))
        # Source ordering and effective model settings are part of cache identity.
        generator.generate_answer("question", list(reversed(self.passages)), question_id="q1", retrieval_method="bm25")
        changed = AnswerGenerator(replace(self.config, max_output_tokens=65), self.root / "cache", transport=transport)
        changed.generate_answer("question", self.passages, question_id="q1", retrieval_method="bm25")
        self.assertEqual(len(calls), 3)

    def test_temporary_errors_retry_with_bounded_backoff(self):
        attempts, sleeps = [], []

        def transport(*args):
            attempts.append(args)
            if len(attempts) < 3:
                headers = Message()
                headers["Retry-After"] = "2"
                raise HTTPError("https://example.test", 429, "contains secret", headers, None)
            return completion()

        result = self.generator(transport=transport, sleep=sleeps.append).generate_answer("question")
        self.assertEqual(sleeps, [2.0, 2.0])
        self.assertEqual(result["request_attempts"], 3)
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["api_error"])
        self.assertEqual(len(result["attempt_errors"]), 2)
        self.assertNotIn("contains secret", json.dumps(result))

    def test_long_retry_after_is_deferred_not_shortened(self):
        headers = Message()
        headers["Retry-After"] = "999"

        def transport(*args):
            raise HTTPError("https://example.test", 429, "private text", headers, None)

        result = self.generator(transport=transport, sleep=lambda _: self.fail("must defer")).generate_answer("question")
        self.assertEqual(result["request_attempts"], 1)
        self.assertTrue(result["api_error"]["retry_deferred"])
        self.assertEqual(result["api_error"]["retry_after_seconds"], 999)

    def test_quota_errors_are_terminal_and_messages_are_not_logged(self):
        def transport(*args):
            body = io.BytesIO(json.dumps({"error": {"code": "insufficient_quota", "type": "insufficient_quota",
                                                   "message": "secret-key-never-log"}}).encode())
            raise HTTPError("https://example.test", 429, "private text", Message(), body)

        result = self.generator(transport=transport, sleep=lambda _: self.fail("must not retry quota")).generate_answer("question")
        self.assertEqual(result["request_attempts"], 1)
        self.assertFalse(result["api_error"]["retryable"])
        self.assertEqual(result["api_error"]["provider_code"], "insufficient_quota")
        self.assertEqual(result["api_error"]["action"], "check_billing_or_usage_limits")
        self.assertNotIn("secret-key-never-log", json.dumps(result))

    def test_pacing_applies_to_calls_and_retries_but_not_cache_hits(self):
        now, starts, sleeps = [0.0], [], []

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        def transport(*args):
            starts.append(now[0])
            if len(starts) == 2:
                raise HTTPError("https://example.test", 429, "rate limited", Message(), None)
            return completion()

        generator = AnswerGenerator(replace(self.config, requests_per_minute=8), self.root / "cache",
                                    transport=transport, sleep=sleep, clock=lambda: now[0])
        generator.generate_answer("first")
        generator.generate_answer("second")
        self.assertEqual(starts, [0.0, 7.5, 15.0])
        previous_sleeps = list(sleeps)
        cached = generator.generate_answer("first")
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(sleeps, previous_sleeps)
        self.assertEqual(starts, [0.0, 7.5, 15.0])
        slower = AnswerGenerator(replace(self.config, requests_per_minute=2), self.root / "cache",
                                 transport=lambda *args: self.fail("pacing change must preserve cache"))
        self.assertTrue(slower.generate_answer("first")["cache_hit"])

    def test_terminal_errors_not_cached_and_not_retried(self):
        attempts = []

        def transport(*args):
            attempts.append(args)
            raise HTTPError("https://example.test", 401, "secret-key-never-log", Message(), None)

        generator = self.generator(transport=transport, sleep=lambda _: self.fail("must not sleep"))
        result = generator.generate_answer("question")
        self.assertEqual(len(attempts), 1)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["api_error"]["http_status"], 401)
        self.assertFalse((self.root / "cache").exists())
        self.assertNotIn("secret-key-never-log", json.dumps(result))
        generator.generate_answer("question")
        self.assertEqual(len(attempts), 2)

    def test_network_failures_stop_after_max_attempts(self):
        sleeps = []

        def fail(*args):
            raise URLError("private error body")

        result = self.generator(transport=fail, sleep=sleeps.append).generate_answer("question")
        self.assertEqual(result["request_attempts"], 3)
        self.assertEqual(sleeps, [1.0, 2.0])
        self.assertEqual(result["status"], "error")
        self.assertNotIn("private error body", json.dumps(result))

    def test_invalid_response_and_unknown_usage_not_fabricated(self):
        for malformed in ({"choices": []}, ["not an object"], {"usage": [1]}, {"choices": [{"message": "invalid"}]}):
            bad = self.generator(transport=lambda *args: (malformed, {})).generate_answer("question")
            self.assertEqual(bad["api_error"]["type"], "invalid_response")
        response, headers = completion()
        response.pop("usage")
        result = self.generator(transport=lambda *args: (response, headers)).generate_answer("another question")
        self.assertIsNone(result["input_tokens"])
        self.assertIsNone(result["estimated_cost_usd"])

    def test_env_loader_precedence_and_no_code_execution(self):
        env = self.root / ".env"
        env.write_text("# comment\nLLM_MODEL='file-model'\nLLM_API_KEY=abc123 # comment\n")
        with patch.dict(os.environ, {"LLM_MODEL": "existing-model"}, clear=True):
            load_env(env)
            self.assertEqual(os.environ["LLM_MODEL"], "existing-model")
            self.assertEqual(os.environ["LLM_API_KEY"], "abc123")
        env.write_text("LLM_MODEL=$(do-not-run)\n")
        with patch.dict(os.environ, {}, clear=True):
            load_env(env)
            self.assertEqual(os.environ["LLM_MODEL"], "$(do-not-run)")
        env.write_text("invalid line\n")
        with self.assertRaises(ValueError):
            load_env(env)

    def test_missing_credentials_and_invalid_config_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "No API key"):
            replace(self.config, api_key="").require_key()
        with self.assertRaises(ValueError):
            replace(self.config, base_url="https://key:secret@example.test/v1")
        with self.assertRaises(ValueError):
            replace(self.config, max_output_tokens=0)
        with self.assertRaises(ValueError):
            replace(self.config, max_attempts=10)

    def test_five_validation_questions_twenty_conditions_and_resume(self):
        val = self.root / "val.jsonl"
        val.write_text("".join(json.dumps({"question_id": f"q{i}", "question": f"question {i}",
                                         "answers": {"text": ["DO_NOT_SEND_REFERENCE"]},
                                         "relevant_passage_id": "DO_NOT_SEND_LABEL"}) + "\n" for i in range(5)))
        corpus, model = self.root / "corpus.jsonl", self.root / "model.joblib"
        corpus.write_text("fixed corpus identity")
        model.write_bytes(b"fixed saved model identity")
        calls, retrieved = [], []
        passages = self.passages

        class Retrieval:
            def retrieve(self, question, top_k):
                retrieved.append((question, top_k))
                return {condition: {"passages": None if condition == "llm_only" else passages,
                                    "retrieval_latency_ms": 0.0 if condition == "llm_only" else 2.0}
                        for condition in CONDITIONS}

        def transport(url, headers, payload, timeout):
            calls.append(payload)
            self.assertNotIn("DO_NOT_SEND", json.dumps(payload))
            return completion()

        generator = self.generator(transport=transport)
        output = self.root / "output"
        with patch("experiments.generate_answers.VAL_PATH", val), \
             patch("experiments.generate_answers.CORPUS_PATH", corpus), \
             patch("experiments.generate_answers.MODEL_PATH", model), \
             redirect_stdout(io.StringIO()):
            summary, records = run_generation(config=self.config, generator=generator, retrieval=Retrieval(), output_dir=output)
            repeated, _ = run_generation(config=self.config, generator=generator, retrieval=Retrieval(), output_dir=output)
        self.assertEqual(summary["completed"], 20)
        self.assertEqual(len(calls), 20)
        self.assertEqual(repeated["cache_hits"], 20)
        self.assertEqual(repeated["new_input_tokens"], 0)
        for question_id in {row["question_id"] for row in records}:
            subset = [row for row in records if row["question_id"] == question_id]
            self.assertEqual({row["retrieval_method"] for row in subset}, set(CONDITIONS))
            self.assertEqual(len({json.dumps(row["model_settings"], sort_keys=True) for row in subset}), 1)
        self.assertEqual(len(list(output.glob("*.jsonl"))), 1)
        self.assertEqual(len(next(output.glob("*.jsonl")).read_text().splitlines()), 20)


if __name__ == "__main__":
    unittest.main()
