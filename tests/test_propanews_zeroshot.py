"""Protocol, label safety, crash recovery and metric regression tests (no GPU)."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import random
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from propanews_common import PromptArticle
from propanews_zeroshot_backend import LocalBackend, MockBackend
from propanews_zeroshot_checkpoint import CheckpointError, Ledger
from propanews_zeroshot_checkpoint import ModelCallBudgetExhausted
from propanews_zeroshot_data import (
    DEFAULT_DATA_PATH, EXPECTED_COUNTS, OfficialTestError,
    load_official_test, validate_test_bytes,
)
from propanews_zeroshot_metrics import classification_metrics, summarize_trials
from propanews_zeroshot_early_stop import summarize_early_stop_trials
from propanews_zeroshot_prompts import (
    ResponseParseError, classification_prompt, parse_classification,
    parse_reflection, reasoning_only, reflection_prompt,
)
from propanews_zeroshot_three_calls import integrated_reflexion_prompt, parse_integrated_reflexion
from run_propanews_zeroshot import (
    article_state, build_provenance, export_state, main, run_inference,
)


VALID = "Rationale: The dates and named sources are consistent with the reported event.\nFinish[real]"


class CountingBackend:
    last_call_info = {"test": True}

    def __init__(self, failure=None):
        self.prompts = []
        self.failure = failure

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.failure:
            raise self.failure
        return VALID


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = DEFAULT_DATA_PATH.read_bytes()

    def test_exact_split_and_mapping_without_cleaning(self):
        dataset = validate_test_bytes(self.payload)
        rows = [json.loads(line) for line in self.payload.decode().splitlines()]
        self.assertEqual(dataset.validation["class_counts"], EXPECTED_COUNTS)
        self.assertEqual(len(dataset.articles), 500)
        for index, (article, row) in enumerate(zip(dataset.articles, rows)):
            self.assertEqual(article, PromptArticle(index, row["txt"]))
            self.assertEqual(dataset.gold[index], "real" if row["label"] else "fake")
            self.assertEqual(set(article.__dataclass_fields__), {"index", "article"})

    def test_truncation_corruption_reordering_duplicates_rejected(self):
        lines = self.payload.splitlines(keepends=True)
        for payload in (self.payload[:-1], self.payload.replace(b"George", b"Jorge", 1),
                        b"".join(lines[::-1]), b"".join(lines[:-1] + lines[:1])):
            with self.subTest(size=len(payload)), self.assertRaises(OfficialTestError):
                validate_test_bytes(payload)

    def test_missing_file_has_no_random_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(OfficialTestError, "No random or substitute"):
                load_official_test(Path(tmp) / "absent.jsonl")

    def test_failed_download_never_publishes_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "test.jsonl"
            with patch("urllib.request.urlopen", side_effect=OSError("offline")):
                with self.assertRaisesRegex(OfficialTestError, "cannot download"):
                    load_official_test(target, download=True)
            self.assertFalse(target.exists())

    def test_incorrect_existing_data_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "test.jsonl"
            target.write_bytes(b"bad")
            with self.assertRaises(OfficialTestError):
                load_official_test(target, download=True)
            self.assertEqual(target.read_bytes(), b"bad")


class PromptTests(unittest.TestCase):
    def test_integrated_review_and_prediction_parse_from_one_response(self):
        raw = "Reflection: Re-check the date and attribution.\n" + VALID
        result = parse_integrated_reflexion(raw)
        self.assertEqual(result["prediction"], "real")
        self.assertEqual(result["feedback"], "Re-check the date and attribution.")

    def test_integrated_prompt_removes_previous_classification(self):
        prompt = integrated_reflexion_prompt("Article sentinel", "The previous prediction was fake. Re-check the attribution.")
        previous = prompt.split("PREVIOUS BRIEF RATIONALE:\n")[1].split("\nRESPONSE:")[0]
        self.assertNotIn("fake", previous.lower())
        self.assertNotIn("Finish[", previous)
        self.assertIn("attribution", previous)

    def test_integrated_response_requires_both_review_and_classification(self):
        for response in (VALID, "Reflection: Re-check the date.\nFinish[real]", "Reflection:\n" + VALID):
            with self.subTest(response=response), self.assertRaises(ResponseParseError):
                parse_integrated_reflexion(response)

    def test_direct_actor_no_gt_epm_or_fewshot(self):
        article = "Article sentinel 987654321 with its full ending.\n  "
        prompt = classification_prompt(article)
        self.assertIn(article, prompt)
        for forbidden in ("STEP 1", "STEP 5", "AI-generated fake news can be created",
                          "EPISODIC MEMORY", "EXAMPLE", "ground truth", "chain-of-thought"):
            self.assertNotIn(forbidden, prompt)
        self.assertEqual(prompt.count("NEWS ARTICLE:"), 1)

    def test_reflector_gets_rationale_without_prior_label(self):
        response = "The previous prediction was fake. The date needs scrutiny. Finish[fake]"
        prompt = reflection_prompt("Article sentinel", response)
        previous = prompt.split("PREVIOUS BRIEF RATIONALE:\n")[1].split("\nSELF-REFLECTION:")[0]
        self.assertIn("date needs scrutiny", previous)
        self.assertNotIn("fake", previous.lower())
        self.assertNotIn("Finish", previous)
        self.assertNotIn("previous prediction", previous)

    def test_feedback_sanitization_is_idempotent(self):
        samples = [
            "The last prediction was fake. Check the date and named source.",
            "The prior answer was wrong. Reverse the previous classification. Re-check the quoted number.",
            "Finish[real]\nThe article is real because the timeline is consistent.",
            "The evidence suggests fake reporting. The actor's role was overlooked.",
        ]
        for sample in samples:
            result = reasoning_only(sample)
            self.assertEqual(result, reasoning_only(result))
            self.assertNotRegex(result.lower(), r"\b(real|fake|wrong|correct|incorrect)\b|finish\[")

    def test_new_actor_sees_only_latest_feedback(self):
        prompt = classification_prompt("Article sentinel", "The prediction was fake. Re-check the attribution.")
        self.assertIn("Re-check the attribution", prompt)
        self.assertNotIn("prediction was fake", prompt)
        self.assertNotIn("PREVIOUS BRIEF RATIONALE", prompt)

    def test_parse_compatible_finish_case(self):
        from evaluator_propanews import extract_prediction
        for label in ("real", "fake", "REAL", "FAKE"):
            response = VALID.replace("real]", label + "]")
            self.assertEqual(parse_classification(response)["prediction"], label.lower())
            self.assertEqual(extract_prediction(response), label.lower())

    def test_malformed_outputs_are_errors_not_default_classes(self):
        for response in ("", "real", "Finish[fake]", VALID + "\ntrailing text",
                         VALID + "\nFinish[fake]", VALID.replace("real]", "unknown]"),
                         "Rationale: <think>hidden</think>\nFinish[real]",
                         "Rationale: STEP 1 analyze.\nFinish[real]"):
            with self.subTest(response=response), self.assertRaises(ResponseParseError):
                parse_classification(response)

    def test_reflection_with_only_classification_rejected(self):
        with self.assertRaises(ResponseParseError):
            parse_reflection("Finish[fake]")
        with self.assertRaises(ResponseParseError):
            parse_classification("Rationale: The article is fake.\nFinish[fake]")


class MetricTests(unittest.TestCase):
    def test_known_confusion_matrix_fake_positive(self):
        value = classification_metrics(["real", "real", "fake", "fake", "fake"],
                                       ["real", "fake", "fake", "real", "fake"])
        self.assertEqual(value["confusion_matrix"], [[1, 1], [1, 2]])
        self.assertEqual(value["accuracy"], 3 / 5)
        self.assertEqual(value["f1_fake"], 2 / 3)
        self.assertEqual(value["precision_fake"], 2 / 3)
        self.assertEqual(value["recall_fake"], 2 / 3)
        self.assertAlmostEqual(value["macro_f1"], (1 / 2 + 2 / 3) / 2)

    def test_zero_division_and_invalid_prediction(self):
        value = classification_metrics(["real", "fake"], ["real", "real"])
        self.assertEqual(value["precision_fake"], 0)
        self.assertEqual(value["f1_fake"], 0)
        with self.assertRaises(ValueError):
            classification_metrics(["fake"], [None])

    def test_metrics_against_sklearn(self):
        try:
            from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
        except ImportError:
            self.skipTest("scikit-learn optional numerical cross-check unavailable")
        rng = random.Random(42)
        gold = ["real"] * 243 + ["fake"] * 257
        for _ in range(4):
            predicted = [rng.choice(["real", "fake"]) for _ in gold]
            metric = classification_metrics(gold, predicted)
            self.assertAlmostEqual(metric["accuracy"], accuracy_score(gold, predicted))
            self.assertAlmostEqual(metric["f1_fake"], f1_score(gold, predicted, pos_label="fake"))
            self.assertAlmostEqual(metric["macro_f1"], f1_score(gold, predicted, average="macro"))
            self.assertAlmostEqual(metric["precision_fake"], precision_score(gold, predicted, pos_label="fake"))
            self.assertAlmostEqual(metric["recall_fake"], recall_score(gold, predicted, pos_label="fake"))
            self.assertEqual(metric["confusion_matrix"], confusion_matrix(gold, predicted, labels=["real", "fake"]).tolist())

    def test_trial_3_actual_output_replaces_oracle_success(self):
        records = []
        for index in range(500):
            gold = "real" if index < 243 else "fake"
            opposite = "fake" if gold == "real" else "real"
            records.append({"index": index, "gold_label": gold, "trials": {
                "1": {"prediction": gold}, "2": {"prediction": gold}, "3": {"prediction": opposite},
            }})
        value = summarize_trials(records)
        self.assertEqual([t["accuracy"] for t in value["trials"]], [1, 1, 0])
        self.assertEqual(value["trials"][2]["f1_fake"], 0)
        self.assertEqual(value["final_regressions"], 500)
        self.assertEqual(value["final_corrected_errors"], 0)
        self.assertEqual(value["transitions"][-1]["accuracy_delta"], -1)
        with self.assertRaises(ValueError):
            summarize_trials(records[:2])


class BackendTests(unittest.TestCase):
    def test_context_overflow_fails_without_truncation_or_generation(self):
        from types import SimpleNamespace
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.options = kwargs
                return messages[0]["content"]

            def __call__(self, text, **kwargs):
                self.tokenize_options = kwargs
                return {"input_ids": list(range(2000))}
        tokenizer = Tokenizer()
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2500))
        backend = LocalBackend.__new__(LocalBackend)
        with patch("llm._load_backend", return_value=(None, tokenizer, model)), patch("llm.call_llm") as generate:
            with self.assertRaisesRegex(RuntimeError, "NOT truncated"):
                backend("full article")
            generate.assert_not_called()
        self.assertFalse(tokenizer.tokenize_options["truncation"])
        self.assertFalse(tokenizer.options["enable_thinking"])

    def test_local_backend_delegates_unchanged_generation(self):
        from types import SimpleNamespace
        tokenizer = unittest.mock.Mock()
        tokenizer.apply_chat_template.return_value = "rendered"
        tokenizer.return_value = {"input_ids": [1, 2, 3]}
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960), hf_device_map={"": 0})
        backend = LocalBackend.__new__(LocalBackend)
        with patch("llm._load_backend", return_value=(None, tokenizer, model)), patch("llm.call_llm", return_value=VALID) as generate:
            self.assertEqual(backend("complete prompt"), VALID)
            generate.assert_called_once_with("complete prompt")
        self.assertEqual(backend.last_call_info["input_tokens"], 3)
        self.assertFalse(backend.last_call_info["input_truncated"])


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.provenance = {"dataset": "pinned", "model": "test"}
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)
        self.temp.cleanup()

    def test_successful_calls_never_duplicated_on_resume(self):
        backend = CountingBackend()
        with Ledger(self.directory, self.provenance) as ledger:
            ledger.call(0, 1, "classify", "prompt", parse_classification, backend)
        with Ledger(self.directory, self.provenance) as ledger:
            ledger.call(0, 1, "classify", "prompt", parse_classification, backend)
            self.assertEqual(len(ledger.all_calls()), 1)
        self.assertEqual(len(backend.prompts), 1)

    def test_raw_response_survives_crash_before_parse(self):
        backend = CountingBackend()
        with Ledger(self.directory, self.provenance) as ledger:
            with patch.object(ledger, "_parse", side_effect=RuntimeError("crash after output commit")):
                with self.assertRaises(RuntimeError):
                    ledger.call(0, 1, "classify", "prompt", parse_classification, backend)
        with Ledger(self.directory, self.provenance) as ledger:
            result = ledger.call(0, 1, "classify", "prompt", parse_classification, backend)
            self.assertEqual(result["prediction"], "real")
        self.assertEqual(len(backend.prompts), 1)

    def test_parse_failure_preserved_without_regeneration_or_label_fallback(self):
        count = []
        def invalid(prompt):
            count.append(prompt)
            return "unparseable response"
        with Ledger(self.directory, self.provenance) as ledger:
            self.assertIsNone(ledger.call(0, 1, "classify", "prompt", parse_classification, invalid))
            self.assertIsNone(ledger.call(0, 1, "classify", "prompt", parse_classification, invalid, retry_errors=True))
            row = ledger.latest(0, 1, "classify")
            self.assertEqual(row["status"], "parse_error")
            self.assertEqual(row["response"], "unparseable response")
            self.assertIsNone(row["parsed"])
        self.assertEqual(len(count), 1)

    def test_execution_error_retry_is_explicit_and_history_retained(self):
        backend = CountingBackend(RuntimeError("CUDA out of memory"))
        with Ledger(self.directory, self.provenance) as ledger:
            with patch("propanews_zeroshot_checkpoint.recover_from_oom") as recover:
                self.assertIsNone(ledger.call(0, 1, "classify", "prompt", parse_classification, backend))
                recover.assert_called_once()
            backend.failure = None
            self.assertIsNone(ledger.call(0, 1, "classify", "prompt", parse_classification, backend))
            self.assertEqual(len(backend.prompts), 1)
            result = ledger.call(0, 1, "classify", "prompt", parse_classification, backend, retry_errors=True)
            self.assertEqual(result["prediction"], "real")
            self.assertEqual(ledger.all_calls()[0]["error_type"], "CUDA_OOM")
            self.assertEqual(len(ledger.all_calls()), 2)

    def test_interrupted_calls_require_explicit_retry(self):
        backend = CountingBackend(KeyboardInterrupt())
        with Ledger(self.directory, self.provenance) as ledger:
            with self.assertRaises(KeyboardInterrupt):
                ledger.call(0, 1, "classify", "prompt", parse_classification, backend)
        backend.failure = None
        with Ledger(self.directory, self.provenance) as ledger:
            self.assertIsNone(ledger.call(0, 1, "classify", "prompt", parse_classification, backend))
            result = ledger.call(0, 1, "classify", "prompt", parse_classification, backend, retry_uncertain=True)
            self.assertEqual(result["prediction"], "real")
        self.assertEqual(len(backend.prompts), 2)

    def test_running_status_recovered_as_uncertain_after_hard_exit(self):
        with Ledger(self.directory, self.provenance) as ledger:
            with ledger.db:
                ledger.db.execute("INSERT INTO calls (article_index,trial,kind,attempt,prompt,prompt_sha256,status,started_at) VALUES (0,1,'classify',1,'prompt',?,'running','test')", (hashlib.sha256(b"prompt").hexdigest(),))
        backend = CountingBackend()
        with Ledger(self.directory, self.provenance) as ledger:
            self.assertIsNone(ledger.call(0, 1, "classify", "prompt", parse_classification, backend))
            self.assertEqual(ledger.latest(0, 1, "classify")["status"], "uncertain")
        self.assertFalse(backend.prompts)

    def test_provenance_prompt_and_concurrent_writer_mismatch(self):
        backend = CountingBackend()
        with Ledger(self.directory, self.provenance) as ledger:
            with self.assertRaises(BlockingIOError):
                Ledger(self.directory, self.provenance)
            ledger.call(0, 1, "classify", "prompt", parse_classification, backend)
            with self.assertRaises(CheckpointError):
                ledger.call(0, 1, "classify", "different prompt", parse_classification, backend)
        with self.assertRaises(CheckpointError):
            Ledger(self.directory, {**self.provenance, "model": "changed"})

    def test_baseline_shared_and_wrong_answers_retry_up_to_3_trials(self):
        articles = (PromptArticle(0, "Article sentinel"),)
        backend = MockBackend()
        calls = []
        def counting(prompt):
            calls.append(prompt)
            return backend(prompt)
        with Ledger(self.directory, self.provenance) as ledger:
            run_inference(articles, ledger, counting, phase="baseline")
            first = ledger.latest(0, 1, "classify")
        with Ledger(self.directory, self.provenance) as ledger:
            run_inference(articles, ledger, counting, is_correct=lambda i, p: False)
            self.assertEqual(first, ledger.latest(0, 1, "classify"))
            self.assertEqual(len(ledger.all_calls()), 3)
            state = article_state(ledger, articles[0])
            self.assertTrue(all(t["status"] == "ok" for t in state["trials"].values()))
            run_inference(articles, ledger, counting, is_correct=lambda i, p: False)
            self.assertEqual(len(ledger.all_calls()), 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sum(p.startswith("Review the previous brief rationale") for p in calls), 2)

    def test_baseline_only_is_once_per_article_including_resume(self):
        articles = tuple(PromptArticle(i, f"Article sentinel {i}") for i in range(3))
        backend = CountingBackend()
        with Ledger(self.directory, self.provenance) as ledger:
            run_inference(articles, ledger, backend, phase="baseline")
            run_inference(articles, ledger, backend, phase="baseline")
            calls = ledger.all_calls()
            self.assertEqual(len(calls), 3)
            self.assertEqual({(c["trial"], c["kind"]) for c in calls}, {(1, "classify")})
        self.assertEqual(backend.prompts, [classification_prompt(a.article) for a in articles])

    def test_only_reflexion_assisted_classification_is_repeated(self):
        articles = tuple(PromptArticle(i, f"Article sentinel {i}") for i in range(2))
        prompts = []
        mock = MockBackend()
        def backend(prompt):
            prompts.append(prompt)
            return mock(prompt)
        with Ledger(self.directory, self.provenance) as ledger:
            run_inference(articles, ledger, backend, is_correct=lambda i, p: False)
            run_inference(articles, ledger, backend, is_correct=lambda i, p: False)
            for article in articles:
                baseline = classification_prompt(article.article)
                self.assertEqual(prompts.count(baseline), 1)
                self.assertNotIn("SELF-REFLECTION FEEDBACK:", ledger.latest(article.index, 1, "classify")["prompt"])
                for trial in (2, 3):
                    self.assertIn("PREVIOUS BRIEF RATIONALE:", ledger.latest(article.index, trial, "classify")["prompt"])
            self.assertEqual({c["kind"] for c in ledger.all_calls()}, {"classify"})
        self.assertEqual(len(prompts), 6)

    def test_failed_initial_classification_does_not_start_reflexion(self):
        prompts = []
        def backend(prompt):
            prompts.append(prompt)
            return "malformed baseline"
        with Ledger(self.directory, self.provenance) as ledger:
            run_inference((PromptArticle(0, "Article sentinel"),), ledger, backend,
                          is_correct=lambda i, p: False)
            self.assertEqual(len(ledger.all_calls()), 1)
            self.assertEqual(ledger.latest(0, 1, "classify")["status"], "parse_error")
            self.assertIsNone(ledger.latest(0, 1, "reflect"))
        self.assertEqual(len(prompts), 1)

    def test_fixed_trial_cache_is_incompatible_with_early_stopping(self):
        dataset = load_official_test()
        provenance = build_provenance(dataset.validation, MockBackend.metadata)
        previous = json.loads(json.dumps(provenance))
        previous["protocol"]["early_stop_on_correct"] = False
        previous["protocol"]["oracle_scoring"] = False
        with Ledger(self.directory, previous) as ledger:
            self.assertFalse(ledger.all_calls())
        with self.assertRaises(CheckpointError):
            Ledger(self.directory, provenance)

    def test_early_stop_cache_rejects_experiment_changes(self):
        dataset = load_official_test()
        provenance = build_provenance(dataset.validation, MockBackend.metadata)
        with Ledger(self.directory, provenance):
            pass
        changes = [
            ("dataset", "sha256"), ("backend", "version"),
            ("source_sha256", "propanews_zeroshot_prompts.py"),
            ("source_sha256", "propanews_zeroshot_metrics.py"),
            ("protocol", "trials"),
        ]
        for field, key in changes:
            changed = json.loads(json.dumps(provenance))
            changed[field][key] = "changed"
            with self.subTest(field=field, key=key), self.assertRaises(CheckpointError):
                Ledger(self.directory, changed)
        with Ledger(self.directory, provenance) as ledger:
            self.assertFalse(ledger.all_calls())


class EarlyStoppingTests(unittest.TestCase):
    def _run_predictions(self, labels):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.prompts = []
        predictions = iter(labels)
        def backend(prompt):
            self.prompts.append(prompt)
            response = VALID.replace("Finish[real]", f"Finish[{next(predictions)}]")
            if prompt.startswith("Review the previous brief rationale"):
                response = "Reflection: Re-check the date and attribution against the reported event.\n" + response
            return response
        self.check = lambda index, prediction: prediction == "real"
        self.articles = (PromptArticle(0, "Article sentinel"),)
        with contextlib.redirect_stdout(io.StringIO()), Ledger(self.directory, {"early_stop": True}) as ledger:
            run_inference(self.articles, ledger, backend, is_correct=self.check)
            return article_state(ledger, self.articles[0], is_correct=self.check)

    def test_correct_trial_1_stops_before_reflection(self):
        state = self._run_predictions(["real"])
        self.assertEqual(len(self.prompts), 1)
        self.assertEqual(state["success_trial"], 1)
        self.assertTrue(state["complete"])
        for trial in (2, 3):
            t = state["trials"][str(trial)]
            self.assertEqual(t["status"], "skipped_after_correct")
            self.assertIsNone(t["prediction"])
            self.assertIsNone(t["response"])
            self.assertIsNone(t["reflexion_feedback"])
            self.assertEqual(t["effective_prediction"], "real")
            self.assertEqual(t["carried_from_trial"], 1)

    def test_correct_trial_2_stops_at_two_total_model_calls(self):
        state = self._run_predictions(["fake", "real"])
        self.assertEqual(len(self.prompts), 2)
        self.assertEqual(state["success_trial"], 2)
        self.assertEqual(state["trials"]["3"]["status"], "skipped_after_correct")
        self.assertEqual(state["trials"]["3"]["carried_from_trial"], 2)
        previous = self.prompts[1].split("PREVIOUS BRIEF RATIONALE:\n")[1]
        self.assertNotIn("Finish[", previous)
        self.assertNotIn("previous answer was wrong", self.prompts[1])

    def test_correct_trial_3_finishes_with_actual_prediction(self):
        state = self._run_predictions(["fake", "fake", "real"])
        self.assertEqual(len(self.prompts), 3)
        self.assertEqual(state["success_trial"], 3)
        self.assertEqual(state["final_prediction_source_trial"], 3)
        self.assertFalse(state["final_prediction_was_carried"])

    def test_all_wrong_runs_maximum_3_total_model_calls_without_separate_reflection(self):
        state = self._run_predictions(["fake", "fake", "fake"])
        self.assertEqual(len(self.prompts), 3)
        self.assertIsNone(state["success_trial"])
        self.assertTrue(state["complete"])
        self.assertEqual(state["stop_reason"], "max_trials")
        self.assertEqual(state["final_prediction"], "fake")
        self.assertFalse(state["final_prediction_was_carried"])
        self.assertEqual(state["total_model_calls"], 3)
        with Ledger(self.directory, {"early_stop": True}) as ledger:
            self.assertEqual({c["kind"] for c in ledger.all_calls()}, {"classify"})
            self.assertEqual([c["trial"] for c in ledger.all_calls()], [1, 2, 3])
            ledger.limit_calls_per_article(3)
            with self.assertRaises(ModelCallBudgetExhausted):
                ledger.call(0, 4, "classify", "forbidden", parse_classification,
                            lambda p: self.fail("a fourth call is forbidden"))

    def test_resume_after_correct_answer_makes_no_calls(self):
        state = self._run_predictions(["fake", "real"])
        with contextlib.redirect_stdout(io.StringIO()), Ledger(self.directory, {"early_stop": True}) as ledger:
            run_inference(self.articles, ledger, lambda p: self.fail("duplicate call"), is_correct=self.check)
            self.assertEqual(len(ledger.all_calls()), 2)
            self.assertEqual(article_state(ledger, self.articles[0], is_correct=self.check), state)

    def test_resume_after_wrong_answer_needs_one_combined_call(self):
        self._run_predictions(["real"])
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp)
            with Ledger(path, {"early_stop": True}) as ledger:
                ledger.call(0, 1, "classify", classification_prompt("Article sentinel"),
                            parse_classification, lambda p: VALID.replace("Finish[real]", "Finish[fake]"))
            prompts = []
            def backend(prompt):
                prompts.append(prompt)
                return "Reflection: Re-check the attribution.\n" + VALID
            with Ledger(path, {"early_stop": True}) as ledger:
                run_inference(self.articles, ledger, backend, is_correct=self.check)
                self.assertEqual(len(ledger.all_calls()), 2)
            self.assertEqual(len(prompts), 1)

    def test_error_retries_cannot_exceed_lifetime_budget_of_three(self):
        prompts = []
        def failing(prompt):
            prompts.append(prompt)
            raise RuntimeError("simulated execution failure")
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for _ in range(5):
                with Ledger(Path(tmp), {"three_calls": True}) as ledger:
                    run_inference((PromptArticle(0, "Article"),), ledger, failing,
                                  retry_errors=True, is_correct=lambda i, p: False)
                    self.assertLessEqual(ledger.call_count(0), 3)
            with Ledger(Path(tmp), {"three_calls": True}) as ledger:
                state = article_state(ledger, PromptArticle(0, "Article"), is_correct=lambda i, p: False)
                self.assertEqual(state["stop_reason"], "call_budget_exhausted")
                self.assertFalse(state["complete"])
        self.assertEqual(len(prompts), 3)

    def test_uncertain_retries_cannot_create_a_fourth_call(self):
        prompts = []
        def interrupted(prompt):
            prompts.append(prompt)
            raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for _ in range(3):
                with Ledger(Path(tmp), {"three_calls": True}) as ledger:
                    with self.assertRaises(KeyboardInterrupt):
                        run_inference((PromptArticle(0, "Article"),), ledger, interrupted,
                                      retry_uncertain=True, is_correct=lambda i, p: False)
            with Ledger(Path(tmp), {"three_calls": True}) as ledger:
                run_inference((PromptArticle(0, "Article"),), ledger, interrupted,
                              retry_uncertain=True, is_correct=lambda i, p: False)
                self.assertEqual(ledger.call_count(0), 3)
        self.assertEqual(len(prompts), 3)

    def test_missing_evaluator_cannot_silently_run_fixed_trials(self):
        with tempfile.TemporaryDirectory() as tmp, Ledger(Path(tmp), {}) as ledger:
            with self.assertRaisesRegex(ValueError, "requires.*is_correct"):
                run_inference((PromptArticle(0, "Article"),), ledger, MockBackend())
            self.assertFalse(ledger.all_calls())


class IntegrationTests(unittest.TestCase):
    def test_full_500_mock_export_resume_and_incomplete_guard(self):
        dataset = load_official_test()
        backend = MockBackend()
        provenance = build_provenance(dataset.validation, backend.metadata)
        check = lambda index, prediction: prediction == dataset.gold[index]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp)
            with Ledger(path, provenance) as ledger:
                run_inference(dataset.articles[:2], ledger, backend, is_correct=check)
                self.assertFalse(export_state(ledger, dataset))
                self.assertFalse((path / "trial_summary.json").exists())
                self.assertFalse((path / "results.json").exists())
            with Ledger(path, provenance) as ledger:
                run_inference(dataset.articles, ledger, backend, is_correct=check)
                self.assertTrue(export_state(ledger, dataset))
                calls = len(ledger.all_calls())
                self.assertLessEqual(calls, 1500)
                self.assertGreaterEqual(calls, 500)
                self.assertEqual({c["status"] for c in ledger.all_calls()}, {"ok"})
                call_counts = Counter(c["article_index"] for c in ledger.all_calls())
                for article in dataset.articles:
                    state = article_state(ledger, article, is_correct=check)
                    self.assertEqual(call_counts[article.index], state["success_trial"] or 3)
                    self.assertLessEqual(call_counts[article.index], 3)
                run_inference(dataset.articles[:2], ledger, backend, is_correct=check)
                self.assertEqual(len(ledger.all_calls()), calls)
                self.assertTrue(export_state(ledger, dataset))
            summary = json.loads((path / "trial_summary.json").read_text())
            self.assertFalse(summary["research_result"])
            self.assertTrue(summary["oracle_scoring"])
            self.assertEqual(summary["evaluation_mode"], "gold_gated_early_stop_cumulative")
            self.assertEqual([r["n"] for r in summary["trials"]], [500, 500, 500])
            accuracies = summary["cumulative_success"]
            self.assertEqual(accuracies, sorted(accuracies))
            import csv
            with (path / "results.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 500)
            self.assertEqual(rows[0]["trial_1_reflexion_feedback"], "")
            self.assertTrue(all(row["oracle_scoring"] == "True" for row in rows))
            self.assertEqual(sum(json.loads((path / "articles" / f"{i:03d}.json").read_text())["gold_label"] == "real" for i in range(500)), 243)
            records = json.loads((path / "results.json").read_text())["records"]
            invalid = json.loads(json.dumps(records))
            skipped = next(t for r in invalid for t in r["trials"].values() if t["status"] == "skipped_after_correct")
            skipped["prediction"] = skipped["effective_prediction"]
            with self.assertRaisesRegex(ValueError, "null actual output"):
                summarize_early_stop_trials(invalid)

    def test_prompt_dataset_settings_changes_invalidate_manifest(self):
        dataset = load_official_test()
        base = build_provenance(dataset.validation, MockBackend.metadata)
        with tempfile.TemporaryDirectory() as tmp:
            with Ledger(Path(tmp), base):
                pass
            for field in ("prompt_version", "dataset", "backend", "source_sha256"):
                changed = {**base, field: "changed"}
                with self.subTest(field=field), self.assertRaises(CheckpointError):
                    Ledger(Path(tmp), changed)


if __name__ == "__main__":
    unittest.main()
