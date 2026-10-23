"""Training-free AdSent and gold-blind Reflexion pipelines for PropaNews.

Benchmark labels never enter an LLM prompt. The runner may provide an opaque boolean
predicate that gates per-example retries after comparing an Actor prediction with
private gold. That oracle-gated protocol is deliberately distinct from deployable,
gold-blind inference.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import wraps
from typing import Any

from adsent_propanews_prompts import (
    CANONICAL_LABELS,
    SENTIMENTS,
    build_consensus_prompt,
    build_detection_prompt,
    build_fact_preservation_prompt,
    build_invariance_feedback_prompt,
    build_plain_detection_prompt,
    build_reflection_prompt,
    build_single_detection_feedback_prompt,
    build_single_reflection_prompt,
    build_sentiment_rewrite_prompt,
)


LLMCallable = Callable[[str], str]
SuccessPredicate = Callable[[str | None], bool]
VIEW_NAMES = ("original", "positive", "negative", "neutral")
REFLEXION_FEEDBACK_MODE = "gold_blind_self_consistency_no_correctness_reward"
CONTROLLED_REFLEXION_FEEDBACK_MODE = (
    "gold_blind_single_candidate_no_correctness_reward"
)


def _oracle_retry_protocol(max_attempts: int) -> dict[str, Any]:
    return {
        "name": "per_example_oracle_early_stop",
        "max_attempts": max_attempts,
        "gold_used_for": "external_retry_gate_only",
        "oracle_label_in_prompt": False,
        "oracle_correctness_message_in_prompt": False,
    }


class PipelineError(RuntimeError):
    """Raised when an LLM stage cannot produce a usable response."""

    def __init__(
        self,
        message: str,
        call_records: Sequence["CallRecord"] = (),
    ) -> None:
        super().__init__(message)
        self.call_records = list(call_records)


@dataclass(frozen=True)
class PipelineConfig:
    """Inference-only configuration; no field controls model training."""

    max_rewrite_attempts: int = 2
    max_format_retries: int = 1
    max_reflexion_trials: int = 5
    reflection_memory_size: int = 3
    require_target_sentiment: bool = True

    def __post_init__(self) -> None:
        if self.max_rewrite_attempts < 1:
            raise ValueError("max_rewrite_attempts must be at least 1")
        if self.max_format_retries < 0:
            raise ValueError("max_format_retries must not be negative")
        if self.max_reflexion_trials < 1:
            raise ValueError("max_reflexion_trials must be at least 1")
        if self.reflection_memory_size < 1:
            raise ValueError("reflection_memory_size must be at least 1")


@dataclass
class CallRecord:
    stage: str
    prompt_sha256: str
    latency_ms: float
    raw_response: str | None
    parse_status: str
    parse_error: str | None = None
    retry: int = 0
    backend_metadata: dict[str, Any] | None = None


@dataclass
class RewriteAttempt:
    attempt: int
    rewritten_text: str | None
    quality_report: dict[str, Any] | None
    accepted: bool
    calls: list[CallRecord] = field(default_factory=list)


@dataclass
class ViewArtifact:
    name: str
    text: str | None
    valid: bool
    target_sentiment: str | None
    attempts: list[RewriteAttempt] = field(default_factory=list)
    failure_reason: str | None = None


def _capture_method_call_ledger(
    method: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Attach every completed LLM call when a multi-stage method fails late."""

    @wraps(method)
    def wrapped(self: "TrainingFreeAdSent", *args: Any, **kwargs: Any) -> dict[str, Any]:
        parent_ledger = self._active_call_ledger
        ledger: list[CallRecord] = []
        self._active_call_ledger = ledger
        try:
            return method(self, *args, **kwargs)
        except PipelineError as exc:
            raise PipelineError(str(exc), ledger or exc.call_records) from exc
        finally:
            self._active_call_ledger = parent_ledger

    return wrapped


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_single_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(
        r"```(?:json|text)?\s*\n?(.*?)\n?```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else stripped


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.strip(), object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("response must be one JSON object")
    return value


def _validate_string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return value


def _validate_preservation(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "facts_preserved",
        "target_sentiment_achieved",
        "changed_claims",
        "brief_reason",
    }
    if set(value) != expected:
        raise ValueError(f"preservation JSON keys must be exactly {sorted(expected)}")
    if type(value.get("facts_preserved")) is not bool:
        raise ValueError("facts_preserved must be a boolean")
    if type(value.get("target_sentiment_achieved")) is not bool:
        raise ValueError("target_sentiment_achieved must be a boolean")
    changed_claims = _validate_string_list(value.get("changed_claims"), "changed_claims")
    if not isinstance(value.get("brief_reason"), str) or not value["brief_reason"].strip():
        raise ValueError("brief_reason must be a non-empty string")
    if value["facts_preserved"] and changed_claims:
        raise ValueError("changed_claims must be empty when facts_preserved is true")
    if not value["facts_preserved"] and not changed_claims:
        raise ValueError("changed_claims must identify a change when facts_preserved is false")
    return value


def _validate_detection(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "label",
        "sentiment_used_as_evidence",
        "key_claims",
        "brief_reason",
    }
    if set(value) != expected:
        raise ValueError(f"detection JSON keys must be exactly {sorted(expected)}")
    if value.get("label") not in CANONICAL_LABELS:
        raise ValueError(f"label must be one of {CANONICAL_LABELS}")
    if type(value.get("sentiment_used_as_evidence")) is not bool:
        raise ValueError("sentiment_used_as_evidence must be a boolean")
    if value["sentiment_used_as_evidence"] is not False:
        raise ValueError("sentiment_used_as_evidence must be false")
    key_claims = _validate_string_list(value.get("key_claims"), "key_claims")
    if not 1 <= len(key_claims) <= 3 or any(not claim.strip() for claim in key_claims):
        raise ValueError("key_claims must contain one to three non-empty strings")
    if not isinstance(value.get("brief_reason"), str) or not value["brief_reason"].strip():
        raise ValueError("brief_reason must be a non-empty string")
    return value


def _validate_plain_detection(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != {"label"}:
        raise ValueError("plain detection JSON must contain exactly the label key")
    if value.get("label") not in CANONICAL_LABELS:
        raise ValueError(f"label must be one of {CANONICAL_LABELS}")
    return value


def _validate_feedback(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "prediction_consistent",
        "reasoning_pass",
        "tone_dependence_detected",
        "issues",
        "recommended_checks",
    }
    if set(value) != expected:
        raise ValueError(f"feedback JSON keys must be exactly {sorted(expected)}")
    for name in (
        "prediction_consistent",
        "reasoning_pass",
        "tone_dependence_detected",
    ):
        if type(value.get(name)) is not bool:
            raise ValueError(f"{name} must be a boolean")
    _validate_string_list(value.get("issues"), "issues")
    _validate_string_list(value.get("recommended_checks"), "recommended_checks")
    return value


def _validate_single_feedback(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "reasoning_pass",
        "sentiment_shortcut_detected",
        "issues",
        "recommended_checks",
    }
    if set(value) != expected:
        raise ValueError(
            f"single feedback JSON keys must be exactly {sorted(expected)}"
        )
    for name in ("reasoning_pass", "sentiment_shortcut_detected"):
        if type(value.get(name)) is not bool:
            raise ValueError(f"{name} must be a boolean")
    issues = _validate_string_list(value.get("issues"), "issues")
    recommended = _validate_string_list(
        value.get("recommended_checks"), "recommended_checks"
    )
    if value["reasoning_pass"] and value["sentiment_shortcut_detected"]:
        raise ValueError(
            "reasoning_pass cannot be true when a sentiment shortcut was detected"
        )
    if value["reasoning_pass"] and issues:
        raise ValueError("issues must be empty when reasoning_pass is true")
    if not value["reasoning_pass"] and not issues:
        raise ValueError("issues must identify why reasoning_pass is false")
    if not value["reasoning_pass"] and not any(item.strip() for item in recommended):
        raise ValueError(
            "recommended_checks must contain an actionable retry check when reasoning fails"
        )
    return value


def _validate_consensus(value: dict[str, Any]) -> dict[str, Any]:
    expected = {"label", "sentiment_used_as_evidence", "brief_reason"}
    if set(value) != expected:
        raise ValueError(f"consensus JSON keys must be exactly {sorted(expected)}")
    if value.get("label") not in CANONICAL_LABELS:
        raise ValueError(f"label must be one of {CANONICAL_LABELS}")
    if type(value.get("sentiment_used_as_evidence")) is not bool:
        raise ValueError("sentiment_used_as_evidence must be a boolean")
    if value["sentiment_used_as_evidence"] is not False:
        raise ValueError("sentiment_used_as_evidence must be false")
    if not isinstance(value.get("brief_reason"), str) or not value["brief_reason"].strip():
        raise ValueError("brief_reason must be a non-empty string")
    return value


def _reflection_validation_error(text: str) -> str | None:
    word_count = len(text.split())
    if not 5 <= word_count <= 180:
        return "reflection must contain between 5 and 180 words"
    if re.search(r"\b(?:human_original|ai_manipulated)\b", text, re.IGNORECASE):
        return "reflection must not name a canonical label"
    if re.search(r"\bfinal\s+(?:answer|verdict|prediction)\b", text, re.IGNORECASE):
        return "reflection must not contain a final verdict"
    if re.search(r'[`{\[]?\s*["\']?label["\']?\s*:', text, re.IGNORECASE):
        return "reflection must not contain a classification JSON field"
    if re.search(
        r"\b(?:switch|flip|choose|select|predict|return)\b.{0,32}(?:"
        r"\b(?:other|opposite)\b(?:\s+(?:class|label))?|"
        r"\b(?:first|second|real|fake|authentic|original|manipulated|synthetic)"
        r"\s+(?:class|label)\b)",
        text,
        re.IGNORECASE,
    ):
        return "reflection must not encode an indirect class-selection instruction"
    return None


class TrainingFreeAdSent:
    """Prompt-only AdSent actor with an optional gold-blind Reflexion loop."""

    def __init__(self, llm: LLMCallable, config: PipelineConfig | None = None):
        if not callable(llm):
            raise TypeError("llm must be callable")
        self.llm = llm
        self.config = config or PipelineConfig()
        self._active_call_ledger: list[CallRecord] | None = None

    def _record_active_call(self, call: CallRecord) -> None:
        if self._active_call_ledger is not None:
            self._active_call_ledger.append(call)

    @staticmethod
    def _validate_condition_reference(
        condition: str,
        reference_context: str | None,
    ) -> None:
        if condition == "article_only":
            if reference_context is not None:
                raise ValueError(
                    "article_only pipeline must not receive reference_context"
                )
            return
        if condition == "reference_assisted":
            if not (reference_context or "").strip():
                raise ValueError(
                    "reference_assisted pipeline requires reference_context"
                )
            return
        raise ValueError(f"unsupported condition: {condition!r}")

    def _take_backend_metadata(self) -> dict[str, Any] | None:
        getter = getattr(self.llm, "take_last_call_metadata", None)
        if not callable(getter):
            return None
        value = getter()
        return value if isinstance(value, dict) else None

    def _call_text(self, prompt: str, *, stage: str) -> tuple[str, CallRecord]:
        started = time.perf_counter()
        try:
            response = self.llm(prompt)
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            backend_metadata = self._take_backend_metadata()
            call = CallRecord(
                stage=stage,
                prompt_sha256=_sha256(prompt),
                latency_ms=round(latency, 3),
                raw_response=None,
                parse_status="call_error",
                parse_error=str(exc),
                backend_metadata=backend_metadata,
            )
            self._record_active_call(call)
            raise PipelineError(f"{stage} LLM call failed: {exc}", [call]) from exc
        latency = (time.perf_counter() - started) * 1000
        backend_metadata = self._take_backend_metadata()
        if not isinstance(response, str) or not response.strip():
            call = CallRecord(
                stage=stage,
                prompt_sha256=_sha256(prompt),
                latency_ms=round(latency, 3),
                raw_response=response if isinstance(response, str) else repr(response),
                parse_status="invalid",
                parse_error="empty or non-string response",
                backend_metadata=backend_metadata,
            )
            self._record_active_call(call)
            raise PipelineError(
                f"{stage} returned an empty or non-string response",
                [call],
            )
        cleaned = _strip_single_markdown_fence(response)
        call = CallRecord(
            stage=stage,
            prompt_sha256=_sha256(prompt),
            latency_ms=round(latency, 3),
            raw_response=response,
            parse_status="not_applicable",
            backend_metadata=backend_metadata,
        )
        self._record_active_call(call)
        return cleaned, call

    def _call_json(
        self,
        prompt: str,
        *,
        stage: str,
        validator: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, list[CallRecord]]:
        calls: list[CallRecord] = []
        current_prompt = prompt
        last_error = "unknown parse error"

        for retry in range(self.config.max_format_retries + 1):
            started = time.perf_counter()
            try:
                response = self.llm(current_prompt)
                latency = (time.perf_counter() - started) * 1000
                backend_metadata = self._take_backend_metadata()
            except Exception as exc:
                latency = (time.perf_counter() - started) * 1000
                backend_metadata = self._take_backend_metadata()
                call = CallRecord(
                        stage=stage,
                        prompt_sha256=_sha256(current_prompt),
                        latency_ms=round(latency, 3),
                        raw_response=None,
                        parse_status="call_error",
                        parse_error=str(exc),
                        retry=retry,
                        backend_metadata=backend_metadata,
                    )
                calls.append(call)
                self._record_active_call(call)
                raise PipelineError(
                    f"{stage} LLM call failed: {exc}",
                    calls,
                ) from exc

            try:
                if not isinstance(response, str):
                    raise ValueError("response must be a string")
                parsed = validator(_parse_json_object(response))
            except ValueError as exc:
                last_error = str(exc)
                call = CallRecord(
                        stage=stage,
                        prompt_sha256=_sha256(current_prompt),
                        latency_ms=round(latency, 3),
                        raw_response=response if isinstance(response, str) else repr(response),
                        parse_status="invalid",
                        parse_error=last_error,
                        retry=retry,
                        backend_metadata=backend_metadata,
                    )
                calls.append(call)
                self._record_active_call(call)
                current_prompt = (
                    prompt
                    + "\n\nYour previous response did not match the required JSON schema "
                    + f"({last_error}). Return only one corrected JSON object."
                )
                continue

            call = CallRecord(
                    stage=stage,
                    prompt_sha256=_sha256(current_prompt),
                    latency_ms=round(latency, 3),
                    raw_response=response,
                    parse_status="valid",
                    retry=retry,
                    backend_metadata=backend_metadata,
                )
            calls.append(call)
            self._record_active_call(call)
            return parsed, calls

        return None, calls

    def _call_reflection_text(
        self,
        prompt: str,
    ) -> tuple[str, list[CallRecord]]:
        calls: list[CallRecord] = []
        current_prompt = prompt
        last_error = "invalid reflection"
        for retry in range(self.config.max_format_retries + 1):
            try:
                reflection, call = self._call_text(
                    current_prompt,
                    stage="self_reflection",
                )
            except PipelineError as exc:
                raise PipelineError(str(exc), [*calls, *exc.call_records]) from exc
            call.retry = retry
            error = _reflection_validation_error(reflection)
            if error is None:
                call.parse_status = "valid"
                calls.append(call)
                return reflection, calls
            last_error = error
            call.parse_status = "invalid"
            call.parse_error = error
            calls.append(call)
            current_prompt = (
                prompt
                + "\n\nThe previous reflection was invalid "
                + f"({error}). Return only a concise, label-free actionable lesson."
            )
        raise PipelineError(
            f"self_reflection remained invalid: {last_error}",
            calls,
        )

    def prepare_views(
        self,
        article: str,
        requested_views: Sequence[str] = VIEW_NAMES,
    ) -> dict[str, ViewArtifact]:
        """Generate and validate requested sentiment views exactly once."""

        if not article.strip():
            raise ValueError("article must not be empty")
        unknown = set(requested_views) - set(VIEW_NAMES)
        if unknown:
            raise ValueError(f"unknown view names: {sorted(unknown)}")

        views: dict[str, ViewArtifact] = {}
        for view_name in dict.fromkeys(requested_views):
            if view_name == "original":
                views[view_name] = ViewArtifact(
                    name="original",
                    text=article,
                    valid=True,
                    target_sentiment=None,
                )
                continue

            repair_feedback: str | None = None
            attempts: list[RewriteAttempt] = []
            accepted_text: str | None = None
            failure_reason: str | None = None

            for attempt_number in range(1, self.config.max_rewrite_attempts + 1):
                rewrite_prompt = build_sentiment_rewrite_prompt(
                    article,
                    view_name,
                    repair_feedback=repair_feedback,
                )
                try:
                    rewritten, rewrite_call = self._call_text(
                        rewrite_prompt,
                        stage=f"rewrite:{view_name}",
                    )
                except PipelineError as exc:
                    failure_reason = str(exc)
                    attempts.append(
                        RewriteAttempt(
                            attempt=attempt_number,
                            rewritten_text=None,
                            quality_report=None,
                            accepted=False,
                            calls=list(exc.call_records),
                        )
                    )
                    repair_feedback = "The prior generation did not produce a usable article."
                    continue
                judge_prompt = build_fact_preservation_prompt(
                    article,
                    rewritten,
                    target_sentiment=view_name,
                )
                try:
                    report, judge_calls = self._call_json(
                        judge_prompt,
                        stage=f"preservation_judge:{view_name}",
                        validator=_validate_preservation,
                    )
                except PipelineError as exc:
                    failure_reason = str(exc)
                    attempts.append(
                        RewriteAttempt(
                            attempt=attempt_number,
                            rewritten_text=rewritten,
                            quality_report=None,
                            accepted=False,
                            calls=[rewrite_call, *exc.call_records],
                        )
                    )
                    repair_feedback = (
                        "The prior rewrite could not be verified; reproduce the article "
                        "while preserving every factual detail."
                    )
                    continue
                accepted = bool(
                    report
                    and report["facts_preserved"]
                    and (
                        report["target_sentiment_achieved"]
                        or not self.config.require_target_sentiment
                    )
                )
                attempts.append(
                    RewriteAttempt(
                        attempt=attempt_number,
                        rewritten_text=rewritten,
                        quality_report=report,
                        accepted=accepted,
                        calls=[rewrite_call, *judge_calls],
                    )
                )
                if accepted:
                    accepted_text = rewritten
                    break

                if report is None:
                    failure_reason = "quality judge did not return valid JSON"
                else:
                    failure_reason = str(report.get("brief_reason") or "quality check failed")
                    changed = report.get("changed_claims") or []
                    if changed:
                        failure_reason += "; changed claims: " + "; ".join(changed)
                repair_feedback = failure_reason

            views[view_name] = ViewArtifact(
                name=view_name,
                text=accepted_text,
                valid=accepted_text is not None,
                target_sentiment=view_name,
                attempts=attempts,
                failure_reason=None if accepted_text is not None else failure_reason,
            )

        return views

    def _detect(
        self,
        view: ViewArtifact,
        *,
        condition: str,
        reference_context: str | None,
        reflections: Sequence[str],
        reveal_sentiment_view: bool = True,
    ) -> dict[str, Any]:
        if not view.valid or view.text is None:
            return {
                "view": view.name,
                "prediction": None,
                "parse_status": "invalid_view",
                "parse_error": view.failure_reason or "view failed quality control",
                "calls": [],
            }

        prompt = build_detection_prompt(
            view.text,
            condition=condition,
            reference_context=reference_context,
            sentiment_view=view.name if reveal_sentiment_view else None,
            reflections=reflections,
        )
        parsed, calls = self._call_json(
            prompt,
            stage=f"detector:{view.name}",
            validator=_validate_detection,
        )
        return {
            "view": view.name,
            "prediction": parsed,
            "parse_status": "valid" if parsed else "invalid",
            "parse_error": None if parsed else "detector response remained invalid",
            "calls": [asdict(call) for call in calls],
        }

    def _evaluate_single_detection(
        self,
        view: ViewArtifact,
        detection: Mapping[str, Any],
        *,
        condition: str,
        reference_context: str | None,
    ) -> dict[str, Any]:
        if view.text is None:
            raise ValueError("single detection evaluator requires a valid text")
        prompt = build_single_detection_feedback_prompt(
            neutral_article=view.text,
            detection={
                "parse_status": detection.get("parse_status"),
                "prediction": detection.get("prediction"),
            },
            condition=condition,
            reference_context=reference_context,
        )
        parsed, calls = self._call_json(
            prompt,
            stage="single_detection_evaluator",
            validator=_validate_single_feedback,
        )
        return {
            "feedback": parsed,
            "parse_status": "valid" if parsed else "invalid",
            "calls": [asdict(call) for call in calls],
        }

    def _reflect_single_detection(
        self,
        view: ViewArtifact,
        detection: Mapping[str, Any],
        evaluator: Mapping[str, Any],
        reflections: Sequence[str],
        *,
        condition: str,
        reference_context: str | None,
    ) -> tuple[str, dict[str, Any]]:
        if view.text is None:
            raise ValueError("single reflection requires a valid text")
        feedback = evaluator.get("feedback")
        if not isinstance(feedback, dict):
            feedback = {
                "reasoning_pass": False,
                "sentiment_shortcut_detected": False,
                "issues": ["Evaluator output was invalid."],
                "recommended_checks": ["Produce schema-valid claim-level reasoning."],
            }
        prompt = build_single_reflection_prompt(
            neutral_article=view.text,
            detection={
                "parse_status": detection.get("parse_status"),
                "prediction": detection.get("prediction"),
            },
            evaluator_feedback=feedback,
            condition=condition,
            reference_context=reference_context,
            prior_reflections=reflections,
        )
        reflection, calls = self._call_reflection_text(prompt)
        return reflection, {
            "parse_status": "valid",
            "calls": [asdict(call) for call in calls],
        }

    @staticmethod
    def _single_reasoning_pass(
        detection: Mapping[str, Any],
        evaluator: Mapping[str, Any],
    ) -> bool:
        prediction = detection.get("prediction")
        feedback = evaluator.get("feedback")
        return bool(
            isinstance(prediction, dict)
            and isinstance(feedback, dict)
            and feedback.get("reasoning_pass") is True
            and feedback.get("sentiment_shortcut_detected") is False
        )

    @staticmethod
    def _reused_detection(
        detection: Mapping[str, Any],
        source_pipeline: str,
    ) -> dict[str, Any]:
        source_calls = detection.get("calls")
        prompt_hashes = [
            call.get("prompt_sha256")
            for call in source_calls
            if isinstance(call, dict) and isinstance(call.get("prompt_sha256"), str)
        ] if isinstance(source_calls, list) else []
        return {
            "view": detection.get("view", "neutral"),
            "prediction": detection.get("prediction"),
            "parse_status": detection.get("parse_status"),
            "parse_error": detection.get("parse_error"),
            "calls": [],
            "reused": True,
            "reused_from_pipeline": source_pipeline,
            "source_prompt_sha256": prompt_hashes[-1] if prompt_hashes else None,
        }

    @_capture_method_call_ledger
    def run_controlled_adsent(
        self,
        views: Mapping[str, ViewArtifact],
        *,
        condition: str,
        reference_context: str | None = None,
        success_predicate: SuccessPredicate | None = None,
        max_trials: int = 1,
    ) -> dict[str, Any]:
        """Run the paper arm, optionally using an external oracle stop predicate."""

        self._validate_condition_reference(condition, reference_context)
        if max_trials < 1:
            raise ValueError("max_trials must be at least 1")
        if success_predicate is None and max_trials != 1:
            raise ValueError("repeated AdSent trials require a success_predicate")
        if "neutral" not in views:
            raise ValueError("controlled AdSent requires a neutral view")
        neutral = views["neutral"]
        self._raise_for_failed_view_calls("controlled_adsent", {"neutral": neutral})
        trials: list[dict[str, Any]] = []
        final_prediction: str | None = None
        stop_reason = "single_pass"
        effective_trials = max_trials if neutral.valid else 1
        for trial_number in range(1, effective_trials + 1):
            detection = self._detect(
                neutral,
                condition=condition,
                reference_context=reference_context,
                reflections=(),
                reveal_sentiment_view=False,
            )
            parsed = detection.get("prediction")
            final_prediction = (
                parsed.get("label") if isinstance(parsed, dict) else None
            )
            oracle_correct = (
                bool(success_predicate(final_prediction))
                if success_predicate is not None
                else None
            )
            trials.append(
                {
                    "trial": trial_number,
                    "detections": {"neutral": detection},
                    "oracle_correct": oracle_correct,
                }
            )
            if not neutral.valid:
                stop_reason = "neutralization_failed"
                break
            if success_predicate is None:
                stop_reason = (
                    "single_pass"
                    if isinstance(parsed, dict)
                    else "invalid_actor_output"
                )
                break
            if oracle_correct:
                stop_reason = "oracle_correct"
                break
            if trial_number == max_trials:
                stop_reason = "max_attempts_exhausted"
                break
        first_prediction = None
        if trials:
            first_parsed = trials[0]["detections"]["neutral"].get("prediction")
            if isinstance(first_parsed, dict):
                first_prediction = first_parsed.get("label")
        first_correct_attempt = next(
            (
                trial["trial"]
                for trial in trials
                if trial.get("oracle_correct") is True
            ),
            None,
        )
        return {
            "method": "adsent",
            "condition": condition,
            "factors": {
                "gt_context": condition == "reference_assisted",
                "reflexion": False,
            },
            "detector_input_view": "neutral",
            "stopping_policy": (
                "oracle_gated_retry" if success_predicate is not None else "single_pass"
            ),
            "max_trials": max_trials,
            "retry_protocol": (
                _oracle_retry_protocol(max_trials)
                if success_predicate is not None
                else None
            ),
            "attempt_count": len(trials),
            "first_correct_attempt": first_correct_attempt,
            "first_prediction": first_prediction,
            "oracle_label_in_prompt": False,
            "oracle_correctness_message_in_prompt": False,
            "oracle_success": stop_reason == "oracle_correct",
            "final_prediction": final_prediction,
            "trials": trials,
            "stop_reason": stop_reason,
        }

    @_capture_method_call_ledger
    def run_controlled_reflexion(
        self,
        views: Mapping[str, ViewArtifact],
        *,
        condition: str,
        reference_context: str | None = None,
        initial_detection: Mapping[str, Any] | None = None,
        initial_detection_source: str | None = None,
        success_predicate: SuccessPredicate | None = None,
        max_trials: int | None = None,
    ) -> dict[str, Any]:
        """Add label-blind Reflexion to the same single neutral-article Actor."""

        self._validate_condition_reference(condition, reference_context)
        effective_max_trials = (
            self.config.max_reflexion_trials if max_trials is None else max_trials
        )
        if effective_max_trials < 1:
            raise ValueError("max_trials must be at least 1")
        if "neutral" not in views:
            raise ValueError("controlled Reflexion requires a neutral view")
        neutral = views["neutral"]
        self._raise_for_failed_view_calls("controlled_reflexion", {"neutral": neutral})
        if initial_detection is not None and initial_detection.get("view") != "neutral":
            raise ValueError("initial_detection must come from the shared neutral view")
        if not neutral.valid:
            detection = (
                self._reused_detection(
                    initial_detection,
                    initial_detection_source or "paired_non_reflexion",
                )
                if initial_detection is not None
                else self._detect(
                    neutral,
                    condition=condition,
                    reference_context=reference_context,
                    reflections=(),
                    reveal_sentiment_view=False,
                )
            )
            return {
                "method": "adsent_reflexion",
                "condition": condition,
                "feedback_mode": CONTROLLED_REFLEXION_FEEDBACK_MODE,
                "factors": {
                    "gt_context": condition == "reference_assisted",
                    "reflexion": True,
                },
                "detector_input_view": "neutral",
                "stopping_policy": (
                    "oracle_gated_retry"
                    if success_predicate is not None
                    else "gold_blind_reasoning_pass"
                ),
                "max_trials": effective_max_trials,
                "retry_protocol": (
                    _oracle_retry_protocol(effective_max_trials)
                    if success_predicate is not None
                    else None
                ),
                "attempt_count": 1,
                "first_correct_attempt": None,
                "first_prediction": None,
                "oracle_label_in_prompt": False,
                "oracle_correctness_message_in_prompt": False,
                "oracle_success": False,
                "initial_detection_reused": initial_detection is not None,
                "initial_detection_source": initial_detection_source,
                "final_prediction": None,
                "trials": [
                    {
                        "trial": 1,
                        "detections": {"neutral": detection},
                        "oracle_correct": (
                            bool(success_predicate(None))
                            if success_predicate is not None
                            else None
                        ),
                    }
                ],
                "final_memory": [],
                "reasoning_pass_converged": False,
                "stop_reason": "neutralization_failed",
            }

        reflections: list[str] = []
        trials: list[dict[str, Any]] = []
        final_prediction: str | None = None
        stop_reason = "max_trials"
        reused_initial = initial_detection is not None

        oracle_success = False
        for trial_number in range(1, effective_max_trials + 1):
            if trial_number == 1 and initial_detection is not None:
                detection = self._reused_detection(
                    initial_detection,
                    initial_detection_source or "paired_non_reflexion",
                )
            else:
                detection = self._detect(
                    neutral,
                    condition=condition,
                    reference_context=reference_context,
                    reflections=tuple(reflections),
                    reveal_sentiment_view=False,
                )
            parsed = detection.get("prediction")
            final_prediction = (
                parsed.get("label") if isinstance(parsed, dict) else None
            )
            oracle_correct = (
                bool(success_predicate(final_prediction))
                if success_predicate is not None
                else None
            )
            if success_predicate is not None and oracle_correct:
                trials.append(
                    {
                        "trial": trial_number,
                        "memory_before": list(reflections),
                        "detections": {"neutral": detection},
                        "evaluator": None,
                        "reflection": None,
                        "reflection_call": None,
                        "oracle_correct": True,
                    }
                )
                oracle_success = True
                stop_reason = "oracle_correct"
                break
            if (
                success_predicate is not None
                and trial_number == effective_max_trials
            ):
                trials.append(
                    {
                        "trial": trial_number,
                        "memory_before": list(reflections),
                        "detections": {"neutral": detection},
                        "evaluator": None,
                        "reflection": None,
                        "reflection_call": None,
                        "oracle_correct": False,
                    }
                )
                stop_reason = "max_attempts_exhausted"
                break
            evaluator = self._evaluate_single_detection(
                neutral,
                detection,
                condition=condition,
                reference_context=reference_context,
            )
            trial_record: dict[str, Any] = {
                "trial": trial_number,
                "memory_before": list(reflections),
                "detections": {"neutral": detection},
                "evaluator": evaluator,
                "reflection": None,
                "reflection_call": None,
                "oracle_correct": oracle_correct,
            }
            trials.append(trial_record)

            if success_predicate is None and self._single_reasoning_pass(
                detection, evaluator
            ):
                stop_reason = "gold_blind_reasoning_pass"
                break
            if trial_number == effective_max_trials:
                if evaluator.get("parse_status") != "valid":
                    stop_reason = "invalid_evaluator_output"
                break

            reflection, reflection_call = self._reflect_single_detection(
                neutral,
                detection,
                evaluator,
                tuple(reflections),
                condition=condition,
                reference_context=reference_context,
            )
            trial_record["reflection"] = reflection
            trial_record["reflection_call"] = reflection_call
            reflections.append(reflection)
            if len(reflections) > self.config.reflection_memory_size:
                reflections = reflections[-self.config.reflection_memory_size :]

        first_prediction = None
        if trials:
            first_parsed = trials[0]["detections"]["neutral"].get("prediction")
            if isinstance(first_parsed, dict):
                first_prediction = first_parsed.get("label")
        first_correct_attempt = next(
            (
                trial["trial"]
                for trial in trials
                if trial.get("oracle_correct") is True
            ),
            None,
        )
        return {
            "method": "adsent_reflexion",
            "condition": condition,
            "feedback_mode": CONTROLLED_REFLEXION_FEEDBACK_MODE,
            "factors": {
                "gt_context": condition == "reference_assisted",
                "reflexion": True,
            },
            "detector_input_view": "neutral",
            "stopping_policy": (
                "oracle_gated_retry"
                if success_predicate is not None
                else "gold_blind_reasoning_pass"
            ),
            "max_trials": effective_max_trials,
            "retry_protocol": (
                _oracle_retry_protocol(effective_max_trials)
                if success_predicate is not None
                else None
            ),
            "attempt_count": len(trials),
            "first_correct_attempt": first_correct_attempt,
            "first_prediction": first_prediction,
            "oracle_label_in_prompt": False,
            "oracle_correctness_message_in_prompt": False,
            "oracle_success": oracle_success,
            "initial_detection_reused": reused_initial,
            "initial_detection_source": initial_detection_source,
            "final_prediction": final_prediction,
            "trials": trials,
            "final_memory": reflections,
            "reasoning_pass_converged": (
                None
                if success_predicate is not None
                else stop_reason == "gold_blind_reasoning_pass"
            ),
            "stop_reason": stop_reason,
        }

    def _detect_plain(
        self,
        view: ViewArtifact,
        *,
        reference_context: str | None = None,
    ) -> dict[str, Any]:
        """Run the same metadata-blind black-box prompt on any sentiment view."""

        if not view.valid or view.text is None:
            return {
                "view": view.name,
                "prediction": None,
                "parse_status": "invalid_view",
                "parse_error": view.failure_reason or "view failed quality control",
                "calls": [],
            }
        prompt = build_plain_detection_prompt(
            view.text,
            reference_context=reference_context,
        )
        parsed, calls = self._call_json(
            prompt,
            stage=f"plain_detector:{view.name}",
            validator=_validate_plain_detection,
        )
        return {
            "view": view.name,
            "prediction": parsed,
            "parse_status": "valid" if parsed else "invalid",
            "parse_error": None if parsed else "plain detector response remained invalid",
            "calls": [asdict(call) for call in calls],
        }

    @staticmethod
    def _preservation_summaries(
        views: Mapping[str, ViewArtifact],
    ) -> dict[str, dict[str, Any]]:
        summaries: dict[str, dict[str, Any]] = {}
        for name, view in views.items():
            if name == "original":
                summaries[name] = {
                    "facts_preserved": True,
                    "target_sentiment_achieved": None,
                    "valid": True,
                }
                continue
            final_report = view.attempts[-1].quality_report if view.attempts else None
            summaries[name] = {
                "facts_preserved": (
                    final_report.get("facts_preserved") if final_report else None
                ),
                "target_sentiment_achieved": (
                    final_report.get("target_sentiment_achieved")
                    if final_report
                    else None
                ),
                "valid": view.valid,
                "brief_reason": (
                    final_report.get("brief_reason") if final_report else view.failure_reason
                ),
            }
        return summaries

    @staticmethod
    def _raise_for_failed_view_calls(
        method: str,
        views: Mapping[str, ViewArtifact],
    ) -> None:
        """Distinguish backend failures from ordinary quality-control rejection."""

        failed_names = [
            name
            for name, view in views.items()
            if not view.valid
            and any(
                call.parse_status == "call_error"
                for attempt in view.attempts
                for call in attempt.calls
            )
        ]
        if failed_names:
            raise PipelineError(
                f"{method} required view generation failed because of an LLM call "
                f"error: {', '.join(sorted(failed_names))}"
            )

    @staticmethod
    def _detections_cover_valid_views(
        detections: Mapping[str, Mapping[str, Any]],
        views: Mapping[str, ViewArtifact],
    ) -> bool:
        """Require every quality-approved view to have a parse-valid Actor output."""

        expected = [name for name, view in views.items() if view.valid]
        return bool(expected) and all(
            isinstance(detections.get(name, {}).get("prediction"), dict)
            for name in expected
        )

    @staticmethod
    def _prediction_payload(
        detections: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, object]]:
        return {
            name: {
                "parse_status": detection.get("parse_status"),
                "prediction": detection.get("prediction"),
            }
            for name, detection in detections.items()
        }

    def _evaluate_invariance(
        self,
        detections: Mapping[str, Mapping[str, Any]],
        views: Mapping[str, ViewArtifact],
        *,
        condition: str,
        reference_context: str | None,
    ) -> dict[str, Any]:
        prediction_payload = self._prediction_payload(detections)
        prompt = build_invariance_feedback_prompt(
            view_articles={
                name: view.text if view.valid else None for name, view in views.items()
            },
            view_predictions=prediction_payload,
            preservation_reports=self._preservation_summaries(views),
            condition=condition,
            reference_context=reference_context,
        )
        parsed, calls = self._call_json(
            prompt,
            stage="invariance_evaluator",
            validator=_validate_feedback,
        )
        return {
            "feedback": parsed,
            "parse_status": "valid" if parsed else "invalid",
            "calls": [asdict(call) for call in calls],
        }

    def _resolve(
        self,
        detections: Mapping[str, Mapping[str, Any]],
        views: Mapping[str, ViewArtifact],
        evaluator: Mapping[str, Any] | None,
        *,
        condition: str,
        reference_context: str | None,
    ) -> dict[str, Any]:
        prompt = build_consensus_prompt(
            view_articles={
                name: view.text if view.valid else None for name, view in views.items()
            },
            view_predictions=self._prediction_payload(detections),
            evaluator_feedback=(evaluator or {}).get("feedback"),
            condition=condition,
            reference_context=reference_context,
        )
        parsed, calls = self._call_json(
            prompt,
            stage="consensus_resolver",
            validator=_validate_consensus,
        )
        return {
            "prediction": parsed,
            "parse_status": "valid" if parsed else "invalid",
            "calls": [asdict(call) for call in calls],
        }

    def _reflect(
        self,
        detections: Mapping[str, Mapping[str, Any]],
        evaluator: Mapping[str, Any],
        reflections: Sequence[str],
    ) -> tuple[str, dict[str, Any]]:
        feedback = evaluator.get("feedback")
        if not isinstance(feedback, dict):
            feedback = {
                "prediction_consistent": False,
                "reasoning_pass": False,
                "tone_dependence_detected": False,
                "issues": ["Evaluator output was invalid."],
                "recommended_checks": ["Produce schema-valid, claim-level reasoning."],
            }
        prompt = build_reflection_prompt(
            view_predictions=self._prediction_payload(detections),
            evaluator_feedback=feedback,
            prior_reflections=reflections,
        )
        reflection, calls = self._call_reflection_text(prompt)
        return reflection, {
            "parse_status": "valid",
            "calls": [asdict(call) for call in calls],
        }

    @staticmethod
    def _gold_blind_stop(
        detections: Mapping[str, Mapping[str, Any]],
        evaluator: Mapping[str, Any],
        consensus: Mapping[str, Any],
    ) -> bool:
        """Use consistency and reasoning checks only; never benchmark correctness."""

        labels = [
            detection["prediction"]["label"]
            for detection in detections.values()
            if isinstance(detection.get("prediction"), dict)
        ]
        predictions = [
            detection["prediction"]
            for detection in detections.values()
            if isinstance(detection.get("prediction"), dict)
        ]
        feedback = evaluator.get("feedback")
        resolved = consensus.get("prediction")
        return bool(
            len(labels) >= 2
            and len(set(labels)) == 1
            and all(
                prediction.get("sentiment_used_as_evidence") is False
                for prediction in predictions
            )
            and isinstance(feedback, dict)
            and feedback.get("prediction_consistent") is True
            and feedback.get("reasoning_pass") is True
            and feedback.get("tone_dependence_detected") is False
            and isinstance(resolved, dict)
            and resolved.get("label") == labels[0]
            and resolved.get("sentiment_used_as_evidence") is False
        )

    @_capture_method_call_ledger
    def run_vanilla(
        self,
        views: Mapping[str, ViewArtifact],
        *,
        condition: str,
        reference_context: str | None = None,
    ) -> dict[str, Any]:
        """Classify the original article once (zero-shot baseline)."""

        if "original" not in views:
            raise ValueError("vanilla requires an original view")
        detection = self._detect_plain(
            views["original"],
            reference_context=reference_context,
        )
        parsed = detection.get("prediction")
        return {
            "method": "vanilla",
            "condition": condition,
            "final_prediction": parsed.get("label") if isinstance(parsed, dict) else None,
            "trials": [{"trial": 1, "detections": {"original": detection}}],
            "stop_reason": "single_pass",
        }

    @_capture_method_call_ledger
    def run_adsent(
        self,
        views: Mapping[str, ViewArtifact],
        *,
        condition: str,
        reference_context: str | None = None,
    ) -> dict[str, Any]:
        """Paper-nearest training-free defense: neutralize, then zero-shot classify."""

        if "neutral" not in views:
            raise ValueError("adsent requires a neutral view")
        selected = views["neutral"]
        self._raise_for_failed_view_calls("adsent", {"neutral": selected})
        if not selected.valid:
            detection = self._detect_plain(
                selected,
                reference_context=reference_context,
            )
            return {
                "method": "adsent",
                "condition": condition,
                "final_prediction": None,
                "trials": [{"trial": 1, "detections": {"neutral": detection}}],
                "neutralization_valid": False,
                "fallback_to_original": False,
                "stop_reason": "neutralization_failed",
            }
        detection = self._detect_plain(
            selected,
            reference_context=reference_context,
        )
        parsed = detection.get("prediction")
        return {
            "method": "adsent",
            "condition": condition,
            "final_prediction": parsed.get("label") if isinstance(parsed, dict) else None,
            "trials": [{"trial": 1, "detections": {selected.name: detection}}],
            "neutralization_valid": views["neutral"].valid,
            "fallback_to_original": False,
            "stop_reason": "single_pass",
        }

    @_capture_method_call_ledger
    def run_attack_benchmark(
        self,
        views: Mapping[str, ViewArtifact],
        *,
        condition: str,
        reference_context: str | None = None,
    ) -> dict[str, Any]:
        """Classify every sentiment view independently for paper-style flip metrics."""

        missing = set(VIEW_NAMES) - set(views)
        if missing:
            raise ValueError(f"attack benchmark is missing views: {sorted(missing)}")
        self._raise_for_failed_view_calls("attack", views)

        detections = {
            name: self._detect_plain(
                view,
                reference_context=reference_context,
            )
            for name, view in views.items()
        }
        original = detections.get("original", {}).get("prediction")
        return {
            "method": "attack_benchmark",
            "condition": condition,
            "final_prediction": (
                original.get("label") if isinstance(original, dict) else None
            ),
            "trials": [{"trial": 1, "detections": detections}],
            "stop_reason": "single_pass",
        }

    @_capture_method_call_ledger
    def run_invariant(
        self,
        views: Mapping[str, ViewArtifact],
        *,
        condition: str,
        reference_context: str | None = None,
    ) -> dict[str, Any]:
        """Classify all views and resolve disagreements once, without reflection."""

        self._raise_for_failed_view_calls("invariant", views)

        detections = {
            name: self._detect(
                view,
                condition=condition,
                reference_context=reference_context,
                reflections=(),
            )
            for name, view in views.items()
        }
        valid_transformed = [
            view for name, view in views.items() if name != "original" and view.valid
        ]
        original_view = views.get("original")
        if original_view is None or not original_view.valid or not valid_transformed:
            return {
                "method": "invariant",
                "condition": condition,
                "final_prediction": None,
                "trials": [{"trial": 1, "detections": detections}],
                "stop_reason": "insufficient_valid_views",
            }
        if not self._detections_cover_valid_views(detections, views):
            return {
                "method": "invariant",
                "condition": condition,
                "final_prediction": None,
                "trials": [{"trial": 1, "detections": detections}],
                "stop_reason": "insufficient_valid_detections",
            }
        evaluator = self._evaluate_invariance(
            detections,
            views,
            condition=condition,
            reference_context=reference_context,
        )
        if evaluator.get("parse_status") != "valid" or not isinstance(
            evaluator.get("feedback"), dict
        ):
            return {
                "method": "invariant",
                "condition": condition,
                "final_prediction": None,
                "trials": [
                    {
                        "trial": 1,
                        "detections": detections,
                        "evaluator": evaluator,
                        "consensus": None,
                    }
                ],
                "stop_reason": "invalid_evaluator_output",
            }
        consensus = self._resolve(
            detections,
            views,
            evaluator,
            condition=condition,
            reference_context=reference_context,
        )
        parsed = consensus.get("prediction")
        return {
            "method": "invariant",
            "condition": condition,
            "final_prediction": parsed.get("label") if isinstance(parsed, dict) else None,
            "trials": [
                {
                    "trial": 1,
                    "detections": detections,
                    "evaluator": evaluator,
                    "consensus": consensus,
                }
            ],
            "stop_reason": (
                "single_pass_consensus"
                if isinstance(parsed, dict)
                else "invalid_consensus_output"
            ),
        }

    @_capture_method_call_ledger
    def run_reflexion(
        self,
        views: Mapping[str, ViewArtifact],
        *,
        condition: str,
        reference_context: str | None = None,
    ) -> dict[str, Any]:
        """Run Actor -> gold-blind Evaluator -> Reflection with memory capped at 3."""

        if len(views) < 2:
            raise ValueError("reflexion requires at least two requested sentiment views")
        self._raise_for_failed_view_calls("reflexion", views)

        has_original = bool(views.get("original") and views["original"].valid)
        has_transformed = any(
            name != "original" and view.valid for name, view in views.items()
        )
        if not has_original or not has_transformed:
            detections = {
                name: self._detect(
                    view,
                    condition=condition,
                    reference_context=reference_context,
                    reflections=(),
                )
                for name, view in views.items()
            }
            return {
                "method": "adsent_reflexion",
                "condition": condition,
                "feedback_mode": REFLEXION_FEEDBACK_MODE,
                "final_prediction": None,
                "trials": [{"trial": 1, "detections": detections}],
                "final_memory": [],
                "consistency_converged": False,
                "stop_reason": "insufficient_valid_views",
            }

        reflections: list[str] = []
        trials: list[dict[str, Any]] = []
        final_prediction: str | None = None
        stop_reason = "max_trials"

        for trial_number in range(1, self.config.max_reflexion_trials + 1):
            detections = {
                name: self._detect(
                    view,
                    condition=condition,
                    reference_context=reference_context,
                    reflections=tuple(reflections),
                )
                for name, view in views.items()
            }
            if not self._detections_cover_valid_views(detections, views):
                trials.append(
                    {
                        "trial": trial_number,
                        "memory_before": list(reflections),
                        "detections": detections,
                        "evaluator": None,
                        "consensus": None,
                        "reflection": None,
                        "reflection_call": None,
                    }
                )
                final_prediction = None
                stop_reason = "insufficient_valid_detections"
                break
            evaluator = self._evaluate_invariance(
                detections,
                views,
                condition=condition,
                reference_context=reference_context,
            )
            if evaluator.get("parse_status") != "valid" or not isinstance(
                evaluator.get("feedback"), dict
            ):
                invalid_evaluator_trial = {
                    "trial": trial_number,
                    "memory_before": list(reflections),
                    "detections": detections,
                    "evaluator": evaluator,
                    "consensus": None,
                    "reflection": None,
                    "reflection_call": None,
                }
                trials.append(invalid_evaluator_trial)
                final_prediction = None
                if trial_number == self.config.max_reflexion_trials:
                    stop_reason = "invalid_evaluator_output"
                    break
                reflection, reflection_call = self._reflect(
                    detections,
                    evaluator,
                    tuple(reflections),
                )
                invalid_evaluator_trial["reflection"] = reflection
                invalid_evaluator_trial["reflection_call"] = reflection_call
                reflections.append(reflection)
                if len(reflections) > self.config.reflection_memory_size:
                    reflections = reflections[-self.config.reflection_memory_size :]
                continue
            consensus = self._resolve(
                detections,
                views,
                evaluator,
                condition=condition,
                reference_context=reference_context,
            )
            parsed = consensus.get("prediction")
            final_prediction = parsed.get("label") if isinstance(parsed, dict) else None

            trial_record: dict[str, Any] = {
                "trial": trial_number,
                "memory_before": list(reflections),
                "detections": detections,
                "evaluator": evaluator,
                "consensus": consensus,
                "reflection": None,
                "reflection_call": None,
            }
            trials.append(trial_record)

            if self._gold_blind_stop(detections, evaluator, consensus):
                stop_reason = "gold_blind_consistency_pass"
                break
            if trial_number == self.config.max_reflexion_trials:
                if not isinstance(parsed, dict):
                    stop_reason = "invalid_consensus_output"
                break

            reflection, reflection_call = self._reflect(
                detections,
                evaluator,
                tuple(reflections),
            )
            trial_record["reflection"] = reflection
            trial_record["reflection_call"] = reflection_call
            reflections.append(reflection)
            if len(reflections) > self.config.reflection_memory_size:
                reflections = reflections[-self.config.reflection_memory_size :]

        return {
            "method": "adsent_reflexion",
            "condition": condition,
            "feedback_mode": REFLEXION_FEEDBACK_MODE,
            "final_prediction": final_prediction,
            "trials": trials,
            "final_memory": reflections,
            "consistency_converged": stop_reason == "gold_blind_consistency_pass",
            "stop_reason": stop_reason,
        }


def serialize_views(views: Mapping[str, ViewArtifact]) -> dict[str, dict[str, Any]]:
    """Convert dataclass artifacts to JSON-serializable dictionaries."""

    return {name: asdict(view) for name, view in views.items()}


def deserialize_views(
    payload: Mapping[str, Mapping[str, Any]],
) -> dict[str, ViewArtifact]:
    """Restore cached view artifacts after validating their nested shape."""

    restored: dict[str, ViewArtifact] = {}
    for name, raw_view in payload.items():
        if name not in VIEW_NAMES or raw_view.get("name") != name:
            raise ValueError(f"invalid cached view name: {name!r}")
        raw_attempts = raw_view.get("attempts", [])
        if not isinstance(raw_attempts, list):
            raise ValueError(f"cached attempts for {name!r} must be a list")
        attempts: list[RewriteAttempt] = []
        for raw_attempt in raw_attempts:
            if not isinstance(raw_attempt, dict):
                raise ValueError(f"cached rewrite attempt for {name!r} must be an object")
            raw_calls = raw_attempt.get("calls", [])
            if not isinstance(raw_calls, list):
                raise ValueError(f"cached calls for {name!r} must be a list")
            calls = [CallRecord(**raw_call) for raw_call in raw_calls]
            attempts.append(
                RewriteAttempt(
                    attempt=raw_attempt["attempt"],
                    rewritten_text=raw_attempt.get("rewritten_text"),
                    quality_report=raw_attempt.get("quality_report"),
                    accepted=raw_attempt["accepted"],
                    calls=calls,
                )
            )
        restored[name] = ViewArtifact(
            name=name,
            text=raw_view.get("text"),
            valid=raw_view["valid"],
            target_sentiment=raw_view.get("target_sentiment"),
            attempts=attempts,
            failure_reason=raw_view.get("failure_reason"),
        )
    return restored


def count_method_calls(result: Mapping[str, Any]) -> int:
    """Count stored LLM call records in a method result recursively."""

    count = 0

    def visit(value: object) -> None:
        nonlocal count
        if isinstance(value, dict):
            if {
                "stage",
                "prompt_sha256",
                "latency_ms",
                "parse_status",
            }.issubset(value):
                count += 1
                return
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result)
    return count


def majority_label(detections: Mapping[str, Mapping[str, Any]]) -> str | None:
    """Diagnostic-only majority helper; the main resolver remains claim based."""

    labels = [
        detection["prediction"]["label"]
        for detection in detections.values()
        if isinstance(detection.get("prediction"), dict)
        and detection["prediction"].get("label") in CANONICAL_LABELS
    ]
    if not labels:
        return None
    counts = Counter(labels)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]
