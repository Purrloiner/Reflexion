"""CLI for training-free AdSent with per-example oracle-gated retries.

Private gold controls only the stop/retry gate (up to five Actor attempts); it is
never placed in an LLM prompt. Reflexion feedback itself remains label-blind.

Examples:

    python src/run_adsent_propanews.py --validate-only
    python src/run_adsent_propanews.py --methods adsent adsent_gt --limit 2
    python src/run_adsent_propanews.py --methods adsent adsent_reflexion --run-name no-gt

The module has a main guard and performs no data loading or model calls on import.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adsent_propanews_data import (
    CanonicalLabel,
    DEFAULT_ARTICLES_PATH,
    DEFAULT_REFERENCES_PATH,
    InputMode,
    PropaNewsRecord,
    load_propanews_dataset,
    validate_records,
)
from adsent_propanews_llm import (
    ImportedCallableLLM,
    VertexGeminiConfig,
    VertexGeminiLLM,
)
from adsent_propanews_pipeline import (
    CONTROLLED_REFLEXION_FEEDBACK_MODE,
    PipelineConfig,
    TrainingFreeAdSent,
    deserialize_views,
    serialize_views,
)
from adsent_propanews_prompts import PROMPT_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "adsent_propanews"
SCHEMA_VERSION = "adsent-propanews-run-v3"
EXPERIMENT_DESIGN = "adsent_gt_reflexion_2x2_oracle_retry5"
SUBSET_EXPERIMENT_DESIGN = "adsent_gt_reflexion_subset_oracle_retry5_diagnostic"
DEFAULT_COHORT_SIZE = 100
MAX_ORACLE_ATTEMPTS = 5
ORACLE_RETRY_PROTOCOL = "per_example_oracle_early_stop"


@dataclass(frozen=True)
class ExperimentPipelineSpec:
    use_gt: bool
    use_reflexion: bool
    condition: str
    result_method: str


PIPELINE_SPECS = {
    "adsent": ExperimentPipelineSpec(False, False, "article_only", "adsent"),
    "adsent_gt": ExperimentPipelineSpec(
        True, False, "reference_assisted", "adsent_gt"
    ),
    "adsent_reflexion": ExperimentPipelineSpec(
        False, True, "article_only", "adsent_reflexion"
    ),
    "adsent_gt_reflexion": ExperimentPipelineSpec(
        True, True, "reference_assisted", "adsent_gt_reflexion"
    ),
}
METHODS = tuple(PIPELINE_SPECS)
PAIRED_BASE_METHOD = {
    "adsent_reflexion": "adsent",
    "adsent_gt_reflexion": "adsent_gt",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _callable_source_sha256(spec: str) -> str | None:
    module_name = spec.split(":", 1)[0]
    try:
        module_spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return None
    origin = getattr(module_spec, "origin", None)
    if not origin or not Path(origin).is_file():
        return None
    return _file_sha256(Path(origin))


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(
                _jsonable(value),
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_latest_events(path: Path) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    nonempty_positions = [position for position, line in enumerate(lines) if line.strip()]
    last_nonempty = nonempty_positions[-1] if nonempty_positions else None
    valid_lines: list[str] = []
    for position, line in enumerate(lines):
        line_number = position + 1
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            if position != last_nonempty:
                raise RuntimeError(
                    f"corrupt JSONL at {path}:{line_number}: {exc.msg}"
                ) from exc
            backup = path.with_name(
                path.name
                + ".trailing-partial."
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                + ".bak"
            )
            shutil.copy2(path, backup)
            repaired = "".join(valid_lines)
            if repaired and not repaired.endswith("\n"):
                repaired += "\n"
            _atomic_write_text(path, repaired)
            print(
                f"Warning: ignored a trailing partial JSONL record at line "
                f"{line_number}; original preserved at {backup}",
                file=sys.stderr,
            )
            break
        if not isinstance(event, dict) or type(event.get("example_id")) is not int:
            raise RuntimeError(f"invalid event at {path}:{line_number}")
        latest[event["example_id"]] = event
        valid_lines.append(line if line.endswith("\n") else line + "\n")
    return latest


def _parse_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    indices: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            index = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid index {part!r}") from exc
        if index < 0:
            raise argparse.ArgumentTypeError("indices must be non-negative")
        indices.append(index)
    if not indices:
        raise argparse.ArgumentTypeError("--indices must contain at least one index")
    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("--indices must not contain duplicates")
    return indices


def _select_records(
    records: tuple[PropaNewsRecord, ...],
    *,
    raw_indices: str | None,
    limit: int | None,
) -> tuple[PropaNewsRecord, ...]:
    requested = _parse_indices(raw_indices)
    by_index = {record.index: record for record in records}
    if requested is None:
        selected = list(records)
    else:
        missing = sorted(set(requested) - set(by_index))
        if missing:
            raise ValueError(f"requested indices are absent from the dataset: {missing}")
        selected = [by_index[index] for index in requested]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        if limit > DEFAULT_COHORT_SIZE:
            raise ValueError(
                f"--limit must not exceed the fixed cohort size "
                f"({DEFAULT_COHORT_SIZE})"
            )
        selected = selected[:limit]
    if not selected:
        raise ValueError("the selected evaluation cohort is empty")
    return tuple(selected)


def _required_views(methods: tuple[str, ...]) -> tuple[str, ...]:
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown pipelines: {sorted(unknown)}")
    return ("original", "neutral")


def _validate_method_dependencies(methods: tuple[str, ...]) -> None:
    """Keep every Reflexion arm paired with its identical single-pass trial 1."""

    selected = set(methods)
    missing_pairs = [
        f"{reflexion} requires {base}"
        for reflexion, base in PAIRED_BASE_METHOD.items()
        if reflexion in selected and base not in selected
    ]
    if missing_pairs:
        raise ValueError("; ".join(missing_pairs))


def _dataset_load_mode(methods: tuple[str, ...]) -> InputMode:
    return (
        "reference_assisted"
        if any(PIPELINE_SPECS[method].use_gt for method in methods)
        else "article_only"
    )


def _experiment_design(
    methods: tuple[str, ...],
    cohort_size: int = DEFAULT_COHORT_SIZE,
) -> str:
    return (
        EXPERIMENT_DESIGN
        if (
            len(methods) == len(METHODS)
            and set(methods) == set(METHODS)
            and cohort_size == DEFAULT_COHORT_SIZE
        )
        else SUBSET_EXPERIMENT_DESIGN
    )


def _estimate_logical_calls(
    methods: tuple[str, ...],
    config: PipelineConfig,
) -> dict[str, int]:
    """Bound prompt-level calls, excluding backend transport retries."""

    json_attempts = config.max_format_retries + 1
    _validate_method_dependencies(methods)
    transformed_views = len(_required_views(methods)) - 1
    shared_minimum = 2 * transformed_views
    shared_maximum = (
        transformed_views
        * config.max_rewrite_attempts
        * (1 + json_attempts)
    )
    minimum = shared_minimum
    maximum = shared_maximum
    for method in methods:
        spec = PIPELINE_SPECS[method]
        if not spec.use_reflexion:
            minimum += 1
            maximum += MAX_ORACLE_ATTEMPTS * json_attempts
        else:
            # A paired non-Reflexion arm supplies the identical trial-1 Actor
            # prediction. An oracle-correct trial 1 stops without another call.
            # Otherwise, trials 1..T-1 each need an evaluator/reflection pair and
            # trials 2..T need a new Actor call; no feedback is generated after the
            # final exhausted attempt.
            maximum += (
                3 * (MAX_ORACLE_ATTEMPTS - 1) * json_attempts
            )
    return {"minimum": minimum, "maximum": maximum}


def _cohort_text_stats(records: tuple[PropaNewsRecord, ...]) -> dict[str, int]:
    longest = max(records, key=lambda record: len(record.prompt.article))
    return {
        "total_article_characters": sum(
            len(record.prompt.article) for record in records
        ),
        "maximum_article_characters": len(longest.prompt.article),
        "maximum_article_index": longest.index,
        "maximum_article_words": max(
            len(record.prompt.article.split()) for record in records
        ),
    }


def _build_llm(args: argparse.Namespace) -> tuple[object, dict[str, object]]:
    llm: ImportedCallableLLM | VertexGeminiLLM
    if args.llm_backend == "existing":
        module_name = args.llm_callable.split(":", 1)[0]
        try:
            module_spec = importlib.util.find_spec(module_name)
        except (ImportError, AttributeError, ValueError) as exc:
            raise RuntimeError(
                f"cannot resolve LLM callable module {module_name!r}: {exc}"
            ) from exc
        if module_spec is None:
            raise RuntimeError(
                f"cannot resolve LLM callable module {module_name!r}"
            )
        llm = ImportedCallableLLM(args.llm_callable)
    else:
        llm = VertexGeminiLLM(
            VertexGeminiConfig(
                model=args.vertex_model,
                project=args.vertex_project,
                location=args.vertex_location,
                temperature=args.temperature,
                seed=args.seed,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.model_retries,
            )
        )
    provenance = dict(llm.provenance)
    if args.llm_backend == "existing":
        provenance.update(
            {
                "decoding_control": "imported_callable",
                "runner_generation_flags_applied": False,
            }
        )
    else:
        provenance.update(
            {
                "decoding_control": "runner_vertex_config",
                "runner_generation_flags_applied": True,
            }
        )
    return llm, provenance


def _config_payload(
    args: argparse.Namespace,
    *,
    records: tuple[PropaNewsRecord, ...],
    llm_provenance: dict[str, object],
    pipeline_config: PipelineConfig,
) -> dict[str, object]:
    stable: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "articles_path": str(Path(args.data).resolve()),
        "articles_sha256": _file_sha256(Path(args.data)),
        "references_path": str(Path(args.gt_data).resolve()),
        "references_sha256": _file_sha256(Path(args.gt_data)),
        "experiment_design": _experiment_design(
            tuple(args.methods), len(records)
        ),
        "dataset_load_mode": _dataset_load_mode(tuple(args.methods)),
        "methods": list(args.methods),
        "inference_protocol": {
            "name": ORACLE_RETRY_PROTOCOL,
            "max_attempts": MAX_ORACLE_ATTEMPTS,
            "cohort_size": len(records),
            "gold_used_for": "external_retry_gate_only",
            "oracle_label_in_prompt": False,
            "oracle_correctness_message_in_prompt": False,
        },
        "pipeline_specs": {
            method: asdict(PIPELINE_SPECS[method]) for method in args.methods
        },
        "cohort_indices": [record.index for record in records],
        "requested_views": list(_required_views(tuple(args.methods))),
        "pipeline": asdict(pipeline_config),
        "llm": llm_provenance,
        "implementation_sha256": {
            name: _file_sha256(Path(__file__).with_name(name))
            for name in (
                "adsent_propanews_data.py",
                "adsent_propanews_llm.py",
                "adsent_propanews_metrics.py",
                "adsent_propanews_pipeline.py",
                "adsent_propanews_prompts.py",
                "run_adsent_propanews.py",
            )
        },
    }
    if args.llm_backend == "existing":
        stable["imported_callable_source_sha256"] = _callable_source_sha256(
            args.llm_callable
        )
    return {
        **stable,
        "fingerprint": _stable_hash(stable),
        "created_at": _utc_now(),
        "scientific_scope": (
            "human original vs AI-manipulated synthetic disinformation; provenance "
            "and veracity are confounded in this PropaNews sample; predictions are a "
            "private oracle-evaluated artifact because retry timing reveals correctness"
        ),
        "reflexion_feedback": CONTROLLED_REFLEXION_FEEDBACK_MODE,
        "logical_call_estimate_per_example": _estimate_logical_calls(
            tuple(args.methods),
            pipeline_config,
        ),
        "cohort_text_stats": _cohort_text_stats(records),
    }


def _ensure_run_config(path: Path, config: dict[str, object], *, resume: bool) -> None:
    if not path.exists():
        _atomic_write_json(path, config)
        return
    with path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    if not resume:
        raise RuntimeError(
            f"run directory already contains {path.name}; choose another --run-name or "
            "enable --resume"
        )
    if existing.get("fingerprint") != config.get("fingerprint"):
        raise RuntimeError(
            "existing run configuration does not match this invocation; choose a new "
            "--run-name so results from different prompts/models are not mixed"
        )


def _method_is_complete(wrapper: object, method: str) -> bool:
    if not isinstance(wrapper, dict) or wrapper.get("status") != "ok":
        return False
    result = wrapper.get("result")
    if not isinstance(result, dict):
        return False
    if method not in PIPELINE_SPECS:
        return False
    spec = PIPELINE_SPECS[method]
    trials = result.get("trials")
    if not isinstance(trials, list) or not trials or not all(
        isinstance(trial, dict) for trial in trials
    ):
        return False
    trial_labels: list[str | None] = []
    for trial in trials:
        detections = trial.get("detections")
        if not isinstance(detections, dict):
            return False
        detection = detections.get("neutral")
        if not isinstance(detection, dict):
            return False
        actor_prediction = detection.get("prediction")
        if actor_prediction is None:
            trial_labels.append(None)
            continue
        if not (
            isinstance(actor_prediction, dict)
            and actor_prediction.get("label")
            in {"human_original", "ai_manipulated"}
        ):
            return False
        trial_labels.append(actor_prediction["label"])
    if (
        result.get("stopping_policy") != "oracle_gated_retry"
        or result.get("max_trials") != MAX_ORACLE_ATTEMPTS
        or result.get("oracle_label_in_prompt") is not False
        or result.get("oracle_correctness_message_in_prompt") is not False
        or result.get("attempt_count") != len(trials)
        or not 1 <= len(trials) <= MAX_ORACLE_ATTEMPTS
        or [trial.get("trial") for trial in trials]
        != list(range(1, len(trials) + 1))
        or any(type(trial.get("oracle_correct")) is not bool for trial in trials)
    ):
        return False
    retry_protocol = result.get("retry_protocol")
    if not isinstance(retry_protocol, dict) or retry_protocol != {
        "name": ORACLE_RETRY_PROTOCOL,
        "max_attempts": MAX_ORACLE_ATTEMPTS,
        "gold_used_for": "external_retry_gate_only",
        "oracle_label_in_prompt": False,
        "oracle_correctness_message_in_prompt": False,
    }:
        return False
    oracle_success = result.get("oracle_success") is True
    stop_reason = result.get("stop_reason")
    first_correct = next(
        (
            trial["trial"]
            for trial in trials
            if trial.get("oracle_correct") is True
        ),
        None,
    )
    if result.get("first_correct_attempt") != first_correct:
        return False
    if oracle_success != (first_correct is not None):
        return False
    if sum(trial.get("oracle_correct") is True for trial in trials) > 1:
        return False
    first_label = trial_labels[0]
    last_label = trial_labels[-1]
    if (
        result.get("first_prediction") != first_label
        or result.get("final_prediction") != last_label
    ):
        return False
    if stop_reason == "oracle_correct":
        if not oracle_success or trials[-1].get("oracle_correct") is not True:
            return False
    elif stop_reason == "max_attempts_exhausted":
        if oracle_success or len(trials) != MAX_ORACLE_ATTEMPTS:
            return False
    elif stop_reason == "neutralization_failed":
        if oracle_success or len(trials) != 1:
            return False
    else:
        return False
    if spec.use_reflexion and (
        result.get("initial_detection_reused") is not True
        or result.get("initial_detection_source") != PAIRED_BASE_METHOD[method]
    ):
        return False
    return bool(
        result.get("pipeline") == method
        and result.get("method") == spec.result_method
        and result.get("condition") == spec.condition
        and result.get("factors")
        == {"gt_context": spec.use_gt, "reflexion": spec.use_reflexion}
        and result.get("detector_input_view") == "neutral"
        and (
            result.get("final_prediction") is None
            or result.get("final_prediction")
            in {"human_original", "ai_manipulated"}
        )
        and isinstance(stop_reason, str)
        and bool(stop_reason)
    )


def _paired_checkpoint_matches(
    method_results: dict[str, Any],
    reflexion_method: str,
) -> bool:
    """Verify that cached Reflexion trial 1 is the paired base detection."""

    source_method = PAIRED_BASE_METHOD.get(reflexion_method)
    if source_method is None:
        return True
    source_wrapper = method_results.get(source_method)
    reflexion_wrapper = method_results.get(reflexion_method)
    if (
        not isinstance(source_wrapper, dict)
        or not isinstance(reflexion_wrapper, dict)
        or not _method_is_complete(source_wrapper, source_method)
        or not _method_is_complete(reflexion_wrapper, reflexion_method)
    ):
        return False
    source_result = source_wrapper.get("result")
    reflexion_result = reflexion_wrapper.get("result")
    if not isinstance(source_result, dict) or not isinstance(reflexion_result, dict):
        return False
    source_detection = source_result["trials"][0]["detections"]["neutral"]
    reused_detection = reflexion_result["trials"][0]["detections"]["neutral"]
    source_calls = source_detection.get("calls")
    source_hashes = (
        [
            call.get("prompt_sha256")
            for call in source_calls
            if isinstance(call, dict)
            and isinstance(call.get("prompt_sha256"), str)
        ]
        if isinstance(source_calls, list)
        else []
    )
    expected_hash = source_hashes[-1] if source_hashes else None
    return bool(
        reused_detection.get("reused") is True
        and reused_detection.get("reused_from_pipeline") == source_method
        and reused_detection.get("source_prompt_sha256") == expected_hash
        and reused_detection.get("prediction") == source_detection.get("prediction")
    )


def _method_result_set_complete(
    method_results: dict[str, Any],
    method: str,
) -> bool:
    return bool(
        _method_is_complete(method_results.get(method), method)
        and (
            method not in PAIRED_BASE_METHOD
            or _paired_checkpoint_matches(method_results, method)
        )
    )


def _run_one_method(
    method: str,
    pipeline: TrainingFreeAdSent,
    views: dict[str, Any],
    record: PropaNewsRecord,
    *,
    initial_detection: dict[str, Any] | None = None,
    initial_detection_source: str | None = None,
) -> dict[str, object]:
    spec = PIPELINE_SPECS[method]
    reference = record.prompt.reference_context if spec.use_gt else None
    if spec.use_gt and not (reference or "").strip():
        raise ValueError(f"pipeline {method!r} requires a non-empty GT reference")
    function = (
        pipeline.run_controlled_reflexion
        if spec.use_reflexion
        else pipeline.run_controlled_adsent
    )
    started = _utc_now()
    try:
        kwargs: dict[str, object] = {
            "condition": spec.condition,
            "reference_context": reference,
            "success_predicate": (
                lambda prediction: prediction == record.gold.label
            ),
            "max_trials": MAX_ORACLE_ATTEMPTS,
        }
        if spec.use_reflexion:
            expected_source = PAIRED_BASE_METHOD[method]
            if initial_detection is None or initial_detection_source != expected_source:
                raise RuntimeError(
                    f"pipeline {method!r} requires the completed paired trial-1 "
                    f"detection from {expected_source!r}"
                )
            kwargs["initial_detection"] = initial_detection
            kwargs["initial_detection_source"] = initial_detection_source
        result = function(views, **kwargs)
        result["base_algorithm"] = result.get("method")
        result["method"] = method
        result["pipeline"] = method
        result["inference_protocol"] = ORACLE_RETRY_PROTOCOL
    except Exception as exc:
        return {
            "status": "error",
            "started_at": started,
            "finished_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "calls": list(getattr(exc, "call_records", [])),
        }
    return {
        "status": "ok",
        "started_at": started,
        "finished_at": _utc_now(),
        "result": result,
    }


def _paired_initial_detection(
    method_results: dict[str, Any],
    reflexion_method: str,
) -> tuple[dict[str, Any] | None, str | None]:
    source_method = PAIRED_BASE_METHOD.get(reflexion_method)
    if source_method is None:
        return None, None
    wrapper = method_results.get(source_method)
    if not isinstance(wrapper, dict) or not _method_is_complete(
        wrapper, source_method
    ):
        return None, None
    result = wrapper.get("result")
    if not isinstance(result, dict):
        return None, None
    trials = result.get("trials", [])
    if not trials:
        return None, None
    detection = trials[0].get("detections", {}).get("neutral")
    return (detection, source_method) if isinstance(detection, dict) else (None, None)


def _article_hash(record: PropaNewsRecord) -> str:
    return hashlib.sha256(record.prompt.article.encode("utf-8")).hexdigest()


def _prediction_event(
    record: PropaNewsRecord,
    *,
    run_fingerprint: object,
    experiment_design: str,
    views: dict[str, Any],
    method_results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "event_time": _utc_now(),
        "example_id": record.index,
        "experiment_design": experiment_design,
        "artifact_visibility": "private_oracle_gated_evaluation",
        "article_sha256": _article_hash(record),
        "article_char_count": len(record.prompt.article),
        "article_word_count": len(record.prompt.article.split()),
        "views": serialize_views(views),
        "method_results": method_results,
    }


def _event_has_reusable_views(
    event: dict[str, Any] | None,
    record: PropaNewsRecord,
    required_views: tuple[str, ...],
) -> bool:
    if not event or event.get("article_sha256") != _article_hash(record):
        return False
    cached = event.get("views")
    if not isinstance(cached, dict) or not set(required_views).issubset(cached):
        return False
    for name in required_views:
        raw_view = cached.get(name)
        if not isinstance(raw_view, dict):
            return False
        if raw_view.get("valid") is False and _contains_call_error(raw_view):
            return False
    return True


def _contains_call_error(value: object) -> bool:
    if isinstance(value, dict):
        if value.get("parse_status") == "call_error":
            return True
        return any(_contains_call_error(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_call_error(child) for child in value)
    return False


def _validate_resume_events(
    events: dict[int, dict[str, Any]],
    records: tuple[PropaNewsRecord, ...],
    *,
    run_fingerprint: str,
    experiment_design: str,
) -> None:
    """Reject cached events that do not belong to this exact run and article."""

    expected = {record.index: record for record in records}
    unexpected = sorted(set(events) - set(expected))
    if unexpected:
        raise RuntimeError(
            "predictions JSONL contains example IDs outside this run cohort: "
            f"{unexpected}"
        )
    for index, event in events.items():
        problems: list[str] = []
        if event.get("schema_version") != SCHEMA_VERSION:
            problems.append("schema_version")
        if event.get("run_fingerprint") != run_fingerprint:
            problems.append("run_fingerprint")
        if event.get("experiment_design") != experiment_design:
            problems.append("experiment_design")
        if event.get("article_sha256") != _article_hash(expected[index]):
            problems.append("article_sha256")
        if not isinstance(event.get("method_results"), dict):
            problems.append("method_results")
        else:
            for method, wrapper in event["method_results"].items():
                if not isinstance(wrapper, dict):
                    continue
                result = wrapper.get("result")
                if not isinstance(result, dict):
                    continue
                trials = result.get("trials")
                if not isinstance(trials, list):
                    continue
                for trial in trials:
                    if not isinstance(trial, dict):
                        continue
                    detection = (
                        trial.get("detections", {}).get("neutral", {})
                        if isinstance(trial.get("detections"), dict)
                        else {}
                    )
                    parsed = (
                        detection.get("prediction")
                        if isinstance(detection, dict)
                        else None
                    )
                    label = parsed.get("label") if isinstance(parsed, dict) else None
                    expected_correct = label == expected[index].gold.label
                    if trial.get("oracle_correct") is not expected_correct:
                        problems.append(f"oracle_correct:{method}")
                        break
        if problems:
            raise RuntimeError(
                f"cached event for example {index} does not match this run "
                f"({', '.join(problems)}); choose a new --run-name or remove the "
                "foreign checkpoint"
            )


def _extract_final_prediction(event: dict[str, Any], method: str) -> str | None:
    method_results = event.get("method_results", {})
    if not isinstance(method_results, dict) or not _method_result_set_complete(
        method_results, method
    ):
        return None
    wrapper = method_results[method]
    prediction = wrapper["result"].get("final_prediction")
    return prediction if prediction in {"human_original", "ai_manipulated"} else None


def _extract_view_predictions(
    event: dict[str, Any], method: str, *, last_trial: bool = True
) -> dict[str, str | None]:
    wrapper = event.get("method_results", {}).get(method)
    if not _method_is_complete(wrapper, method):
        return {}
    trials = wrapper["result"].get("trials", [])
    if not trials:
        return {}
    trial = trials[-1] if last_trial else trials[0]
    predictions: dict[str, str | None] = {}
    for view, detection in trial.get("detections", {}).items():
        parsed = detection.get("prediction")
        label = parsed.get("label") if isinstance(parsed, dict) else None
        predictions[view] = (
            label if label in {"human_original", "ai_manipulated"} else None
        )
    return predictions


def _metrics_to_dict(value: object) -> object:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return _jsonable(value)


def _count_call_records(value: object) -> int:
    return _call_usage_summary(value)["logical_calls"]


def _call_usage_summary(value: object) -> dict[str, int]:
    count = 0
    transport_attempts = 0
    recorded_total_tokens = 0
    token_usage_records = 0

    def visit(node: object) -> None:
        nonlocal count, transport_attempts, recorded_total_tokens, token_usage_records
        if is_dataclass(node) and not isinstance(node, type):
            visit(asdict(node))
            return
        if isinstance(node, dict):
            if {
                "stage",
                "prompt_sha256",
                "latency_ms",
                "parse_status",
            }.issubset(node):
                count += 1
                metadata = node.get("backend_metadata")
                attempts = (
                    metadata.get("transport_attempts")
                    if isinstance(metadata, dict)
                    else None
                )
                transport_attempts += (
                    attempts if type(attempts) is int and attempts >= 1 else 1
                )
                usage = metadata.get("usage") if isinstance(metadata, dict) else None
                total_tokens = (
                    usage.get("total_token_count")
                    if isinstance(usage, dict)
                    else None
                )
                if type(total_tokens) is int and total_tokens >= 0:
                    recorded_total_tokens += total_tokens
                    token_usage_records += 1
                return
            for child in node.values():
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)
    return {
        "logical_calls": count,
        "transport_attempts": transport_attempts,
        "recorded_total_tokens": recorded_total_tokens,
        "token_usage_records": token_usage_records,
    }


def _build_metrics(
    records: tuple[PropaNewsRecord, ...],
    events: dict[int, dict[str, Any]],
    methods: tuple[str, ...],
) -> dict[str, object]:
    """Score the private-gold retry trajectory without exposing gold in prompts."""

    from adsent_propanews_metrics import compute_classification_metrics

    gold = [record.gold.label for record in records]
    missing_events = sorted(
        record.index for record in records if record.index not in events
    )
    failed_any_method = sorted(
        record.index
        for record in records
        if any(
            not _method_result_set_complete(
                events.get(record.index, {}).get("method_results", {}),
                method,
            )
            for method in methods
        )
    )
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_examples": len(records),
        "missing_event_examples": missing_events,
        "failed_any_method_examples": failed_any_method,
        "missing_or_failed_examples": failed_any_method,
        "methods": {},
    }

    view_quality: dict[str, object] = {}
    for view in ("original", "neutral"):
        cached_views = [
            events.get(record.index, {}).get("views", {}).get(view)
            for record in records
        ]
        requested_count = sum(isinstance(item, dict) for item in cached_views)
        valid_count = sum(
            isinstance(item, dict) and item.get("valid") is True
            for item in cached_views
        )
        attempts = [
            len(item.get("attempts", []))
            for item in cached_views
            if isinstance(item, dict)
        ]
        view_quality[view] = {
            "requested_count": requested_count,
            "valid_count": valid_count,
            "valid_rate": valid_count / requested_count if requested_count else None,
            "average_rewrite_attempts": (
                sum(attempts) / len(attempts) if attempts else None
            ),
        }
    report["view_quality"] = view_quality
    shared_usage = [
        _call_usage_summary(events.get(record.index, {}).get("views", {}))
        for record in records
    ]
    average_shared_logical_calls = sum(
        usage["logical_calls"] for usage in shared_usage
    ) / len(records)
    report["average_shared_view_generation_calls"] = average_shared_logical_calls
    report["shared_view_generation_usage"] = {
        "average_logical_calls": average_shared_logical_calls,
        "average_transport_attempts": sum(
            usage["transport_attempts"] for usage in shared_usage
        )
        / len(records),
        "average_recorded_total_tokens": sum(
            usage["recorded_total_tokens"] for usage in shared_usage
        )
        / len(records),
        "token_usage_record_coverage": (
            sum(usage["token_usage_records"] for usage in shared_usage)
            / sum(usage["logical_calls"] for usage in shared_usage)
            if sum(usage["logical_calls"] for usage in shared_usage)
            else 0.0
        ),
    }

    method_reports: dict[str, object] = {}
    for method in methods:
        predictions = [
            _extract_final_prediction(events.get(record.index, {}), method)
            for record in records
        ]
        metrics = compute_classification_metrics(gold, predictions)
        failed_or_missing = [
            record.index
            for record in records
            if not _method_result_set_complete(
                events.get(record.index, {}).get("method_results", {}),
                method,
            )
        ]
        method_usage = [
            _call_usage_summary(
                events.get(record.index, {})
                .get("method_results", {})
                .get(method, {})
            )
            for record in records
        ]
        average_method_logical_calls = sum(
            usage["logical_calls"] for usage in method_usage
        ) / len(records)
        final_metric_payload = _metrics_to_dict(metrics)
        method_report: dict[str, object] = {
            "final": final_metric_payload,
            "oracle_stopped_final": final_metric_payload,
            "final_metric_scope": "oracle_stopped_final_secondary",
            "primary_metric_scope": "trial1",
            "failed_or_missing_examples": failed_or_missing,
            "average_method_llm_calls": average_method_logical_calls,
            "usage": {
                "average_logical_calls": average_method_logical_calls,
                "average_transport_attempts": sum(
                    usage["transport_attempts"] for usage in method_usage
                )
                / len(records),
                "average_recorded_total_tokens": sum(
                    usage["recorded_total_tokens"] for usage in method_usage
                )
                / len(records),
                "token_usage_record_coverage": (
                    sum(usage["token_usage_records"] for usage in method_usage)
                    / sum(usage["logical_calls"] for usage in method_usage)
                    if sum(usage["logical_calls"] for usage in method_usage)
                    else 0.0
                ),
            },
        }
        first_predictions: list[str | None] = []
        checkpoint_predictions: dict[int, list[str | None]] = {
            attempt: [] for attempt in range(1, MAX_ORACLE_ATTEMPTS + 1)
        }
        attempted_count_by_trial: Counter[int] = Counter()
        attempt_count_distribution: Counter[int] = Counter()
        first_correct_counts: Counter[int] = Counter()
        stop_reasons: Counter[str] = Counter()
        resolved_ids: list[int] = []
        unresolved_ids: list[int] = []
        total_attempts = 0
        reused_trial1_count = 0
        paired_trial1_eligible = 0
        paired_trial1_agreement = 0
        reflection_count = 0

        for record in records:
            event = events.get(record.index, {})
            method_results = event.get("method_results", {})
            wrapper = method_results.get(method, {})
            result = (
                wrapper.get("result", {})
                if _method_result_set_complete(method_results, method)
                else {}
            )
            trials = result.get("trials", [])
            labels: list[str | None] = []
            for trial in trials:
                detection = trial.get("detections", {}).get("neutral", {})
                parsed = detection.get("prediction") if isinstance(detection, dict) else None
                label = parsed.get("label") if isinstance(parsed, dict) else None
                labels.append(
                    label if label in {"human_original", "ai_manipulated"} else None
                )
                expected_correct = labels[-1] == record.gold.label
                if trial.get("oracle_correct") is not expected_correct:
                    raise RuntimeError(
                        f"oracle correctness mismatch for example {record.index}, "
                        f"method {method}, trial {trial.get('trial')}"
                    )
                if trial.get("reflection") is not None:
                    reflection_count += 1

            total_attempts += len(labels)
            attempt_count_distribution[len(labels)] += 1
            for attempt in range(1, len(labels) + 1):
                attempted_count_by_trial[attempt] += 1
            first = labels[0] if labels else None
            first_predictions.append(first)
            first_correct = next(
                (
                    position
                    for position, label in enumerate(labels, start=1)
                    if label == record.gold.label
                ),
                None,
            )
            if first_correct is None:
                unresolved_ids.append(record.index)
            else:
                first_correct_counts[first_correct] += 1
                resolved_ids.append(record.index)

            for attempt in range(1, MAX_ORACLE_ATTEMPTS + 1):
                if first_correct is not None and first_correct <= attempt:
                    checkpoint = labels[first_correct - 1]
                elif len(labels) >= attempt:
                    checkpoint = labels[attempt - 1]
                else:
                    checkpoint = None
                checkpoint_predictions[attempt].append(checkpoint)

            stop_reason = result.get("stop_reason")
            if isinstance(stop_reason, str):
                stop_reasons[stop_reason] += 1
            if result.get("initial_detection_reused") is True:
                reused_trial1_count += 1

            if method in PAIRED_BASE_METHOD:
                source_method = PAIRED_BASE_METHOD[method]
                source_wrapper = method_results.get(source_method, {})
                if _method_is_complete(source_wrapper, source_method):
                    source_trials = source_wrapper["result"].get("trials", [])
                    source_first = None
                    if source_trials:
                        source_parsed = (
                            source_trials[0]
                            .get("detections", {})
                            .get("neutral", {})
                            .get("prediction")
                        )
                        if isinstance(source_parsed, dict):
                            source_first = source_parsed.get("label")
                    if (
                        first in {"human_original", "ai_manipulated"}
                        and source_first in {"human_original", "ai_manipulated"}
                    ):
                        paired_trial1_eligible += 1
                        if first == source_first:
                            paired_trial1_agreement += 1

        trial1_metrics = compute_classification_metrics(gold, first_predictions)
        checkpoint_metrics = {
            str(attempt): _metrics_to_dict(
                compute_classification_metrics(
                    gold,
                    checkpoint_predictions[attempt],
                )
            )
            for attempt in range(1, MAX_ORACLE_ATTEMPTS + 1)
        }
        success_at_k = {
            str(attempt): sum(
                count
                for first_attempt, count in first_correct_counts.items()
                if first_attempt <= attempt
            )
            / len(records)
            for attempt in range(1, MAX_ORACLE_ATTEMPTS + 1)
        }
        method_report["trial1"] = _metrics_to_dict(trial1_metrics)
        method_report["first_trial"] = method_report["trial1"]
        method_report["oracle_retry"] = {
            "protocol": ORACLE_RETRY_PROTOCOL,
            "max_attempts": MAX_ORACLE_ATTEMPTS,
            "success_at_k": success_at_k,
            "success_within_k": dict(success_at_k),
            "checkpoint_at_k": checkpoint_metrics,
            "attempted_count_by_trial": {
                str(attempt): attempted_count_by_trial[attempt]
                for attempt in range(1, MAX_ORACLE_ATTEMPTS + 1)
            },
            "attempt_count_distribution": {
                str(attempt): attempt_count_distribution[attempt]
                for attempt in range(0, MAX_ORACLE_ATTEMPTS + 1)
            },
            "first_correct_attempt_counts": {
                **{
                    str(attempt): first_correct_counts[attempt]
                    for attempt in range(1, MAX_ORACLE_ATTEMPTS + 1)
                },
                "unresolved": len(unresolved_ids),
            },
            "total_attempts": total_attempts,
            "average_attempts": total_attempts / len(records),
            "resolved_count": len(resolved_ids),
            "resolved_rate": len(resolved_ids) / len(records),
            "unresolved_count": len(unresolved_ids),
            "unresolved_rate": len(unresolved_ids) / len(records),
            "unresolved_example_ids": unresolved_ids,
            "strict_denominator": len(records),
            "strict_denominator_includes_failed_or_missing": True,
            "stop_reason_counts": dict(sorted(stop_reasons.items())),
            "oracle_label_in_prompt": False,
            "oracle_correctness_message_in_prompt": False,
            "private_artifact_warning": (
                "retry timing plus predictions can reveal binary gold labels"
            ),
        }
        if PIPELINE_SPECS[method].use_reflexion:
            method_report["reflexion"] = {
                "feedback_mode": CONTROLLED_REFLEXION_FEEDBACK_MODE,
                "reflection_count": reflection_count,
                "paired_trial1_reuse_count": reused_trial1_count,
                "paired_trial1_reuse_rate": reused_trial1_count / len(records),
                "paired_trial1_eligible_count": paired_trial1_eligible,
                "paired_trial1_agreement_count": paired_trial1_agreement,
                "paired_trial1_agreement_rate": (
                    paired_trial1_agreement / paired_trial1_eligible
                    if paired_trial1_eligible
                    else None
                ),
            }

        if PIPELINE_SPECS[method].use_gt:
            retrieval_groups: dict[str, object] = {}
            for status in ("FULL", "PARTIAL"):
                positions = [
                    position
                    for position, record in enumerate(records)
                    if record.metadata.retrieval_status == status
                ]
                retrieval_groups[status] = {
                    "support": len(positions),
                    "trial1": (
                        _metrics_to_dict(
                            compute_classification_metrics(
                                [gold[position] for position in positions],
                                [first_predictions[position] for position in positions],
                            )
                        )
                        if positions
                        else None
                    ),
                    "oracle_stopped_final": (
                        _metrics_to_dict(
                            compute_classification_metrics(
                                [gold[position] for position in positions],
                                [predictions[position] for position in positions],
                            )
                        )
                        if positions
                        else None
                    ),
                }
            method_report["reference_retrieval_status"] = retrieval_groups

        subtype_recall: dict[str, object] = {}
        for subtype in ("app_auth_final", "sent_fake", "ll_filled"):
            positions = [
                position
                for position, record in enumerate(records)
                if record.metadata.manipulation_subtype == subtype
            ]
            correct = sum(predictions[position] == "ai_manipulated" for position in positions)
            subtype_recall[subtype] = {
                "support": len(positions),
                "trial1_ai_manipulated_recall": (
                    sum(
                        first_predictions[position] == "ai_manipulated"
                        for position in positions
                    )
                    / len(positions)
                    if positions
                    else None
                ),
                "oracle_stopped_ai_manipulated_recall": (
                    correct / len(positions) if positions else None
                ),
            }
        method_report["path_derived_subtype_recall"] = subtype_recall
        method_reports[method] = method_report

    report["methods"] = method_reports

    def factorial_effects(
        scores: dict[str, dict[str, Any]],
    ) -> dict[str, object]:
        effects: dict[str, object] = {}

        def add_delta(name: str, left: str, right: str) -> None:
            if left not in scores or right not in scores:
                return
            effects[name] = {
                "left": left,
                "right": right,
                "accuracy_delta": (
                    scores[right]["accuracy"] - scores[left]["accuracy"]
                ),
                "macro_f1_delta": (
                    scores[right]["macro_f1"] - scores[left]["macro_f1"]
                ),
            }

        add_delta("gt_effect_without_reflexion", "adsent", "adsent_gt")
        add_delta("reflexion_effect_without_gt", "adsent", "adsent_reflexion")
        add_delta(
            "reflexion_effect_with_gt", "adsent_gt", "adsent_gt_reflexion"
        )
        add_delta(
            "gt_effect_with_reflexion",
            "adsent_reflexion",
            "adsent_gt_reflexion",
        )
        if set(METHODS).issubset(scores):
            base = scores["adsent"]
            gt = scores["adsent_gt"]
            reflexion = scores["adsent_reflexion"]
            both = scores["adsent_gt_reflexion"]
            effects["gt_x_reflexion_interaction"] = {
                "accuracy": (
                    both["accuracy"]
                    - gt["accuracy"]
                    - reflexion["accuracy"]
                    + base["accuracy"]
                ),
                "macro_f1": (
                    both["macro_f1"]
                    - gt["macro_f1"]
                    - reflexion["macro_f1"]
                    + base["macro_f1"]
                ),
            }
        return effects

    trial1_scores = {
        method: method_reports[method]["trial1"]  # type: ignore[index]
        for method in methods
    }
    final_scores = {
        method: method_reports[method]["final"]  # type: ignore[index]
        for method in methods
    }
    checkpoint_effects = {
        str(attempt): factorial_effects(
            {
                method: method_reports[method]["oracle_retry"][  # type: ignore[index]
                    "checkpoint_at_k"
                ][str(attempt)]
                for method in methods
            }
        )
        for attempt in range(1, MAX_ORACLE_ATTEMPTS + 1)
    }
    report["comparisons"] = {
        "trial1": factorial_effects(trial1_scores),
        "oracle_stopped_final": factorial_effects(final_scores),
        "checkpoint_at_k": checkpoint_effects,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled 2x2 AdSent experiment (GT context x Reflexion) "
            "on PropaNews without modifying existing experiment files."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--gt-data", type=Path, default=DEFAULT_REFERENCES_PATH)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument("--indices", help="comma-separated dataset indices")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_COHORT_SIZE,
        help=(
            f"evaluate at most {DEFAULT_COHORT_SIZE} records; the default full "
            "experiment uses indices 0..99"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-name",
        default=None,
        help="stable name enables exact-config resume; default is a UTC timestamp",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--validate-only", action="store_true")

    parser.add_argument(
        "--llm-backend",
        choices=("existing", "vertex"),
        default="existing",
    )
    parser.add_argument("--llm-callable", default="llm:call_llm")
    parser.add_argument("--vertex-model", default="gemini-2.5-flash")
    parser.add_argument("--vertex-project")
    parser.add_argument("--vertex-location", default="global")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Vertex backend only; imported callables control their own decoding",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Vertex backend only",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="optional Vertex output cap; leave unset to avoid silent article truncation",
    )
    parser.add_argument("--model-retries", type=int, default=10)

    parser.add_argument("--max-rewrite-attempts", type=int, default=2)
    parser.add_argument("--max-format-retries", type=int, default=1)
    parser.add_argument(
        "--max-reflexion-trials",
        type=int,
        choices=(MAX_ORACLE_ATTEMPTS,),
        default=MAX_ORACLE_ATTEMPTS,
        help="fixed at 5 for the per-example oracle retry protocol",
    )
    parser.add_argument("--reflection-memory-size", type=int, default=3)
    parser.add_argument(
        "--require-target-sentiment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.methods) != len(set(args.methods)):
        parser.error("--methods must not contain duplicates")
    try:
        _validate_method_dependencies(tuple(args.methods))
    except ValueError as exc:
        parser.error(str(exc))

    try:
        load_mode = _dataset_load_mode(tuple(args.methods))
        all_records = load_propanews_dataset(
            args.data,
            args.gt_data,
            mode=load_mode,
        )
        selected = _select_records(
            all_records,
            raw_indices=args.indices,
            limit=args.limit,
        )
    except Exception as exc:
        parser.error(str(exc))

    integrity = validate_records(selected)
    if (
        args.llm_backend == "vertex"
        and not args.vertex_project
        and not os.environ.get("GOOGLE_CLOUD_PROJECT")
    ):
        parser.error(
            "vertex backend requires --vertex-project or GOOGLE_CLOUD_PROJECT"
        )
    try:
        pipeline_config = PipelineConfig(
            max_rewrite_attempts=args.max_rewrite_attempts,
            max_format_retries=args.max_format_retries,
            max_reflexion_trials=args.max_reflexion_trials,
            reflection_memory_size=args.reflection_memory_size,
            require_target_sentiment=args.require_target_sentiment,
        )
        llm, llm_provenance = _build_llm(args)
    except (TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    if args.validate_only:
        print(
            json.dumps(
                {
                    "dataset": integrity.to_dict(),
                    "selected_indices": [record.index for record in selected],
                    "methods": args.methods,
                    "experiment_design": _experiment_design(
                        tuple(args.methods), len(selected)
                    ),
                    "inference_protocol": {
                        "name": ORACLE_RETRY_PROTOCOL,
                        "max_attempts": MAX_ORACLE_ATTEMPTS,
                        "gold_used_for": "external_retry_gate_only",
                        "oracle_label_in_prompt": False,
                        "oracle_correctness_message_in_prompt": False,
                    },
                    "pipeline_specs": {
                        method: asdict(PIPELINE_SPECS[method])
                        for method in args.methods
                    },
                    "requested_views": list(_required_views(tuple(args.methods))),
                    "pipeline": asdict(pipeline_config),
                    "llm": llm_provenance,
                    "logical_call_estimate_per_example": _estimate_logical_calls(
                        tuple(args.methods),
                        pipeline_config,
                    ),
                    "cohort_text_stats": _cohort_text_stats(selected),
                    "model_calls_made": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.run_name is None:
        args.run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_name):
        parser.error("--run-name may contain only letters, digits, dot, underscore, hyphen")

    pipeline = TrainingFreeAdSent(llm, pipeline_config)  # type: ignore[arg-type]
    run_dir = args.output_root.resolve() / args.run_name
    config_path = run_dir / "config.json"
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    config = _config_payload(
        args,
        records=selected,
        llm_provenance=llm_provenance,
        pipeline_config=pipeline_config,
    )
    experiment_design = str(config["experiment_design"])
    _ensure_run_config(config_path, config, resume=args.resume)

    latest = _read_latest_events(predictions_path)
    _validate_resume_events(
        latest,
        selected,
        run_fingerprint=str(config["fingerprint"]),
        experiment_design=experiment_design,
    )
    requested_views = _required_views(tuple(args.methods))
    failures = 0

    for position, record in enumerate(selected, start=1):
        previous = latest.get(record.index)
        previous_methods = (
            dict(previous.get("method_results", {})) if previous else {}
        )
        missing_methods = [
            method
            for method in METHODS
            if method in args.methods
            if not _method_result_set_complete(previous_methods, method)
        ]
        if not missing_methods:
            print(f"[{position}/{len(selected)}] index={record.index} resume-skip")
            continue

        print(
            f"[{position}/{len(selected)}] index={record.index} "
            f"methods={','.join(missing_methods)}"
        )
        views: dict[str, Any] | None = None
        method_results = previous_methods
        try:
            if _event_has_reusable_views(previous, record, requested_views):
                if previous is None or not isinstance(previous.get("views"), dict):
                    raise RuntimeError("reusable view checkpoint is malformed")
                views = deserialize_views(previous["views"])
            else:
                views = pipeline.prepare_views(record.prompt.article, requested_views)

            event = _prediction_event(
                record,
                run_fingerprint=config["fingerprint"],
                experiment_design=experiment_design,
                views=views,
                method_results=method_results,
            )
            _append_jsonl(predictions_path, event)
            latest[record.index] = event

            for method in missing_methods:
                initial_detection, initial_source = _paired_initial_detection(
                    method_results,
                    method,
                )
                method_results[method] = _run_one_method(
                    method,
                    pipeline,
                    views,
                    record,
                    initial_detection=initial_detection,
                    initial_detection_source=initial_source,
                )
                event = _prediction_event(
                    record,
                    run_fingerprint=config["fingerprint"],
                    experiment_design=experiment_design,
                    views=views,
                    method_results=method_results,
                )
                _append_jsonl(predictions_path, event)
                latest[record.index] = event
                if method_results[method]["status"] != "ok":
                    failures += 1
                    print(
                        f"  method={method} error: {method_results[method]['error']}",
                        file=sys.stderr,
                    )
                    if args.fail_fast:
                        raise RuntimeError(str(method_results[method]["error"]))
        except Exception as exc:
            failures += 1
            if views is not None:
                cached_views: object = serialize_views(views)
            elif previous and isinstance(previous.get("views"), dict):
                cached_views = previous["views"]
            else:
                cached_views = {}
            error_event = {
                "schema_version": SCHEMA_VERSION,
                "run_fingerprint": config["fingerprint"],
                "event_time": _utc_now(),
                "example_id": record.index,
                "experiment_design": experiment_design,
                "artifact_visibility": "private_oracle_gated_evaluation",
                "article_sha256": _article_hash(record),
                "article_char_count": len(record.prompt.article),
                "article_word_count": len(record.prompt.article.split()),
                "views": cached_views,
                "status": "example_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "method_results": method_results,
            }
            _append_jsonl(predictions_path, error_event)
            latest[record.index] = error_event
            print(f"  example error: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise

    metrics = _build_metrics(selected, latest, tuple(args.methods))
    metrics["generated_at"] = _utc_now()
    metrics["run_fingerprint"] = config["fingerprint"]
    metrics["experiment_design"] = experiment_design
    method_payloads = metrics.get("methods")
    if not isinstance(method_payloads, dict):
        raise RuntimeError("generated metrics are missing the methods object")
    total_method_failures = 0
    for method_report in method_payloads.values():
        if not isinstance(method_report, dict) or not isinstance(
            method_report.get("failed_or_missing_examples"), list
        ):
            raise RuntimeError("generated method metrics have an invalid schema")
        total_method_failures += len(method_report["failed_or_missing_examples"])
    metrics["method_failures"] = total_method_failures
    metrics["example_level_failures"] = sum(
        1
        for events_status in latest.values()
        if events_status.get("status") == "example_error"
    )
    _atomic_write_json(metrics_path, metrics)
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    return 1 if total_method_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
