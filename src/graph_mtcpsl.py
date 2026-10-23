"""Run the two article-only SPRS-MTCPSL PropaNews pipelines.

Evaluation is article-major. Each article owns a fresh Reflexion memory, and a
validated progress file is atomically replaced after every completed article so
an interrupted full-dataset run can skip the completed prefix.

A recognised local actor/reflection OOM is saved as an ``oom_skipped`` article
before execution continues with the next index. Other exceptions still abort.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .propanews_common import (
        ARTICLE_TEXT_FIELD,
        DATASET_ID,
        is_oom_error,
        recover_from_oom,
    )
except ImportError:  # Direct execution with ``PYTHONPATH=src``.
    from propanews_common import (
        ARTICLE_TEXT_FIELD,
        DATASET_ID,
        is_oom_error,
        recover_from_oom,
    )


# ``None`` means the complete dataset in its file order. ``--num`` remains a
# deterministic first-N diagnostic option; it never samples or shuffles.
NUM: int | None = None
MAX_TRIALS = 3
MAX_REFLECTIONS = 3
PROMPT_VERSION = "sprs-mtcpsl-propanews-clean-v3-prior-v4"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "propanews_sprs_mtcpsl_qwen8"
AGGREGATE_METRICS_FILENAME = "aggregate_metrics.json"
PLOT_FILENAME = "propanews_sprs_mtcpsl_cumulative_success.png"

RESULT_SCHEMA_VERSION = 3
EVALUATION_MODE = "article_major_failure_retry_cumulative"
_OOM_ERROR_MAX_LENGTH = 500

_REFLECTION_FINISH_PATTERN = re.compile(
    r"Finish\s*\[",
    flags=re.IGNORECASE,
)
_REFLECTION_LABEL_PATTERN = re.compile(
    r"\b(?:real|fake)\b",
    flags=re.IGNORECASE,
)


def _default_generation_metadata() -> Mapping[str, object]:
    """Read model/cache identity from the sole non-PCoT LLM adapter."""

    prefix = f"{__package__}." if __package__ else ""
    module = importlib.import_module(f"{prefix}llm")
    return module.generation_metadata()


def _normalise_generation_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise TypeError("generation metadata must be a mapping")
    normalised = dict(metadata)
    if not normalised:
        raise ValueError("generation metadata must not be empty")
    if any(not isinstance(key, str) for key in normalised):
        raise TypeError("generation metadata keys must be strings")
    try:
        json.dumps(normalised, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise TypeError("generation metadata must be JSON serializable") from error
    return normalised


@dataclass(frozen=True)
class PipelineSpec:
    key: str
    label: str
    runner_name: str
    use_reflection: bool
    use_gt: bool


PIPELINES: tuple[PipelineSpec, ...] = (
    PipelineSpec("sprs", "SPRS", "run_SPRS", False, False),

    PipelineSpec(
        "sprs_gt_reflexion",
        "SPRS + GT + Reflexion",
        "run_SPRS_GT_Re",
        True,
        True,
    ),

)

@dataclass(frozen=True)
class ExperimentDependencies:
    """Injected boundaries for deterministic orchestration tests."""

    run_SPRS: Callable[[Any], str]
    run_SPRS_Re: Callable[[Any, list[str]], str]
    run_SPRS_GT: Callable[[Any], str]
    run_SPRS_GT_Re: Callable[[Any, list[str]], str]
    get_article: Callable[[Any], str]
    get_available_indices: Callable[[], Sequence[Any]]
    extract_prediction: Callable[[str], str]
    get_gold_label: Callable[[Any], Any]
    EM_score: Callable[[str, Any], bool]
    make_reflection_SPRS: Callable[[str, str], str]
    generation_metadata: Callable[[], Mapping[str, object]] = (
        _default_generation_metadata
    )


def load_default_dependencies() -> ExperimentDependencies:
    """Import the non-compatibility modules only when an experiment runs."""

    prefix = f"{__package__}." if __package__ else ""
    main_module = importlib.import_module(f"{prefix}main_mtcpsl")
    actor_module = importlib.import_module(f"{prefix}actor_mtcpsl")
    evaluator_module = importlib.import_module(f"{prefix}evaluator_mtcpsl")
    reflection_module = importlib.import_module(f"{prefix}reflection_mtcpsl")
    llm_module = importlib.import_module(f"{prefix}llm")
    if actor_module.PROMPT_VERSION != PROMPT_VERSION:
        raise RuntimeError("SPRS actor and graph prompt versions do not match")

    return ExperimentDependencies(
    run_SPRS=main_module.run_SPRS,
    run_SPRS_Re=main_module.run_SPRS_Re,
    run_SPRS_GT=main_module.run_SPRS_GT,
    run_SPRS_GT_Re=main_module.run_SPRS_GT_Re,

    get_article=actor_module.get_article,
    get_available_indices=actor_module.get_available_indices,
    extract_prediction=evaluator_module.extract_prediction,
    get_gold_label=evaluator_module.get_gold_label,
    EM_score=evaluator_module.EM_score,
    make_reflection_SPRS=reflection_module.make_reflection_SPRS,
    generation_metadata=llm_module.generation_metadata,
)


def _normalise_label(label: Any) -> str:
    """Canonicalize a private evaluation label without exposing it to actors."""

    if label is True:
        return "real"
    if label is False:
        return "fake"
    if isinstance(label, str):
        normalised = label.strip().casefold()
        if normalised in {"real", "fake"}:
            return normalised
    raise ValueError(f"unsupported PropaNews gold label: {label!r}")


def select_indices(
    available_indices: Sequence[Any],
    num: int | None,
) -> list[Any]:
    """Preserve file order and optionally select a deterministic first-N prefix."""

    indices = list(available_indices)
    if not indices:
        raise ValueError("get_available_indices() returned no indices")
    try:
        if len(indices) != len(set(indices)):
            raise ValueError("get_available_indices() returned duplicate indices")
    except TypeError as error:
        raise ValueError("available indices must be hashable") from error
    if num is None:
        return indices
    if num <= 0:
        raise ValueError("num must be greater than zero")
    if len(indices) < num:
        raise ValueError(
            f"requested {num} samples, but only {len(indices)} are available"
        )
    return indices[:num]


def build_result_config(
    spec: PipelineSpec,
    indices: Sequence[Any],
    max_trials: int,
    *,
    generation: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build the exact cache identity for one pipeline result."""

    if max_trials <= 0:
        raise ValueError("max_trials must be greater than zero")
    generation_values = _normalise_generation_metadata(
        generation if generation is not None else _default_generation_metadata()
    )
    selected_indices = list(indices)
    return {
        "pipeline_key": spec.key,
        "pipeline": spec.label,
        "use_reflection": spec.use_reflection,
        "use_gt": spec.use_gt,
        "num_samples": len(selected_indices),
        "max_trials": max_trials,
        "max_reflections": MAX_REFLECTIONS,
        "indices": selected_indices,
        "dataset": DATASET_ID,
        "article_field": ARTICLE_TEXT_FIELD,
        "prompt_version": PROMPT_VERSION,
        "generation_metadata": generation_values,
    }


def _runner_for(
    spec: PipelineSpec,
    dependencies: ExperimentDependencies,
) -> Callable[..., str]:
    return getattr(dependencies, spec.runner_name)


def _summaries(
    sample_results: Sequence[Mapping[str, Any]],
    *,
    num_samples: int,
    max_trials: int,
) -> tuple[list[int], list[float], list[dict[str, Any]]]:
    solved_at_trial = [0] * max_trials
    for sample in sample_results:
        success_trial = sample.get("success_trial")
        if sample.get("solved") is True and isinstance(success_trial, int):
            solved_at_trial[success_trial - 1] += 1

    cumulative_success: list[float] = []
    trial_summary: list[dict[str, Any]] = []
    total_solved = 0
    for offset, newly_solved in enumerate(solved_at_trial):
        total_solved += newly_solved
        rate = total_solved / num_samples
        cumulative_success.append(rate)
        trial_summary.append(
            {
                "trial": offset + 1,
                "newly_solved": newly_solved,
                "total_solved": total_solved,
                "remaining": num_samples - total_solved,
                "cumulative_success": rate,
            }
        )
    return solved_at_trial, cumulative_success, trial_summary


def _describe_oom(error: BaseException) -> str:
    """Return a short, single-line diagnostic suitable for a JSON checkpoint."""

    message = " ".join(str(error).split())
    description = type(error).__name__
    if message:
        description = f"{description}: {message}"
    return description[:_OOM_ERROR_MAX_LENGTH].rstrip()


def _build_result(
    spec: PipelineSpec,
    indices: Sequence[Any],
    max_trials: int,
    sample_results: Sequence[Mapping[str, Any]],
    *,
    generation: Mapping[str, object],
) -> dict[str, Any]:
    generation_values = _normalise_generation_metadata(generation)
    config = build_result_config(
        spec,
        indices,
        max_trials,
        generation=generation_values,
    )
    results = copy.deepcopy(list(sample_results))
    solved_at_trial, cumulative_success, trial_summary = _summaries(
        results,
        num_samples=len(indices),
        max_trials=max_trials,
    )
    completed_count = len(results)
    oom_skipped_indices = [
        sample["index"] for sample in results if sample.get("oom_skipped") is True
    ]
    complete = completed_count == len(indices)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_mode": EVALUATION_MODE,
        "status": "complete" if complete else "in_progress",
        "complete": complete,
        "completed_count": completed_count,
        "completed_indices": [sample["index"] for sample in results],
        "oom_skipped_count": len(oom_skipped_indices),
        "oom_skipped_indices": oom_skipped_indices,
        "config": config,
        "pipeline_key": spec.key,
        "pipeline": spec.label,
        "num_samples": len(indices),
        "max_trials": max_trials,
        "max_reflections": MAX_REFLECTIONS,
        "prompt_version": PROMPT_VERSION,
        "generation_metadata": generation_values,
        "solved_at_trial": solved_at_trial,
        "cumulative_success": cumulative_success,
        "trial_summary": trial_summary,
        "results": results,
    }


def _validate_result(
    result: Any,
    expected_config: Mapping[str, Any],
    *,
    require_complete: bool,
) -> tuple[bool, str]:
    if not isinstance(result, dict):
        return False, "result is not an object"
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        return False, "schema version mismatch"
    if result.get("evaluation_mode") != EVALUATION_MODE:
        return False, "evaluation mode mismatch"
    expected = dict(expected_config)
    if result.get("config") != expected:
        return False, "experiment configuration mismatch"

    for field in (
        "pipeline_key",
        "pipeline",
        "num_samples",
        "max_trials",
        "max_reflections",
        "prompt_version",
        "generation_metadata",
    ):
        if result.get(field) != expected.get(field):
            return False, f"{field} mismatch"

    indices = list(expected.get("indices", []))
    num_samples = expected.get("num_samples")
    max_trials = expected.get("max_trials")
    use_reflection = expected.get("use_reflection")
    use_gt = expected.get("use_gt")

    if not isinstance(use_gt, bool):
        return False, "invalid GT switch"
    if not isinstance(num_samples, int) or num_samples <= 0:
        return False, "invalid expected sample count"
    if len(indices) != num_samples:
        return False, "invalid expected index list"
    if not isinstance(max_trials, int) or max_trials <= 0:
        return False, "invalid expected trial count"
    if not isinstance(use_reflection, bool):
        return False, "invalid reflection switch"

    results = result.get("results")
    if not isinstance(results, list) or len(results) > num_samples:
        return False, "sample results are invalid"
    if require_complete and len(results) != num_samples:
        return False, "sample results are incomplete"

    for position, sample in enumerate(results):
        if not isinstance(sample, dict) or sample.get("index") != indices[position]:
            return False, "sample indices are not a file-order prefix"
        if sample.get("gold_label") not in {"real", "fake"}:
            return False, "sample gold label is invalid"
        if not isinstance(sample.get("solved"), bool):
            return False, "sample solved flag is invalid"
        if sample.get("success") is not sample.get("solved"):
            return False, "sample success alias mismatch"

        if sample.get("complete") is not True:
            return False, "sample is not marked complete"
        oom_skipped = sample.get("oom_skipped")
        if not isinstance(oom_skipped, bool):
            return False, "sample OOM skip flag is invalid"
        if oom_skipped:
            if sample.get("skip_reason") != "oom":
                return False, "OOM-skipped sample reason is invalid"
            if sample.get("oom_stage") not in {"actor", "reflection"}:
                return False, "OOM-skipped sample stage is invalid"
            oom_trial = sample.get("oom_trial")
            if type(oom_trial) is not int or not 1 <= oom_trial <= max_trials:
                return False, "OOM-skipped sample trial is invalid"
            oom_error = sample.get("oom_error")
            if (
                not isinstance(oom_error, str)
                or not oom_error.strip()
                or oom_error != oom_error.strip()
                or "\n" in oom_error
                or len(oom_error) > _OOM_ERROR_MAX_LENGTH
            ):
                return False, "OOM-skipped sample error is invalid"
            if sample.get("solved") is not False:
                return False, "OOM-skipped sample cannot be solved"
            if sample.get("success_trial") is not None:
                return False, "OOM-skipped sample has a success trial"
        elif any(
            sample.get(field) is not None
            for field in ("skip_reason", "oom_stage", "oom_trial", "oom_error")
        ):
            return False, "non-skipped sample contains OOM metadata"

        attempts = sample.get("attempts")
        minimum_attempts = 0 if oom_skipped else 1
        if (
            not isinstance(attempts, list)
            or not minimum_attempts <= len(attempts) <= max_trials
        ):
            return False, "attempt records are incomplete"
        if sample.get("attempt_count") != len(attempts):
            return False, "attempt count mismatch"

        correct_trials: list[int] = []
        expected_reflections: list[str] = []
        for trial, record in enumerate(attempts, start=1):
            if not isinstance(record, dict):
                return False, "attempt record is not an object"
            if record.get("index") != indices[position] or record.get("trial") != trial:
                return False, "attempt index or order mismatch"
            required = {
                "response",
                "prediction",
                "predicted_label",
                "gold_label",
                "correct",
                "outcome",
                "reflection",
                "reflection_used",
                "reflection_generated",
            }
            if not required.issubset(record):
                return False, "attempt record is missing detailed fields"
            if not isinstance(record.get("response"), str):
                return False, "attempt response is invalid"
            if record.get("predicted_label") not in {"real", "fake", ""}:
                return False, "attempt prediction is invalid"
            if record.get("prediction") != record.get("predicted_label"):
                return False, "prediction alias mismatch"
            finish_values = re.findall(
                r"Finish\[(.*?)\]",
                record["response"],
                flags=re.IGNORECASE | re.DOTALL,
            )
            parsed_prediction = (
                finish_values[-1].strip().casefold() if finish_values else ""
            )
            if parsed_prediction not in {"real", "fake"}:
                parsed_prediction = ""
            if record["predicted_label"] != parsed_prediction:
                return False, "prediction does not match stored response"
            if record.get("gold_label") != sample.get("gold_label"):
                return False, "attempt gold label mismatch"
            if not isinstance(record.get("correct"), bool):
                return False, "attempt correctness is invalid"
            expected_correct = bool(record["predicted_label"]) and (
                record["predicted_label"] == sample["gold_label"]
            )
            if record["correct"] is not expected_correct:
                return False, "attempt correctness does not match prediction and label"
            expected_outcome = "correct" if record["correct"] else "incorrect"
            if record.get("outcome") != expected_outcome:
                return False, "attempt outcome mismatch"

            used = record.get("reflection_used")
            if not isinstance(used, list) or len(used) > MAX_REFLECTIONS:
                return False, "reflection history is invalid"
            expected_used = expected_reflections if use_reflection else []
            if used != expected_used:
                return False, "attempt reflection history does not match"

            generated = record.get("reflection_generated")
            if record.get("reflection") != generated:
                return False, "reflection alias mismatch"
            if generated is not None and not isinstance(generated, str):
                return False, "generated reflection is not null or a string"
            if isinstance(generated, str) and (
                not generated.strip() or generated != generated.strip()
            ):
                return False, "generated reflection was not sanitised"
            if isinstance(generated, str) and (
                _REFLECTION_FINISH_PATTERN.search(generated)
                or _REFLECTION_LABEL_PATTERN.search(generated)
            ):
                return False, "generated reflection contains a classification token"
            if not use_reflection and generated is not None:
                return False, "non-Reflexion pipeline generated a reflection"
            if record["correct"] and generated is not None:
                return False, "successful attempt generated a reflection"
            if trial == max_trials and generated is not None:
                return False, "reflection was generated after the final attempt"
            if isinstance(generated, str):
                expected_reflections = (expected_reflections + [generated])[
                    -MAX_REFLECTIONS:
                ]
            if record["correct"]:
                correct_trials.append(trial)

        solved = sample["solved"]
        success_trial = sample.get("success_trial")
        if oom_skipped:
            if correct_trials:
                return False, "OOM-skipped sample contains a correct attempt"
            oom_trial = sample["oom_trial"]
            if sample["oom_stage"] == "actor" and len(attempts) != oom_trial - 1:
                return False, "actor OOM attempt boundary is invalid"
            if sample["oom_stage"] == "reflection":
                if not use_reflection:
                    return False, "non-Reflexion pipeline recorded reflection OOM"
                if oom_trial >= max_trials or len(attempts) != oom_trial:
                    return False, "reflection OOM attempt boundary is invalid"
                if not attempts or attempts[-1]["correct"] is not False:
                    return False, "reflection OOM must follow an incorrect attempt"
                if attempts[-1]["reflection_generated"] is not None:
                    return False, "reflection OOM unexpectedly generated reflection"
        elif solved:
            if type(success_trial) is not int:
                return False, "success trial is not an integer"
            if len(correct_trials) != 1 or correct_trials[0] != len(attempts):
                return False, "solved sample has an invalid stopping point"
            if success_trial != correct_trials[0]:
                return False, "success trial mismatch"
        elif correct_trials or success_trial is not None or len(attempts) != max_trials:
            return False, "unsolved sample has incomplete attempts"

    canonical = _build_result(
        PipelineSpec(
            key=str(expected["pipeline_key"]),
            label=str(expected["pipeline"]),
            runner_name="validation_only",
            use_reflection=bool(use_reflection),
            use_gt=bool(use_gt),
        ),
        indices,
        int(max_trials),
        results,
        generation=expected["generation_metadata"],
    )
    for field in (
        "status",
        "complete",
        "completed_count",
        "completed_indices",
        "oom_skipped_count",
        "oom_skipped_indices",
        "solved_at_trial",
        "cumulative_success",
        "trial_summary",
    ):
        if result.get(field) != canonical[field]:
            return False, f"{field} does not match completed records"
    if require_complete and result.get("complete") is not True:
        return False, "result is not marked complete"
    return True, "complete" if result.get("complete") else "valid progress"


def _upgrade_legacy_result(result: Any) -> Any:
    """Upgrade schema-v2 normal checkpoints so completed work remains resumable."""

    if not isinstance(result, dict) or result.get("schema_version") != 2:
        return result
    upgraded = copy.deepcopy(result)
    results = upgraded.get("results")
    if not isinstance(results, list):
        return result
    for sample in results:
        if not isinstance(sample, dict):
            return result
        sample.update(
            {
                "complete": True,
                "oom_skipped": False,
                "skip_reason": None,
                "oom_stage": None,
                "oom_trial": None,
                "oom_error": None,
            }
        )
    upgraded["schema_version"] = RESULT_SCHEMA_VERSION
    upgraded["oom_skipped_count"] = 0
    upgraded["oom_skipped_indices"] = []
    return upgraded


def validate_pipeline_result(
    result: Any,
    expected_config: Mapping[str, Any],
) -> tuple[bool, str]:
    """Validate a complete result and all retry/memory invariants."""

    return _validate_result(result, expected_config, require_complete=True)


def validate_progress_result(
    result: Any,
    expected_config: Mapping[str, Any],
) -> tuple[bool, str]:
    """Validate a resumable file-order prefix of completed articles."""

    return _validate_result(result, expected_config, require_complete=False)


def evaluate_pipeline(
    spec: PipelineSpec,
    indices: Sequence[Any],
    dependencies: ExperimentDependencies,
    *,
    max_trials: int = MAX_TRIALS,
    generation: Mapping[str, object] | None = None,
    initial_results: Sequence[Mapping[str, Any]] | None = None,
    checkpoint: Callable[[Mapping[str, Any]], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate in article-major order with per-article Reflexion isolation."""

    if max_trials <= 0:
        raise ValueError("max_trials must be greater than zero")
    selected_indices = list(indices)
    if not selected_indices:
        raise ValueError("indices must not be empty")
    try:
        if len(selected_indices) != len(set(selected_indices)):
            raise ValueError("indices must be unique")
    except TypeError as error:
        raise ValueError("indices must be hashable") from error

    generation_values = _normalise_generation_metadata(
        generation if generation is not None else dependencies.generation_metadata()
    )
    config = build_result_config(
        spec,
        selected_indices,
        max_trials,
        generation=generation_values,
    )
    sample_results = copy.deepcopy(list(initial_results or ()))
    initial_payload = _build_result(
        spec,
        selected_indices,
        max_trials,
        sample_results,
        generation=generation_values,
    )
    valid, reason = validate_progress_result(initial_payload, config)
    if not valid:
        raise ValueError(f"invalid initial results: {reason}")
    if checkpoint is not None:
        checkpoint(initial_payload)

    runner = _runner_for(spec, dependencies)
    for position in range(len(sample_results), len(selected_indices)):
        index = selected_indices[position]
        if progress is not None:
            progress(
                f"[{spec.label}] index={index} "
                f"({position + 1}/{len(selected_indices)})"
            )

        # Memory is local to this iteration and therefore cannot cross articles.
        reflections: list[str] = []
        gold_label = _normalise_label(dependencies.get_gold_label(index))
        attempts: list[dict[str, Any]] = []
        success_trial: int | None = None
        oom_skipped = False
        oom_stage: str | None = None
        oom_trial: int | None = None
        oom_error: str | None = None

        for trial in range(1, max_trials + 1):
            reflection_used = list(reflections) if spec.use_reflection else []
            try:
                response = (
                    runner(index, reflection_used)
                    if spec.use_reflection
                    else runner(index)
                )
            except Exception as error:
                if not is_oom_error(error):
                    raise
                recover_from_oom()
                oom_skipped = True
                oom_stage = "actor"
                oom_trial = trial
                oom_error = _describe_oom(error)
                if progress is not None:
                    progress(
                        f"[{spec.label}] OOM-SKIP index={index} "
                        f"trial={trial}/{max_trials} stage=actor"
                    )
                break

            predicted_label = dependencies.extract_prediction(response)
            predicted_label = (
                predicted_label.strip().casefold()
                if isinstance(predicted_label, str)
                else ""
            )
            if predicted_label not in {"real", "fake"}:
                predicted_label = ""

            correct = bool(dependencies.EM_score(response, index))
            reflection_generated: str | None = None
            if correct:
                success_trial = trial
            elif spec.use_reflection and trial < max_trials:
                try:
                    generated = dependencies.make_reflection_SPRS(
                        dependencies.get_article(index),
                        response,
                        use_gt=spec.use_gt,
                    )
                except Exception as error:
                    if not is_oom_error(error):
                        raise
                    recover_from_oom()
                    oom_skipped = True
                    oom_stage = "reflection"
                    oom_trial = trial
                    oom_error = _describe_oom(error)
                else:
                    sanitised = (
                        generated.strip() if isinstance(generated, str) else ""
                    )
                    if sanitised:
                        reflection_generated = sanitised
                        reflections.append(sanitised)
                        if len(reflections) > MAX_REFLECTIONS:
                            reflections.pop(0)

            attempts.append(
                {
                    "index": index,
                    "trial": trial,
                    "response": response,
                    "prediction": predicted_label,
                    "predicted_label": predicted_label,
                    "gold_label": gold_label,
                    "correct": correct,
                    "outcome": "correct" if correct else "incorrect",
                    "reflection": reflection_generated,
                    "reflection_used": reflection_used,
                    "reflection_generated": reflection_generated,
                }
            )
            if oom_skipped:
                if progress is not None:
                    progress(
                        f"[{spec.label}] OOM-SKIP index={index} "
                        f"trial={trial}/{max_trials} stage=reflection"
                    )
                break
            if progress is not None:
                progress(
                    f"[{spec.label}] index={index} trial={trial}/{max_trials} "
                    f"-> {'SUCCESS' if correct else 'FAIL'}"
                )
            if correct:
                break

        solved = success_trial is not None
        sample_results.append(
            {
                "index": index,
                "gold_label": gold_label,
                "solved": solved,
                "success": solved,
                "success_trial": success_trial,
                "attempt_count": len(attempts),
                "attempts": attempts,
                "complete": True,
                "oom_skipped": oom_skipped,
                "skip_reason": "oom" if oom_skipped else None,
                "oom_stage": oom_stage,
                "oom_trial": oom_trial,
                "oom_error": oom_error,
            }
        )
        if checkpoint is not None:
            checkpoint(
                _build_result(
                    spec,
                    selected_indices,
                    max_trials,
                    sample_results,
                    generation=generation_values,
                )
            )

    result = _build_result(
        spec,
        selected_indices,
        max_trials,
        sample_results,
        generation=generation_values,
    )
    complete, reason = validate_pipeline_result(result, config)
    if not complete:
        raise RuntimeError(f"pipeline produced an invalid result: {reason}")
    return result


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace JSON and fsync it before exposing the checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _supports_resume_call(func: Callable[..., object]) -> bool:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    names = {parameter.name for parameter in parameters}
    has_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    return has_kwargs or {"resume_results", "checkpoint"}.issubset(names)


def load_or_run(
    name: str,
    func: Callable[..., dict[str, Any]],
    *,
    output_dir: str | Path = OUTPUT_DIR,
    expected_config: Mapping[str, Any],
    force: bool = False,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Load a complete cache or resume a validated completed-article prefix."""

    result_path = Path(output_dir) / f"{name}.json"
    resume_results: list[Mapping[str, Any]] = []
    if result_path.exists() and not force:
        try:
            with result_path.open("r", encoding="utf-8") as input_file:
                raw_cached_result = json.load(input_file)
                cached_result = _upgrade_legacy_result(raw_cached_result)
                upgraded_legacy = cached_result is not raw_cached_result
            complete, reason = validate_pipeline_result(cached_result, expected_config)
            if complete:
                if upgraded_legacy:
                    save_json(result_path, cached_result)
                if progress is not None:
                    progress(f"[Load] {name}: {result_path}")
                return cached_result

            resumable, progress_reason = validate_progress_result(
                cached_result,
                expected_config,
            )
            if resumable:
                resume_results = list(cached_result["results"])
                if progress is not None:
                    progress(
                        f"[Resume] {name}: {len(resume_results)}/"
                        f"{expected_config['num_samples']} articles"
                    )
            elif progress is not None:
                progress(
                    f"[Rerun] {name}: {reason}; progress invalid ({progress_reason})"
                )
        except (OSError, json.JSONDecodeError) as error:
            if progress is not None:
                progress(f"[Rerun] {name}: cannot read cache ({error})")
    elif progress is not None:
        progress(f"[Run] {name}")

    def save_checkpoint(payload: Mapping[str, Any]) -> None:
        save_json(result_path, payload)

    if _supports_resume_call(func):
        result = func(
            resume_results=resume_results,
            checkpoint=save_checkpoint,
        )
    else:
        result = func()

    complete, reason = validate_pipeline_result(result, expected_config)
    if not complete:
        raise RuntimeError(f"refusing to save incomplete result for {name}: {reason}")
    save_json(result_path, result)
    if progress is not None:
        progress(f"[Saved] {result_path}")
    return result


def build_aggregate_metrics(
    results_by_key: Mapping[str, Mapping[str, Any]],
    indices: Sequence[Any],
    max_trials: int,
    *,
    generation: Mapping[str, object],
) -> dict[str, Any]:
    """Build the compact plot-ready summary in the fixed two-arm order."""

    pipeline_metrics: list[dict[str, Any]] = []
    for spec in PIPELINES:
        result = results_by_key[spec.key]
        rates = list(result["cumulative_success"])
        pipeline_metrics.append(
            {
                "pipeline_key": spec.key,
                "pipeline": spec.label,
                "cumulative_success": rates,
                "solved_at_trial": list(result["solved_at_trial"]),
                "final_solved": result["trial_summary"][-1]["total_solved"],
                "final_success_rate": rates[-1],
                "oom_skipped_count": result["oom_skipped_count"],
                "oom_skipped_indices": list(result["oom_skipped_indices"]),
            }
        )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_mode": EVALUATION_MODE,
        "num_samples": len(indices),
        "max_trials": max_trials,
        "max_reflections": MAX_REFLECTIONS,
        "indices": list(indices),
        "prompt_version": PROMPT_VERSION,
        "generation_metadata": _normalise_generation_metadata(generation),
        "pipeline_order": [spec.label for spec in PIPELINES],
        "pipelines": pipeline_metrics,
    }


def save_cumulative_success_plot(
    results_by_key: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
    max_trials: int,
) -> Path:
    """Save one PNG containing the two requested cumulative curves."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = list(range(1, max_trials + 1))
    styles = (
        {"marker": "o", "linestyle": "--", "color": "gray"},
        {"marker": "D", "linestyle": "-", "color": "firebrick"},
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    for spec, style in zip(PIPELINES, styles):
        axis.plot(
            trials,
            results_by_key[spec.key]["cumulative_success"],
            label=spec.label,
            linewidth=1.5,
            markersize=5,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
        )
    axis.set_xlabel("Trial Number", fontweight="bold")
    axis.set_ylabel("Proportion of Solved Tasks", fontweight="bold")
    axis.set_xticks(trials)
    axis.set_xlim(1, max_trials if max_trials > 1 else 1.1)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.4)
    axis.legend()
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return destination


def run_experiments(
    *,
    num: int | None = NUM,
    max_trials: int = MAX_TRIALS,
    output_dir: str | Path = OUTPUT_DIR,
    force: bool = False,
    dependencies: ExperimentDependencies | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, dict[str, Any]]:
    """Run/load both pipelines on the full dataset by default."""

    if max_trials <= 0:
        raise ValueError("max_trials must be greater than zero")
    deps = dependencies or load_default_dependencies()
    indices = select_indices(deps.get_available_indices(), num)
    generation_values = _normalise_generation_metadata(
        deps.generation_metadata()
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    results_by_key: dict[str, dict[str, Any]] = {}
    for spec in PIPELINES:
        expected_config = build_result_config(
            spec,
            indices,
            max_trials,
            generation=generation_values,
        )

        def run_current_pipeline(
            *,
            resume_results: Sequence[Mapping[str, Any]],
            checkpoint: Callable[[Mapping[str, Any]], None],
            current_spec: PipelineSpec = spec,
        ) -> dict[str, Any]:
            return evaluate_pipeline(
                current_spec,
                indices,
                deps,
                max_trials=max_trials,
                generation=generation_values,
                initial_results=resume_results,
                checkpoint=checkpoint,
                progress=progress,
            )

        results_by_key[spec.key] = load_or_run(
            spec.key,
            run_current_pipeline,
            output_dir=destination,
            expected_config=expected_config,
            force=force,
            progress=progress,
        )

    aggregate = build_aggregate_metrics(
        results_by_key,
        indices,
        max_trials,
        generation=generation_values,
    )
    aggregate_path = destination / AGGREGATE_METRICS_FILENAME
    save_json(aggregate_path, aggregate)
    plot_path = save_cumulative_success_plot(
        results_by_key,
        destination / PLOT_FILENAME,
        max_trials,
    )
    if progress is not None:
        progress(f"[Saved] {aggregate_path}")
        progress(f"[Saved] {plot_path}")
    return results_by_key


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run article-only SPRS-MTCPSL experiments on PropaNews."
    )
    parser.add_argument(
        "--num",
        type=int,
        default=NUM,
        help="optional first-N diagnostic limit (default: all dataset articles)",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=MAX_TRIALS,
        help=f"maximum attempts per article (default: {MAX_TRIALS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"result directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore cached progress and rerun every article",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_experiments(
        num=args.num,
        max_trials=args.max_trials,
        output_dir=args.output_dir,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_METRICS_FILENAME",
    "EVALUATION_MODE",
    "ExperimentDependencies",
    "MAX_REFLECTIONS",
    "MAX_TRIALS",
    "NUM",
    "OUTPUT_DIR",
    "PIPELINES",
    "PLOT_FILENAME",
    "PROMPT_VERSION",
    "PipelineSpec",
    "build_aggregate_metrics",
    "build_argument_parser",
    "build_result_config",
    "evaluate_pipeline",
    "load_default_dependencies",
    "load_or_run",
    "main",
    "run_experiments",
    "save_cumulative_success_plot",
    "save_json",
    "select_indices",
    "validate_pipeline_result",
    "validate_progress_result",
]
