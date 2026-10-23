"""Run and plot the two article-only PCoT PropaNews experiment arms.

This module owns orchestration only.  Prompt construction and the two LLM calls
live in :mod:`actor_pcot`/:mod:`main_pcot`; final-label parsing lives in
:mod:`evaluator_pcot`; and gold-blind critique generation lives in
:mod:`reflection_pcot`.  Project dependencies are loaded lazily so importing
this module never starts an inference.

Evaluation is article-major: every allowed attempt for one article finishes
before the next article is touched.  A PCoT attempt always regenerates both
Stage 1 and Stage 2.  The Reflexion arm carries only the last three critique
strings for the current article; raw previous trajectories are retained for
debugging but are never actor memory.  A progress checkpoint is atomically
written after each completed article.  A local-memory OOM during either PCoT
generation stage or reflection marks that article as completed/OOM-skipped,
checkpoints it, clears recoverable caches, and continues with the next index;
all other errors still abort the run.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__:
    from .pcot_prompts import (
        PCOT_FINAL_MODE,
        PROMPT_VERSION,
        SUPPORTED_FINAL_MODES,
        normalise_final_mode,
    )
    from .propanews_common import (
        ARTICLE_TEXT_FIELD,
        DATASET_ID,
        is_oom_error,
        recover_from_oom,
    )
else:
    from pcot_prompts import (
        PCOT_FINAL_MODE,
        PROMPT_VERSION,
        SUPPORTED_FINAL_MODES,
        normalise_final_mode,
    )
    from propanews_common import (
        ARTICLE_TEXT_FIELD,
        DATASET_ID,
        is_oom_error,
        recover_from_oom,
    )


# ``None`` means the complete dataset in its stored file order.  ``--num`` is
# retained only for explicit smoke/subset runs.
NUM: int | None = None
MAX_TRIALS = 3
MAX_REFLECTIONS = 3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "propanews_pcot_qwen8"
AGGREGATE_METRICS_FILENAME = "aggregate_metrics.json"
PLOT_FILENAME = "propanews_pcot_cumulative_success.png"

RESULT_SCHEMA_VERSION = 2
EVALUATION_MODE = "article_major_failure_retry_cumulative"
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND = "pcot_article_progress"

_OOM_ERROR_MAX_LENGTH = 500


@dataclass(frozen=True)
class PipelineSpec:
    """The two experimental switches that distinguish a PCoT arm."""

    key: str
    label: str
    runner_name: str
    use_reflection: bool


# These are the complete requested conditions.  Dataset-specific reference
# information is not an experimental factor.
PIPELINES: tuple[PipelineSpec, ...] = (
    PipelineSpec("pcot", "PCoT", "run_PCoT", False),
    PipelineSpec(
        "pcot_reflexion",
        "PCoT + Reflexion",
        "run_PCoT_Re",
        True,
    ),
)


def _default_generation_metadata() -> Mapping[str, object]:
    """Load PCoT generation settings without importing the legacy LLM client."""

    prefix = f"{__package__}." if __package__ else ""
    module = importlib.import_module(f"{prefix}llm_pcot")
    return module.generation_metadata()


@dataclass(frozen=True)
class ExperimentDependencies:
    """Injected experiment boundaries for deterministic orchestration tests."""

    run_PCoT: Callable[..., Mapping[str, object]]
    run_PCoT_Re: Callable[..., Mapping[str, object]]
    get_article: Callable[[Any], str]
    get_available_indices: Callable[[], Sequence[Any]]
    extract_prediction: Callable[[str], str]
    get_gold_label: Callable[[Any], Any]
    EM_score: Callable[[str, Any], bool]
    make_reflection_PCoT: Callable[[str, str, str], str]
    generation_metadata: Callable[[], Mapping[str, object]] = (
        _default_generation_metadata
    )


def load_default_dependencies() -> ExperimentDependencies:
    """Import project modules only when an experiment is actually requested."""

    prefix = f"{__package__}." if __package__ else ""
    main_module = importlib.import_module(f"{prefix}main_pcot")
    actor_module = importlib.import_module(f"{prefix}actor_pcot")
    evaluator_module = importlib.import_module(f"{prefix}evaluator_pcot")
    reflection_module = importlib.import_module(f"{prefix}reflection_pcot")
    llm_module = importlib.import_module(f"{prefix}llm_pcot")

    return ExperimentDependencies(
        run_PCoT=main_module.run_PCoT,
        run_PCoT_Re=main_module.run_PCoT_Re,
        get_article=actor_module.get_article,
        get_available_indices=actor_module.get_available_indices,
        extract_prediction=evaluator_module.extract_prediction,
        get_gold_label=evaluator_module.get_gold_label,
        EM_score=evaluator_module.EM_score,
        make_reflection_PCoT=reflection_module.make_reflection_PCoT,
        generation_metadata=llm_module.generation_metadata,
    )


def _normalise_label(label: Any) -> str:
    """Return the persisted PropaNews label without exposing it to actors."""

    if label is True:
        return "real"
    if label is False:
        return "fake"
    if isinstance(label, str):
        normalised = label.strip().casefold()
        if normalised in {"real", "fake"}:
            return normalised
    raise ValueError(f"unsupported PropaNews gold label: {label!r}")


def _summarise_oom_error(error: BaseException) -> str:
    """Return a compact, single-line diagnostic safe for result JSON."""

    message = " ".join(str(error).split())
    summary = f"{type(error).__name__}: {message}" if message else type(error).__name__
    return summary[:_OOM_ERROR_MAX_LENGTH].rstrip()


def _with_oom_fields(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade a legacy, non-OOM completed sample without losing its fields."""

    materialised = dict(sample)
    materialised.setdefault("complete", True)
    materialised.setdefault("oom_skipped", False)
    materialised.setdefault("skip_reason", None)
    materialised.setdefault("oom_stage", None)
    materialised.setdefault("oom_trial", None)
    materialised.setdefault("oom_error", None)
    return materialised


def select_indices(
    available_indices: Sequence[Any],
    num: int | None,
) -> list[Any]:
    """Select a duplicate-free file-order cohort for every arm."""

    if num is not None and num <= 0:
        raise ValueError("num must be greater than zero")
    indices = list(available_indices)
    try:
        unique_count = len(set(indices))
    except TypeError as error:
        raise ValueError("available indices must be hashable") from error
    if len(indices) != unique_count:
        raise ValueError("get_available_indices() returned duplicate indices")
    if num is None:
        if not indices:
            raise ValueError("no PropaNews indices are available")
        return indices
    if len(indices) < num:
        raise ValueError(
            f"requested {num} samples, but only {len(indices)} are available"
        )
    return indices[:num]


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


def build_result_config(
    spec: PipelineSpec,
    indices: Sequence[Any],
    max_trials: int,
    *,
    final_mode: str = PCOT_FINAL_MODE,
    generation: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build the complete cache identity for one pipeline result."""

    if max_trials <= 0:
        raise ValueError("max_trials must be greater than zero")
    mode = normalise_final_mode(final_mode)
    generation_values = _normalise_generation_metadata(
        generation if generation is not None else _default_generation_metadata()
    )
    selected_indices = list(indices)
    return {
        "pipeline_key": spec.key,
        "pipeline": spec.label,
        "use_reflection": spec.use_reflection,
        "num_samples": len(selected_indices),
        "max_trials": max_trials,
        "max_reflections": MAX_REFLECTIONS,
        "indices": selected_indices,
        "dataset": DATASET_ID,
        "article_field": ARTICLE_TEXT_FIELD,
        "prompt_version": PROMPT_VERSION,
        "final_mode": mode,
        "generation_metadata": generation_values,
    }


def _runner_for(
    spec: PipelineSpec,
    dependencies: ExperimentDependencies,
) -> Callable[..., Mapping[str, object]]:
    return getattr(dependencies, spec.runner_name)


def _supports_final_mode(runner: Callable[..., object]) -> bool:
    """Allow compact legacy-style fakes while always configuring real runners."""

    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return True
    return "final_mode" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _run_trajectory(
    runner: Callable[..., Mapping[str, object]],
    *,
    index: Any,
    reflections: list[str],
    use_reflection: bool,
    final_mode: str,
) -> Mapping[str, object]:
    arguments: list[object] = [index]
    if use_reflection:
        arguments.append(reflections)
    if _supports_final_mode(runner):
        return runner(*arguments, final_mode=final_mode)
    return runner(*arguments)


def _validate_trajectory(trajectory: object) -> tuple[str, dict[str, Any] | None, str]:
    """Validate and unpack the actor's preserved two-stage trajectory."""

    if not isinstance(trajectory, Mapping):
        raise TypeError("PCoT runner must return a trajectory mapping")
    missing = {"persuasion", "persuasion_parsed", "final"} - set(trajectory)
    if missing:
        raise ValueError(f"PCoT trajectory is missing fields: {sorted(missing)}")

    persuasion = trajectory["persuasion"]
    parsed = trajectory["persuasion_parsed"]
    final = trajectory["final"]
    if not isinstance(persuasion, str) or not persuasion.strip():
        raise ValueError("trajectory persuasion output must be a non-empty string")
    if parsed is not None and not isinstance(parsed, dict):
        raise TypeError("trajectory persuasion_parsed must be a dict or None")
    if not isinstance(final, str) or not final.strip():
        raise ValueError("trajectory final output must be a non-empty string")
    return persuasion, parsed, final


def _build_pipeline_result(
    spec: PipelineSpec,
    config: Mapping[str, Any],
    sample_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the legacy-compatible final payload from completed samples."""

    total_samples = int(config["num_samples"])
    max_trials = int(config["max_trials"])
    solved_at_trial = [0] * max_trials
    solved: list[bool] = []
    materialised = [_with_oom_fields(sample) for sample in sample_results]
    oom_skipped_indices = [
        sample["index"]
        for sample in materialised
        if sample.get("oom_skipped") is True
    ]

    for sample in materialised:
        sample_solved = sample.get("solved") is True
        solved.append(sample_solved)
        success_trial = sample.get("success_trial")
        if sample_solved and isinstance(success_trial, int):
            solved_at_trial[success_trial - 1] += 1

    cumulative_success: list[float] = []
    trial_summary: list[dict[str, Any]] = []
    total_solved = 0
    for offset, newly_solved in enumerate(solved_at_trial):
        total_solved += newly_solved
        rate = total_solved / total_samples
        cumulative_success.append(rate)
        trial_summary.append(
            {
                "trial": offset + 1,
                "newly_solved": newly_solved,
                "total_solved": total_solved,
                "remaining": total_samples - total_solved,
                "cumulative_success": rate,
            }
        )

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_mode": EVALUATION_MODE,
        "config": dict(config),
        "pipeline_key": spec.key,
        "pipeline": spec.label,
        "num_samples": total_samples,
        "max_trials": max_trials,
        "max_reflections": MAX_REFLECTIONS,
        "prompt_version": config["prompt_version"],
        "final_mode": config["final_mode"],
        "generation_metadata": dict(config["generation_metadata"]),
        "solved": solved,
        "solved_at_trial": solved_at_trial,
        "cumulative_success": cumulative_success,
        "trial_summary": trial_summary,
        "oom_skipped_count": len(oom_skipped_indices),
        "oom_skipped_indices": oom_skipped_indices,
        "results": materialised,
    }


def evaluate_pipeline(
    spec: PipelineSpec,
    indices: Sequence[Any],
    dependencies: ExperimentDependencies,
    *,
    max_trials: int = MAX_TRIALS,
    final_mode: str = PCOT_FINAL_MODE,
    generation: Mapping[str, object] | None = None,
    progress: Callable[[str], None] | None = None,
    initial_results: Sequence[Mapping[str, Any]] | None = None,
    on_sample_complete: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate one PCoT arm in strict article-major retry order.

    ``initial_results`` must be a valid completed prefix produced by an earlier
    checkpoint.  Those articles are skipped without any model call.  Only the
    final Stage-2 response reaches the evaluator, and neither the gold label nor
    an evaluator verdict is passed to the reflection model.  A recognised local
    OOM is recorded as a completed skipped article and never retried on resume.
    """

    if max_trials <= 0:
        raise ValueError("max_trials must be greater than zero")
    if not indices:
        raise ValueError("indices must not be empty")
    selected_indices = list(indices)
    try:
        if len(selected_indices) != len(set(selected_indices)):
            raise ValueError("indices must be unique")
    except TypeError as error:
        raise ValueError("indices must be hashable") from error

    mode = normalise_final_mode(final_mode)
    generation_values = _normalise_generation_metadata(
        generation if generation is not None else dependencies.generation_metadata()
    )
    config = build_result_config(
        spec,
        selected_indices,
        max_trials,
        final_mode=mode,
        generation=generation_values,
    )
    runner = _runner_for(spec, dependencies)
    sample_results = [dict(sample) for sample in (initial_results or ())]
    valid_prefix, prefix_reason = _validate_completed_samples(
        sample_results,
        config,
        require_complete=False,
    )
    if not valid_prefix:
        raise ValueError(f"invalid initial checkpoint results: {prefix_reason}")
    sample_results = [_with_oom_fields(sample) for sample in sample_results]

    for position in range(len(sample_results), len(selected_indices)):
        index = selected_indices[position]
        if progress is not None:
            progress(
                f"[{spec.label}] index={index} "
                f"({position + 1}/{len(selected_indices)})"
            )

        gold_label = _normalise_label(dependencies.get_gold_label(index))
        attempts: list[dict[str, Any]] = []
        reflections: list[str] = []
        success_trial: int | None = None
        oom_stage: str | None = None
        oom_trial: int | None = None
        oom_error: str | None = None

        for trial in range(1, max_trials + 1):
            reflection_used = list(reflections) if spec.use_reflection else []
            try:
                raw_trajectory = _run_trajectory(
                    runner,
                    index=index,
                    reflections=reflection_used,
                    use_reflection=spec.use_reflection,
                    final_mode=mode,
                )
            except Exception as error:
                if not is_oom_error(error):
                    raise
                oom_stage = "stage1_or_stage2"
                oom_trial = trial
                oom_error = _summarise_oom_error(error)
            if oom_stage is not None:
                # Leave the exception handler first so traceback-held tensors
                # can be collected before clearing the accelerator cache.
                recover_from_oom()
                if progress is not None:
                    progress(
                        f"[{spec.label}] index={index} trial={trial}/{max_trials} "
                        f"stage={oom_stage} -> OOM-SKIP"
                    )
                break
            persuasion, persuasion_parsed, final_response = _validate_trajectory(
                raw_trajectory
            )
            prediction = dependencies.extract_prediction(final_response)
            prediction = (
                prediction.strip().casefold() if isinstance(prediction, str) else ""
            )
            if prediction not in {"real", "fake"}:
                prediction = ""

            correct = bool(dependencies.EM_score(final_response, index))
            reflection_generated: str | None = None
            if correct:
                success_trial = trial
            elif spec.use_reflection and trial < max_trials:
                try:
                    generated = dependencies.make_reflection_PCoT(
                        dependencies.get_article(index),
                        persuasion,
                        final_response,
                    )
                except Exception as error:
                    if not is_oom_error(error):
                        raise
                    oom_stage = "reflection"
                    oom_trial = trial
                    oom_error = _summarise_oom_error(error)
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
                    "persuasion_analysis": persuasion,
                    "persuasion_analysis_parsed": persuasion_parsed,
                    "final_response": final_response,
                    "prediction": prediction,
                    "predicted_label": prediction,
                    "gold_label": gold_label,
                    "correct": correct,
                    "outcome": "correct" if correct else "incorrect",
                    "reflection": reflection_generated,
                    "reflection_used": reflection_used,
                    "reflection_generated": reflection_generated,
                }
            )

            if oom_stage is not None:
                # As above, recover only after the exception traceback is no
                # longer retained by the active handler.
                recover_from_oom()
            if progress is not None:
                if oom_stage is not None:
                    progress(
                        f"[{spec.label}] index={index} trial={trial}/{max_trials} "
                        f"stage={oom_stage} -> OOM-SKIP"
                    )
                else:
                    progress(
                        f"[{spec.label}] index={index} trial={trial}/{max_trials} "
                        f"-> {'SUCCESS' if correct else 'FAIL'}"
                    )
            if oom_stage is not None:
                break
            if correct:
                break

        sample_results.append(
            {
                "index": index,
                "gold_label": gold_label,
                "complete": True,
                "oom_skipped": oom_stage is not None,
                "skip_reason": "oom" if oom_stage is not None else None,
                "oom_stage": oom_stage,
                "oom_trial": oom_trial,
                "oom_error": oom_error,
                "solved": success_trial is not None and oom_stage is None,
                "success_trial": success_trial,
                "attempt_count": len(attempts),
                "attempts": attempts,
            }
        )
        if on_sample_complete is not None:
            on_sample_complete(tuple(sample_results))

    return _build_pipeline_result(spec, config, sample_results)


_LEGACY_SAMPLE_KEYS = {
    "index",
    "gold_label",
    "solved",
    "success_trial",
    "attempt_count",
    "attempts",
}
_OOM_SAMPLE_KEYS = {
    "complete",
    "oom_skipped",
    "skip_reason",
    "oom_stage",
    "oom_trial",
    "oom_error",
}
_SAMPLE_KEYS = _LEGACY_SAMPLE_KEYS | _OOM_SAMPLE_KEYS
_ATTEMPT_KEYS = {
    "index",
    "trial",
    "persuasion_analysis",
    "persuasion_analysis_parsed",
    "final_response",
    "prediction",
    "predicted_label",
    "gold_label",
    "correct",
    "outcome",
    "reflection",
    "reflection_used",
    "reflection_generated",
}


def _validate_completed_samples(
    results: Any,
    expected_config: Mapping[str, Any],
    *,
    require_complete: bool,
) -> tuple[bool, str]:
    """Validate a complete-prefix sequence without trusting checkpoint data."""

    if not isinstance(results, (list, tuple)):
        return False, "sample results are not a sequence"
    indices = list(expected_config.get("indices", []))
    num_samples = expected_config.get("num_samples")
    max_trials = expected_config.get("max_trials")
    use_reflection = expected_config.get("use_reflection")
    if not isinstance(num_samples, int) or num_samples <= 0:
        return False, "invalid expected sample count"
    if len(indices) != num_samples:
        return False, "invalid expected index list"
    if not isinstance(max_trials, int) or max_trials <= 0:
        return False, "invalid expected trial count"
    if not isinstance(use_reflection, bool):
        return False, "invalid reflection switch"
    if len(results) > num_samples:
        return False, "too many sample results"
    if require_complete and len(results) != num_samples:
        return False, "sample results are incomplete"

    for position, sample in enumerate(results):
        if not isinstance(sample, Mapping):
            return False, "sample is not an object"
        sample_keys = set(sample)
        legacy_normal_sample = sample_keys == _LEGACY_SAMPLE_KEYS
        if not legacy_normal_sample and sample_keys != _SAMPLE_KEYS:
            return False, "sample fields do not match the result schema"
        index = indices[position]
        if sample.get("index") != index:
            return False, "sample indices or file order do not match"
        gold_label = sample.get("gold_label")
        if gold_label not in {"real", "fake"}:
            return False, "sample gold label is invalid"

        if legacy_normal_sample:
            oom_skipped = False
            skip_reason = None
            oom_stage = None
            oom_trial = None
            oom_error = None
        else:
            if sample.get("complete") is not True:
                return False, "completed sample is not marked complete"
            oom_skipped = sample.get("oom_skipped")
            skip_reason = sample.get("skip_reason")
            oom_stage = sample.get("oom_stage")
            oom_trial = sample.get("oom_trial")
            oom_error = sample.get("oom_error")
            if not isinstance(oom_skipped, bool):
                return False, "sample OOM skipped flag is invalid"
            if oom_skipped:
                if skip_reason != "oom":
                    return False, "OOM-skipped sample has an invalid skip reason"
                if oom_stage not in {"stage1_or_stage2", "reflection"}:
                    return False, "OOM-skipped sample has an invalid stage"
                if type(oom_trial) is not int or not 1 <= oom_trial <= max_trials:
                    return False, "OOM-skipped sample has an invalid trial"
                if (
                    not isinstance(oom_error, str)
                    or not oom_error.strip()
                    or oom_error != oom_error.strip()
                    or len(oom_error) > _OOM_ERROR_MAX_LENGTH
                ):
                    return False, "OOM-skipped sample has an invalid error summary"
            elif any(
                value is not None
                for value in (skip_reason, oom_stage, oom_trial, oom_error)
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

        expected_reflections: list[str] = []
        correct_trials: list[int] = []
        for trial, record in enumerate(attempts, start=1):
            if not isinstance(record, Mapping) or set(record) != _ATTEMPT_KEYS:
                return False, "attempt fields do not match the result schema"
            if record.get("index") != index or record.get("trial") != trial:
                return False, "attempt index or order mismatch"
            persuasion = record.get("persuasion_analysis")
            final_response = record.get("final_response")
            if not isinstance(persuasion, str) or not persuasion.strip():
                return False, "persuasion analysis is invalid"
            parsed = record.get("persuasion_analysis_parsed")
            if parsed is not None and not isinstance(parsed, dict):
                return False, "parsed persuasion analysis is invalid"
            if not isinstance(final_response, str) or not final_response.strip():
                return False, "final response is invalid"

            prediction = record.get("prediction")
            if prediction not in {"real", "fake", ""}:
                return False, "attempt prediction is invalid"
            if record.get("predicted_label") != prediction:
                return False, "prediction alias mismatch"
            finish_values = re.findall(
                r"Finish\[(.*?)\]",
                final_response,
                flags=re.IGNORECASE | re.DOTALL,
            )
            parsed_prediction = (
                finish_values[-1].strip().casefold() if finish_values else ""
            )
            if parsed_prediction not in {"real", "fake"}:
                parsed_prediction = ""
            if prediction != parsed_prediction:
                return False, "prediction does not match stored final response"
            if record.get("gold_label") != gold_label:
                return False, "attempt gold label mismatch"
            correct = record.get("correct")
            if not isinstance(correct, bool):
                return False, "attempt correctness is invalid"
            if correct != (bool(prediction) and prediction == gold_label):
                return False, "attempt correctness does not match labels"
            expected_outcome = "correct" if correct else "incorrect"
            if record.get("outcome") != expected_outcome:
                return False, "attempt outcome mismatch"

            used = record.get("reflection_used")
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
            if not use_reflection and generated is not None:
                return False, "plain PCoT generated a reflection"
            if correct and generated is not None:
                return False, "successful attempt generated a reflection"
            if trial == max_trials and generated is not None:
                return False, "reflection was generated after the final attempt"
            if isinstance(generated, str):
                expected_reflections = (expected_reflections + [generated])[
                    -MAX_REFLECTIONS:
                ]
            if correct:
                correct_trials.append(trial)

        solved = sample.get("solved")
        success_trial = sample.get("success_trial")
        if oom_skipped:
            if solved is not False or success_trial is not None or correct_trials:
                return False, "OOM-skipped sample has success data"
            if oom_stage == "stage1_or_stage2":
                if oom_trial != len(attempts) + 1:
                    return False, "trajectory OOM trial does not match attempts"
            elif (
                not use_reflection
                or oom_trial != len(attempts)
                or oom_trial >= max_trials
            ):
                return False, "reflection OOM trial does not match attempts"
        elif solved is True:
            if correct_trials != [len(attempts)] or success_trial != len(attempts):
                return False, "solved sample has an invalid stopping point"
        elif solved is False:
            if correct_trials or success_trial is not None or len(attempts) != max_trials:
                return False, "unsolved sample has incomplete attempts"
        else:
            return False, "sample solved flag is invalid"

    return True, "complete" if require_complete else "valid completed prefix"


def build_progress_checkpoint(
    expected_config: Mapping[str, Any],
    completed_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the small sidecar atomically saved after every article."""

    valid, reason = _validate_completed_samples(
        completed_results,
        expected_config,
        require_complete=False,
    )
    if not valid:
        raise ValueError(f"cannot checkpoint invalid results: {reason}")
    materialised = [_with_oom_fields(sample) for sample in completed_results]
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": CHECKPOINT_KIND,
        "config": dict(expected_config),
        "completed_count": len(materialised),
        "results": materialised,
    }


def validate_progress_checkpoint(
    checkpoint: Any,
    expected_config: Mapping[str, Any],
) -> tuple[bool, str]:
    """Accept only a matching, article-complete prefix checkpoint."""

    if not isinstance(checkpoint, Mapping):
        return False, "checkpoint is not an object"
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return False, "checkpoint schema version mismatch"
    if checkpoint.get("checkpoint_kind") != CHECKPOINT_KIND:
        return False, "checkpoint kind mismatch"
    if checkpoint.get("config") != dict(expected_config):
        return False, "checkpoint configuration mismatch"
    results = checkpoint.get("results")
    if not isinstance(results, list) or checkpoint.get("completed_count") != len(results):
        return False, "checkpoint completed count mismatch"
    return _validate_completed_samples(
        results,
        expected_config,
        require_complete=False,
    )


def validate_pipeline_result(
    result: Any,
    expected_config: Mapping[str, Any],
) -> tuple[bool, str]:
    """Validate cache identity, attempt records, metrics, and stopping rules."""

    if not isinstance(result, Mapping):
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
        "final_mode",
        "generation_metadata",
    ):
        if result.get(field) != expected.get(field):
            return False, f"{field} mismatch"

    results = result.get("results")
    valid, reason = _validate_completed_samples(
        results,
        expected,
        require_complete=True,
    )
    if not valid:
        return False, reason

    if type(result.get("oom_skipped_count")) is not int:
        return False, "oom_skipped_count is invalid"
    if not isinstance(result.get("oom_skipped_indices"), list):
        return False, "oom_skipped_indices is invalid"

    spec = PipelineSpec(
        key=str(expected["pipeline_key"]),
        label=str(expected["pipeline"]),
        runner_name="",
        use_reflection=bool(expected["use_reflection"]),
    )
    recomputed = _build_pipeline_result(spec, expected, results)
    for field in (
        "solved",
        "solved_at_trial",
        "cumulative_success",
        "trial_summary",
        "oom_skipped_count",
        "oom_skipped_indices",
    ):
        if result.get(field) != recomputed[field]:
            return False, f"{field} does not match sample records"
    return True, "complete"


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically so interruption cannot create a valid cache."""

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


def load_or_run(
    name: str,
    func: Callable[[], dict[str, Any]],
    *,
    output_dir: str | Path = OUTPUT_DIR,
    expected_config: Mapping[str, Any],
    force: bool = False,
    progress: Callable[[str], None] | None = print,
    resume_func: Callable[
        [
            Sequence[Mapping[str, Any]],
            Callable[[Sequence[Mapping[str, Any]]], None],
        ],
        dict[str, Any],
    ]
    | None = None,
) -> dict[str, Any]:
    """Load a final result or resume from an article-complete sidecar."""

    result_path = Path(output_dir) / f"{name}.json"
    checkpoint_path = Path(output_dir) / f"{name}.progress.json"
    if result_path.exists() and not force:
        try:
            with result_path.open("r", encoding="utf-8") as input_file:
                cached_result = json.load(input_file)
            complete, reason = validate_pipeline_result(
                cached_result,
                expected_config,
            )
            if complete:
                if progress is not None:
                    progress(f"[Load] {name}: {result_path}")
                return cached_result
            if progress is not None:
                progress(f"[Rerun] {name}: {reason}")
        except (OSError, json.JSONDecodeError) as error:
            if progress is not None:
                progress(f"[Rerun] {name}: cannot read cache ({error})")
    initial_results: Sequence[Mapping[str, Any]] = ()
    if resume_func is not None and checkpoint_path.exists() and not force:
        try:
            with checkpoint_path.open("r", encoding="utf-8") as input_file:
                checkpoint = json.load(input_file)
            reusable, reason = validate_progress_checkpoint(
                checkpoint,
                expected_config,
            )
            if reusable:
                initial_results = checkpoint["results"]
                if progress is not None:
                    progress(
                        f"[Resume] {name}: {len(initial_results)}/"
                        f"{expected_config['num_samples']} articles"
                    )
            elif progress is not None:
                progress(f"[Rerun] {name}: {reason}")
        except (OSError, json.JSONDecodeError) as error:
            if progress is not None:
                progress(f"[Rerun] {name}: cannot read checkpoint ({error})")
    elif progress is not None:
        progress(f"[Run] {name}")

    if resume_func is None:
        result = func()
    else:

        def checkpoint_completed(
            completed_results: Sequence[Mapping[str, Any]],
        ) -> None:
            payload = build_progress_checkpoint(
                expected_config,
                completed_results,
            )
            save_json(checkpoint_path, payload)
            if progress is not None:
                progress(
                    f"[Checkpoint] {name}: {len(completed_results)}/"
                    f"{expected_config['num_samples']} articles"
                )

        result = resume_func(initial_results, checkpoint_completed)

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
    final_mode: str,
    generation: Mapping[str, object],
) -> dict[str, Any]:
    """Build a compact plot-ready summary in the fixed legend order."""

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
        "final_mode": normalise_final_mode(final_mode),
        "generation_metadata": _normalise_generation_metadata(generation),
        "pipeline_order": [spec.label for spec in PIPELINES],
        "pipelines": pipeline_metrics,
    }


def save_cumulative_success_plot(
    results_by_key: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
    max_trials: int,
) -> Path:
    """Save one PNG containing the two requested PCoT lines."""

    if max_trials <= 0:
        raise ValueError("max_trials must be greater than zero")

    # Delay matplotlib import and force a headless backend for CLI/server runs.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = list(range(1, max_trials + 1))
    styles: tuple[dict[str, str], ...] = (
        {"marker": "o", "linestyle": "--", "color": "gray"},
        {"marker": "D", "linestyle": "-", "color": "firebrick"},
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    for spec, style in zip(PIPELINES, styles):
        rates = results_by_key[spec.key]["cumulative_success"]
        if not isinstance(rates, Sequence) or len(rates) != max_trials:
            plt.close(figure)
            raise ValueError(f"{spec.label} has incomplete cumulative metrics")
        axis.plot(
            trials,
            rates,
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
    axis.set_xlim(1, max_trials)
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
    final_mode: str = PCOT_FINAL_MODE,
    force: bool = False,
    dependencies: ExperimentDependencies | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, dict[str, Any]]:
    """Run/load both arms and save their shared metrics and plot."""

    if max_trials <= 0:
        raise ValueError("max_trials must be greater than zero")
    mode = normalise_final_mode(final_mode)
    deps = dependencies or load_default_dependencies()
    indices = select_indices(deps.get_available_indices(), num)
    generation_values = _normalise_generation_metadata(deps.generation_metadata())
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    results_by_key: dict[str, dict[str, Any]] = {}
    for spec in PIPELINES:
        expected_config = build_result_config(
            spec,
            indices,
            max_trials,
            final_mode=mode,
            generation=generation_values,
        )

        def run_current_pipeline(
            current_spec: PipelineSpec = spec,
        ) -> dict[str, Any]:
            return evaluate_pipeline(
                current_spec,
                indices,
                deps,
                max_trials=max_trials,
                final_mode=mode,
                generation=generation_values,
                progress=progress,
            )

        def resume_current_pipeline(
            initial_results: Sequence[Mapping[str, Any]],
            on_sample_complete: Callable[
                [Sequence[Mapping[str, Any]]],
                None,
            ],
            current_spec: PipelineSpec = spec,
        ) -> dict[str, Any]:
            return evaluate_pipeline(
                current_spec,
                indices,
                deps,
                max_trials=max_trials,
                final_mode=mode,
                generation=generation_values,
                progress=progress,
                initial_results=initial_results,
                on_sample_complete=on_sample_complete,
            )

        results_by_key[spec.key] = load_or_run(
            spec.key,
            run_current_pipeline,
            output_dir=destination,
            expected_config=expected_config,
            force=force,
            progress=progress,
            resume_func=resume_current_pipeline,
        )

    aggregate = build_aggregate_metrics(
        results_by_key,
        indices,
        max_trials,
        final_mode=mode,
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
        description="Run two-stage PCoT experiments on PropaNews."
    )
    parser.add_argument(
        "--num",
        type=int,
        default=NUM,
        help="number of file-order articles to evaluate (default: all)",
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
        "--final-mode",
        choices=SUPPORTED_FINAL_MODES,
        default=PCOT_FINAL_MODE,
        help=f"Stage-2 prompt variant (default: {PCOT_FINAL_MODE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore complete cached results and rerun every arm",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_experiments(
        num=args.num,
        max_trials=args.max_trials,
        output_dir=args.output_dir,
        final_mode=args.final_mode,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_METRICS_FILENAME",
    "CHECKPOINT_KIND",
    "CHECKPOINT_SCHEMA_VERSION",
    "EVALUATION_MODE",
    "ExperimentDependencies",
    "MAX_REFLECTIONS",
    "MAX_TRIALS",
    "NUM",
    "OUTPUT_DIR",
    "PIPELINES",
    "PLOT_FILENAME",
    "PipelineSpec",
    "build_aggregate_metrics",
    "build_argument_parser",
    "build_progress_checkpoint",
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
    "validate_progress_checkpoint",
]
