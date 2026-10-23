"""Run the two requested CoT conditions on all stored PropaNews rows.

Execution remains article-major: one article receives up to ``TRIAL`` attempts
before the next stored index begins.  The result file is updated atomically
after every attempt, so completed indices and the current article's own EPM
and Reflexion memories can be resumed without additional model calls.

A recognised local inference OOM completes that article as ``oom_skipped``,
atomically checkpoints the marker, clears recoverable caches, and continues
with the next stored index. Other exceptions still abort the run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from propanews_common import (
    ARTICLE_TEXT_FIELD,
    DATASET_ID,
    EPM_MEMORY_VERSION,
    extract_epm_reasoning,
    is_oom_error,
    recover_from_oom,
)
from reflection_propanews import sanitize_reflection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qwen8" / "van_re"
PLOT_FILENAME = "propanews_cumulative.png"

# ``None`` means the complete dataset in its stored file order. ``--num`` is
# retained only as an optional deterministic first-N diagnostic limit.
NUM: int | None = None
TRIAL = 3
MAX_TRAJECTORIES = 1
MAX_REFLECTIONS = 3
RESULT_SCHEMA_VERSION = 5
EVALUATION_MODE = "article_major_failure_retry_cumulative"
PROMPT_VERSION = "cot-propanews-clean-v3-gt-epm-reflexion-v4"
OOM_ERROR_MAX_LENGTH = 500


def _default_generation_metadata() -> Mapping[str, object]:
    """Read model/cache identity from the sole non-PCoT LLM adapter."""

    module = importlib.import_module("llm")
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
class MethodSpec:
    key: str
    label: str
    runner_name: str
    use_reflection: bool
    use_gt: bool
    use_epm: bool


PipelineSpec = MethodSpec


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        "cot",
        "CoT",
        "run_CoT",
        use_reflection=False,
        use_gt=False,
        use_epm=False,
    ),
    MethodSpec(
        "cot_gt_epm_reflexion",
        "CoT + GT + EPM + Reflexion",
        "run_CoT_GT_EPM_Re",
        use_reflection=True,
        use_gt=True,
        use_epm=True,
    ),
)

# CoT historically called this collection ``METHODS``.  Expose the graph-style
# name too while keeping existing imports working.
PIPELINES = METHODS


@dataclass(frozen=True)
class ExperimentDependencies:
    run_CoT: Callable[[int], str]
    get_article: Callable[[int], str]
    get_available_indices: Callable[[], Sequence[int]]
    extract_prediction: Callable[[str], str]
    get_gold_label_bool: Callable[[int], bool]
    EM_score: Callable[[str, int], bool]
    make_reflection_propanews: Callable[[str, str], str]
    run_CoT_GT_EPM_Re: Callable[[int, list[str], list[str]], str]
    generation_metadata: Callable[[], Mapping[str, object]] = (
        _default_generation_metadata
    )


def load_default_dependencies() -> ExperimentDependencies:
    """Load model-facing modules only when an experiment is requested."""

    from actor_propanews import get_article, get_available_indices
    from evaluator_propanews import EM_score, extract_prediction, get_gold_label_bool
    from main_propanews import run_CoT, run_CoT_GT_EPM_Re
    from reflection_propanews import make_reflection_propanews
    from llm import generation_metadata

    return ExperimentDependencies(
        run_CoT=run_CoT,
        get_article=get_article,
        get_available_indices=get_available_indices,
        extract_prediction=extract_prediction,
        get_gold_label_bool=get_gold_label_bool,
        EM_score=EM_score,
        make_reflection_propanews=make_reflection_propanews,
        run_CoT_GT_EPM_Re=run_CoT_GT_EPM_Re,
        generation_metadata=generation_metadata,
    )


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


def success_trials_to_acc(
    success_trials: Sequence[int | None],
    *,
    max_trials: int = TRIAL,
) -> list[float]:
    """Convert first-success trials to cumulative success rates."""

    denominator = len(success_trials)
    if denominator == 0:
        return [0.0] * max_trials
    return [
        sum(
            value is not None and value <= trial
            for value in success_trials
        )
        / denominator
        for trial in range(1, max_trials + 1)
    ]


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist a checkpoint or complete result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_result_config(
    spec: MethodSpec,
    indices: Sequence[int],
    max_trials: int,
    *,
    generation: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    switches = (spec.use_gt, spec.use_epm, spec.use_reflection)
    if any(type(value) is not bool for value in switches):
        raise TypeError("pipeline switches must be booleans")
    if switches not in {(False, False, False), (True, True, True)}:
        raise ValueError("only baseline or GT + EPM + Reflexion is supported")
    generation_values = _normalise_generation_metadata(
        generation if generation is not None else _default_generation_metadata()
    )
    return {
        "pipeline_key": spec.key,
        "pipeline": spec.label,
        "method_key": spec.key,
        "method": spec.label,
        "use_reflection": spec.use_reflection,
        "use_gt": spec.use_gt,
        "use_epm": spec.use_epm,
        "num_samples": len(indices),
        "max_trials": max_trials,
        "max_trajectories": MAX_TRAJECTORIES,
        "max_reflections": MAX_REFLECTIONS,
        "indices": list(indices),
        "dataset": DATASET_ID,
        "article_field": ARTICLE_TEXT_FIELD,
        "epm_memory_version": EPM_MEMORY_VERSION,
        "prompt_version": PROMPT_VERSION,
        "generation_metadata": generation_values,
    }


def _new_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_mode": EVALUATION_MODE,
        "config": dict(config),
        "pipeline_key": config["pipeline_key"],
        "pipeline": config["pipeline"],
        "method_key": config["method_key"],
        "method": config["method"],
        "use_reflection": config["use_reflection"],
        "use_gt": config["use_gt"],
        "use_epm": config["use_epm"],
        "num_samples": config["num_samples"],
        "max_trials": config["max_trials"],
        "max_trajectories": config["max_trajectories"],
        "max_reflections": config["max_reflections"],
        "epm_memory_version": config["epm_memory_version"],
        "prompt_version": config["prompt_version"],
        "generation_metadata": dict(config["generation_metadata"]),
        "complete": False,
        "results": [],
        "solved_at_trial": [0] * int(config["max_trials"]),
        "cumulative_success": [0.0] * int(config["max_trials"]),
        "oom_skipped_count": 0,
        "oom_skipped_indices": [],
    }


def _refresh_metrics(payload: dict[str, Any]) -> None:
    max_trials = int(payload["max_trials"])
    success_trials = [
        sample.get("success_trial")
        for sample in payload["results"]
        if isinstance(sample, dict) and sample.get("complete") is True
    ]
    solved_at_trial = [0] * max_trials
    for success_trial in success_trials:
        if isinstance(success_trial, int) and 1 <= success_trial <= max_trials:
            solved_at_trial[success_trial - 1] += 1

    total_samples = int(payload["num_samples"])
    cumulative: list[float] = []
    solved = 0
    for newly_solved in solved_at_trial:
        solved += newly_solved
        cumulative.append(solved / total_samples)
    payload["solved_at_trial"] = solved_at_trial
    payload["cumulative_success"] = cumulative
    oom_skipped_indices = [
        sample["index"]
        for sample in payload["results"]
        if isinstance(sample, dict) and sample.get("oom_skipped") is True
    ]
    payload["oom_skipped_count"] = len(oom_skipped_indices)
    payload["oom_skipped_indices"] = oom_skipped_indices


def _short_oom_error(error: BaseException) -> str:
    """Return a compact, single-line diagnostic suitable for a JSON result."""

    message = " ".join(str(error).split())
    diagnostic = f"{type(error).__name__}: {message}" if message else type(error).__name__
    return diagnostic[:OOM_ERROR_MAX_LENGTH]


def _mark_oom_skipped(
    sample: dict[str, Any],
    *,
    stage: str,
    trial_number: int,
    diagnostic: str,
) -> None:
    """Mark one article complete-but-skipped after a recognised local OOM."""

    sample.update(
        {
            "success": False,
            "success_trial": None,
            "complete": True,
            "oom_skipped": True,
            "skip_reason": "oom",
            "oom_stage": stage,
            "oom_trial": trial_number,
            "oom_error": diagnostic,
        }
    )


def _validate_checkpoint(
    payload: object,
    expected_config: Mapping[str, Any],
) -> tuple[bool, str]:
    """Reject legacy/foreign caches before any completed index is skipped."""

    if not isinstance(payload, dict):
        return False, "result is not an object"
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        return False, "schema version mismatch"
    if payload.get("evaluation_mode") != EVALUATION_MODE:
        return False, "evaluation mode mismatch"
    if payload.get("config") != dict(expected_config):
        return False, "experiment configuration mismatch"

    for field in (
        "pipeline_key",
        "pipeline",
        "method_key",
        "method",
        "use_reflection",
        "use_gt",
        "use_epm",
        "num_samples",
        "max_trials",
        "max_trajectories",
        "max_reflections",
        "epm_memory_version",
        "prompt_version",
        "generation_metadata",
    ):
        if payload.get(field) != expected_config.get(field):
            return False, f"{field} mismatch"

    indices = list(expected_config["indices"])
    max_trials = int(expected_config["max_trials"])
    use_reflection = expected_config.get("use_reflection")
    use_gt = expected_config.get("use_gt")
    use_epm = expected_config.get("use_epm")
    if any(type(value) is not bool for value in (use_reflection, use_gt, use_epm)):
        return False, "invalid pipeline switches"
    if (use_gt, use_epm, use_reflection) not in {
        (False, False, False),
        (True, True, True),
    }:
        return False, "unsupported intermediate pipeline switches"
    if expected_config.get("max_trajectories") != MAX_TRAJECTORIES:
        return False, "max trajectory count mismatch"
    results = payload.get("results")
    if not isinstance(results, list) or len(results) > len(indices):
        return False, "invalid result prefix"

    found_incomplete = False
    for position, sample in enumerate(results):
        if not isinstance(sample, dict) or sample.get("index") != indices[position]:
            return False, "result indices do not match file order"
        if found_incomplete:
            return False, "an incomplete sample must be the final checkpoint entry"
        if type(sample.get("label")) is not bool:
            return False, "sample label is invalid"
        oom_skipped = sample.get("oom_skipped", False)
        if type(oom_skipped) is not bool:
            return False, "oom_skipped flag is invalid"
        trials = sample.get("trials")
        if not isinstance(trials, list):
            return False, "sample trials are invalid"
        if oom_skipped:
            if not 0 <= len(trials) <= max_trials:
                return False, "OOM-skipped sample trials are invalid"
        elif not 1 <= len(trials) <= max_trials:
            return False, "sample trials are invalid"

        expected_reflections: list[str] = []
        expected_trajectories: list[str] = []
        correct_trials: list[int] = []
        for trial_number, trial in enumerate(trials, start=1):
            if not isinstance(trial, dict):
                return False, "trial is not an object"
            if trial.get("trial") != trial_number:
                return False, "trial order mismatch"
            if trial.get("prediction") not in {"real", "fake", ""}:
                return False, "prediction is invalid"
            if not isinstance(trial.get("response"), str):
                return False, "response is invalid"
            if not isinstance(trial.get("correct"), bool):
                return False, "correct flag is invalid"
            if trial.get("reflection_used") != (
                expected_reflections if use_reflection else []
            ):
                return False, "reflection memory mismatch"
            if trial.get("trajectory_used") != (
                expected_trajectories if use_epm else []
            ):
                return False, "trajectory memory mismatch"

            trajectory_generated = trial.get("trajectory_generated")
            should_generate_trajectory = bool(
                use_epm and not trial["correct"] and trial_number < max_trials
            )

            reusable_reasoning = (
                extract_epm_reasoning(trial["response"])
                if should_generate_trajectory
                else None
            )

            if should_generate_trajectory and reusable_reasoning:
                if not isinstance(trajectory_generated, str):
                    return False, "failed retry did not generate an EPM trajectory"

                if trajectory_generated != reusable_reasoning:
                    return False, "EPM trajectory is not canonical reasoning-only memory"

                expected_trajectories = [trajectory_generated]

            elif should_generate_trajectory and not reusable_reasoning:
                # reasoning이 없는 실패 응답은 EPM 없이 다음 trial 진행
                if trajectory_generated is not None:
                    return False, "empty reasoning unexpectedly generated EPM memory"

                expected_trajectories = []

            elif trajectory_generated is not None:
                return False, "unexpected EPM trajectory was generated"

            reflection_generated = trial.get("reflection_generated")
            if trial.get("reflection") != reflection_generated:
                return False, "reflection alias mismatch"
            if reflection_generated is not None and not isinstance(
                reflection_generated, str
            ):
                return False, "reflection is invalid"
            if not use_reflection and reflection_generated is not None:
                return False, "non-Reflexion method contains reflection"
            if trial["correct"] and reflection_generated is not None:
                return False, "successful trial generated reflection"
            if trial_number == max_trials and reflection_generated is not None:
                return False, "final trial generated unused reflection"
            reflection_oom_attempt = bool(
                oom_skipped
                and sample.get("oom_stage") == "reflection"
                and sample.get("oom_trial") == trial_number
            )
            should_generate_reflection = bool(
                use_reflection
                and not trial["correct"]
                and trial_number < max_trials
                and reusable_reasoning
                and not reflection_oom_attempt
            )
            if should_generate_reflection and not isinstance(
                reflection_generated,
                str,
            ):
                return False, "failed retry did not generate a reflection"
            if isinstance(reflection_generated, str):
                if (
                    not reflection_generated.strip()
                    or reflection_generated != reflection_generated.strip()
                ):
                    return False, "reflection was not sanitised"
                if sanitize_reflection(reflection_generated) != reflection_generated:
                    return False, "reflection contains classification feedback"
                expected_reflections = (
                    expected_reflections + [reflection_generated]
                )[-MAX_REFLECTIONS:]
            if trial["correct"]:
                correct_trials.append(trial_number)

            expected_prediction = "real" if sample["label"] else "fake"
            expected_correct = bool(trial["prediction"]) and (
                trial["prediction"] == expected_prediction
            )
            if trial["correct"] is not expected_correct:
                return False, "correct flag does not match prediction and label"

            finish_values = re.findall(
                r"Finish\[(.*?)\]",
                trial["response"],
                flags=re.IGNORECASE | re.DOTALL,
            )
            parsed_prediction = (
                finish_values[-1].strip().casefold() if finish_values else ""
            )
            if parsed_prediction not in {"real", "fake"}:
                parsed_prediction = ""
            if trial["prediction"] != parsed_prediction:
                return False, "prediction does not match stored response"

        complete = sample.get("complete")
        success = sample.get("success")
        success_trial = sample.get("success_trial")
        if oom_skipped:
            if complete is not True or success is not False or success_trial is not None:
                return False, "OOM-skipped sample state is invalid"
            if correct_trials:
                return False, "OOM-skipped sample contains a successful trial"
            if sample.get("skip_reason") != "oom":
                return False, "OOM skip reason is invalid"
            oom_stage = sample.get("oom_stage")
            if oom_stage not in {"actor", "reflection"}:
                return False, "OOM stage is invalid"
            if oom_stage == "reflection" and not use_reflection:
                return False, "non-Reflexion method has a reflection-stage OOM"
            oom_trial = sample.get("oom_trial")
            if type(oom_trial) is not int or not 1 <= oom_trial <= max_trials:
                return False, "OOM trial is invalid"
            if oom_stage == "actor" and oom_trial != len(trials) + 1:
                return False, "actor OOM trial does not match stored attempts"
            if oom_stage == "reflection" and (
                oom_trial >= max_trials or oom_trial != len(trials)
            ):
                return False, "reflection OOM trial does not match stored attempts"
            oom_error = sample.get("oom_error")
            if (
                not isinstance(oom_error, str)
                or not oom_error.strip()
                or len(oom_error) > OOM_ERROR_MAX_LENGTH
            ):
                return False, "OOM diagnostic is invalid"
        elif any(
            sample.get(field) is not None
            for field in ("skip_reason", "oom_stage", "oom_trial", "oom_error")
        ):
            return False, "non-skipped sample contains OOM metadata"
        elif complete is True:
            if success is True:
                if correct_trials != [len(trials)] or success_trial != len(trials):
                    return False, "successful sample stopping point is invalid"
            elif success is False:
                if correct_trials or success_trial is not None or len(trials) != max_trials:
                    return False, "failed sample is incomplete"
            else:
                return False, "success flag is invalid"
        elif complete is False:
            found_incomplete = True
            if correct_trials or success is not False or success_trial is not None:
                return False, "incomplete sample state is invalid"
            if len(trials) >= max_trials:
                return False, "incomplete sample has no remaining trial"
        else:
            return False, "complete flag is invalid"

    if payload.get("complete") is True:
        if len(results) != len(indices) or found_incomplete:
            return False, "complete result has missing samples"

    expected_metrics: dict[str, Any] = {
        "max_trials": max_trials,
        "num_samples": len(indices),
        "results": results,
    }
    _refresh_metrics(expected_metrics)
    if payload.get("solved_at_trial") != expected_metrics["solved_at_trial"]:
        return False, "per-trial success counts do not match samples"
    if payload.get("cumulative_success") != expected_metrics["cumulative_success"]:
        return False, "cumulative success does not match samples"
    # The summary fields were added without invalidating otherwise compatible
    # pre-OOM checkpoints.  When present, however, they must be exact.
    if "oom_skipped_count" in payload:
        if (
            type(payload.get("oom_skipped_count")) is not int
            or payload.get("oom_skipped_count")
            != expected_metrics["oom_skipped_count"]
        ):
            return False, "OOM skip count does not match samples"
    if (
        "oom_skipped_indices" in payload
        and payload.get("oom_skipped_indices")
        != expected_metrics["oom_skipped_indices"]
    ):
        return False, "OOM skip indices do not match samples"
    return True, "complete" if payload.get("complete") is True else "resumable"


def _load_checkpoint(
    path: Path,
    expected_config: Mapping[str, Any],
    *,
    force: bool,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    if path.exists() and not force:
        try:
            with path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            valid, reason = _validate_checkpoint(cached, expected_config)
            if valid:
                if progress is not None:
                    progress(f"[Load] {path.name}: {reason}")
                return cached
            if progress is not None:
                progress(f"[Rerun] {path.name}: {reason}")
        except (OSError, json.JSONDecodeError) as exc:
            if progress is not None:
                progress(f"[Rerun] {path.name}: cannot read cache ({exc})")
    return _new_payload(expected_config)


def _runner_for(
    spec: MethodSpec,
    dependencies: ExperimentDependencies,
) -> Callable[..., str]:
    runner = getattr(dependencies, spec.runner_name)
    if runner is not None:
        return runner
    raise ValueError(f"missing runner dependency: {spec.runner_name}")


def evaluate_method(
    spec: MethodSpec,
    indices: Sequence[int],
    dependencies: ExperimentDependencies,
    *,
    max_trials: int = TRIAL,
    generation: Mapping[str, object] | None = None,
    result_path: str | Path | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Evaluate and checkpoint one method in strict article-major order."""

    selected_indices = list(indices)
    if max_trials <= 0:
        raise ValueError("max_trials must be greater than zero")
    if not selected_indices or len(selected_indices) != len(set(selected_indices)):
        raise ValueError("indices must be a non-empty duplicate-free sequence")

    generation_values = _normalise_generation_metadata(
        generation if generation is not None else dependencies.generation_metadata()
    )
    config = build_result_config(
        spec,
        selected_indices,
        max_trials,
        generation=generation_values,
    )
    destination = Path(result_path) if result_path is not None else None
    payload = (
        _load_checkpoint(
            destination,
            config,
            force=force,
            progress=progress,
        )
        if destination is not None
        else _new_payload(config)
    )
    # Populate the OOM summary when resuming a compatible checkpoint created
    # before these fields existed.
    _refresh_metrics(payload)
    if payload.get("complete") is True:
        return payload
    if destination is not None:
        # Expose a resumable state before the first model call. This also makes
        # --force invalidate a previous complete cache immediately.
        save_json(destination, payload)

    results: list[dict[str, Any]] = payload["results"]
    runner = _runner_for(spec, dependencies)

    for position, index in enumerate(selected_indices):
        if position < len(results):
            sample = results[position]
            if sample["complete"] is True:
                continue
        else:
            sample = {
                "index": index,
                "label": dependencies.get_gold_label_bool(index),
                "trials": [],
                "success": False,
                "success_trial": None,
                "complete": False,
                "oom_skipped": False,
                "skip_reason": None,
                "oom_stage": None,
                "oom_trial": None,
                "oom_error": None,
            }
            results.append(sample)

        if progress is not None:
            progress(
                f"[{spec.label}] index={index} "
                f"({position + 1}/{len(selected_indices)})"
            )

        trials: list[dict[str, Any]] = sample["trials"]

        # Both memories belong to this article only. Reconstructing them from
        # this sample's checkpoint preserves resume without allowing another
        # article's state to cross the boundary.
        trajectories: list[str] = []
        reflections: list[str] = []
        if spec.use_epm:
            generated_trajectories = [
                trial["trajectory_generated"]
                for trial in trials
                if isinstance(trial.get("trajectory_generated"), str)
            ]
            trajectories = generated_trajectories[-MAX_TRAJECTORIES:]
        if spec.use_reflection:
            reflections = [
                trial["reflection_generated"]
                for trial in trials
                if isinstance(trial.get("reflection_generated"), str)
            ][-MAX_REFLECTIONS:]

        for trial_number in range(len(trials) + 1, max_trials + 1):
            reflection_used = list(reflections) if spec.use_reflection else []
            trajectory_used = list(trajectories) if spec.use_epm else []
            actor_oom_diagnostic: str | None = None
            try:
                response = (
                    runner(index, trajectory_used, reflection_used)
                    if spec.use_gt or spec.use_epm or spec.use_reflection
                    else runner(index)
                )
            except Exception as error:
                if not is_oom_error(error):
                    raise
                actor_oom_diagnostic = _short_oom_error(error)
            if actor_oom_diagnostic is not None:
                _mark_oom_skipped(
                    sample,
                    stage="actor",
                    trial_number=trial_number,
                    diagnostic=actor_oom_diagnostic,
                )
                _refresh_metrics(payload)
                try:
                    if destination is not None:
                        save_json(destination, payload)
                    if progress is not None:
                        progress(
                            f"[{spec.label}] OOM-SKIP index={index} "
                            f"trial={trial_number}/{max_trials} stage=actor"
                        )
                finally:
                    recover_from_oom()
                break
            prediction = dependencies.extract_prediction(response)
            if prediction not in {"real", "fake"}:
                prediction = ""
            correct = bool(dependencies.EM_score(response, index))

            trajectory_generated: str | None = None
            reflection_generated: str | None = None
            reflection_oom_diagnostic: str | None = None
            if not correct and trial_number < max_trials:
                epm_reasoning: str | None = extract_epm_reasoning(response)

                if spec.use_epm:
                    if epm_reasoning:
                        # 정상적으로 reasoning을 얻은 경우에만 EPM으로 사용
                        trajectory_generated = epm_reasoning
                        trajectories = [epm_reasoning]
                    else:
                        # 이번 응답에는 재사용 가능한 reasoning이 없음.
                        # 이전 trial의 trajectory를 잘못 재사용하지 않도록 비움.
                        trajectories = []

                        if progress is not None:
                            progress(
                                f"[{spec.label}] index={index} "
                                f"trial={trial_number}/{max_trials} "
                                f"-> no reusable EPM reasoning; retry without EPM"
                            )

                if spec.use_reflection:
                    if epm_reasoning:
                        try:
                            reflection = dependencies.make_reflection_propanews(
                                dependencies.get_article(index),
                                epm_reasoning,
                            )
                            reflection = sanitize_reflection(reflection)

                            if not reflection:
                                raise ValueError(
                                    "reflection model returned no label-blind content"
                                )

                            reflection_generated = reflection
                            reflections.append(reflection)
                            reflections = reflections[-MAX_REFLECTIONS:]

                        except Exception as error:
                            if not is_oom_error(error):
                                raise
                            reflection_oom_diagnostic = _short_oom_error(error)

                    else:
                        # reasoning 자체가 없으므로 이번 trial에서는
                        # Reflection도 만들지 않고 다음 trial로 진행
                        if progress is not None:
                            progress(
                                f"[{spec.label}] index={index} "
                                f"trial={trial_number}/{max_trials} "
                                f"-> reflection skipped because reasoning is empty"
                            )

            trials.append(
                {
                    "trial": trial_number,
                    "prediction": prediction,
                    "correct": correct,
                    "response": response,
                    "trajectory_used": trajectory_used,
                    "trajectory_generated": trajectory_generated,
                    "reflection_used": reflection_used,
                    "reflection_generated": reflection_generated,
                    # Preserve the existing result-field alias.
                    "reflection": reflection_generated,
                }
            )
            sample["success"] = correct
            sample["success_trial"] = trial_number if correct else None
            sample["complete"] = correct or trial_number == max_trials

            if reflection_oom_diagnostic is not None:
                _mark_oom_skipped(
                    sample,
                    stage="reflection",
                    trial_number=trial_number,
                    diagnostic=reflection_oom_diagnostic,
                )
                _refresh_metrics(payload)
                try:
                    if destination is not None:
                        save_json(destination, payload)
                    if progress is not None:
                        progress(
                            f"[{spec.label}] OOM-SKIP index={index} "
                            f"trial={trial_number}/{max_trials} stage=reflection"
                        )
                finally:
                    recover_from_oom()
                break

            _refresh_metrics(payload)
            if destination is not None:
                save_json(destination, payload)
            if progress is not None:
                progress(
                    f"[{spec.label}] index={index} "
                    f"trial={trial_number}/{max_trials} -> "
                    f"{'SUCCESS' if correct else 'FAIL'}"
                )
            if sample["complete"]:
                break

    payload["complete"] = True
    _refresh_metrics(payload)
    valid, reason = _validate_checkpoint(payload, config)
    if not valid:
        raise RuntimeError(f"refusing to save invalid result: {reason}")
    if destination is not None:
        save_json(destination, payload)
        if progress is not None:
            progress(f"[Saved] {destination}")
    return payload


def _evaluate_named(
    key: str,
    *,
    indices: Sequence[int] | None = None,
    result_path: str | Path | None = None,
    max_trials: int = TRIAL,
    generation: Mapping[str, object] | None = None,
    force: bool = False,
    dependencies: ExperimentDependencies | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    deps = dependencies or load_default_dependencies()
    selected_indices = select_indices(
        deps.get_available_indices() if indices is None else indices,
        None,
    )
    spec = next(method for method in METHODS if method.key == key)
    return evaluate_method(
        spec,
        selected_indices,
        deps,
        max_trials=max_trials,
        generation=generation,
        result_path=result_path,
        force=force,
        progress=progress,
    )


def evaluate_CoT(**kwargs: Any) -> dict[str, Any]:
    return _evaluate_named("cot", **kwargs)


def evaluate_CoT_GT_EPM_Re(**kwargs: Any) -> dict[str, Any]:
    return _evaluate_named("cot_gt_epm_reflexion", **kwargs)


def evaluate_CoT_Re(**kwargs: Any) -> dict[str, Any]:
    """Legacy alias for the sole active Reflexion-containing condition."""

    return evaluate_CoT_GT_EPM_Re(**kwargs)


def load_or_run(
    name: str,
    func: Callable[..., dict[str, Any]],
    *,
    output_dir: str | Path = OUTPUT_DIR,
    indices: Sequence[int] | None = None,
    max_trials: int = TRIAL,
    generation: Mapping[str, object] | None = None,
    force: bool = False,
    dependencies: ExperimentDependencies | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Keep the legacy helper name while using resumable detailed results."""

    return func(
        indices=indices,
        result_path=Path(output_dir) / f"{name}.json",
        max_trials=max_trials,
        generation=generation,
        force=force,
        dependencies=dependencies,
        progress=progress,
    )


def save_cumulative_plot(
    results_by_key: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
    max_trials: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = list(range(1, max_trials + 1))
    styles = (
        {"marker": "o", "linestyle": "--"},
        {"marker": "D", "linestyle": "-"},
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    for spec, style in zip(METHODS, styles):
        axis.plot(
            trials,
            results_by_key[spec.key]["cumulative_success"],
            label=spec.label,
            **style,
        )
    axis.set_xlabel("Trial Number", fontweight="bold")
    axis.set_ylabel("Cumulative Success Rate", fontweight="bold")
    axis.set_xticks(trials)
    axis.set_xlim(1, max_trials)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.5)
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
    output_dir: str | Path = OUTPUT_DIR,
    max_trials: int = TRIAL,
    force: bool = False,
    dependencies: ExperimentDependencies | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, dict[str, Any]]:
    deps = dependencies or load_default_dependencies()
    indices = select_indices(deps.get_available_indices(), num)
    generation_values = _normalise_generation_metadata(
        deps.generation_metadata()
    )

    evaluators = {
        "cot": evaluate_CoT,
        "cot_gt_epm_reflexion": evaluate_CoT_GT_EPM_Re,
    }
    results: dict[str, dict[str, Any]] = {}
    for spec in METHODS:
        results[spec.key] = load_or_run(
            spec.key,
            evaluators[spec.key],
            output_dir=output_dir,
            indices=indices,
            max_trials=max_trials,
            generation=generation_values,
            force=force,
            dependencies=deps,
            progress=progress,
        )

    plot_path = save_cumulative_plot(
        results,
        Path(output_dir) / PLOT_FILENAME,
        max_trials,
    )
    if progress is not None:
        progress(f"[Saved] {plot_path}")
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the two CoT experiments on the full PropaNews dataset."
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
        default=TRIAL,
        help=f"maximum attempts per article (default: {TRIAL})",
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
        help="ignore matching checkpoints and rerun every method",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_experiments(
        num=args.num,
        output_dir=args.output_dir,
        max_trials=args.max_trials,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVALUATION_MODE",
    "ExperimentDependencies",
    "MAX_REFLECTIONS",
    "MAX_TRAJECTORIES",
    "METHODS",
    "PIPELINES",
    "PipelineSpec",
    "NUM",
    "OUTPUT_DIR",
    "PROMPT_VERSION",
    "RESULT_SCHEMA_VERSION",
    "TRIAL",
    "MethodSpec",
    "build_argument_parser",
    "build_result_config",
    "evaluate_CoT",
    "evaluate_CoT_GT_EPM_Re",
    "evaluate_CoT_Re",
    "evaluate_method",
    "load_default_dependencies",
    "load_or_run",
    "main",
    "run_experiments",
    "save_cumulative_plot",
    "save_json",
    "select_indices",
    "success_trials_to_acc",
]
