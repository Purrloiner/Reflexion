"""Zero-shot once, then gold-gated Reflexion retries with early stopping."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import llm
from propanews_common import PromptArticle
from propanews_zeroshot_backend import LocalBackend, MockBackend, runtime_metadata
from propanews_zeroshot_checkpoint import CheckpointError, Ledger, ModelCallBudgetExhausted, timestamp
from propanews_zeroshot_data import (
    DEFAULT_DATA_PATH, ROOT, OfficialTestError, atomic_write, load_official_test, write_json,
)
from propanews_zeroshot_early_stop import EARLY_STOP_PROTOCOL, summarize_early_stop_trials
from propanews_zeroshot_prompts import (
    classification_prompt, parse_classification,
)
from propanews_zeroshot_three_calls import (
    PROMPT_VERSION, integrated_reflexion_prompt, parse_integrated_reflexion,
)


def build_provenance(validation: dict, backend_metadata: dict) -> dict:
    sources = [
        "llm.py", "propanews_common.py", "reflection_propanews.py",
        "propanews_zeroshot_data.py", "propanews_zeroshot_prompts.py",
        "propanews_zeroshot_checkpoint.py", "propanews_zeroshot_backend.py",
        "propanews_zeroshot_metrics.py", "run_propanews_zeroshot.py",
        "propanews_zeroshot_early_stop.py",
        "propanews_zeroshot_three_calls.py",
    ]
    return {
        "schema_version": 3, "prompt_version": PROMPT_VERSION,
        "dataset": validation, "backend": backend_metadata,
        "protocol": dict(EARLY_STOP_PROTOCOL),
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in sources
        },
    }


def article_state(ledger: Ledger, article: PromptArticle, *,
                  is_correct: Callable[[int, str], bool] | None = None) -> dict:
    trials = {}
    success_trial, successful_prediction = None, None
    for trial in (1, 2, 3):
        if success_trial is not None:
            trials[str(trial)] = {
                "prediction": None, "rationale": None, "response": None,
                "status": "skipped_after_correct", "classification_call_id": None,
                "reflexion_feedback": None, "reflexion_response": None,
                "reflexion_status": "not_run", "reflexion_call_id": None,
                "correct": None, "executed": False,
                "effective_prediction": successful_prediction, "effective_correct": True,
                "carried_from_trial": success_trial, "skip_reason": "previous_trial_correct",
            }
            continue
        call = ledger.latest(article.index, trial, "classify")
        parsed = json.loads(call["parsed"]) if call and call["status"] == "ok" else {}
        prediction = parsed.get("prediction")
        correct = bool(is_correct(article.index, prediction)) if is_correct and prediction else None
        trials[str(trial)] = {
            "prediction": prediction, "rationale": parsed.get("rationale"),
            "response": call["response"] if call else None,
            "status": call["status"] if call else "pending",
            "classification_call_id": call["id"] if call else None,
            "reflexion_feedback": parsed.get("feedback") if trial > 1 else None,
            "reflexion_response": parsed.get("reflection_response") if trial > 1 else None,
            "reflexion_status": ("ok" if parsed.get("feedback") else "pending") if trial > 1 else "not_applicable",
            "reflexion_call_id": call["id"] if trial > 1 and parsed.get("feedback") else None,
            "correct": correct, "executed": call is not None,
            "effective_prediction": prediction, "effective_correct": correct,
            "carried_from_trial": None, "skip_reason": None,
        }
        if correct and (trial == 1 or trials[str(trial)]["reflexion_status"] == "ok"):
            success_trial, successful_prediction = trial, prediction
    exhausted = all(t["status"] == "ok" and
                    (key == "1" or t["reflexion_status"] == "ok") for key, t in trials.items())
    complete = success_trial is not None or exhausted
    call_count = ledger.call_count(article.index)
    exhausted_budget = not complete and call_count >= 3
    return {
        "index": article.index, "article_sha256": hashlib.sha256(article.article.encode()).hexdigest(),
        "trials": trials, "complete": complete, "success_trial": success_trial,
        "stop_reason": "correct" if success_trial else ("max_trials" if exhausted else ("call_budget_exhausted" if exhausted_budget else "incomplete")),
        "total_model_calls": call_count, "max_total_model_calls": 3,
        "attempted_classification_trials": sum(t["executed"] for t in trials.values()),
        "final_prediction": trials["3"]["effective_prediction"] if complete else None,
        "final_prediction_source_trial": (success_trial or 3) if complete else None,
        "final_prediction_was_carried": complete and success_trial in {1, 2},
    }


def save_article(ledger: Ledger, article: PromptArticle, *, is_correct=None) -> None:
    write_json(ledger.directory / "articles" / f"{article.index:03d}.json",
               article_state(ledger, article, is_correct=is_correct))


def run_inference(articles: tuple[PromptArticle, ...], ledger: Ledger, backend,
                  *, phase: str = "all", retry_errors: bool = False,
                  retry_uncertain: bool = False,
                  is_correct: Callable[[int, str], bool] | None = None) -> None:
    """Gold correctness controls retries only; it never enters model prompts."""
    if phase not in {"baseline", "all"}:
        raise ValueError(f"unknown phase: {phase}")
    if phase == "all" and is_correct is None:
        raise ValueError("gold-gated early stopping requires an evaluation-only is_correct callback")
    options = {"retry_errors": retry_errors, "retry_uncertain": retry_uncertain}
    ledger.limit_calls_per_article(3)
    def model_call(*args, **kwargs):
        try:
            return ledger.call(*args, **kwargs)
        except ModelCallBudgetExhausted as exc:
            print(f"[call_budget_exhausted] {exc}", flush=True)
            return None
    for article in articles:
        # Zero-shot is a SINGLE call, outside the Reflexion trial loop.
        # Its saved output is also Reflexion's Trial 1, never regenerated.
        parsed = model_call(
            article.index, 1, "classify", classification_prompt(article.article),
            parse_classification, backend, **options,
        )
        save_article(ledger, article, is_correct=is_correct)
        if parsed is not None and phase == "all":
            for trial in (2, 3):
                # Correctness is checked before the next single combined call.
                if is_correct(article.index, parsed["prediction"]):
                    break
                parsed = model_call(
                    article.index, trial, "classify",
                    integrated_reflexion_prompt(article.article, parsed["safe_rationale"]),
                    parse_integrated_reflexion, backend, **options,
                )
                save_article(ledger, article, is_correct=is_correct)
                if parsed is None:
                    break
        print(f"[checkpoint] article {article.index + 1}/{len(articles)}", flush=True)


def csv_file(path: Path, rows: list[dict]) -> None:
    if not rows:
        atomic_write(path, b"")
        return
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, buffer.getvalue().encode("utf-8"))


def export_state(ledger: Ledger, dataset) -> bool:
    # The evaluator also supplies the stopping signal; prompts stay label-blind.
    is_correct = lambda index, prediction: prediction == dataset.gold[index]
    records = [article_state(ledger, article, is_correct=is_correct) for article in dataset.articles]
    complete_articles = sum(record["complete"] for record in records)
    for record in records:
        expected = dataset.gold[record["index"]]
        record["gold_label"] = expected
        record["gold_source_label"] = expected == "real"
        write_json(ledger.directory / "articles" / f"{record['index']:03d}.json", record)
    calls = ledger.all_calls()
    errors = [{k: call[k] for k in (
        "id", "article_index", "trial", "kind", "attempt", "status", "error_type", "error_message",
        "response", "started_at", "completed_at",
    )} for call in calls if call["status"] in {"parse_error", "execution_error", "uncertain"}]
    counts = Counter(call["status"] for call in calls)
    complete = len(records) == complete_articles == 500
    write_json(ledger.directory / "progress.json", {
        "updated_at": timestamp(), "complete_articles": complete_articles, "required_articles": 500,
        "call_status_counts": dict(counts), "complete": complete,
        "evaluation_mode": "gold_gated_early_stop_cumulative", "oracle_scoring": True,
        "backend": ledger.manifest["provenance"]["backend"],
    })
    write_json(ledger.directory / "records.partial.json", records)
    write_json(ledger.directory / "errors.json", errors)
    csv_file(ledger.directory / "errors.csv", errors)
    if not complete:
        print(f"[incomplete] {complete_articles}/500 articles reached correctness or Trial 3. No final metrics generated.", flush=True)
        return False
    summary = summarize_early_stop_trials(records)
    summary["run_fingerprint"] = ledger.manifest["fingerprint"]
    summary["research_result"] = ledger.manifest["provenance"]["backend"].get("backend") != "mock"
    summary["dataset_sha256"] = dataset.validation["sha256"]
    write_json(ledger.directory / "results.json", {
        "run_fingerprint": ledger.manifest["fingerprint"],
        "evaluation_mode": summary["evaluation_mode"], "oracle_scoring": True, "records": records,
    })
    write_json(ledger.directory / "trial_summary.json", summary)
    metrics_rows = []
    for metric in summary["trials"]:
        row = {k: v for k, v in metric.items() if k not in {"confusion_matrix", "confusion_matrix_labels"}}
        row["confusion_matrix"] = json.dumps(metric["confusion_matrix"])
        row["matrix_order"] = "rows=gold, columns=prediction; real,fake"
        row["research_result"] = summary["research_result"]
        metrics_rows.append(row)
    csv_file(ledger.directory / "trial_summary.csv", metrics_rows)
    csv_file(ledger.directory / "trial_transitions.csv", [
        {**transition, "oracle_scoring": True} for transition in summary["transitions"]
    ])
    article_rows = []
    for record in records:
        row = {"index": record["index"], "gold_label": record["gold_label"],
               "gold_source_label": record["gold_source_label"], "oracle_scoring": True,
               "success_trial": record["success_trial"], "stop_reason": record["stop_reason"],
               "final_prediction": record["final_prediction"],
               "final_prediction_source_trial": record["final_prediction_source_trial"]}
        for trial in (1, 2, 3):
            for key, value in record["trials"][str(trial)].items():
                row[f"trial_{trial}_{key}"] = value
        article_rows.append(row)
    csv_file(ledger.directory / "results.csv", article_rows)
    print(f"[complete] Final evaluation saved in {ledger.directory}", flush=True)
    return True


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--backend", choices=("local", "mock"), default="local")
    p.add_argument("--phase", choices=("all", "baseline"), default="all",
                   help="all: at most 3 total model calls, with integrated self-review and early stopping; baseline: 1 call")
    p.add_argument("--limit", type=int, help="smoke only: first N official rows; no final metrics below 500")
    p.add_argument("--prepare-data", action="store_true", help="download and validate official data, then exit")
    p.add_argument("--check-environment", action="store_true", help="inspect existing local inference environment without loading weights")
    p.add_argument("--retry-errors", action="store_true", help="explicitly retry failed execution calls; successful calls are never repeated")
    p.add_argument("--retry-uncertain", action="store_true", help="explicitly retry calls interrupted before a durable response (may repeat generation)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.limit is not None and not 1 <= args.limit <= 500:
        parser().error("--limit must be between 1 and 500")
    if args.check_environment:
        report = {"runtime": runtime_metadata(), "generation": llm.generation_metadata()}
        try:
            backend = LocalBackend()
            import torch
            report.update({"cuda_available": torch.cuda.is_available(), "model_files_verified": len(backend.metadata["model_files"])})
            if not report["cuda_available"]:
                report["note"] = "CUDA unavailable; unchanged device_map=auto may use CPU"
        except Exception as exc:
            report["error"] = str(exc)
            print(json.dumps(report, indent=2))
            return 2
        print(json.dumps(report, indent=2))
        return 0
    try:
        dataset = load_official_test(args.data_path, download=args.prepare_data)
        if args.prepare_data:
            write_json(args.data_path.parent / "validation.json", dataset.validation)
            print(json.dumps(dataset.validation, indent=2))
            return 0
        backend = MockBackend() if args.backend == "mock" else LocalBackend()
        directory = args.output_dir or ROOT / "outputs" / f"propanews_zeroshot_three_calls_{args.backend}"
        provenance = build_provenance(dataset.validation, backend.metadata)
        # The stopping protocol changed: fixed-three-trial caches are not compatible.
        with Ledger(directory, provenance) as ledger:
            write_json(directory / "dataset_validation.json", dataset.validation)
            try:
                run_inference(
                    dataset.articles[:args.limit], ledger, backend, phase=args.phase,
                    retry_errors=args.retry_errors, retry_uncertain=args.retry_uncertain,
                    is_correct=lambda index, prediction: prediction == dataset.gold[index],
                )
            except KeyboardInterrupt:
                export_state(ledger, dataset)
                print("[interrupted] Saved. Re-run the same command to resume; inspect uncertain calls.", file=sys.stderr)
                return 130
            complete = export_state(ledger, dataset)
            if complete:
                return 0
            # A deliberately limited or baseline-only run is successful if its requested calls succeeded.
            for article in dataset.articles[:args.limit]:
                state = article_state(ledger, article,
                                      is_correct=lambda index, prediction: prediction == dataset.gold[index])
                if ((args.phase == "baseline" and state["trials"]["1"]["status"] != "ok") or
                        (args.phase == "all" and not state["complete"])):
                    return 2
            return 0
    except (OfficialTestError, CheckpointError, RuntimeError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
