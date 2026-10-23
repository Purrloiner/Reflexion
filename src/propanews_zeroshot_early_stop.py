"""Gold-gated early stopping: explicitly oracle/cumulative evaluation."""

from __future__ import annotations

from propanews_zeroshot_metrics import summarize_trials


EARLY_STOP_PROTOCOL = {
    "name": "gold_gated_early_stop_integrated_reflexion",
    "trials": 3,
    "max_classification_calls_per_article": 3,
    "max_total_model_calls_per_article": 3,
    "failed_and_interrupted_attempts_consume_budget": True,
    "zero_shot_classification_calls_per_article": 1,
    "max_reflexion_additional_classification_calls_per_article": 2,
    "max_reflection_calls_per_article": 0,
    "reflexion_mode": "integrated_in_classification_calls_2_and_3",
    "trial_1_shared": True,
    "early_stop_on_correct": True,
    "oracle_scoring": True,
    "gold_correctness_used_for_control": True,
    "gold_label_in_model_prompt": False,
    "correctness_signal_in_reflection_prompt": False,
    "skipped_trial_prediction": "null; successful prediction carried in effective_prediction only",
    "gt_knowledge": False, "epm": False, "few_shot": False,
    "reflection_memory": "latest sanitized rationale reviewed within the next classification call",
}


def summarize_early_stop_trials(records: list[dict]) -> dict:
    """Score the stopped policy, preserving null actual predictions on skips."""
    effective_records = []
    for record in records:
        if not record.get("complete"):
            raise ValueError("early-stop evaluation requires every article to reach correctness or Trial 3")
        source_trial = None
        carried_prediction = None
        effective_trials = {}
        for trial in (1, 2, 3):
            state = record["trials"][str(trial)]
            if source_trial is not None:
                if (state["status"] != "skipped_after_correct" or
                        state["prediction"] is not None or state["response"] is not None or
                        state["carried_from_trial"] != source_trial or
                        state["effective_prediction"] != carried_prediction):
                    raise ValueError("skipped trials must retain null actual output and an explicit successful source")
            else:
                if state["status"] != "ok" or (trial > 1 and state["reflexion_status"] != "ok"):
                    raise ValueError("an executed trial requires a valid classification and preceding reflection")
                if state["effective_prediction"] != state["prediction"]:
                    raise ValueError("executed trial must be scored using its actual prediction")
                if state["prediction"] == record["gold_label"]:
                    source_trial, carried_prediction = trial, state["prediction"]
            effective_trials[str(trial)] = {"prediction": state["effective_prediction"]}
        effective_records.append({
            "index": record["index"], "gold_label": record["gold_label"], "trials": effective_trials,
        })
    summary = summarize_trials(effective_records)
    summary.update({
        "evaluation_mode": "gold_gated_early_stop_cumulative",
        "oracle_scoring": True, "early_stop_on_correct": True,
        "score_source": "effective_prediction_after_gold_gated_stopping",
        "interpretation": "Gold labels decide which articles are retried. These are oracle/cumulative scores, not label-free fixed-Trial-3 performance.",
    })
    for metric in summary["trials"]:
        trial = metric["trial"]
        metric["condition"] = "zero_shot" if trial == 1 else "zero_shot_reflexion_oracle_early_stop"
        metric["oracle_scoring"] = True
        metric["executed_articles"] = sum(r["trials"][str(trial)]["status"] == "ok" for r in records)
        metric["carried_articles"] = sum(r["trials"][str(trial)]["status"] == "skipped_after_correct" for r in records)
    summary["solved_at_trial"] = [sum(r["success_trial"] == t for r in records) for t in (1, 2, 3)]
    summary["cumulative_success"] = [metric["accuracy"] for metric in summary["trials"]]
    return summary
