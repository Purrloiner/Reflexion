"""Evaluation only: actual predictions per trial, never cumulative oracle scores."""

from __future__ import annotations


def classification_metrics(gold: list[str], predictions: list[str]) -> dict:
    if not gold or len(gold) != len(predictions):
        raise ValueError("evaluation requires equal nonempty gold and prediction sequences")
    if any(value not in {"real", "fake"} for value in gold + predictions):
        raise ValueError("missing/invalid predictions cannot be silently mapped to a class")
    labels = {"real": 0, "fake": 1}
    matrix = [[0, 0], [0, 0]]
    for expected, actual in zip(gold, predictions):
        matrix[labels[expected]][labels[actual]] += 1
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    ratio = lambda n, d: n / d if d else 0.0
    f1_fake = ratio(2 * tp, 2 * tp + fp + fn)
    f1_real = ratio(2 * tn, 2 * tn + fp + fn)
    return {
        "n": len(gold), "accuracy": (tp + tn) / len(gold),
        "precision_fake": ratio(tp, tp + fp), "recall_fake": ratio(tp, tp + fn),
        "f1_fake": f1_fake, "macro_f1": (f1_real + f1_fake) / 2,
        "confusion_matrix": matrix, "confusion_matrix_labels": ["real", "fake"],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "positive_class": "fake", "zero_division": 0,
    }


def summarize_trials(records: list[dict]) -> dict:
    if len(records) != 500 or [r["index"] for r in records] != list(range(500)):
        raise ValueError("final evaluation requires all 500 official articles in source order")
    gold = [r["gold_label"] for r in records]
    if gold.count("real") != 243 or gold.count("fake") != 257:
        raise ValueError("gold distribution must be real=243, fake=257")
    trial_metrics = []
    predictions = {}
    for trial in (1, 2, 3):
        predictions[trial] = [r["trials"][str(trial)]["prediction"] for r in records]
        trial_metrics.append({
            "trial": trial,
            "condition": "zero_shot" if trial == 1 else "zero_shot_reflexion",
            **classification_metrics(gold, predictions[trial]),
        })
    transitions = []
    for before, after in ((1, 2), (2, 3), (1, 3)):
        previous = [p == y for p, y in zip(predictions[before], gold)]
        current = [p == y for p, y in zip(predictions[after], gold)]
        fixed = sum(not old and new for old, new in zip(previous, current))
        regressed = sum(old and not new for old, new in zip(previous, current))
        a, b = trial_metrics[before - 1], trial_metrics[after - 1]
        transitions.append({
            "from_trial": before, "to_trial": after,
            "wrong_to_correct": fixed, "correct_to_wrong": regressed,
            "net_correct_change": fixed - regressed,
            "correct_to_correct": sum(old and new for old, new in zip(previous, current)),
            "wrong_to_wrong": sum(not old and not new for old, new in zip(previous, current)),
            "accuracy_delta": b["accuracy"] - a["accuracy"],
            "f1_fake_delta": b["f1_fake"] - a["f1_fake"],
            "macro_f1_delta": b["macro_f1"] - a["macro_f1"],
        })
    return {
        "evaluation_mode": "per_trial_actual_prediction_no_oracle",
        "reflexion_final_trial": 3,
        "trials": trial_metrics, "transitions": transitions,
        "final_corrected_errors": transitions[-1]["wrong_to_correct"],
        "final_regressions": transitions[-1]["correct_to_wrong"],
        "final_net_correct_change": transitions[-1]["net_correct_change"],
    }
