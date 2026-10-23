"""Evaluation-only label access for the full PropaNews test set."""

from __future__ import annotations

import re

from propanews_common import load_gold_labels


_FINISH_PATTERN = re.compile(
    r"Finish\[(.*?)\]",
    flags=re.IGNORECASE | re.DOTALL,
)
_GOLD_LABELS: dict[int, bool] = load_gold_labels()


def extract_finish(response: str) -> str:
    if not isinstance(response, str):
        return ""
    matches = _FINISH_PATTERN.findall(response)
    return matches[-1].strip().casefold() if matches else ""


def extract_prediction(response: str) -> str:
    prediction = extract_finish(response)
    return prediction if prediction in {"real", "fake"} else ""


def get_gold_label_bool(index: int) -> bool:
    try:
        return _GOLD_LABELS[index]
    except KeyError as exc:
        raise IndexError(f"unknown PropaNews index: {index}") from exc


def get_gold_label(index: int) -> str:
    return "real" if get_gold_label_bool(index) else "fake"


def classification_score(response: str, index: int) -> int:
    prediction = extract_prediction(response)
    return int(bool(prediction) and prediction == get_gold_label(index))


def EM_score(response: str, index: int) -> int:
    """Keep the evaluator API used by the existing experiment runners."""

    return classification_score(response, index)


__all__ = [
    "EM_score",
    "classification_score",
    "extract_finish",
    "extract_prediction",
    "get_gold_label",
    "get_gold_label_bool",
]
