"""Strict Finish[real|fake] evaluation for the PropaNews SPRS experiment."""

from __future__ import annotations

import re

if __package__:
    from .propanews_common import load_gold_labels
else:
    from propanews_common import load_gold_labels


_FINISH_PATTERN = re.compile(
    r"Finish\[(.*?)\]",
    flags=re.IGNORECASE | re.DOTALL,
)

_GOLD_LABELS: dict[int, str] = {
    index: "real" if label else "fake"
    for index, label in load_gold_labels().items()
}


def extract_finish(response: str) -> str:
    """Reuse the existing evaluator convention: return the last Finish value."""

    if not isinstance(response, str):
        return ""
    matches = _FINISH_PATTERN.findall(response)
    if not matches:
        return ""
    return matches[-1].strip()


def extract_prediction(response: str) -> str:
    """Return a canonical prediction, or ``""`` for malformed output."""

    prediction = extract_finish(response).casefold()
    return prediction if prediction in {"real", "fake"} else ""


def get_gold_label(index: int) -> str:
    """Return the private evaluation label for a stable dataset index."""

    try:
        return _GOLD_LABELS[index]
    except KeyError as exc:
        raise IndexError(f"unknown PropaNews index: {index}") from exc


def classification_score(response: str, index: int) -> int:
    prediction = extract_prediction(response)
    if not prediction:
        return 0
    return int(prediction == get_gold_label(index))


# Keep the evaluator API used by the existing graph code.
def EM_score(response: str, index: int) -> int:
    return classification_score(response, index)


__all__ = [
    "EM_score",
    "classification_score",
    "extract_finish",
    "extract_prediction",
    "get_gold_label",
]
