"""Strict Finish[real|fake] evaluator for the PCoT PropaNews experiment."""

from __future__ import annotations

import re

from propanews_common import load_gold_labels


_FINISH_PATTERN = re.compile(
    r"Finish\[(.*?)\]",
    flags=re.IGNORECASE | re.DOTALL,
)

# The evaluator owns labels independently of actors.  The supplied data maps a
# JSON true label to an original/real article and false to manipulated/fake.
_GOLD_LABELS: dict[int, str] = {
    index: "real" if label else "fake"
    for index, label in load_gold_labels().items()
}


def extract_finish(response: str) -> str:
    """Return the last Finish value, matching the legacy evaluator behavior."""

    if not isinstance(response, str):
        return ""
    matches = _FINISH_PATTERN.findall(response)
    return matches[-1].strip() if matches else ""


def extract_prediction(response: str) -> str:
    """Return real/fake, or an empty string for every malformed response."""

    prediction = extract_finish(response).casefold()
    return prediction if prediction in {"real", "fake"} else ""


def get_gold_label(index: int) -> str:
    """Return the private canonical evaluation target for an index."""

    try:
        return _GOLD_LABELS[index]
    except KeyError as exc:
        raise IndexError(f"unknown PropaNews index: {index}") from exc


def classification_score(response: str, index: int) -> int:
    prediction = extract_prediction(response)
    return int(bool(prediction) and prediction == get_gold_label(index))


def EM_score(response: str, index: int) -> int:
    """Keep the legacy evaluator name used by graph orchestration."""

    return classification_score(response, index)


__all__ = [
    "EM_score",
    "classification_score",
    "extract_finish",
    "extract_prediction",
    "get_gold_label",
]
