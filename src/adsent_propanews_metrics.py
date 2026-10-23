"""Strict, dependency-free metrics for prompt-based PropaNews detection.

Generic label outputs can be parsed as exact one-field JSON documents with
``parse_label_json``.  The richer detector responses used by this repository are
parsed with ``parse_detection_json``, which enforces their complete four-field
schema.  Invalid and ambiguous outputs remain an explicit abstention category;
they are never coerced to either substantive class.

All metric values are fractions in ``[0, 1]`` except MCC and signed accuracy
deltas, which can range from ``-1`` to ``1``.  Invalid outputs count as wrong for
accuracy, recall, F1, and robustness calculations while ``coverage`` reports how
often a valid class was produced.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import (
    Callable,
    Final,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    TypeAlias,
    cast,
)


CanonicalLabel: TypeAlias = Literal["human_original", "ai_manipulated"]
PredictionLabel: TypeAlias = CanonicalLabel | None

HUMAN_ORIGINAL: Final[CanonicalLabel] = "human_original"
AI_MANIPULATED: Final[CanonicalLabel] = "ai_manipulated"
INVALID: Final[str] = "invalid"
CANONICAL_LABELS: Final[tuple[CanonicalLabel, CanonicalLabel]] = (
    HUMAN_ORIGINAL,
    AI_MANIPULATED,
)
PREDICTION_COLUMNS: Final[tuple[str, str, str]] = (
    HUMAN_ORIGINAL,
    AI_MANIPULATED,
    INVALID,
)


class MetricsInputError(ValueError):
    """Raised when gold labels or paired evaluation inputs are malformed."""


class _DuplicateJSONKey(ValueError):
    """Internal signal used to reject ambiguous JSON objects."""


@dataclass(frozen=True, slots=True)
class ParsedPrediction:
    """Result of strict model-output parsing.

    ``label`` is ``None`` for every invalid result.  ``error`` is a compact,
    stable error code suitable for aggregating generation failures.
    """

    label: PredictionLabel
    error: str | None = None

    def __post_init__(self) -> None:
        if self.label is not None and self.label not in CANONICAL_LABELS:
            raise MetricsInputError(f"invalid parsed label: {self.label!r}")
        if self.label is None and self.error is None:
            raise MetricsInputError("an invalid ParsedPrediction needs an error code")
        if self.label is not None and self.error is not None:
            raise MetricsInputError("a valid ParsedPrediction cannot have an error")

    @property
    def valid(self) -> bool:
        """Whether parsing produced exactly one canonical label."""

        return self.label is not None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {"label": self.label, "valid": self.valid, "error": self.error}


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """One-vs-rest metrics for one canonical class."""

    label: CanonicalLabel
    precision: float
    recall: float
    f1: float
    support: int
    predicted: int
    true_positive: int
    false_positive: int
    false_negative: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
            "predicted": self.predicted,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
        }


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Binary classification metrics with an explicit invalid prediction column."""

    total: int
    correct: int
    valid_predictions: int
    invalid_predictions: int
    accuracy: float
    coverage: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    mcc: float
    per_class: Mapping[CanonicalLabel, ClassMetrics]
    confusion: Mapping[CanonicalLabel, Mapping[str, int]]

    @property
    def invalid_rate(self) -> float:
        """Fraction of all predictions that were invalid or ambiguous."""

        return _safe_divide(self.invalid_predictions, self.total)

    def to_dict(self) -> dict[str, object]:
        """Return a fully JSON-serializable representation."""

        return {
            "total": self.total,
            "correct": self.correct,
            "valid_predictions": self.valid_predictions,
            "invalid_predictions": self.invalid_predictions,
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "invalid_rate": self.invalid_rate,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "mcc": self.mcc,
            "per_class": {
                label: metrics.to_dict()
                for label, metrics in self.per_class.items()
            },
            "confusion": {
                gold: dict(columns) for gold, columns in self.confusion.items()
            },
        }


@dataclass(frozen=True, slots=True)
class VariantMetrics:
    """Paired original-versus-one-variant robustness statistics."""

    name: str
    classification: ClassificationMetrics
    paired_count: int
    valid_pair_count: int
    flip_count: int
    stable_valid_count: int
    strict_agreement_rate: float
    valid_pair_flip_rate: float
    disruption_rate: float
    robust_correct_count: int
    robust_correct_rate: float
    original_correct_variant_wrong: int
    original_wrong_variant_correct: int
    both_wrong: int
    accuracy_drop: float
    macro_f1_drop: float
    transition_counts: Mapping[str, int]

    @property
    def robustness_rate(self) -> float:
        """Alias for strict valid prediction agreement over every paired item."""

        return self.strict_agreement_rate

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "name": self.name,
            "classification": self.classification.to_dict(),
            "paired_count": self.paired_count,
            "valid_pair_count": self.valid_pair_count,
            "valid_pair_coverage": _safe_divide(
                self.valid_pair_count, self.paired_count
            ),
            "flip_count": self.flip_count,
            "flip_rate": _safe_divide(self.flip_count, self.paired_count),
            "valid_pair_flip_rate": self.valid_pair_flip_rate,
            "stable_valid_count": self.stable_valid_count,
            "strict_agreement_rate": self.strict_agreement_rate,
            "robustness_rate": self.robustness_rate,
            "disruption_rate": self.disruption_rate,
            "robust_correct_count": self.robust_correct_count,
            "robust_correct_rate": self.robust_correct_rate,
            "original_correct_variant_wrong": self.original_correct_variant_wrong,
            "original_wrong_variant_correct": self.original_wrong_variant_correct,
            "both_wrong": self.both_wrong,
            "accuracy_drop": self.accuracy_drop,
            "macro_f1_drop": self.macro_f1_drop,
            "transition_counts": dict(self.transition_counts),
        }


@dataclass(frozen=True, slots=True)
class VariantRobustnessSummary:
    """Baseline metrics plus paired summaries for named text variants."""

    baseline: ClassificationMetrics
    variants: Mapping[str, VariantMetrics]
    mean_variant_accuracy: float
    mean_variant_macro_f1: float
    mean_accuracy_drop: float
    mean_macro_f1_drop: float
    mean_strict_agreement_rate: float
    total_flips: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "baseline": self.baseline.to_dict(),
            "variants": {
                name: metrics.to_dict() for name, metrics in self.variants.items()
            },
            "mean_variant_accuracy": self.mean_variant_accuracy,
            "mean_variant_macro_f1": self.mean_variant_macro_f1,
            "mean_accuracy_drop": self.mean_accuracy_drop,
            "mean_macro_f1_drop": self.mean_macro_f1_drop,
            "mean_strict_agreement_rate": self.mean_strict_agreement_rate,
            "total_flips": self.total_flips,
        }


def parse_label_json(response: object, *, label_key: str = "label") -> ParsedPrediction:
    """Parse an exact one-field JSON classification object.

    Parsing is intentionally strict:

    * surrounding Markdown fences or explanatory prose are rejected;
    * the top-level JSON value must be an object with exactly ``label_key``;
    * duplicate keys at any object level are rejected;
    * the label value must exactly match a canonical lowercase label.

    The function never guesses from substrings, so a response mentioning both
    classes remains invalid instead of being forced into one of them.
    """

    if not isinstance(label_key, str) or not label_key:
        raise MetricsInputError("label_key must be a non-empty string")
    if not isinstance(response, str):
        return ParsedPrediction(label=None, error="non_string_response")
    if not response.strip():
        return ParsedPrediction(label=None, error="empty_response")

    try:
        payload = json.loads(
            response,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except _DuplicateJSONKey:
        return ParsedPrediction(label=None, error="duplicate_json_key")
    except (json.JSONDecodeError, ValueError, TypeError):
        return ParsedPrediction(label=None, error="invalid_json")

    if not isinstance(payload, dict):
        return ParsedPrediction(label=None, error="non_object_json")
    if set(payload) != {label_key}:
        return ParsedPrediction(label=None, error="wrong_json_schema")
    label = payload[label_key]
    if not isinstance(label, str):
        return ParsedPrediction(label=None, error="non_string_label")
    if label not in CANONICAL_LABELS:
        return ParsedPrediction(label=None, error="unknown_label")
    return ParsedPrediction(label=cast(CanonicalLabel, label))


def parse_label_json_many(
    responses: Iterable[object], *, label_key: str = "label"
) -> tuple[ParsedPrediction, ...]:
    """Strictly parse a batch while preserving one result per response."""

    return tuple(
        parse_label_json(response, label_key=label_key) for response in responses
    )


def parse_detection_json(response: object) -> ParsedPrediction:
    """Parse the exact four-field JSON emitted by the AdSent detector prompt."""

    if not isinstance(response, str):
        return ParsedPrediction(label=None, error="non_string_response")
    if not response.strip():
        return ParsedPrediction(label=None, error="empty_response")
    try:
        payload = json.loads(
            response,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except _DuplicateJSONKey:
        return ParsedPrediction(label=None, error="duplicate_json_key")
    except (json.JSONDecodeError, ValueError, TypeError):
        return ParsedPrediction(label=None, error="invalid_json")
    if not isinstance(payload, dict):
        return ParsedPrediction(label=None, error="non_object_json")
    expected = {
        "label",
        "sentiment_used_as_evidence",
        "key_claims",
        "brief_reason",
    }
    if set(payload) != expected:
        return ParsedPrediction(label=None, error="wrong_detection_schema")
    label = payload["label"]
    if not isinstance(label, str):
        return ParsedPrediction(label=None, error="non_string_label")
    if label not in CANONICAL_LABELS:
        return ParsedPrediction(label=None, error="unknown_label")
    if payload["sentiment_used_as_evidence"] is not False:
        return ParsedPrediction(label=None, error="sentiment_evidence_violation")
    key_claims = payload["key_claims"]
    if (
        not isinstance(key_claims, list)
        or not 1 <= len(key_claims) <= 3
        or not all(isinstance(claim, str) and claim.strip() for claim in key_claims)
    ):
        return ParsedPrediction(label=None, error="invalid_key_claims")
    reason = payload["brief_reason"]
    if not isinstance(reason, str) or not reason.strip():
        return ParsedPrediction(label=None, error="invalid_brief_reason")
    return ParsedPrediction(label=cast(CanonicalLabel, label))


def parse_detection_json_many(
    responses: Iterable[object],
) -> tuple[ParsedPrediction, ...]:
    """Parse a batch of repository detector responses without substring guessing."""

    return tuple(parse_detection_json(response) for response in responses)


def parser_error_counts(
    predictions: Iterable[ParsedPrediction],
) -> dict[str, int]:
    """Count invalid parser outcomes by stable error code."""

    counts: Counter[str] = Counter()
    for prediction in predictions:
        if not isinstance(prediction, ParsedPrediction):
            raise MetricsInputError("parser_error_counts expects ParsedPrediction values")
        if prediction.error is not None:
            counts[prediction.error] += 1
    return dict(sorted(counts.items()))


def compute_classification_metrics(
    gold_labels: Iterable[CanonicalLabel],
    predictions: Iterable[PredictionLabel | ParsedPrediction | object],
) -> ClassificationMetrics:
    """Compute accuracy, coverage, per-class/macro scores, MCC, and confusion.

    Invalid predictions use the third ``invalid`` confusion column and are wrong
    in the accuracy numerator.  Precision/recall/F1 are computed one-vs-rest for
    the two canonical classes, with invalid predictions contributing false
    negatives to their gold class.

    MCC uses the standard multiclass covariance formula over the two gold rows
    and three prediction columns.  This preserves invalid as an abstention-like
    outcome instead of mapping it to the opposite binary class; with no invalid
    outputs it is exactly ordinary binary MCC.
    """

    gold = tuple(_validate_gold_label(label) for label in gold_labels)
    predicted = tuple(_normalize_prediction(value) for value in predictions)
    if len(gold) != len(predicted):
        raise MetricsInputError(
            f"gold/prediction length mismatch: {len(gold)} != {len(predicted)}"
        )
    if not gold:
        raise MetricsInputError("at least one gold/prediction pair is required")

    confusion: dict[CanonicalLabel, dict[str, int]] = {
        label: {column: 0 for column in PREDICTION_COLUMNS}
        for label in CANONICAL_LABELS
    }
    for target, prediction in zip(gold, predicted):
        column = prediction if prediction is not None else INVALID
        confusion[target][column] += 1

    total = len(gold)
    correct = sum(confusion[label][label] for label in CANONICAL_LABELS)
    invalid_predictions = sum(confusion[label][INVALID] for label in CANONICAL_LABELS)
    valid_predictions = total - invalid_predictions

    class_results: dict[CanonicalLabel, ClassMetrics] = {}
    for label in CANONICAL_LABELS:
        true_positive = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_count = sum(confusion[row][label] for row in CANONICAL_LABELS)
        false_positive = predicted_count - true_positive
        false_negative = support - true_positive
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _f1(precision, recall)
        class_results[label] = ClassMetrics(
            label=label,
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
            predicted=predicted_count,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
        )

    return ClassificationMetrics(
        total=total,
        correct=correct,
        valid_predictions=valid_predictions,
        invalid_predictions=invalid_predictions,
        accuracy=_safe_divide(correct, total),
        coverage=_safe_divide(valid_predictions, total),
        macro_precision=_mean(
            metric.precision for metric in class_results.values()
        ),
        macro_recall=_mean(metric.recall for metric in class_results.values()),
        macro_f1=_mean(metric.f1 for metric in class_results.values()),
        mcc=_multiclass_mcc(confusion),
        per_class=class_results,
        confusion=confusion,
    )


def evaluate_json_responses(
    gold_labels: Iterable[CanonicalLabel],
    responses: Iterable[object],
    *,
    label_key: str = "label",
) -> ClassificationMetrics:
    """Strictly parse raw JSON model responses and compute all metrics."""

    parsed = parse_label_json_many(responses, label_key=label_key)
    return compute_classification_metrics(gold_labels, parsed)


def evaluate_detection_responses(
    gold_labels: Iterable[CanonicalLabel],
    responses: Iterable[object],
) -> ClassificationMetrics:
    """Parse rich detector JSON responses and compute classification metrics."""

    return compute_classification_metrics(
        gold_labels,
        parse_detection_json_many(responses),
    )


def summarize_variant_robustness(
    gold_labels: Sequence[CanonicalLabel] | Mapping[Hashable, CanonicalLabel],
    original_predictions: Sequence[object] | Mapping[Hashable, object],
    variant_predictions: Mapping[
        str, Sequence[object] | Mapping[Hashable, object]
    ],
) -> VariantRobustnessSummary:
    """Summarize paired flips and robustness for one or more text variants.

    Inputs may all be aligned sequences, or they may be ID-keyed mappings.  The
    mapping form performs an exact key-set join and is preferred for experiment
    artifacts.  Mixing the two forms, missing IDs, extra IDs, and length
    mismatches are rejected.

    Prediction values must already be canonical strings, ``None``, or
    :class:`ParsedPrediction` instances.  Use
    :func:`summarize_json_variant_robustness` for raw model responses.
    """

    gold, original, variants = _align_variant_inputs(
        gold_labels, original_predictions, variant_predictions
    )
    baseline = compute_classification_metrics(gold, original)
    results: dict[str, VariantMetrics] = {}

    for name, variant_values in variants.items():
        classification = compute_classification_metrics(gold, variant_values)
        valid_pair_count = 0
        flip_count = 0
        stable_valid_count = 0
        robust_correct_count = 0
        original_correct_variant_wrong = 0
        original_wrong_variant_correct = 0
        both_wrong = 0
        transitions: Counter[str] = Counter()

        for target, original_value, variant_value in zip(
            gold, original, variant_values
        ):
            original_label = _normalize_prediction(original_value)
            variant_label = _normalize_prediction(variant_value)
            original_valid = original_label is not None
            variant_valid = variant_label is not None
            if original_valid and variant_valid:
                valid_pair_count += 1
                if original_label == variant_label:
                    stable_valid_count += 1
                else:
                    flip_count += 1

            original_correct = original_label == target
            variant_correct = variant_label == target
            if original_correct and variant_correct:
                robust_correct_count += 1
            elif original_correct:
                original_correct_variant_wrong += 1
            elif variant_correct:
                original_wrong_variant_correct += 1
            else:
                both_wrong += 1

            transitions[
                _transition_key(target, original_label, variant_label)
            ] += 1

        paired_count = len(gold)
        strict_agreement_rate = _safe_divide(stable_valid_count, paired_count)
        results[name] = VariantMetrics(
            name=name,
            classification=classification,
            paired_count=paired_count,
            valid_pair_count=valid_pair_count,
            flip_count=flip_count,
            stable_valid_count=stable_valid_count,
            strict_agreement_rate=strict_agreement_rate,
            valid_pair_flip_rate=_safe_divide(flip_count, valid_pair_count),
            disruption_rate=1.0 - strict_agreement_rate,
            robust_correct_count=robust_correct_count,
            robust_correct_rate=_safe_divide(robust_correct_count, paired_count),
            original_correct_variant_wrong=original_correct_variant_wrong,
            original_wrong_variant_correct=original_wrong_variant_correct,
            both_wrong=both_wrong,
            accuracy_drop=baseline.accuracy - classification.accuracy,
            macro_f1_drop=baseline.macro_f1 - classification.macro_f1,
            transition_counts=dict(sorted(transitions.items())),
        )

    return VariantRobustnessSummary(
        baseline=baseline,
        variants=results,
        mean_variant_accuracy=_mean(
            variant.classification.accuracy for variant in results.values()
        ),
        mean_variant_macro_f1=_mean(
            variant.classification.macro_f1 for variant in results.values()
        ),
        mean_accuracy_drop=_mean(
            variant.accuracy_drop for variant in results.values()
        ),
        mean_macro_f1_drop=_mean(
            variant.macro_f1_drop for variant in results.values()
        ),
        mean_strict_agreement_rate=_mean(
            variant.strict_agreement_rate for variant in results.values()
        ),
        total_flips=sum(variant.flip_count for variant in results.values()),
    )


def summarize_json_variant_robustness(
    gold_labels: Sequence[CanonicalLabel] | Mapping[Hashable, CanonicalLabel],
    original_responses: Sequence[object] | Mapping[Hashable, object],
    variant_responses: Mapping[
        str, Sequence[object] | Mapping[Hashable, object]
    ],
    *,
    label_key: str = "label",
) -> VariantRobustnessSummary:
    """Strictly parse raw JSON responses, then compute paired robustness."""

    parsed_original = _map_container(
        original_responses,
        lambda response: parse_label_json(response, label_key=label_key),
    )
    parsed_variants = {
        name: _map_container(
            responses,
            lambda response: parse_label_json(response, label_key=label_key),
        )
        for name, responses in variant_responses.items()
    }
    return summarize_variant_robustness(
        gold_labels,
        parsed_original,
        parsed_variants,
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


def _validate_gold_label(value: object) -> CanonicalLabel:
    if not isinstance(value, str) or value not in CANONICAL_LABELS:
        raise MetricsInputError(
            "gold labels must be exactly 'human_original' or 'ai_manipulated'; "
            f"got {value!r}"
        )
    return cast(CanonicalLabel, value)


def _normalize_prediction(value: object) -> PredictionLabel:
    if isinstance(value, ParsedPrediction):
        return value.label
    if isinstance(value, str) and value in CANONICAL_LABELS:
        return cast(CanonicalLabel, value)
    return None


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_divide(2.0 * precision * recall, precision + recall)


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    return _safe_divide(sum(materialized), len(materialized))


def _multiclass_mcc(
    confusion: Mapping[CanonicalLabel, Mapping[str, int]],
) -> float:
    # Add a zero-support invalid gold row conceptually, producing a square 3x3
    # table while keeping invalid predictions distinct from either real class.
    classes = PREDICTION_COLUMNS
    row_sums: dict[str, int] = {
        HUMAN_ORIGINAL: sum(confusion[HUMAN_ORIGINAL].values()),
        AI_MANIPULATED: sum(confusion[AI_MANIPULATED].values()),
        INVALID: 0,
    }
    column_sums = {
        label: sum(confusion[row][label] for row in CANONICAL_LABELS)
        for label in classes
    }
    sample_count = sum(row_sums.values())
    correct = sum(
        confusion[label][label] for label in CANONICAL_LABELS
    )
    covariance = correct * sample_count - sum(
        row_sums[label] * column_sums[label] for label in classes
    )
    true_variance = sample_count * sample_count - sum(
        count * count for count in row_sums.values()
    )
    predicted_variance = sample_count * sample_count - sum(
        count * count for count in column_sums.values()
    )
    denominator = math.sqrt(true_variance * predicted_variance)
    return _safe_divide(covariance, denominator)


def _align_variant_inputs(
    gold_labels: Sequence[CanonicalLabel] | Mapping[Hashable, CanonicalLabel],
    original_predictions: Sequence[object] | Mapping[Hashable, object],
    variant_predictions: Mapping[
        str, Sequence[object] | Mapping[Hashable, object]
    ],
) -> tuple[
    tuple[CanonicalLabel, ...],
    tuple[object, ...],
    dict[str, tuple[object, ...]],
]:
    if not isinstance(variant_predictions, Mapping) or not variant_predictions:
        raise MetricsInputError("at least one named variant is required")
    for name in variant_predictions:
        if not isinstance(name, str) or not name:
            raise MetricsInputError("variant names must be non-empty strings")

    if isinstance(gold_labels, Mapping):
        if not isinstance(original_predictions, Mapping):
            raise MetricsInputError(
                "mapping gold labels require mapping original predictions"
            )
        ids = tuple(gold_labels.keys())
        expected = set(ids)
        _require_exact_keys("original predictions", original_predictions, expected)
        aligned_variants: dict[str, tuple[object, ...]] = {}
        for name, values in variant_predictions.items():
            if not isinstance(values, Mapping):
                raise MetricsInputError(
                    "mapping gold labels require mapping variant predictions"
                )
            _require_exact_keys(f"variant {name!r}", values, expected)
            aligned_variants[name] = tuple(values[item_id] for item_id in ids)
        gold = tuple(_validate_gold_label(gold_labels[item_id]) for item_id in ids)
        original = tuple(original_predictions[item_id] for item_id in ids)
        if not gold:
            raise MetricsInputError("at least one paired item is required")
        return gold, original, aligned_variants

    if isinstance(original_predictions, Mapping):
        raise MetricsInputError(
            "sequence gold labels require sequence original predictions"
        )
    if not _is_nonstring_sequence(gold_labels) or not _is_nonstring_sequence(
        original_predictions
    ):
        raise MetricsInputError("aligned inputs must be sequences or mappings")
    gold = tuple(_validate_gold_label(label) for label in gold_labels)
    original = tuple(original_predictions)
    if not gold:
        raise MetricsInputError("at least one paired item is required")
    if len(gold) != len(original):
        raise MetricsInputError(
            f"gold/original length mismatch: {len(gold)} != {len(original)}"
        )
    aligned_variants = {}
    for name, values in variant_predictions.items():
        if isinstance(values, Mapping) or not _is_nonstring_sequence(values):
            raise MetricsInputError(
                "sequence gold labels require sequence variant predictions"
            )
        materialized = tuple(values)
        if len(materialized) != len(gold):
            raise MetricsInputError(
                f"gold/variant {name!r} length mismatch: "
                f"{len(gold)} != {len(materialized)}"
            )
        aligned_variants[name] = materialized
    return gold, original, aligned_variants


def _require_exact_keys(
    description: str,
    values: Mapping[Hashable, object],
    expected: set[Hashable],
) -> None:
    actual = set(values)
    if actual == expected:
        return
    missing = expected - actual
    extra = actual - expected
    raise MetricsInputError(
        f"{description} ID set differs: missing={_stable_repr(missing)}, "
        f"extra={_stable_repr(extra)}"
    )


def _stable_repr(values: set[Hashable]) -> str:
    return repr(sorted((repr(value) for value in values)))


def _transition_key(
    gold: CanonicalLabel,
    original: PredictionLabel,
    variant: PredictionLabel,
) -> str:
    original_text = original if original is not None else INVALID
    variant_text = variant if variant is not None else INVALID
    return f"gold={gold}|original={original_text}->variant={variant_text}"


def _map_container(
    values: Sequence[object] | Mapping[Hashable, object],
    function: Callable[[object], object],
) -> Sequence[object] | Mapping[Hashable, object]:
    if isinstance(values, Mapping):
        return {key: function(value) for key, value in values.items()}
    if not _is_nonstring_sequence(values):
        raise MetricsInputError("response inputs must be sequences or mappings")
    return tuple(function(value) for value in values)


def _is_nonstring_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


# Concise aliases for experiment drivers.
compute_metrics = compute_classification_metrics
evaluate_json_predictions = evaluate_json_responses
summarize_variants = summarize_variant_robustness


__all__ = [
    "AI_MANIPULATED",
    "CANONICAL_LABELS",
    "CanonicalLabel",
    "ClassMetrics",
    "ClassificationMetrics",
    "HUMAN_ORIGINAL",
    "INVALID",
    "MetricsInputError",
    "PREDICTION_COLUMNS",
    "ParsedPrediction",
    "PredictionLabel",
    "VariantMetrics",
    "VariantRobustnessSummary",
    "compute_classification_metrics",
    "compute_metrics",
    "evaluate_detection_responses",
    "evaluate_json_predictions",
    "evaluate_json_responses",
    "parse_detection_json",
    "parse_detection_json_many",
    "parse_label_json",
    "parse_label_json_many",
    "parser_error_counts",
    "summarize_json_variant_robustness",
    "summarize_variant_robustness",
    "summarize_variants",
]
