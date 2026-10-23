"""Plot trial-1 and oracle-gated retry metrics from the PropaNews runner.

Unlike the original ``graph.py``, importing this module does not execute experiments or
write files.  The oracle-stopped results are deliberately presented as secondary,
private evaluation artifacts rather than ordinary final model performance.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "adsent": "AdSent",
    "adsent_gt": "AdSent + GT",
    "adsent_reflexion": "AdSent + Reflexion",
    "adsent_gt_reflexion": "AdSent + GT + Reflexion",
}
MAX_ORACLE_ATTEMPTS = 5
ORACLE_FINAL_SCOPE = "oracle_stopped_final_secondary"


def _comparison_scores(
    method_name: str,
    method_payload: Mapping[str, object],
    metric_scope: str = "trial1",
) -> tuple[float, float]:
    """Return accuracy and macro-F1 for an explicitly named metric scope."""

    try:
        source = method_payload[metric_scope]
        if not isinstance(source, Mapping):
            raise TypeError
        accuracy = source["accuracy"]
        macro_f1 = source["macro_f1"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"metrics for method {method_name!r} do not contain a valid "
            f"{metric_scope!r} score object"
        ) from exc
    scores = (accuracy, macro_f1)
    if any(
        type(score) not in (int, float)
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
        for score in scores
    ):
        raise ValueError(
            f"method {method_name!r} contains an invalid {metric_scope!r} score"
        )
    return float(accuracy), float(macro_f1)


def _oracle_stopped_scores(
    method_name: str,
    method_payload: Mapping[str, object],
) -> tuple[float, float]:
    """Read ``final`` only when it is explicitly scoped as oracle-stopped."""

    if method_payload.get("final_metric_scope") != ORACLE_FINAL_SCOPE:
        raise ValueError(
            f"method {method_name!r} does not identify final as an "
            "oracle-stopped secondary metric"
        )
    return _comparison_scores(method_name, method_payload, "final")


def _success_within_k(
    method_name: str,
    method_payload: Mapping[str, object],
) -> tuple[float, ...]:
    """Return the monotonic oracle success-within-k curve for k=1..5."""

    retry = method_payload.get("oracle_retry")
    if not isinstance(retry, Mapping):
        raise ValueError(f"method {method_name!r} has no oracle_retry object")
    if retry.get("max_attempts") != MAX_ORACLE_ATTEMPTS:
        raise ValueError(
            f"method {method_name!r} must use exactly {MAX_ORACLE_ATTEMPTS} attempts"
        )
    raw_scores = retry.get("success_within_k")
    if not isinstance(raw_scores, Mapping):
        raw_scores = retry.get("success_at_k")
    if not isinstance(raw_scores, Mapping):
        raise ValueError(
            f"method {method_name!r} has no success_within_k object"
        )

    values: list[float] = []
    for attempt in range(1, MAX_ORACLE_ATTEMPTS + 1):
        value = raw_scores.get(str(attempt))
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(
                f"method {method_name!r} contains an invalid success@{attempt}"
            )
        values.append(float(value))
    if any(later < earlier for earlier, later in zip(values, values[1:])):
        raise ValueError(
            f"method {method_name!r} contains a non-monotonic success-within-k curve"
        )
    return tuple(values)


def _load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("methods"), dict):
        raise ValueError("metrics file does not contain a methods object")
    return payload


def plot_metrics(metrics_path: Path, output_path: Path) -> None:
    """Render primary trial-1 and explicitly labelled oracle-gated results."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to create the plot") from exc

    payload = _load_metrics(metrics_path)
    methods: dict[str, Any] = payload["methods"]
    ordered_methods = [name for name in METHOD_LABELS if name in methods]
    if not ordered_methods:
        raise ValueError("metrics file has no supported methods")

    trial1_pairs = [
        _comparison_scores(name, methods[name], "trial1")
        for name in ordered_methods
    ]
    success_curves = [
        _success_within_k(name, methods[name]) for name in ordered_methods
    ]
    oracle_pairs = [
        _oracle_stopped_scores(name, methods[name]) for name in ordered_methods
    ]

    figure, axes = plt.subplots(1, 3, figsize=(19.5, 5.6))
    x_positions = list(range(len(ordered_methods)))
    x_labels = [METHOD_LABELS[name] for name in ordered_methods]

    # Panel 1: the only ordinary predictive-performance comparison is trial 1.
    width = 0.38
    axes[0].bar(
        [position - width / 2 for position in x_positions],
        [scores[0] for scores in trial1_pairs],
        width,
        label="Trial-1 Accuracy",
        color="#4C78A8",
    )
    axes[0].bar(
        [position + width / 2 for position in x_positions],
        [scores[1] for scores in trial1_pairs],
        width,
        label="Trial-1 Macro-F1",
        color="#F58518",
    )
    axes[0].set_xticks(x_positions, x_labels, rotation=22, ha="right")
    axes[0].set_title("(a) Trial-1 performance\n(primary; before oracle retry)")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    # Panel 2: cumulative success is explicitly an external-gold stopping metric.
    attempts = list(range(1, MAX_ORACLE_ATTEMPTS + 1))
    colors = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")
    markers = ("o", "s", "^", "D")
    for color, marker, method, curve in zip(
        colors, markers, ordered_methods, success_curves
    ):
        axes[1].plot(
            attempts,
            curve,
            marker=marker,
            linewidth=2,
            color=color,
            label=METHOD_LABELS[method],
        )
    axes[1].set_xticks(attempts)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Maximum attempts allowed (k)")
    axes[1].set_ylabel("Oracle success within k")
    axes[1].set_title("(b) Success within k\n(oracle-gated; private evaluation)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7.5)

    # Panel 3: show both metrics, but label the post-retry values as oracle-stopped.
    narrow_width = 0.19
    series = (
        (
            "Trial-1 Accuracy (primary)",
            [scores[0] for scores in trial1_pairs],
            "#9ECAE1",
            -1.5,
        ),
        (
            "Oracle-stopped <=5 Accuracy",
            [scores[0] for scores in oracle_pairs],
            "#2171B5",
            -0.5,
        ),
        (
            "Trial-1 Macro-F1 (primary)",
            [scores[1] for scores in trial1_pairs],
            "#FDD0A2",
            0.5,
        ),
        (
            "Oracle-stopped <=5 Macro-F1",
            [scores[1] for scores in oracle_pairs],
            "#E6550D",
            1.5,
        ),
    )
    for label, values, color, offset in series:
        axes[2].bar(
            [position + offset * narrow_width for position in x_positions],
            values,
            narrow_width,
            label=label,
            color=color,
        )
    axes[2].set_xticks(x_positions, x_labels, rotation=22, ha="right")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Score")
    axes[2].set_title(
        "(c) Trial 1 vs oracle-stopped within 5\n"
        "(post-retry values are secondary/private)"
    )
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=6.8, ncol=2)

    figure.suptitle(
        "PropaNews AdSent 2x2 — trial-1 primary; "
        "oracle-gated retry <=5 secondary/private",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="path to a run's metrics.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.metrics.with_name("adsent_propanews_summary.png")
    plot_metrics(args.metrics, output)
    print(f"Graph saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
