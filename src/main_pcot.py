"""Thin run wrappers for the two PCoT PropaNews experiment arms."""

from __future__ import annotations

from collections.abc import Sequence

from actor_pcot import PCoT_actor
from pcot_prompts import PCOT_FINAL_MODE


def _run_pcot(
    index: int,
    *,
    trajectories: str | Sequence[str] | None,
    reflections: str | Sequence[str] | None,
    use_gt: bool,
    final_mode: str,
) -> dict[str, object]:
    return PCoT_actor(
        index,
        trajectories=trajectories,
        reflections=reflections,
        use_gt=use_gt,
        final_mode=final_mode,
    )


def run_PCoT(
    index: int,
    *,
    final_mode: str = PCOT_FINAL_MODE,
) -> dict[str, object]:
    return _run_pcot(
        index,
        trajectories=None,
        reflections=None,
        use_gt=False,
        final_mode=final_mode,
    )


def run_PCoT_GT_EPM_Re(
    index: int,
    trajectories: str | Sequence[str] | None,
    reflections: str | Sequence[str] | None,
    *,
    final_mode: str = PCOT_FINAL_MODE,
) -> dict[str, object]:
    return _run_pcot(
        index,
        trajectories=trajectories,
        reflections=reflections,
        use_gt=True,
        final_mode=final_mode,
    )


def run_PCoT_Re(
    index: int,
    reflections: str | Sequence[str] | None,
    *,
    final_mode: str = PCOT_FINAL_MODE,
) -> dict[str, object]:
    """Compatibility wrapper; the graph no longer schedules this arm."""

    return run_PCoT_GT_EPM_Re(
        index,
        trajectories=None,
        reflections=reflections,
        final_mode=final_mode,
    )


__all__ = ["run_PCoT", "run_PCoT_GT_EPM_Re", "run_PCoT_Re"]
