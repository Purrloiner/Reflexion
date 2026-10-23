"""Thin run wrappers for the two article-only SPRS experiment arms."""

from __future__ import annotations

from collections.abc import Sequence

if __package__:
    from .actor_mtcpsl import SPRS_actor
else:
    from actor_mtcpsl import SPRS_actor


def run_SPRS(index: int) -> str:
    return SPRS_actor(index, use_gt=False)


def run_SPRS_Re(
    index: int,
    reflection: str | Sequence[str] | None,
) -> str:
    return SPRS_actor(
        index,
        reflection=reflection,
        use_gt=False,
    )


def run_SPRS_GT(index: int) -> str:
    return SPRS_actor(index, use_gt=True)


def run_SPRS_GT_Re(
    index: int,
    reflection: str | Sequence[str] | None,
) -> str:
    return SPRS_actor(
        index,
        reflection=reflection,
        use_gt=True,
    )

__all__ = ["run_SPRS", "run_SPRS_Re"]
