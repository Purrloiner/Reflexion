"""Two-stage PCoT actors for the PropaNews experiment.

The public flow mirrors the legacy actor module:

    prompt builder -> actor -> call_llm

Stage 1 receives the target article plus explicitly enabled GT/EPM/Reflexion
context.  Stage 2 receives the original article again, the new raw Stage-1
response, and the same optional memories.  Dataset labels and paths never
enter this module.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pcot_prompts import (
    PCOT_FINAL_MODE,
    build_final_prompt,
    build_persuasion_prompt,
)
from propanews_common import PromptArticle, load_prompt_articles


def call_llm(prompt: str) -> str:
    """Lazily use the PCoT-only deterministic LLM adapter."""

    from llm_pcot import call_llm as pcot_call_llm

    return pcot_call_llm(prompt)


# Retain only model-visible fields.  Keep a separate order tuple because the
# experiment must follow the JSON file order rather than sorting or reindexing.
_PROMPT_ARTICLES = load_prompt_articles()
_PROMPT_INPUTS: dict[int, PromptArticle] = {
    record.index: record for record in _PROMPT_ARTICLES
}
_AVAILABLE_INDICES = tuple(record.index for record in _PROMPT_ARTICLES)
del _PROMPT_ARTICLES


_JSON_FENCE = re.compile(
    r"```(?:json|python)?\s*(\{.*?\})\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)
_PERSUASION_KEYS = {
    "Attack_on_reputation",
    "Justification",
    "Simplification",
    "Distraction",
    "Call",
    "Manipulative_wording",
}


def get_available_indices() -> tuple[int, ...]:
    """Return stored PropaNews indices in their original JSON file order."""

    return _AVAILABLE_INDICES


def _get_prompt_input(index: int) -> PromptArticle:
    try:
        return _PROMPT_INPUTS[index]
    except KeyError as exc:
        raise IndexError(f"unknown PropaNews index: {index}") from exc


def get_article(index: int) -> str:
    """Return only the target ``txt`` field for an index."""

    return _get_prompt_input(index).article


def PCoT_persuasion_prompt(
    index: int,
    reflections: str | Sequence[str] | None = None,
    trajectories: str | Sequence[str] | None = None,
    *,
    use_gt: bool = False,
) -> str:
    """Build the official-derived DMT persuasion prompt for Stage 1."""

    return build_persuasion_prompt(
        article=get_article(index),
        reflections=reflections,
        trajectories=trajectories,
        use_gt=use_gt,
    )


def PCoT_persuasion_actor(
    index: int,
    reflections: str | Sequence[str] | None = None,
    trajectories: str | Sequence[str] | None = None,
    *,
    use_gt: bool = False,
) -> str:
    """Run one fresh Stage-1 inference."""

    return call_llm(
        PCoT_persuasion_prompt(
            index,
            reflections=reflections,
            trajectories=trajectories,
            use_gt=use_gt,
        )
    )


def PCoT_final_prompt(
    index: int,
    persuasion_analysis: str,
    reflections: str | Sequence[str] | None = None,
    trajectories: str | Sequence[str] | None = None,
    use_gt: bool = False,
    final_mode: str = PCOT_FINAL_MODE,
) -> str:
    """Build Stage 2 with the original article and raw persuasion analysis."""

    return build_final_prompt(
        article=get_article(index),
        persuasion_analysis=persuasion_analysis,
        reflections=reflections,
        trajectories=trajectories,
        use_gt=use_gt,
        final_mode=final_mode,
    )


def PCoT_final_actor(
    index: int,
    persuasion_analysis: str,
    reflections: str | Sequence[str] | None = None,
    trajectories: str | Sequence[str] | None = None,
    use_gt: bool = False,
    final_mode: str = PCOT_FINAL_MODE,
) -> str:
    """Run the second and final PCoT inference."""

    prompt = PCoT_final_prompt(
        index,
        persuasion_analysis=persuasion_analysis,
        reflections=reflections,
        trajectories=trajectories,
        use_gt=use_gt,
        final_mode=final_mode,
    )
    return call_llm(prompt)


def _candidate_objects(response: str) -> tuple[str, ...]:
    candidates: list[str] = []
    stripped = response.strip()
    if stripped:
        candidates.append(stripped)
    candidates.extend(match.strip() for match in _JSON_FENCE.findall(response))
    start = response.find("{")
    end = response.rfind("}")
    if start >= 0 and end > start:
        candidates.append(response[start : end + 1].strip())

    # Preserve order while avoiding repeated parser work.
    return tuple(dict.fromkeys(candidates))


def _valid_persuasion_mapping(value: object) -> bool:
    if not isinstance(value, Mapping) or not _PERSUASION_KEYS.issubset(value):
        return False
    for key in _PERSUASION_KEYS:
        item = value[key]
        if not isinstance(item, Mapping):
            return False
        if "is_used" not in item or "explanation" not in item:
            return False
        is_used = item["is_used"]
        explanation = item["explanation"]
        if not isinstance(is_used, str) or is_used.strip().casefold() not in {
            "yes",
            "no",
        }:
            return False
        if not isinstance(explanation, str) or not explanation.strip():
            return False
    return True


def parse_persuasion_analysis(response: str) -> dict[str, Any] | None:
    """Best-effort Stage-1 parser; raw text remains authoritative context."""

    if not isinstance(response, str) or not response.strip():
        return None
    for candidate in _candidate_objects(response):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if _valid_persuasion_mapping(parsed):
                return dict(parsed)
    return None


def PCoT_actor(
    index: int,
    reflections: str | Sequence[str] | None = None,
    trajectories: str | Sequence[str] | None = None,
    use_gt: bool = False,
    final_mode: str = PCOT_FINAL_MODE,
) -> dict[str, object]:
    """Run both PCoT stages and preserve both outputs.

    Parsing never controls execution: even malformed JSON-like Stage-1 output
    is passed verbatim to Stage 2 and merely receives a ``None`` parsed value.
    """

    persuasion = PCoT_persuasion_actor(
        index,
        reflections=reflections,
        trajectories=trajectories,
        use_gt=use_gt,
    )
    final = PCoT_final_actor(
        index,
        persuasion_analysis=persuasion,
        reflections=reflections,
        trajectories=trajectories,
        use_gt=use_gt,
        final_mode=final_mode,
    )
    return {
        "persuasion": persuasion,
        "persuasion_parsed": parse_persuasion_analysis(persuasion),
        "final": final,
    }


__all__ = [
    "PCoT_actor",
    "PCoT_final_actor",
    "PCoT_final_prompt",
    "PCoT_persuasion_actor",
    "PCoT_persuasion_prompt",
    "get_article",
    "get_available_indices",
    "parse_persuasion_analysis",
]
