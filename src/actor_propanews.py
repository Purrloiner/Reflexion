"""Vanilla and chain-of-thought actors for the full PropaNews dataset."""

from __future__ import annotations

from collections.abc import Sequence

from propanews_common import (
    COMMON_NEWS_INSTRUCTION,
    PromptArticle,
    extract_epm_reasoning,
    load_prompt_articles,
)
from reflection_propanews import sanitize_reflection


# Keep only model-visible fields in the actor process. Dataset labels and paths
# are loaded exclusively by evaluator_propanews.
_PROMPT_ROWS: tuple[PromptArticle, ...] = load_prompt_articles()
_ARTICLES: dict[int, str] = {row.index: row.article for row in _PROMPT_ROWS}
_ARTICLE_ORDER: tuple[int, ...] = tuple(row.index for row in _PROMPT_ROWS)
del _PROMPT_ROWS


def call_llm(prompt: str) -> str:
    """Lazily use the project's unchanged model configuration."""

    from llm import call_llm as project_call_llm

    return project_call_llm(prompt)


def get_available_indices() -> tuple[int, ...]:
    """Return stored indices in exactly the JSON file's order."""

    return _ARTICLE_ORDER


def get_article(index: int) -> str:
    try:
        return _ARTICLES[index]
    except KeyError as exc:
        raise IndexError(f"unknown PropaNews index: {index}") from exc


def _format_reflections(reflections: str | Sequence[str] | None) -> str:
    if reflections is None:
        return ""
    candidates = (reflections,) if isinstance(reflections, str) else tuple(reflections)
    notes: list[str] = []
    for position, note in enumerate(candidates):
        if not isinstance(note, str):
            raise TypeError(f"reflections[{position}] must be a string")
        if note.strip():
            safe_note = sanitize_reflection(note)
            if not safe_note:
                raise ValueError(
                    f"reflections[{position}] contains no reasoning-only content"
                )
            notes.append(safe_note)
    if not notes:
        return ""

    rendered = "\n\n".join(
        f"Reflection {position}:\n{note}"
        for position, note in enumerate(notes, start=1)
    )
    return f"""
SELF-REFLECTION NOTES FROM EARLIER ATTEMPTS:
These notes are reasoning critiques, not verified facts or evaluator feedback.
Re-check them against the current article and do not preserve a previous
conclusion by default.

{rendered}
"""


def _format_previous_trajectory(previous_trajectory: str | None) -> str:
    """Render only the reasoning from the latest failed CoT attempt as EPM."""

    if previous_trajectory is None:
        return ""
    if not isinstance(previous_trajectory, str):
        raise TypeError("previous_trajectory must be a string or None")
    if not previous_trajectory.strip():
        return ""
    reasoning = extract_epm_reasoning(previous_trajectory)
    if not reasoning:
        raise ValueError("previous trajectory contains no reasoning-only EPM content")
    return f"""
EPISODIC MEMORY FROM THE PREVIOUS ATTEMPT:

The following contains only reasoning from your previous attempt on the same
article. The previous classification has been removed.

Review it critically and use it only to improve the current attempt.
Do not blindly repeat the previous conclusion.

{reasoning}
"""


def Vanilla_prompt(index: int) -> str:
    article = get_article(index)
    return f"""{COMMON_NEWS_INSTRUCTION}

News Article:
{article}

Determine whether the News Article is REAL or FAKE.

REAL:
- The article is an original, unmanipulated news article.

FAKE:
- The article has been AI-generated, rewritten, filled in, or otherwise
  manipulated so that important factual content differs from the original.

Use only the News Article. Do not use or speculate about a file path, filename,
dataset label, original article, or hidden metadata.

Do not provide reasoning or explanation. Return exactly one classification:

Finish[real]

or

Finish[fake]
"""


def Vanilla_actor(index: int) -> str:
    return call_llm(Vanilla_prompt(index))


def CoT_prompt(
    index: int,
    reflection: str | Sequence[str] | None = None,
    *,
    previous_trajectory: str | None = None,
    use_gt: bool = False,
) -> str:
    article = get_article(index)
    reflection_text = _format_reflections(reflection)
    trajectory_text = _format_previous_trajectory(previous_trajectory)
    gt_text = f"{COMMON_NEWS_INSTRUCTION}\n\n" if use_gt else ""
    if trajectory_text and reflection_text:
        memory_guidance = (
            "Use only the News Article, previous trajectory, and self-reflection "
            "notes below."
        )
    elif trajectory_text:
        memory_guidance = "Use only the News Article and previous trajectory below."
    elif reflection_text:
        memory_guidance = (
            "Use only the News Article and self-reflection notes below."
        )
    else:
        memory_guidance = "Use only the News Article."

    return f"""{gt_text}News Article:
{article}

Your task is to determine whether the News Article is REAL or FAKE.

REAL:
- The article is an original, unmanipulated news article.

FAKE:
- The article has been AI-generated, rewritten, filled in, or otherwise
  manipulated so that important factual content differs from the original.

You do NOT have the original source article. Therefore, do not claim that a
factual statement is false merely because it cannot be externally verified.

{memory_guidance}

Analyze the article in the following order.

STEP 1 — IMPORTANT FACTUAL CLAIMS

Identify the most important factual claims involving:

- people or organizations,
- dates or times,
- numbers,
- events or actions,
- outcomes,
- quotations or attribution,
- causal relations.

STEP 2 — INTERNAL FACTUAL CONSISTENCY

Check whether the important claims are internally consistent. Look for:

- conflicting entities,
- conflicting dates or numbers,
- inconsistent event descriptions,
- conflicting outcomes,
- quotation or attribution problems,
- causal inconsistencies.

A claim that cannot be verified from the article alone is UNVERIFIABLE, not
automatically false.

STEP 3 — MANIPULATION CUES

Check whether the article contains signs that important factual content may
have been awkwardly generated, rewritten, substituted, or distorted.

Do not classify FAKE based only on writing style, emotional wording,
persuasion, or grammatical quality.

STEP 4 — EVIDENCE BALANCE

Briefly state:

- strongest evidence suggesting factual manipulation,
- strongest evidence against factual manipulation.

STEP 5 — FINAL DECISION

Decide REAL or FAKE based primarily on factual integrity.

Do not use or speculate about file paths, filenames, dataset labels, an
original article, or hidden metadata.

{trajectory_text}

{reflection_text}

End with exactly one Finish action. The final non-whitespace line must be:

Finish[real]

or

Finish[fake]

Reasoning:
"""


def CoT_actor(
    index: int,
    reflection: str | Sequence[str] | None = None,
    *,
    previous_trajectory: str | None = None,
    use_gt: bool = False,
) -> str:
    return call_llm(
        CoT_prompt(
            index,
            reflection,
            previous_trajectory=previous_trajectory,
            use_gt=use_gt,
        )
    )


__all__ = [
    "CoT_actor",
    "CoT_prompt",
    "Vanilla_actor",
    "Vanilla_prompt",
    "get_article",
    "get_available_indices",
]
