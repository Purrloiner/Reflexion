"""Actors for the prompt-based SPRS-MTCPSL adaptation on PropaNews.

The actor retains only the stored dataset index and article text. Gold labels,
source paths, and article-specific reference context are owned outside this
module.
"""

from __future__ import annotations

from collections.abc import Sequence

if __package__:
    from .propanews_common import (
        COMMON_NEWS_INSTRUCTION,
        PromptArticle,
        load_prompt_articles,
    )
else:
    from propanews_common import (
        COMMON_NEWS_INSTRUCTION,
        PromptArticle,
        load_prompt_articles,
    )


PROMPT_VERSION = "sprs-mtcpsl-propanews-clean-v3-prior-v4"


def call_llm(prompt: str) -> str:
    """Lazily delegate to the project's unchanged LLM adapter."""

    from llm import call_llm as project_call_llm

    return project_call_llm(prompt)


# Preserve JSON file order separately from index lookup. PromptArticle has no
# label, path, or other private dataset field.
_PROMPT_ARTICLES: tuple[PromptArticle, ...] = load_prompt_articles()
_ARTICLES_BY_INDEX: dict[int, str] = {
    record.index: record.article for record in _PROMPT_ARTICLES
}


def get_available_indices() -> tuple[int, ...]:
    """Return every stored index in the dataset file's original order."""

    return tuple(record.index for record in _PROMPT_ARTICLES)


def get_article(index: int) -> str:
    """Return only the target article text for ``index``."""

    try:
        return _ARTICLES_BY_INDEX[index]
    except KeyError as exc:
        raise IndexError(f"unknown PropaNews index: {index}") from exc


def _normalize_reflections(
    reflection: str | Sequence[str] | None,
) -> tuple[str, ...]:
    if reflection is None:
        return ()
    candidates = (reflection,) if isinstance(reflection, str) else tuple(reflection)
    normalized: list[str] = []
    for position, item in enumerate(candidates):
        if not isinstance(item, str):
            raise TypeError(f"reflection[{position}] must be a string")
        if item.strip():
            normalized.append(item.strip())
    return tuple(normalized)


def _format_reflections(
    reflection: str | Sequence[str] | None,
) -> str:
    reflections = _normalize_reflections(reflection)
    if not reflections:
        return ""

    numbered = "\n\n".join(
        f"Reflection {position}:\n{text}"
        for position, text in enumerate(reflections, start=1)
    )
    return f"""
SELF-REFLECTION NOTES FROM EARLIER ATTEMPTS ON THIS ARTICLE:
These notes are fallible reasoning critiques, not facts or evaluator verdicts.
Re-check every note against the NEWS ARTICLE before using it.

{numbered}
"""


def _build_sprs_prompt(
    *,
    article: str,
    reflection: str | Sequence[str] | None,
    use_gt: bool = True,
) -> str:

    ...

    reflection_text = _format_reflections(reflection)
    gt_text = COMMON_NEWS_INSTRUCTION if use_gt else ""

    return f"""{gt_text}

NEWS ARTICLE:
{article}

SPRS-MTCPSL ANALYSIS INSTRUCTIONS:

You are classifying the NEWS ARTICLE as REAL or FAKE using a prompt-based
adaptation inspired by SPRS-MTCPSL.

This is structured semantic reasoning, not the paper's original training
procedure.

Do not claim to compute embeddings, vector subtraction, losses, or fine-tuning.


PROPANEWS LABEL MEANING:

REAL:
An original, unmanipulated article, or an article whose important factual
content is consistent with the original source.

FAKE:
An AI-generated, rewritten, filled, substituted, or otherwise manipulated
article in which important factual content materially differs from the
original source.


CRITICAL RULE:

Propaganda and factual falsity are different.

REAL news may contain emotional, persuasive, fearful, exaggerated, or loaded
language.

Never classify an article as FAKE merely because propaganda is present.

The final decision must focus primarily on meaningful factual manipulation,
contradictions, substitutions, or distortions.


Keep STEP 1 through STEP 4 concise.

Spend most of the reasoning effort on STEP 5 through STEP 7.

STEP 1 — PROPAGANDA SEMANTIC ANALYSIS

Analyze the NEWS ARTICLE for:

1. Loaded Language (LL)
2. Appeal to Fear (AF)
3. Exaggeration (E)

For each technique classify the evidence as:

- present,
- weak,
- absent.

Use only evidence actually contained in the NEWS ARTICLE.

STEP 2 — SUMMARIZED PROPAGANDA

For LL, AF, and E, summarize ONLY the propaganda expression actually supported
by the NEWS ARTICLE.

Do NOT invent a hypothetical example of how a propaganda technique could have
been used.

If there is no concrete evidence for a technique, state NONE.

STEP 3 — PROPAGANDA REASON

For each technique briefly identify the wording, claim, framing, exaggeration,
fear appeal, or emotional expression that supports the assessment.

Keep every explanation tied directly to textual evidence.

STEP 4 — PROPAGANDA LEVEL SCORE

Assign LL, AF, and E separate scores from 0.0 to 1.0.

A score measures rhetorical intensity only.

It is NOT a fake-news probability.

Select a dominant propaganda technique only when meaningful textual evidence
exists.

If all propaganda signals are weak, output:

Dominant technique: NONE

Do not force a propaganda interpretation.

STEP 5 — NEUTRAL ATOMIC FACTUAL CLAIMS

Remove rhetorical framing and extract up to SEVEN important atomic factual
claims from the NEWS ARTICLE.

Each item must contain only ONE factual proposition.

Prioritize and preserve the article's claimed:

- entities,
- dates/times,
- numbers,
- events/actions,
- outcomes,
- quotations/attribution,
- causal relations.

Do not merge several facts into one vague summary.

Do not add information that is not stated in the NEWS ARTICLE.

These claims represent what the NEWS ARTICLE claims.

They are NOT automatically verified facts.

STEP 6 — FACTUAL INTEGRITY ANALYSIS

No external reference source is available.

Do NOT treat the neutral factual claims from STEP 5 as independently verified
truth.

Instead, examine whether the NEWS ARTICLE is internally consistent and whether
a small but material claim may have been generated, rewritten, substituted, or
distorted.

Check explicitly for:

- entity inconsistencies,
- date/time inconsistencies,
- number inconsistencies,
- event/action inconsistencies,
- outcome inconsistencies,
- quotation or attribution problems,
- causal inconsistencies,
- factual assertions conflicting with other supplied information.

If a claim cannot be verified from the NEWS ARTICLE alone, classify it as
UNVERIFIABLE rather than FALSE.

Do not invent a contradiction merely because external confirmation is
unavailable.

Separate rhetorical propaganda from factual manipulation.

STEP 7 — FINAL FAKE NEWS DETECTION

Use the following evidence priority:

1. Supported material factual inconsistencies or contradictions.
2. Other supported evidence of factual rewriting, substitution, or distortion.
3. Propaganda and rhetorical cues only as secondary context.

Propaganda must NEVER independently determine FAKE.

Before the final answer briefly state:

- Material inconsistencies or manipulation evidence:
- Strongest evidence suggesting factual manipulation:
- Strongest counter-evidence:

Then decide REAL or FAKE.


{reflection_text}

OUTPUT REQUIREMENTS:

- Write STEP 1 through STEP 7 in order.
- Do not use or speculate about dataset paths, filenames, hidden labels, or
  manipulation subtypes.
- Use exactly one Finish action.
- The final non-whitespace line must be exactly:

Finish[real]

or

Finish[fake]

STEP 1 — PROPAGANDA SEMANTIC ANALYSIS:
"""


def SPRS_prompt(
    index: int,
    reflection: str | Sequence[str] | None = None,
    use_gt: bool = True,
) -> str:

    return _build_sprs_prompt(
        article=get_article(index),
        reflection=reflection,
        use_gt=use_gt,
    )


def SPRS_actor(
    index: int,
    reflection: str | Sequence[str] | None = None,
    use_gt: bool = True, 
) -> str:

    return call_llm(
        SPRS_prompt(
            index,
            reflection=reflection,
            use_gt=use_gt,
        )
    )


__all__ = [
    "PROMPT_VERSION",
    "SPRS_actor",
    "SPRS_prompt",
    "get_article",
    "get_available_indices",
]
