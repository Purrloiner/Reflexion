"""Label-blind Reflexion for two-stage PCoT trajectories."""

from __future__ import annotations

import re

from propanews_common import COMMON_NEWS_INSTRUCTION, extract_epm_reasoning


_FINISH_PATTERN = re.compile(
    r"Finish\s*\[[^\]]*(?:\]|$)",
    flags=re.IGNORECASE,
)

_LABEL_TOKEN_PATTERN = re.compile(
    r"\b(?:real|fake)\b",
    flags=re.IGNORECASE,
)
_PRIOR_VERDICT_PATTERN = re.compile(
    r"\b(?:(?:the|your|this|that)\s+)?"
    r"(?:(?:previous|prior|earlier|last)\s+)?"
    r"(?:answer|prediction|classification|conclusion|verdict|final\s+decision)\s*"
    r"(?:(?:was|is|seems?(?:\s+to\s+be)?|appears?(?:\s+to\s+be)?|"
    r"turned\s+out(?:\s+to\s+be)?|(?:may|might|could)"
    r"(?:\s+be|(?:\s+have)?\s+been))\s+|[:=-]\s*)"
    r"(?:(?:not|[a-z]+ly|very|mostly|partly|overall)\s+){0,3}"
    r"(?:real|fake|true|false|correct|incorrect|right|wrong)\b|"
    r"\b(?:real|fake|true|false|correct|incorrect|right|wrong)\s+"
    r"(?:answer|prediction|classification|conclusion|verdict|final\s+decision)\b",
    flags=re.IGNORECASE,
)

def call_llm(prompt: str) -> str:
    """Lazily use the same deterministic adapter as both PCoT stages."""

    from llm_pcot import call_llm as pcot_call_llm

    return pcot_call_llm(prompt)


def PCoT_reflection_prompt(
    article: str,
    persuasion_analysis: str,
    final_response: str,
) -> str:

    if not isinstance(article, str) or not article.strip():
        raise ValueError("article must be a non-empty string")
    if not isinstance(persuasion_analysis, str):
        raise TypeError("persuasion_analysis must be a string")
    if not isinstance(final_response, str):
        raise TypeError("final_response must be a string")

    # The graph already supplies reasoning-only text. Apply the shared filter
    # again to both stages so direct callers cannot expose a malformed Stage-1
    # prediction or a previous Stage-2 prediction either.
    stage_one_reasoning = extract_epm_reasoning(persuasion_analysis)
    stage_two_reasoning = extract_epm_reasoning(final_response)
    if not stage_one_reasoning and not stage_two_reasoning:
        raise ValueError("PCoT trajectory contains no reusable reflection reasoning")
    stage_one_block = ""
    if stage_one_reasoning:
        stage_one_block = f"""
STAGE-1 ANALYSIS:
{stage_one_reasoning}

"""
    stage_two_block = ""
    if stage_two_reasoning:
        stage_two_block = f"""
PREVIOUS STAGE-2 REASONING:
{stage_two_reasoning}

"""

    return f"""{COMMON_NEWS_INSTRUCTION}

TARGET ARTICLE:
{article}

Review one previous PropaNews-adapted PCoT trajectory and produce a concise
self-reflection that can improve the next attempt.

You are NOT given a gold label, evaluator verdict, expected answer, or any
signal telling you whether the previous prediction was correct.

Do not recommend simply changing or retaining the previous classification.

Identify at most THREE specific reasoning problems that could materially affect
the next attempt.

For each problem provide:

1. ISSUE
What specific reasoning was weak, unsupported, incomplete, or potentially
misleading?

2. EVIDENCE TO RE-CHECK
Identify the exact factual claim, article evidence, or Stage-1 analysis that
should be checked again.

3. CORRECTION
Explain specifically how the next attempt should reason differently.


PRIORITIZE CHECKING:

- important factual evidence that the trajectory overlooked,
- incorrect entity, date, number, event, outcome, quotation, or causal analysis,
- factual claims incorrectly inferred from the article,
- UNVERIFIABLE claims incorrectly treated as false,
- weak or unsupported persuasion classifications or explanations,
- over-reliance on persuasion as evidence of factual manipulation,
- a material contradiction that was overlooked.

Do not give generic advice such as "analyze more carefully."

Do not output or recommend a final classification.

Do not output any Finish action.

{stage_one_block}{stage_two_block}
SELF-REFLECTION:
"""

def sanitize_reflection(reflection: str) -> str:
    """Return only critique reasoning safe to show to the next attempt."""

    if not isinstance(reflection, str):
        return ""
    reasoning = extract_epm_reasoning(reflection)
    without_finish = _FINISH_PATTERN.sub("", reasoning)
    without_verdict = _PRIOR_VERDICT_PATTERN.sub(
        "[unsupported classification assessment omitted]",
        without_finish,
    )
    without_labels = _LABEL_TOKEN_PATTERN.sub(
        "[classification label omitted]",
        without_verdict,
    )
    return without_labels.strip()


def make_reflection_PCoT(
    article: str,
    persuasion_analysis: str,
    final_response: str,
) -> str:
    prompt = PCoT_reflection_prompt(
        article=article,
        persuasion_analysis=persuasion_analysis,
        final_response=final_response,
    )
    return sanitize_reflection(call_llm(prompt))


__all__ = [
    "PCoT_reflection_prompt",
    "make_reflection_PCoT",
    "sanitize_reflection",
]
