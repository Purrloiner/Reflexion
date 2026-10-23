"""Label-blind self-reflection for prompt-based SPRS attempts."""

from __future__ import annotations

import re

if __package__:
    from .propanews_common import COMMON_NEWS_INSTRUCTION
else:
    from propanews_common import COMMON_NEWS_INSTRUCTION


def call_llm(prompt: str) -> str:
    """Lazily delegate to the same unchanged adapter as the SPRS actor."""

    from llm import call_llm as project_call_llm

    return project_call_llm(prompt)


_FINISH_PATTERN = re.compile(
    r"Finish\s*\[[^\]]*(?:\]|$)",
    flags=re.IGNORECASE,
)
_LABEL_TOKEN_PATTERN = re.compile(
    r"\b(?:real|fake)\b",
    flags=re.IGNORECASE,
)


def SPRS_reflection_prompt(
    article: str,
    response: str,
    use_gt: bool = True,
) -> str:

    ...

    gt_text = COMMON_NEWS_INSTRUCTION if use_gt else ""

    return f"""{gt_text}

NEWS ARTICLE:
{article}

Review the previous SPRS-MTCPSL-inspired reasoning trajectory for this one NEWS
ARTICLE and produce a concise self-reflection that can improve a later attempt.

You are not given a gold label, expected answer, evaluator verdict, or any signal
indicating whether the previous prediction was correct.

Focus primarily on STEP 5, STEP 6, and STEP 7. Identify at most THREE specific
reasoning problems that could materially affect the next attempt.

For each problem provide:

1. PREVIOUS REASONING
Identify the specific reasoning that may be weak or incomplete.

2. EVIDENCE TO RE-CHECK
Identify the exact factual claim or article evidence that should be checked.

3. WHY IT MAY BE UNRELIABLE
Explain the unsupported assumption, missed inconsistency, or faulty comparison.

4. CORRECTION
Explain exactly how the next attempt should reason differently.

PRIORITIZE:

- important factual claims omitted or incorrectly extracted in STEP 5,
- facts accidentally changed while producing neutral atomic claims,
- entity/date/number/event/outcome inconsistencies,
- quotation, attribution, or causal inconsistencies,
- treating UNVERIFIABLE as FALSE,
- overlooking a small but material alteration,
- relying on propaganda intensity instead of factual manipulation.

Do not spend reflection effort on small propaganda-score differences unless they
materially affected the factual-manipulation decision.

Do not use dataset labels, paths, filenames, hidden metadata, or external
article-specific reference information. Do not state whether the previous answer
was correct or incorrect. Do not recommend a final classification or output any
Finish action. Return only the concise reasoning critique.

PREVIOUS REASONING TRAJECTORY:
{response}

SELF-REFLECTION:
"""


def sanitize_reflection(reflection: str) -> str:
    """Remove accidental classification actions/tokens before actor memory."""

    if not isinstance(reflection, str):
        return ""
    without_finish = _FINISH_PATTERN.sub("", reflection)
    without_labels = _LABEL_TOKEN_PATTERN.sub(
        "[classification label omitted]",
        without_finish,
    )
    return without_labels.strip()


def make_reflection_SPRS(
    article: str,
    response: str,
    use_gt: bool = True,
) -> str:
    prompt = SPRS_reflection_prompt(
        article=article,
        response=response,
        use_gt=use_gt,
    )
    return sanitize_reflection(call_llm(prompt))


__all__ = [
    "SPRS_reflection_prompt",
    "make_reflection_SPRS",
    "sanitize_reflection",
]
