import re

from propanews_common import COMMON_NEWS_INSTRUCTION, extract_epm_reasoning


_PRIOR_VERDICT_PATTERN = re.compile(
    r"\b(?:(?:the|your|this|that)\s+)?"
    r"(?:(?:previous|prior|earlier|last)\s+)?"
    r"(?:answer|prediction|classification|conclusion|verdict|final\s+decision)\s*"
    r"(?:(?:was|is|seems?(?:\s+to\s+be)?|appears?(?:\s+to\s+be)?|"
    r"turned\s+out(?:\s+to\s+be)?|(?:may|might|could)"
    r"(?:\s+be|(?:\s+have)?\s+been))\s+"
    r"(?:(?:not|[a-z]+ly|very|mostly|partly|overall)\s+){0,3}|"
    r"[:=-]\s*)"
    r"(?:correct|incorrect|right|wrong)\b|"
    r"\b(?:correct|incorrect|right|wrong)\s+"
    r"(?:answer|prediction|classification|conclusion|verdict|final\s+decision)\b",
    flags=re.IGNORECASE,
)
_PRIOR_CLASSIFICATION_PATTERN = re.compile(
    r"""
    \b(?:(?:the|your|this|that|a)\s+)?
    (?:(?:previous|prior|earlier|last)\s+)?
    (?:attempt(?:'s)?\s+)?
    (?:answer|prediction|classification|conclusion|verdict|
       (?:final\s+)?decision)\b
    (?:
        \s*(?:is|was|seems?(?:\s+to\s+be)?|appears?(?:\s+to\s+be)?)\s+
        (?:likely\s+|probably\s+|possibly\s+)?
        (?:real|fake|true|false|authentic|genuine|AI[\s-]*generated|
           generated\s+by\s+(?:an?\s+)?AI)\b
      |
        [^.!?\n]{0,30}\b
        (?:classif(?:y|ied)|predict(?:s|ed)?|conclud(?:e|ed)|decid(?:e|ed)|
           call(?:s|ed)?|label(?:s|led)?|cho(?:ose|se)|selected|returned|
           output(?:s|ted)?)\b
        \s+(?:that\s+)?
        (?:(?:(?:the|this)\s+(?:news\s+)?(?:article|text)|it)\s+
           (?:(?:is|was|as)\s+)?|as\s+)?
        (?:likely\s+|probably\s+|possibly\s+)?
        (?:real|fake|true|false|authentic|genuine|AI[\s-]*generated|
           generated\s+by\s+(?:an?\s+)?AI)\b
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
_PRIOR_ATTEMPT_OUTCOME_PATTERN = re.compile(
    r"\b(?:(?:the|your|this|that|a)\s+)?"
    r"(?:(?:previous|prior|earlier|last)\s+)"
    r"(?:attempt|response|output|model)\b"
    r"[^.!?\n]{0,60}\b"
    r"(?:classif(?:y|ied)|predict(?:s|ed)?|conclud(?:e|ed)|decid(?:e|ed)|"
    r"determin(?:e|ed)|judg(?:e|ed)|state(?:s|d)?|say|says|said)\b"
    r"\s+(?:that\s+)?(?:(?:the|this)\s+(?:news\s+)?article\s+|it\s+)?"
    r"(?:(?:is|was|as)\s+)?(?:likely\s+|probably\s+|possibly\s+)?"
    r"(?:real|fake|true|false|authentic|genuine|AI[\s-]*generated)\b",
    flags=re.IGNORECASE,
)
_PRIOR_LABEL_ASSIGNMENT_PATTERN = re.compile(
    r"\b(?:(?:the|your|this|that|a)\s+)?"
    r"(?:(?:previous|prior|earlier|last)\s+)"
    r"(?:answer|prediction|classification|conclusion|verdict|label|"
    r"(?:final\s+)?decision)\s*(?:is|was|[:=\-—])\s*"
    r"(?:likely\s+|probably\s+|possibly\s+)?"
    r"(?:real|fake|true|false|authentic|genuine|AI[\s-]*generated)\b",
    flags=re.IGNORECASE,
)
_PRIOR_DECISION_DIRECTIVE_PATTERN = re.compile(
    r"(?:\b(?:(?:previous|prior|earlier|last)\s+)"
    r"(?:answer|prediction|classification|conclusion|verdict|decision)\b"
    r"[^.!?\n]{0,60}\b(?:reverse|retain|keep|preserve|change|flip|"
    r"repeat|follow)(?:d|ed|ing)?\b|"
    r"\b(?:reverse|retain|keep|preserve|change|flip|repeat|follow)"
    r"(?:d|ed|ing)?\b[^.!?\n]{0,60}\b"
    r"(?:(?:previous|prior|earlier|last)\s+)"
    r"(?:answer|prediction|classification|conclusion|verdict|decision)\b)",
    flags=re.IGNORECASE,
)
_CLASSIFICATION_DIRECTIVE_PATTERN = re.compile(
    r"\b(?:classif(?:y|ied)|predict(?:s|ed)?|conclud(?:e|ed)|"
    r"decid(?:e|ed)|determin(?:e|ed)|judg(?:e|ed)|cho(?:ose|se)|"
    r"select(?:s|ed)?|answer(?:s|ed)?|"
    r"output(?:s|ted)?|return(?:s|ed)?|assume(?:s|d)?)\b"
    r"\s+(?:that\s+)?(?:(?:the|this)\s+(?:news\s+)?article\s+"
    r"(?:is|was)\s+|it\s+(?:is|was)\s+|as\s+)?"
    r"(?:likely\s+|probably\s+|possibly\s+)?"
    r"(?:real|fake|true|false|authentic|genuine|AI[\s-]*generated)\b",
    flags=re.IGNORECASE,
)
_CLASSIFICATION_SWITCH_PATTERN = re.compile(
    r"\b(?:switch|change|flip)(?:ing|ed)?\b"
    r"(?:\s+the\s+(?:answer|prediction|classification|conclusion|verdict))?"
    r"\s+from\s+(?:real|fake|true|false)\s+to\s+"
    r"(?:real|fake|true|false)\b",
    flags=re.IGNORECASE,
)
_BARE_CLASSIFICATION_PATTERN = re.compile(
    r"[*_`#>\-+\s]*[\"']?"
    r"(?:likely\s+|probably\s+|possibly\s+)?"
    r"(?:real|fake|true|false|authentic|genuine|AI[\s-]*generated)"
    r"[\"']?\s*[.!?]*[*_`\s]*",
    flags=re.IGNORECASE,
)
_PRONOUN_OUTCOME_PATTERN = re.compile(
    r"^(?:it|this)\s+(?:is|was|seems?(?:\s+to\s+be)?|"
    r"appears?(?:\s+to\s+be)?)\s+"
    r"(?:likely\s+|probably\s+|possibly\s+)?"
    r"(?:real|fake|true|false|authentic|genuine|AI[\s-]*generated)\b",
    flags=re.IGNORECASE,
)


def _is_classification_feedback(sentence: str) -> bool:
    """Return whether a reflection sentence exposes a prior outcome."""

    candidate = sentence.strip().strip("*_`#> ")
    candidate = re.sub(r"^[-+]\s+", "", candidate)
    return bool(_BARE_CLASSIFICATION_PATTERN.fullmatch(candidate)) or any(
        pattern.search(candidate)
        for pattern in (
            _PRIOR_VERDICT_PATTERN,
            _PRIOR_CLASSIFICATION_PATTERN,
            _PRIOR_ATTEMPT_OUTCOME_PATTERN,
            _PRIOR_LABEL_ASSIGNMENT_PATTERN,
            _PRIOR_DECISION_DIRECTIVE_PATTERN,
            _CLASSIFICATION_DIRECTIVE_PATTERN,
            _CLASSIFICATION_SWITCH_PATTERN,
            _PRONOUN_OUTCOME_PATTERN,
        )
    )


def _remove_classification_feedback(text: str) -> str:
    """Drop whole outcome-bearing sentences while retaining nearby critique."""

    retained_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            retained_lines.append("")
            continue
        sentences = re.split(r"(?<=[.!?])\s+", stripped)
        retained_lines.append(
            " ".join(
                sentence
                for sentence in sentences
                if sentence and not _is_classification_feedback(sentence)
            ).strip()
        )
    return "\n".join(retained_lines)


def call_llm(prompt):
    from llm import call_llm as project_call_llm

    return project_call_llm(prompt)


def sanitize_reflection(reflection):
    """Return canonical reasoning-only reflection memory.

    The shared EPM filter removes final-answer syntax and classification
    conclusions.  Reflection needs one additional guard because a critique can
    reveal the previous outcome indirectly (for example, by saying that the
    previous prediction was wrong or should be reversed).
    """

    if not isinstance(reflection, str):
        return ""
    # Strip reflection-specific verdict sentences before the shared helper can
    # reduce an explicit field to a harmless but meaningless prefix (for
    # example, ``The last prediction was fake`` -> ``The last``).
    without_feedback = _remove_classification_feedback(reflection)
    reasoning_only = extract_epm_reasoning(without_feedback)
    sanitised = _remove_classification_feedback(reasoning_only).strip()
    sanitised = re.sub(r"\n{3,}", "\n\n", sanitised)
    # Keep the function canonical and idempotent if sentence removal exposes a
    # final-answer fragment that was previously embedded in surrounding text.
    return extract_epm_reasoning(sanitised)


def make_reflection_propanews(article, response):
    previous_reasoning = sanitize_reflection(response)
    if not previous_reasoning:
        raise ValueError("previous response contains no reasoning-only content")

    review_instruction = """
No external reference article is available.

Prioritize checking:

- important factual claims that were overlooked,
- internal contradictions,
- entity/date/number/event inconsistencies,
- quotation or attribution problems,
- unsupported reasoning,
- UNVERIFIABLE claims incorrectly treated as false,
- excessive reliance on writing style or rhetoric.
"""

    prompt = f"""{COMMON_NEWS_INSTRUCTION}

News Article:
{article}

Review the previous reasoning trajectory for one PropaNews classification
attempt.

Your task is NOT to make a new REAL/FAKE prediction.

You are not given a gold label, evaluator verdict, or any signal indicating
whether the previous answer was correct.

Identify at most THREE specific reasoning problems that should be reconsidered
in the next attempt.

For each problem provide:

1. ISSUE
What specific part of the previous reasoning was weak, unsupported, or
incomplete?

2. EVIDENCE TO RE-CHECK
Which exact factual claim or article passage should be checked again?

3. CORRECTION
How should the next attempt reason differently?

{review_instruction}

Important:

- Do not state whether the previous answer was correct or incorrect.
- Do not recommend simply reversing or retaining the previous classification.
- Do not output REAL or FAKE.
- Do not output Finish[real] or Finish[fake].
- Do not give generic advice such as "analyze more carefully."
- Return only a concise and specific reasoning critique.

PREVIOUS REASONING TRAJECTORY:
{previous_reasoning}

SELF-REFLECTION:
"""

    return sanitize_reflection(call_llm(prompt))


__all__ = ["make_reflection_propanews", "sanitize_reflection"]
