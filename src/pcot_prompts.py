"""Official-derived PCoT prompt material for the PropaNews experiment.

The persuasion taxonomy and definitions are transcribed from the released
``persuasion_knowledge_infusion.yaml`` at upstream PCoT commit
``b7893141e01872a896ce915429cb26b8754cd904``.  Formatting is adapted because
this project exposes a single-text ``call_llm(prompt)`` interface rather than
separate system and user messages.

The common PropaNews instruction is the experiment's GT condition, so it is
included only when ``use_gt`` is true.  EPM trajectories and Reflexion notes
are independent optional memories; empty memories leave the underlying
two-stage baseline prompt unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from propanews_common import COMMON_NEWS_INSTRUCTION, extract_epm_reasoning
from reflection_pcot import sanitize_reflection


PROMPT_VERSION: Final[str] = "pcot-b7893141-propanews-clean-v3-gt-epm-reflexion-v7"
# Preserve the graph experiment's existing effective Stage-2 default.
PCOT_FINAL_MODE: Final[str] = "van"
SUPPORTED_FINAL_MODES: Final[tuple[str, ...]] = (
    "van",
    "zcot",
    "defspec",
)


# Each tuple is (official technique name, official definition).  The source
# prompt uses commas/"or" in three names for which the experiment request uses
# slashes: Appeal to Fear, Prejudice; False Dilemma or No Choice; and
# Obfuscation, Intentional Vagueness, Confusion.
PERSUASION_TAXONOMY: Final[
    tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...]
] = (
    (
        "Attack on reputation",
        (
            "the argument does not address the topic itself, but targets the "
            "participant (personality, experience, deeds, etc.) in order to "
            "question and/or to undermine his credibility. The object of the "
            "argumentation can also refer to a group of individuals, an "
            "organization, an object, or an activity."
        ),
        (
            (
                "Name Calling or Labelling",
                (
                    "a form of argument in which loaded labels are directed at "
                    "an individual, group, object or activity, typically in an "
                    "insulting or demeaning way, but also using labels the "
                    "target audience finds desirable."
                ),
            ),
            (
                "Guilt by Association",
                (
                    "attacking the opponent or an activity by associating it "
                    "with a another group, activity or concept that has sharp "
                    "negative connotations for the target audience."
                ),
            ),
            (
                "Casting Doubt",
                (
                    "questioning the character or personal attributes of "
                    "someone or something in order to question their general "
                    "credibility or quality."
                ),
            ),
            (
                "Appeal to Hypocrisy",
                (
                    "the target of the technique is attacked on its reputation "
                    "by charging them with hypocrisy/inconsistency."
                ),
            ),
            (
                "Questioning the Reputation",
                (
                    "the target is attacked by making strong negative claims "
                    "about it, focusing specially on undermining its character "
                    "and moral stature rather than relying on an argument "
                    "about the topic."
                ),
            ),
        ),
    ),
    (
        "Justification",
        (
            "the argument is made of two parts, a statement and an explanation "
            "or appeal, where the latter is used to justify and/or to support "
            "the statement."
        ),
        (
            (
                "Flag Waving",
                (
                    "justifying an idea by exhaling the pride of a group or "
                    "highlighting the benefits for that specific group."
                ),
            ),
            (
                "Appeal to Authority",
                (
                    "a weight is given to an argument, an idea or information "
                    "by simply stating that a particular entity considered as "
                    "an authority is the source of the information."
                ),
            ),
            (
                "Appeal to Popularity",
                (
                    "a weight is given to an argument or idea by justifying it "
                    "on the basis that allegedly everybody (or the large "
                    "majority) agrees with it or nobody disagrees with it."
                ),
            ),
            (
                "Appeal to Values",
                (
                    "a weight is given to an idea by linking it to values seen "
                    "by the target audience as positive."
                ),
            ),
            (
                "Appeal to Fear, Prejudice",
                (
                    "promotes or rejects an idea through the repulsion or fear "
                    "of the audience towards this idea."
                ),
            ),
        ),
    ),
    (
        "Simplification",
        (
            "the argument excessively simplifies a problem, usually regarding "
            "the cause, the consequence, or the existence of choices."
        ),
        (
            (
                "Causal Oversimplification",
                (
                    "assuming a single cause or reason when there are actually "
                    "multiple causes for an issue."
                ),
            ),
            (
                "False Dilemma or No Choice",
                (
                    "a logical fallacy that presents only two options or sides "
                    "when there are many options or sides. In extreme, the "
                    "author tells the audience exactly what actions to take, "
                    "eliminating any other possible choices."
                ),
            ),
            (
                "Consequential Oversimplification",
                (
                    "is an assertion one is making of some first event/action "
                    "leading to a domino-like chain of events that have some "
                    "significant negative (positive) effects and consequences "
                    "that appear to be ludicrous or unwarranted or with each "
                    "step in the chain more and more improbable."
                ),
            ),
        ),
    ),
    (
        "Distraction",
        (
            "the argument takes focus away from the main topic or argument to "
            "distract the reader."
        ),
        (
            (
                "Strawman",
                (
                    "consists in making an impression of refuting an argument "
                    "of the opponent's proposition, whereas the real subject "
                    "of the argument was not addressed or refuted, but instead "
                    "replaced with a false one."
                ),
            ),
            (
                "Red Herring",
                (
                    "consists in diverting the attention of the audience from "
                    "the main topic being discussed, by introducing another "
                    "topic, which is irrelevant."
                ),
            ),
            (
                "Whataboutism",
                (
                    "a technique that attempts to discredit an opponent's "
                    "position by charging them with hypocrisy without directly "
                    "disproving their argument."
                ),
            ),
        ),
    ),
    (
        "Call",
        (
            "the text is not an argument but an encouragement to act or to "
            "think in a particular way."
        ),
        (
            (
                "Slogans",
                (
                    "a brief and striking phrase, often acting like emotional "
                    "appeals, that may include labeling and stereotyping."
                ),
            ),
            (
                "Conversation Killer",
                (
                    "words or phrases that discourage critical thought and "
                    "meaningful discussion about a given topic."
                ),
            ),
            (
                "Appeal to Time",
                (
                    "the argument is centred around the idea that time has come "
                    "for a particular action."
                ),
            ),
        ),
    ),
    (
        "Manipulative wording",
        (
            "the text is not an argument per se, but uses specific language, "
            "which contains words or phrases that are either non-neutral, "
            "confusing, exaggerating, loaded, etc., in order to impact the "
            "reader emotionally."
        ),
        (
            (
                "Loaded Language",
                (
                    "use of specific words and phrases with strong emotional "
                    "implications (either positive or negative) to influence "
                    "and convince the audience that an argument is valid."
                ),
            ),
            (
                "Obfuscation, Intentional Vagueness, Confusion",
                (
                    "use of words that are deliberately not clear, vague or "
                    "ambiguous so that the audience may have its own "
                    "interpretations."
                ),
            ),
            (
                "Exaggeration or Minimisation",
                (
                    "consists of either representing something in an excessive "
                    "manner or making something seem less important or smaller "
                    "than it really is."
                ),
            ),
            (
                "Repetition",
                (
                    "the speaker uses the same phrase repeatedly with the "
                    "hopes that the repetition will lead to persuade the "
                    "audience."
                ),
            ),
        ),
    ),
)


_OUTPUT_SCHEMA: Final[str] = """{
  "Attack_on_reputation": {
    "is_used": "Your answer. Use only Yes or No",
    "explanation": "Explain the decision using the analyzed text"
  },
  "Justification": {
    "is_used": "Your answer. Use only Yes or No",
    "explanation": "Explain the decision using the analyzed text"
  },
  "Simplification": {
    "is_used": "Your answer. Use only Yes or No",
    "explanation": "Explain the decision using the analyzed text"
  },
  "Distraction": {
    "is_used": "Your answer. Use only Yes or No",
    "explanation": "Explain the decision using the analyzed text"
  },
  "Call": {
    "is_used": "Your answer. Use only Yes or No",
    "explanation": "Explain the decision using the analyzed text"
  },
  "Manipulative_wording": {
    "is_used": "Your answer. Use only Yes or No",
    "explanation": "Explain the decision using the analyzed text"
  }
}"""


def normalise_final_mode(mode: str) -> str:
    """Return a canonical Stage-2 mode name."""

    if not isinstance(mode, str):
        raise TypeError("final mode must be a string")
    compact = mode.strip().casefold().replace("-", "").replace("_", "")
    aliases = {
        "van": "van",
        "zcot": "zcot",
        "defspec": "defspec",
    }
    try:
        return aliases[compact]
    except KeyError as exc:
        raise ValueError(
            f"unsupported PCoT final mode {mode!r}; expected one of "
            f"{SUPPORTED_FINAL_MODES}"
        ) from exc


def normalise_reflections(
    reflections: str | Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate reflection memory and remove empty entries."""

    if reflections is None:
        return ()
    candidates = (reflections,) if isinstance(reflections, str) else tuple(reflections)
    result: list[str] = []
    for position, reflection in enumerate(candidates):
        if not isinstance(reflection, str):
            raise TypeError(f"reflections[{position}] must be a string")
        sanitised = sanitize_reflection(reflection)
        if sanitised:
            result.append(sanitised)
    return tuple(result)


def normalise_trajectories(
    trajectories: str | Sequence[str] | None,
) -> tuple[str, ...]:
    """Return reasoning only from the most recent non-empty trajectory."""

    if trajectories is None:
        return ()
    candidates = (
        (trajectories,)
        if isinstance(trajectories, str)
        else tuple(trajectories)
    )
    raw_items: list[str] = []
    for position, trajectory in enumerate(candidates):
        if not isinstance(trajectory, str):
            raise TypeError(f"trajectories[{position}] must be a string")
        if trajectory.strip():
            raw_items.append(trajectory)
    if not raw_items:
        return ()

    reasoning = extract_epm_reasoning(raw_items[-1])
    if not reasoning:
        raise ValueError("latest trajectory contains no reasoning-only EPM content")
    return (reasoning,)


def format_pcot_trajectory(
    persuasion_analysis: str,
    final_response: str,
) -> str:
    """Format one reasoning-only, gold-blind EPM item.

    Stage 1 is normally persuasion analysis, but malformed outputs can still
    contain a classification. Both stages therefore pass through the shared
    reasoning extractor before becoming model-facing memory.
    """

    for name, value in (
        ("persuasion_analysis", persuasion_analysis),
        ("final_response", final_response),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    stage_one_reasoning = extract_epm_reasoning(persuasion_analysis)
    stage_two_reasoning = extract_epm_reasoning(final_response)
    if not stage_one_reasoning and not stage_two_reasoning:
        raise ValueError("PCoT trajectory contains no reasoning-only EPM content")

    sections: list[str] = []
    if stage_one_reasoning:
        sections.append(
            "STAGE 1 PERSUASION ANALYSIS:\n"
            f"{stage_one_reasoning}"
        )
    if stage_two_reasoning:
        sections.append(
            "STAGE 2 REASONING:\n"
            f"{stage_two_reasoning}"
        )
    return "PREVIOUS PCoT REASONING:\n\n" + "\n\n".join(sections)


def _gt_block(use_gt: bool) -> str:
    if not isinstance(use_gt, bool):
        raise TypeError("use_gt must be a bool")
    return f"{COMMON_NEWS_INSTRUCTION}\n\n" if use_gt else ""


def _trajectory_block(
    trajectories: str | Sequence[str] | None,
) -> str:
    items = normalise_trajectories(trajectories)
    if not items:
        return ""
    return f"""
EPM GUIDANCE:
The following is reasoning from the model's most recent attempt on this same
article. Its final prediction has been removed. It contains no gold label or
evaluator feedback and may contain reasoning errors. Re-check it against the
target article rather than treating it as fact, and do not try to infer the
omitted prediction.

{items[-1]}
"""


def _reflection_block(reflections: str | Sequence[str] | None) -> str:
    notes = normalise_reflections(reflections)
    if not notes:
        return ""
    rendered = "\n\n".join(
        f"Reflection {position}:\n{note}"
        for position, note in enumerate(notes, start=1)
    )
    return f"""
SELF-REFLECTION NOTES FROM EARLIER ATTEMPTS:
These notes are gold-blind reasoning critiques, not facts or evaluator
verdicts. Re-check them against the supplied text rather than trusting them.

{rendered}
"""


def _persuasion_knowledge() -> str:
    high_level = []
    detailed = []
    for number, (strategy, definition, techniques) in enumerate(
        PERSUASION_TAXONOMY,
        start=1,
    ):
        high_level.append(f"{number}. {strategy}: {definition}")
        technique_lines = [
            f"{chr(97 + offset)}. {name} - {technique_definition}"
            for offset, (name, technique_definition) in enumerate(techniques)
        ]
        detailed.append(
            f"{strategy} includes different persuasion techniques. Below are "
            "examples of techniques and their definitions:\n"
            + "\n".join(technique_lines)
        )
    return (
        "Persuasive text is characterized by a specific use of language in "
        "order to influence readers. We distinguish the following high-level "
        "persuasion approaches:\n"
        + "\n".join(high_level)
        + "\n\n"
        + "\n\n".join(detailed)
    )


PERSUASION_KNOWLEDGE: Final[str] = _persuasion_knowledge()


def build_persuasion_prompt(
    article: str,
    reflections: str | Sequence[str] | None = None,
    trajectories: str | Sequence[str] | None = None,
    *,
    use_gt: bool = False,
) -> str:
    """Build the official-derived DMT Stage-1 prompt."""

    if not isinstance(article, str) or not article.strip():
        raise ValueError("article must be a non-empty string")

    gt_text = _gt_block(use_gt)
    trajectory_text = _trajectory_block(trajectories)
    reflection_text = _reflection_block(reflections)

    return f"""{gt_text}News Article (TARGET ARTICLE):
{article}

PCOT METHOD — STAGE 1:

You are an assistant who detects persuasion in text.

{PERSUASION_KNOWLEDGE}

You are the expert who detects high-level persuasion approaches: Attack on
reputation, Justification, Simplification, Distraction, Call, and Manipulative
wording.

{trajectory_text}
{reflection_text}
Analyze the text and decide if the text contains any high-level persuasion
approaches from the following: Attack on reputation, Justification,
Simplification, Distraction, Call, Manipulative wording. For each high-level
persuasion approach give an explanation of how it appears, or why it is not
supported, in the analyzed text.

You are very conservative in your final decisions and when you are not fully
sure you answer No. Give your answer in the form of this dictionary and use
only Yes or No for every is_used value:

{_OUTPUT_SCHEMA}

Answer:
"""


def _final_instruction(mode: str) -> str:

    if mode == "van":
        return (
            "Analyze the given text and determine if it is real or fake news. "
            "Answer only with the required Finish action."
        )

    if mode == "zcot":
        return (
            "Deeply analyze the given text, think step-by-step, and determine "
            "if it is real or fake news. End with the required Finish action."
        )

    return (
        "Analyze the tone, language, and sources to determine if the text is "
        "real (supported by facts) or fake (misleading information). End with "
        "the required Finish action."
    )


def build_final_prompt(
    *,
    article: str,
    persuasion_analysis: str,
    reflections: str | Sequence[str] | None = None,
    trajectories: str | Sequence[str] | None = None,
    use_gt: bool = False,
    final_mode: str = PCOT_FINAL_MODE,
) -> str:
    """Build Stage 2 from the article, Stage-1 analysis, and reflection."""

    if not isinstance(article, str) or not article.strip():
        raise ValueError("article must be a non-empty string")

    if not isinstance(persuasion_analysis, str) or not persuasion_analysis.strip():
        raise ValueError("persuasion_analysis must be a non-empty string")

    mode = normalise_final_mode(final_mode)

    gt_text = _gt_block(use_gt)
    trajectory_text = _trajectory_block(trajectories)
    reflection_text = _reflection_block(reflections)
    instruction = _final_instruction(mode)

    return f"""{gt_text}News Article (TARGET ARTICLE):
{article}

PCOT METHOD — STAGE 2:

SYSTEM INSTRUCTION:
You are an AI assistant trained to detect fake news.

USER INSTRUCTION:
The text you are about to analyze has already been reviewed to identify any
high-level persuasion approaches. The results of this analysis are presented
below:

PERSUASION ANALYSIS:
{persuasion_analysis}

Keep in mind that the above results may include errors and not all explanations
may be right. Use them as auxiliary reasoning context, not as established
facts. The presence of persuasion or emotional language is not by itself proof
that an article is fake; reliable news can also use persuasion.

{trajectory_text}
{reflection_text}

Knowing this, it is now your turn to analyze the original text and decide if it
includes fake news.

{instruction}

The evaluator adapter accepts exactly one of these outputs:
Finish[real]
Finish[fake]

Use exactly one Finish action. Its value must be real or fake, and the final
non-whitespace line must be that Finish action.

Answer:
"""


__all__ = [
    "PCOT_FINAL_MODE",
    "PERSUASION_KNOWLEDGE",
    "PERSUASION_TAXONOMY",
    "PROMPT_VERSION",
    "SUPPORTED_FINAL_MODES",
    "build_final_prompt",
    "build_persuasion_prompt",
    "format_pcot_trajectory",
    "normalise_final_mode",
    "normalise_reflections",
    "normalise_trajectories",
]
