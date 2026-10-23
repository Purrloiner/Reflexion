# actor.py

from llm import call_llm

import json
import os
import re


# ======================================================================================================================
# DATA
# ======================================================================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "/home/sscc/Humane/data/propanews_150_sample_final.json"
)

GT_PATH = os.path.join(
    BASE_DIR,
    "/home/sscc/Humane/data/propanews_GT_final.json"
)




with open(DATA_PATH, "r", encoding="utf-8") as f:
    propanews_150 = json.load(f)

with open(GT_PATH, "r", encoding="utf-8") as f:
    propanews_GT = json.load(f)




# ======================================================================================================================
# BASIC
# ======================================================================================================================

def get_news(index):

    for item in propanews_150:

        if item["index"] == index:
            return item["txt"]

    raise ValueError(
        f"index {index}를 찾을 수 없습니다."
    )


def get_gt(index):

    for item in propanews_GT:

        if item["index"] == index:
            return item

    raise ValueError(
        f"GT index {index}를 찾을 수 없습니다."
    )


def get_gt_context(index):

    item = get_gt(index)

    context_base = item.get("GT_context")

    if context_base is None:
        return ""

    context = ""

    for title, sentences in zip(
        context_base["title"],
        context_base["sentences"]
    ):

        context += f"Title: {title}\n"

        for sent in sentences:
            context += sent + "\n"

    return context.strip()


# ======================================================================================================================
# LABEL PARSER
#
# PropaNews
# True  = Real
# False = Fake
# ======================================================================================================================

def parse_prediction(response):

    lower = response.lower()

    if "[this is fake news]" in lower:
        return False

    if "[this is real news]" in lower:
        return True

    return None


# ======================================================================================================================
#
# 1. DETECTION MODULE
#
# 논문 Appendix B Detection Prompt
#
# raw_concepts:
# spaCy에서 먼저 뽑은 entity / concept 목록
#
# ======================================================================================================================

# ======================================================================================================================
#
# 1. DETECTION MODULE
#
# ======================================================================================================================

def Detection_text_prompt(
    news,
    raw_concepts
):

    concepts = ", ".join(
        raw_concepts
    )

    prompt = f"""
There is a news article and a list of concepts related to it. Your task is to determine
how relevant these keywords are to the given news. Assign each keyword a relevancy
score between 0 and 1, where 1 indicates the keyword is highly relevant, and 0
indicates it is not relevant at all.
################
Instructions:
1. Analyze the content of the news article to understand its main topics and themes.
2. Compare each concept to the news content and assign a relevancy score between 0
and 1.
3. Output the concepts and their scores in reverse order (from the most relevant to the
least relevant).
4. Don't return your explanation about the concepts and their scores.
################
Output Format:
[["concept 1", <score>], ["concept2", <score>],..., ["concept N", <score>]]
################
Example Input:
News Article:
Scientists have discovered a new method to reduce carbon emissions using advanced
nanotechnology. This breakthrough could significantly impact efforts to combat
climate change globally.
concepts:
Global warming, Carbon emissions, Cryptocurrency, Nanotechnology, Climate change
################
Example Output:
[["Global warming", 0.8], ["Carbon emissions", 1.0], ["Cryptocurrency", 0.0],
["Nanotechnology", 1.0], ["Climate change", 0.9]]
################
Input:
News Article: {news}
concepts: {concepts}
################
Your Output Format:
[["concept 1", #score#], ["concept2", #score#],..., ["concept N", #score#]]
Please replace #score# in Your Output Format with the relevancy score of
corresponding word:
"""

    return prompt


# ======================================================================================================================
# 평가 데이터용
# ======================================================================================================================

def Detection_prompt(
    index,
    raw_concepts
):

    news = get_news(
        index
    )

    return Detection_text_prompt(
        news,
        raw_concepts
    )


def Detection_actor(
    index,
    raw_concepts
):

    prompt = Detection_prompt(
        index,
        raw_concepts
    )

    return call_llm(
        prompt
    )


# ======================================================================================================================
# Retrieval pool용
#
# index가 아니라 뉴스 text를 직접 받음
# ======================================================================================================================

def Detection_text_actor(
    news,
    raw_concepts
):

    prompt = Detection_text_prompt(
        news,
        raw_concepts
    )

    return call_llm(
        prompt
    )

# ======================================================================================================================
#
# 2. INSIDE JUDGE
#
# kNN에서
#
# Real 5개
# Fake 5개
#
# 가져온 뒤 각각 이 prompt를 돌림.
#
# 논문에서는 각각 top 2를 선택.
#
# ======================================================================================================================

def format_passages(passages):

    text = ""

    for i, passage in enumerate(passages):

        if isinstance(passage, dict):
            passage = passage["txt"]

        text += (
            f"\n[Passage {i}]\n"
            f"{passage}\n"
        )

    return text


def Inside_Judge_prompt(passages, concepts):

    passages_text = format_passages(
        passages
    )

    concepts_text = ", ".join(
        concepts
    )

    prompt = f"""
I need your help determining the reliability of some passages, and I've provided some
relevant entities mentioned in the text. Here are some hints:
1. **Entity Accuracy**: Check if the entities (e.g., names, attributes, roles) match the
passages I've provided. Note that we do not require all relevant entities to appear in
the passage, you only need to verify that the entities present in the passage do not
conflict with the entities provided. You also do not need to check the relationships
between the entities. However, if none of the entities appear, then the passage should
be unreliable. Please output your reason and index of the two most relevant passage in
the following JSON format:
{{"reason": <your explanation for the decision>,"index": [<index of passage which is
the most relevant>, <index of passage which is the second relevant>]}}

Here are the Passages:
{passages_text}

Here are the relevant entities:
{concepts_text}

Output:
{{"reason": #Explanation#,"index": [#Index 1#, #Index 2#]}}

Please replace #Explanation# with your explanation for the decision and replace
#Index 1#, #Index 2# with index of passage which is the first relevant and second
relevant.
"""

    return prompt


def Inside_Judge_actor(passages, concepts):

    prompt = Inside_Judge_prompt(
        passages,
        concepts
    )

    response = call_llm(prompt)

    return response


# ======================================================================================================================
#
# 3. OUTSIDE JUDGE
#
# Google/Wikipedia에서 가져온 각각의 문서에 대해
# relation triple 추출
#
# 이후 ComplEx score 계산은 actor.py가 아니라 outside_judge.py 같은 곳에서 처리
#
# ======================================================================================================================

def Outside_Judge_prompt(document, entities):

    entity_text = ", ".join(
        entities
    )

    prompt = f"""
## Task Name: Extract Relationships between Provided Entities from a Text.

## Task Description:
Your task is to extract triples that represent relationships between
a set of **predefined entities** based on the provided text.
The subject and object of each triple must come **only** from the list of entities that I provide.
If no meaningful relationships are found between the entities, return **None**.

### Task Input:
1. A piece of text.
2. A list of possible entities.

### Task Output:
A list of triples in JSON format, or **None** if no valid relationships are found.

### JSON Format for Triples:
[{{"subject": "subject_name", "predicate": "predicate_relation", "object": "object_name"}},
{{"subject": "subject_name", "predicate": "predicate_relation", "object": "object_name"}}]

### Guidelines:
- The **subject** and **object** of the triple must come from the list of predefined
entities. Do not extract any other words as entities.
- If there are no valid relationships between the provided entities, return **None**.

### Examples:

#### Example 1:
**Input Text:** "Paris is the capital of France."
**Entities Provided:** ["Paris", "France"]

**Output JSON:**
[{{"subject": "Paris", "predicate": "is the capital of", "object": "France"}}]

#### Example 2:
**Input Text:** "Einstein developed the theory of relativity."
**Entities Provided:** ["Einstein", "the theory of relativity"]

**Output JSON:**
[{{"subject": "Einstein", "predicate": "developed", "object": "the theory of relativity"}}]

#### Example 3:
**Input Text:** "She is a member of the organization."
**Entities Provided:** ["the organization"]

**Output JSON:**
None

#### Example 4:
**Input Text:** "The Nile is the longest river in the world, flowing through Egypt."
**Entities Provided:** ["The Nile", "Egypt"]

**Output JSON:**
[{{"subject": "The Nile", "predicate": "flows through", "object": "Egypt"}}]

### Real Data:

**Input Text:**
{document}

**Entities Provided:**
{entity_text}

**Output JSON:**
[{{"subject": ##predefined_entity##, "predicate": #Relation#,
"object": ##predefined_entity##}}]

Please replace ##predefined_entity## in **Output JSON** of Real Data with entities
from Entities Provided and replace #Relation# in **Output JSON** of Real Data with
meaningful relationship between entities.

I do not need you give me code, return triples in JSON format, or **None** if no
valid relationships are found.
"""

    return prompt


def Outside_Judge_actor(document, entities):

    prompt = Outside_Judge_prompt(
        document,
        entities
    )

    response = call_llm(prompt)

    return response


# ======================================================================================================================
#
# 4. INSIDE DETERMINATION
#
# Inside Judge가 선택한
#
# Real 2개 + Fake 2개
#
# 총 4개 demonstration 사용
#
# ======================================================================================================================

def format_inside_examples(examples):

    result = ""

    for i, example in enumerate(
        examples,
        start=1
    ):

        label = example["label"]

        if label is True:
            label_text = "[This is real news]"
        else:
            label_text = "[This is fake news]"

        result += f"""
[example {i}]:
[input news]:
[{example["txt"]}]

[output]:
{label_text}

"""

    return result


def Inside_Determination_prompt(
    index,
    examples
):

    news = get_news(index)

    examples_text = format_inside_examples(
        examples
    )

    prompt = f"""
You are an expert of news authenticity evaluation. As an expert of news authenticity
evaluation, you should analyze and evaluate the authenticity of news. I need your
assistance in evaluating the authenticity of a news article. Please focus on whether the
content of the article is true, not on the genre and rhetoric of the article. I will provide
you the news article and additional information about this news. You have to answer
that [This is fake news] or [This is real news] in the first sentence of your output and
give your explanation about [target news].

{examples_text}

[target news]:
[{news}]

[output]:
"""

    return prompt


def Inside_Determination_actor(
    index,
    examples
):

    prompt = Inside_Determination_prompt(
        index,
        examples
    )

    response = call_llm(prompt)

    return response


# ======================================================================================================================
#
# 5. OUTSIDE DETERMINATION
#
# Outside Judge에서 최종 선택된 문서 2개
#
# ======================================================================================================================

def Outside_Determination_prompt(
    index,
    outside_information
):

    news = get_news(index)

    if isinstance(
        outside_information,
        list
    ):

        outside_information = "\n\n".join(
            outside_information
        )

    prompt = f"""
I need your assistance in evaluating the authenticity of a news article. I will provide
you the news article and additional information about this news. Please analyze the
following news and give your decision and reason. The first sentence of your
[Decision and Reason] must be [This is fake news] or [This is real news], and then
give reason.

The news article is:
{news}

The additional information is:
{outside_information}

[Decision and Reason]:
"""

    return prompt


def Outside_Determination_actor(
    index,
    outside_information
):

    prompt = Outside_Determination_prompt(
        index,
        outside_information
    )

    response = call_llm(prompt)

    return response


# ======================================================================================================================
#
# 6. INTEGRATED DETERMINATION
#
# Base DKFND
#
# ======================================================================================================================

def Integrated_Determination_prompt(
    index,
    inside_prediction,
    inside_reason,
    outside_prediction,
    outside_reason
):

    news = get_news(index)

    prompt = f"""
I need your assistance in evaluating the authenticity of a news article. This news
article include news title, news text and news tweet.

Here are the news article:
{news}

There are two different views on this news article.

Some people believe that {inside_prediction}, their explanation is:
{inside_reason}

Others believe that {outside_prediction}, their explanation is:
{outside_reason}

Please judge their opinion, evaluate the authenticity of the news article by analyzing
its logic, reliability, and factual accuracy, and finally give your decision.

The first sentence after [Explanation] must be [This is fake news] or [This is real
news], and then give your explanation.

[Explanation]:
"""

    return prompt


def Integrated_Determination_actor(
    index,
    inside_prediction,
    inside_reason,
    outside_prediction,
    outside_reason
):

    prompt = Integrated_Determination_prompt(
        index,
        inside_prediction,
        inside_reason,
        outside_prediction,
        outside_reason
    )

    response = call_llm(prompt)

    return response


# ======================================================================================================================
#
# EXTENSION
#
# 아래부터는 DKFND 논문 원본 prompt가 아니라
# 우리 실험용 GT / Reflexion / Memory 추가 부분
#
# 원래 prompt 문장은 그대로 두고,
# [Explanation] 직전에 context만 추가한다.
#
# ======================================================================================================================

def make_reflection_block(reflections):

    if not reflections:
        return ""

    return """
You have attempted to evaluate this news article before.
Use the following reflection(s) to reconsider your previous reasoning.
Do not simply repeat the previous decision. Carefully reconsider the evidence.

Previous Reflection(s):
""" + "\n\n".join(reflections)


def make_memory_block(memory):

    if not memory:
        return ""

    result = """
You have previous experience from earlier attempts on this same news article.
Use the following memory to improve the current decision.

Experience Memory:
"""

    for i, episode in enumerate(
        memory,
        start=1
    ):

        result += (
            f"\n[Episode {i}]\n"
            + episode
            + "\n"
        )

    return result


def make_gt_block(index):

    gt = get_gt_context(index)

    if not gt:
        return ""

    return f"""
The following is verified ground-truth context derived from the original source article.
Use it as factual reference when evaluating whether the target news has been altered.

Verified Ground Truth Context:
{gt}
"""


# ======================================================================================================================
#
# Integrated Determination + GT / Reflexion / Memory
#
# ======================================================================================================================

def Integrated_Extended_prompt(
    index,
    inside_prediction,
    inside_reason,
    outside_prediction,
    outside_reason,
    use_gt=False,
    reflections=None,
    memory=None
):

    if reflections is None:
        reflections = []

    if memory is None:
        memory = []

    news = get_news(index)

    gt_block = ""

    if use_gt:
        gt_block = make_gt_block(index)

    reflection_block = make_reflection_block(
        reflections
    )

    memory_block = make_memory_block(
        memory
    )

    # 여기까지의 DKFND 문구는 논문 Integrated Determination Prompt 그대로
    prompt = f"""
I need your assistance in evaluating the authenticity of a news article. This news
article include news title, news text and news tweet.

Here are the news article:
{news}

There are two different views on this news article.

Some people believe that {inside_prediction}, their explanation is:
{inside_reason}

Others believe that {outside_prediction}, their explanation is:
{outside_reason}

Please judge their opinion, evaluate the authenticity of the news article by analyzing
its logic, reliability, and factual accuracy, and finally give your decision.

{gt_block}

{reflection_block}

{memory_block}

The first sentence after [Explanation] must be [This is fake news] or [This is real
news], and then give your explanation.

[Explanation]:
"""

    return prompt


def Integrated_Extended_actor(
    index,
    inside_prediction,
    inside_reason,
    outside_prediction,
    outside_reason,
    use_gt=False,
    reflections=None,
    memory=None
):

    prompt = Integrated_Extended_prompt(
        index=index,
        inside_prediction=inside_prediction,
        inside_reason=inside_reason,
        outside_prediction=outside_prediction,
        outside_reason=outside_reason,
        use_gt=use_gt,
        reflections=reflections,
        memory=memory
    )

    response = call_llm(prompt)

    return response