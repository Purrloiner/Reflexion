import fewshots
from llm import call_llm


def make_reflection_react(question, response):

    fewshot_reflection = fewshots.REFLECTIONS

    prompt = f"""
You are an advanced reasoning agent.

Here are some examples:
{fewshot_reflection}
(END OF EXAMPLES)

You have attempted to answer the following question before and failed.

Question:
{question}

Your previous reasoning:
{response}

Carefully analyze why your previous reasoning failed.
Identify the mistake or missing information.
Write a short reflection that will help you solve the question correctly next time.

Reflection:
"""

    return call_llm(prompt)


def make_reflection_CoT(question, response, context=""):

    fewshot_reflection = fewshots.COT_REFLECT

    context_text = ""

    if context:
        context_text = f"""
Relevant Context:
{context}
"""

    prompt = f"""
You are an advanced reasoning agent.

Here are some examples:
{fewshot_reflection}
(END OF EXAMPLES)

You have attempted to answer the following question before and failed.

{context_text}

Question:
{question}

Your previous reasoning:
{response}

Carefully analyze why your previous reasoning failed.
Identify the mistake or missing information.
Write a short reflection that will help you solve the question correctly next time.

Reflection:
"""

    return call_llm(prompt)