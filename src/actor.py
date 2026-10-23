from llm import call_llm
from dataset import get_question, get_context
from reflection import make_reflection_CoT, make_reflection_react
import json
import fewshots
import re
from wikienv import WikiEnv
from action_parser import parse_action



with open('/home/sscc/Humane/data/hotpotqa_100_GT.json', 'r') as f:
    hotpotqa_100_GT = json.load(f)
    
with open('/home/sscc/Humane/data/hotpotqa_100_context.json', 'r') as f:
    hotpotqa_100_context = json.load(f)
    
with open('/home/sscc/Humane/data/hotpotqa_100.json', 'r') as f:
    hotpotqa_100 = json.load(f)
    
    

#GT X ==================================================================================================================

    
    
#CoT ===================================================================================================================

def CoT_prompt(index, reflection=[]):

    question = hotpotqa_100[index]["question"]
    fewshot_examples = fewshots.COTQA_SIMPLE6

    reflections = ""
    if reflection:
        reflections = (
            "You have attempted to answer this question before and failed.\n"
            "Use the following reflection(s) to improve your strategy.\n\n"
            + "\n".join(reflection)
        )

    prompt = f"""
Solve a question answering task by having a Thought, then Finish with your answer.
Thought can reason about the current situation.
Finish[answer] returns the answer and finishes the task.

Here are some examples:
{fewshot_examples}
(END OF EXAMPLES)

{reflections}

Question: {question}
Thought:
"""
    return prompt


def CoT_actor(index, reflection=[]):
    prompt = CoT_prompt(index, reflection)
    response = call_llm(prompt)
    return response

#===================================================================================================================



#ReAct =============================================================================================================


def ReAct_prompt(index, context="", reflection=[]):

    question = hotpotqa_100[index]["question"]
    fewshot_examples = fewshots.WEBTHINK_SIMPLE6

    reflections = ""
    if reflection:
        reflections = (
            "You have attempted to answer this question before and failed.\n"
            "Use the following reflection(s) to improve your strategy.\n\n"
            + "\n".join(reflection)
        )

    prompt = f"""
Solve a question answering task with interleaving Thought, Action, Observation steps.
Thought can reason about the current situation.
Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.

Here are some examples:
{fewshot_examples}
(END OF EXAMPLES)

{reflections}

Question: {question}

{context}
"""
    return prompt


def ReAct_actor(index, reflection=[], max_steps=6):

    env = WikiEnv()
    context = ""

    for step in range(max_steps):

        response = call_llm(
            ReAct_prompt(index, context, reflection)
        )

        context += response + "\n"
        action, argument = parse_action(response)

        if action == "Finish":
            return context

        elif action == "Search":
            observation = env.search(argument)

            context += (
                f"Observation {step+1}: "
                + observation
                + "\n"
            )

        elif action == "Lookup":
            observation = env.lookup(argument)

            context += (
                f"Observation {step+1}: "
                + observation
                + "\n"
            )

        else:
            context += (
                f"Observation {step+1}: "
                "Invalid Action\n"
            )

    return context

#===================================================================================================================


#GT O ==============================================================================================================


#CoT ===============================================================================================================

def prompt_GT(index):

    context_base = hotpotqa_100_GT[index]["GT_context"]
    question = get_question(index)
    fewshot_examples = fewshots.COT

    context = ""

    for i in range(len(context_base["title"])):
        title = context_base["title"][i]
        context += "Title : " + title + "\n"

        for j in range(len(context_base["sentences"][i])):
            context += context_base["sentences"][i][j] + "\n"

    prompt = f"""
Solve a question answering task by having a Thought, then Finish with your answer.
Thought can reason about the current situation.
Finish[answer] returns the answer and finishes the task.

Here are some examples:
{fewshot_examples}
(END OF EXAMPLES)

Relevant Context:
{context}

Question:
{question}

Thought:
"""

    return prompt


def actor_GT(index):
    prompt = prompt_GT(index)
    response = call_llm(prompt)
    return response

# ====================================================================================================================


#CoT + EPM ============================================================================================================

def prompt_GT_Mem(index, memory):

    context_base = hotpotqa_100_GT[index]["GT_context"]
    question = get_question(index)
    fewshot_examples = fewshots.COT

    context = ""

    for i in range(len(context_base["title"])):
        context += f"Title : {context_base['title'][i]}\n"

        for sent in context_base["sentences"][i]:
            context += sent + "\n"

    reflections = ""

    if memory:
        reflections += (
            "You have attempted to answer this question before and failed.\n"
            "The following reflection(s) or previous episode(s) may help you avoid making the same mistake.\n"
            "Use them to improve your reasoning strategy.\n\n"
        )

        for episode in memory:
            reflections += episode + "\n"

    prompt = f"""
Solve a question answering task by having a Thought, then Finish with your answer.
Thought can reason about the current situation.
Finish[answer] returns the answer and finishes the task.

Here are some examples:
{fewshot_examples}
(END OF EXAMPLES)

{reflections}

Relevant Context:
{context}

Question:
{question}

Thought:
"""

    return prompt


def actor_GT_Mem(index, memory):
    prompt = prompt_GT_Mem(index, memory)
    response = call_llm(prompt)
    return response


# ========================================================================================================================