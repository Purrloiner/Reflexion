from main import *
from evaluator import EM_score
from reflection import make_reflection_CoT, make_reflection_react

import json
import os

with open('/home/sscc/Humane/data/hotpotqa_100.json', 'r') as f:
    hotpotqa_100 = json.load(f)
    
with open('/home/sscc/Humane/data/hotpotqa_100_GT.json', 'r') as f:
    hotpotqa_100_GT = json.load(f)

def get_gt_context_text(index):

    context_base = hotpotqa_100_GT[index]["GT_context"]
    context = ""

    for title, sentences in zip(
        context_base["title"],
        context_base["sentences"]
    ):
        context += f"Title : {title}\n"

        for sentence in sentences:
            context += sentence + "\n"

    return context

def load_or_run(name, func):

    path = f"/home/sscc/Humane/outputs/Qwen8/trial2/{name}.json"

    expected_len = {
        "cot": 5,
        "react": 7,
        "cot_re": 5,
        "react_re": 7,
        "gt": 8,
        "gt_re": 8,
        "gt_mem": 5,
        "gt_mem_re": 5
    }[name]

    if os.path.exists(path):

        with open(path, "r") as f:
            result = json.load(f)

        if len(result) == expected_len:
            print(f"[Load] {name}")
            return result

        print(f"[Rerun] {name} (previous result incomplete)")

    else:
        print(f"[Run] {name}")

    result = func()

    with open(path, "w") as f:
        json.dump(result, f, indent=4)

    return result

TRIAL = 8
NUM = 100

def evaluate_CoT():

    solved = [False]*NUM
    acc=[]

    for trial in range(5):

        correct=0

        for i in range(NUM):

            if solved[i]:
                correct+=1
                continue

            response = run_CoT(i)

            if EM_score(response,i):

                solved[i]=True
                correct+=1

        acc.append(correct/NUM)

    return acc

def evaluate_React():

    solved=[False]*NUM
    acc=[]

    for trial in range(7):

        correct=0

        for i in range(NUM):

            if solved[i]:
                correct+=1
                continue

            response = run_React(i)

            if EM_score(response,i):

                solved[i]=True
                correct+=1

        acc.append(correct/NUM)

    return acc

def evaluate_CoT_Re():

    solved=[False]*NUM
    memories=[[] for _ in range(NUM)]

    acc=[]

    for trial in range(5):

        correct=0

        for i in range(NUM):

            if solved[i]:
                correct+=1
                continue

            response = run_CoT_Re(i, memories[i])

            if EM_score(response,i):

                solved[i]=True
                correct+=1

            else:

                reflection = make_reflection_CoT(
                    hotpotqa_100[i]["question"],
                    response
                )

                memories[i].append(
                    "\n\nReflection:\n"+reflection
                )

                if len(memories[i])>3:
                    memories[i].pop(0)

        acc.append(correct/NUM)

    return acc

def evaluate_React_Re():

    solved=[False]*NUM
    memories=[[] for _ in range(NUM)]

    acc=[]

    for trial in range(7):

        correct=0

        for i in range(NUM):

            if solved[i]:
                correct+=1
                continue

            response = run_React_Re(i, memories[i])

            if EM_score(response,i):

                solved[i]=True
                correct+=1

            else:

                reflection = make_reflection_react(
                    hotpotqa_100[i]["question"],
                    response
                )

                memories[i].append(
                    "\n\nReflection:\n"+reflection
                )

                if len(memories[i])>3:
                    memories[i].pop(0)

        acc.append(correct/NUM)


    return acc

def evaluate_GT():

    solved = [False] * NUM
    acc = []

    for trial in range(TRIAL):

        correct = 0

        for i in range(NUM):

            if solved[i]:
                correct += 1
                continue

            response = run_GT(i)

            if EM_score(response, i):

                solved[i] = True
                correct += 1

        acc.append(correct / NUM)


    return acc

def evaluate_GT_Re():

    solved = [False] * NUM
    memories = [[] for _ in range(NUM)]
    acc = []

    for trial in range(TRIAL):

        correct = 0

        for i in range(NUM):

            if solved[i]:
                correct += 1
                continue

            response = run_GT_Re(i, memories[i])

            if EM_score(response, i):

                solved[i] = True
                correct += 1

            else:

                reflection = make_reflection_CoT(
                    hotpotqa_100[i]["question"],
                    response,
                    context=get_gt_context_text(i)
                )

                memories[i].append(
                    "\n\nReflection:\n" + reflection
                )

                if len(memories[i]) > 3:
                    memories[i].pop(0)

        acc.append(correct / NUM)


    return acc

def evaluate_GT_Mem():

    solved = [False] * NUM
    memories = [[] for _ in range(NUM)]
    acc = []

    for trial in range(5):

        correct = 0

        for i in range(NUM):

            if solved[i]:
                correct += 1
                continue

            response = run_GT_Mem(i, memories[i])

            if EM_score(response, i):

                solved[i] = True
                correct += 1

            else:

                memories[i] = [
                    "Previous trajectory:\n" + response
                ]

        acc.append(correct / NUM)


    return acc

def evaluate_GT_Mem_Re():

    solved = [False] * NUM

    trajectories = [[] for _ in range(NUM)]
    reflections = [[] for _ in range(NUM)]

    acc = []

    for trial in range(5):

        correct = 0

        for i in range(NUM):

            if solved[i]:
                correct += 1
                continue

            memories = trajectories[i] + reflections[i]

            response = run_GT_Mem_Re(i, memories)

            if EM_score(response, i):

                solved[i] = True
                correct += 1

            else:

                reflection = make_reflection_CoT(
                    hotpotqa_100[i]["question"],
                    response,
                    context=get_gt_context_text(i)
                )

                trajectories[i] = [
                    "Previous trajectory:\n" + response
                ]

                reflections[i].append(
                    "\n\nReflection:\n" + reflection
                )

                if len(reflections[i]) > 3:
                    reflections[i].pop(0)

        acc.append(correct / NUM)

    return acc

cot = load_or_run("cot", evaluate_CoT)
react = load_or_run("react", evaluate_React)

cot_re = load_or_run("cot_re", evaluate_CoT_Re)
react_re = load_or_run("react_re", evaluate_React_Re)

gt = load_or_run("gt", evaluate_GT)
gt_re = load_or_run("gt_re", evaluate_GT_Re)

gt_mem = load_or_run("gt_mem", evaluate_GT_Mem)
gt_mem_re = load_or_run("gt_mem_re", evaluate_GT_Mem_Re)


import matplotlib.pyplot as plt


# 논문 Figure 4와 동일한 표시 구간
# (a): 0~6, (b): 0~7, (c): 0~4
n_b = min(
    8,
    len(gt),
    len(gt_re)
)

n_c = min(
    5,
    len(gt),
    len(gt_mem),
    len(gt_mem_re)
)


trial_b = list(range(n_b))
trial_c = list(range(n_c))


fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4.5)
)


# ==================================================
# (a) HotPotQA Success Rate
# ==================================================

axes[0].plot(
    range(len(cot)),
    cot,
    color="lightgray",
    linestyle="--",
    marker="o",
    markersize=3,
    linewidth=1,
    label="CoT only"
)

axes[0].plot(
    range(len(react)),
    react,
    color="gray",
    linestyle="--",
    marker="o",
    markersize=3,
    linewidth=1,
    label="ReAct only"
)

axes[0].plot(
    range(len(cot_re)),
    cot_re,
    color="red",
    linestyle="-",
    marker="D",
    markersize=3,
    linewidth=1.2,
    label="CoT + Reflexion"
)

axes[0].plot(
    range(len(react_re)),
    react_re,
    color="blue",
    linestyle="-",
    marker="D",
    markersize=3,
    linewidth=1.2,
    label="ReAct + Reflexion"
)

axes[0].set_title(
    "(a) HotPotQA Success Rate",
    fontweight="bold"
)

axes[0].set_xlabel(
    "Trial Number",
    fontweight="bold"
)

axes[0].set_ylabel(
    "Proportion of Solved Tasks",
    fontweight="bold"
)

axes[0].set_xticks(range(0, max(len(cot), len(react), len(cot_re), len(react_re)), 2))
axes[0].set_ylim(0.2, 0.8)
axes[0].grid(True, alpha=0.5)

axes[0].legend(
    loc="upper left",
    fontsize=8,
    frameon=True
)


# ==================================================
# (b) HotPotQA CoT (GT)
# ==================================================

axes[1].plot(
    trial_b,
    gt[:n_b],
    color="lightgray",
    linestyle="--",
    marker="o",
    markersize=3,
    linewidth=1,
    label="CoT (GT) only"
)

axes[1].plot(
    trial_b,
    gt_re[:n_b],
    color="firebrick",
    linestyle="-",
    marker="D",
    markersize=3,
    linewidth=1.2,
    label="CoT (GT) + Reflexion"
)

axes[1].set_title(
    "(b) HotPotQA CoT (GT)",
    fontweight="bold"
)

axes[1].set_xlabel(
    "Trial Number",
    fontweight="bold"
)

axes[1].set_ylabel(
    "Proportion of Solved Tasks",
    fontweight="bold"
)

axes[1].set_xticks(range(n_b))
axes[1].set_ylim(0.4, 1.0)
axes[1].grid(True, alpha=0.5)

axes[1].legend(
    loc="upper left",
    fontsize=8,
    frameon=True
)


# ==================================================
# (c) HotPotQA Episodic Memory
# ==================================================

axes[2].plot(
    trial_c,
    gt[:n_c],
    color="lightgray",
    linestyle="--",
    marker="o",
    markersize=3,
    linewidth=1,
    label="CoT (GT) only"
)

axes[2].plot(
    trial_c,
    gt_mem[:n_c],
    color="violet",
    linestyle="--",
    marker="o",
    markersize=3,
    linewidth=1,
    label="CoT (GT) EPM"
)

axes[2].plot(
    trial_c,
    gt_mem_re[:n_c],
    color="purple",
    linestyle="-",
    marker="D",
    markersize=3,
    linewidth=1.2,
    label="CoT (GT) EPM + Reflexion"
)

axes[2].set_title(
    "(c) HotPotQA Episodic Memory",
    fontweight="bold"
)

axes[2].set_xlabel(
    "Trial Number",
    fontweight="bold"
)

axes[2].set_ylabel(
    "Proportion of Solved Tasks",
    fontweight="bold"
)

axes[2].set_xticks(range(n_c))
axes[2].set_ylim(0.5, 1.0)
axes[2].grid(True, alpha=0.5)

axes[2].legend(
    loc="upper left",
    fontsize=8,
    frameon=True
)


plt.tight_layout()

plt.savefig(
    "/home/sscc/Humane/outputs/Qwen8/trial2/hotpotqa_figure4.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Graph saved to "
    "/home/sscc/Humane/outputs/Qwen8/trial2/hotpotqa_figure4.png"
)