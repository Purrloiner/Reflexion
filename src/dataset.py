from datasets import load_dataset
import json

NUM = 100
SAMPLE_SEED = 42

ds = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)

sample = (
    ds["train"]
    .shuffle(seed=SAMPLE_SEED)
    .select(range(NUM))
)


def get_question(index):
    return sample[index]["question"]


def get_answer(index):
    return sample[index]["answer"]


def get_supporting_facts(index):
    return sample[index]["supporting_facts"]


def get_context(index):
    return sample[index]["context"]


import json

extracted_data = []
for idx in range(100):
    data_item = {
        "index": idx,
        "question": get_question(idx),
        "answer": get_answer(idx),
        "supporting_facts": get_supporting_facts(idx),
        "context": get_context(idx)
    }
    extracted_data.append(data_item)

file_path = "/home/sscc/Humane/data/hotpotqa_100.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=4)




extracted_data = []
for idx in range(100):
    data_item = {
        "index": idx,
        "context": get_context(idx)
    }
    extracted_data.append(data_item)

file_path = "/home/sscc/Humane/data/hotpotqa_100_context.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=4)
    
    


extracted_data = []

for idx in range(100):

    context = get_context(idx)
    support = get_supporting_facts(idx)

    gt_titles = set(support["title"])

    gt_context = {
        "title": [],
        "sentences": []
    }

    for title, sents in zip(context["title"], context["sentences"]):
        if title in gt_titles:
            gt_context["title"].append(title)
            gt_context["sentences"].append(sents)

    extracted_data.append({
        "index": idx,
        "question": get_question(idx),
        "answer": get_answer(idx),
        "GT_context": gt_context
    })

with open("/home/sscc/Humane/data/hotpotqa_100_GT.json", "w", encoding="utf-8") as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=4)