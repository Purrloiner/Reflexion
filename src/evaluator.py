import re
from dataset import get_answer


def extract_finish(response):
    matches = re.findall(r"Finish\[(.*?)\]", response,
                         flags=re.IGNORECASE | re.DOTALL)

    if len(matches) == 0:
        return ""

    return matches[-1].strip()


def EM_score(response, index):
    gt = get_answer(index).strip().casefold()

    pred = extract_finish(response).casefold()

    return int(gt == pred)