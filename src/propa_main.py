# main.py

from annotated_types import doc

from propa_prom_DKFND import (
    Detection_actor,
    Inside_Judge_actor,
    Outside_Judge_actor,
    Inside_Determination_actor,
    Outside_Determination_actor,
    Integrated_Extended_actor,
    get_news
)

import json
import os
import re

import spacy
import torch

from transformers import (
    AutoTokenizer,
    AutoModel
)

import numpy as np


# ======================================================================================================================
# PATH
# ======================================================================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RETRIEVAL_PATH = (
    "/home/sscc/Humane/data/"
    "retrieval_pool_64_concepts.json"
)


with open(
    RETRIEVAL_PATH,
    "r",
    encoding="utf-8"
) as f:

    retrieval_pool = json.load(f)


# ======================================================================================================================
# DKFND SETTINGS
#
# 논문 설정
#
# Detection N = 5
# Inside Investigation k = 5 positive + 5 negative
# Inside Judge m = 2 positive + 2 negative
# Outside Investigation L = 8
# Outside Judge n = 2
# ======================================================================================================================

TOP_CONCEPTS = 5

INSIDE_K = 5

INSIDE_M = 2

OUTSIDE_L = 8

OUTSIDE_N = 2


# ======================================================================================================================
# DEVICE
# ======================================================================================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "[DKFND Device]",
    DEVICE
)


# ======================================================================================================================
# SPACY
#
# python -m spacy download en_core_web_sm
# ======================================================================================================================

try:

    nlp = spacy.load(
        "en_core_web_sm"
    )

except OSError:

    raise RuntimeError(
        "\nspaCy 모델이 없습니다.\n"
        "아래 명령어를 먼저 실행하세요:\n\n"
        "python -m spacy download en_core_web_sm\n"
    )


# ======================================================================================================================
# DEBERTA
#
# 논문: DeBERTa-base
# ======================================================================================================================

DEBERTA_NAME = (
    "microsoft/deberta-base"
)


tokenizer = AutoTokenizer.from_pretrained(
    DEBERTA_NAME
)

deberta = AutoModel.from_pretrained(
    DEBERTA_NAME,
    use_safetensors=True
)

deberta = deberta.to(
    DEVICE
)

deberta.eval()


# ======================================================================================================================
# 1. DETECTION
#
# spaCy
#   ↓
# raw concepts
#   ↓
# LLM Detection Prompt
#   ↓
# top 5
# ======================================================================================================================

# ======================================================================================================================
# CONCEPT NORMALIZATION
# ======================================================================================================================

def normalize_concept(text):

    text = str(text).strip().lower()

    # possessive 제거
    text = re.sub(
        r"['’]s\b",
        "",
        text
    )

    # 맨 앞 article 제거
    text = re.sub(
        r"^(the|a|an)\s+",
        "",
        text
    )

    # 특수문자 -> 공백
    text = re.sub(
        r"[^a-z0-9\s-]",
        " ",
        text
    )

    # 공백 정리
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def extract_raw_concepts(text):

    doc = nlp(text)

    concepts = []
    seen = set()


    def add_concept(value):

        value = value.strip()

        if not value:
            return

        if len(value) <= 2:
            return

        normalized = normalize_concept(
            value
        )

        if not normalized:
            return

        if normalized in seen:
            return

        seen.add(
            normalized
        )

        concepts.append(
            value
        )


    # ==================================================================
    # Named entities
    #
    # entity만 너무 많이 들어가서 noun chunk가 밀리지 않도록
    # 최대 25개
    # ==================================================================

    entity_count = 0

    for ent in doc.ents:

        before = len(
        concepts
    )

        add_concept(
        ent.text
    )

        after = len(
        concepts
    )

        if after > before:
            entity_count += 1

        if entity_count >= 25:
            break


    # ==================================================================
    # Noun chunks
    #
    # "weapons of mass destruction" 같은 핵심 표현 확보
    # 최대 25개 추가
    # ==================================================================

    chunk_count = 0

    for chunk in doc.noun_chunks:

        before = len(
            concepts
        )

        add_concept(
            chunk.text
        )

        after = len(
            concepts
        )

        if after > before:

            chunk_count += 1

        if chunk_count >= 25:
            break


    return concepts

# ======================================================================================================================
# Detection actor response parser
#
# 예상:
#
# [["concept", 1.0], ["concept2", 0.8]]
# ======================================================================================================================

def parse_detection_response(
    response
):

    try:

        response = response.replace(
            "```json",
            ""
        )

        response = response.replace(
            "```",
            ""
        )

        response = response.strip()


        data = json.loads(
            response
        )


        parsed = []


        for item in data:

            if (
                isinstance(item, list)
                and len(item) >= 2
            ):

                concept = str(
                    item[0]
                ).strip()

                score = float(
                    item[1]
                )

                parsed.append(
                    (
                        concept,
                        score
                    )
                )


        # 점수 높은 순
        parsed.sort(
            key=lambda x: x[1],
            reverse=True
        )


        # ==================================================================
        # 의미상 중복 concept 제거하면서 Top-5 채우기
        # ==================================================================

        selected = []

        seen = set()


        for concept, score in parsed:

            normalized = normalize_concept(
                concept
            )


            if not normalized:

                continue


            if normalized in seen:

                continue


            seen.add(
                normalized
            )

            selected.append(
                concept
            )


            if len(selected) >= TOP_CONCEPTS:

                break


        return selected


    except Exception as e:

        print(
            "[Warning] Detection parse failed:",
            e
        )

        return []

def run_detection(index):

    news = get_news(
        index
    )


    raw_concepts = extract_raw_concepts(
        news
    )


    response = Detection_actor(
        index,
        raw_concepts
    )


    concepts = parse_detection_response(
        response
    )


    # parsing 실패 시 spaCy 앞 5개 사용
    if not concepts:

        concepts = raw_concepts[
            :TOP_CONCEPTS
        ]


    return {
        "raw_concepts":
            raw_concepts,

        "detection_response":
            response,

        "concepts":
            concepts
    }


# ======================================================================================================================
# DEBERTA EMBEDDING
# ======================================================================================================================

def get_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )


    inputs = {
        key: value.to(DEVICE)
        for key, value
        in inputs.items()
    }


    with torch.no_grad():

        outputs = deberta(
            **inputs
        )


    # CLS embedding
    embedding = (
        outputs.last_hidden_state[
            :,
            0,
            :
        ]
        .squeeze(0)
        .cpu()
        .numpy()
    )


    return embedding


# ======================================================================================================================
# Retrieval pool embedding cache
#
# retrieval sample과 target 모두
#
# Top-5 concepts
#     ↓
# concatenate
#     ↓
# DeBERTa
#
# 동일한 representation 사용
# ======================================================================================================================

RETRIEVAL_EMBEDDINGS = None


def build_retrieval_embeddings():

    global RETRIEVAL_EMBEDDINGS


    if RETRIEVAL_EMBEDDINGS is not None:

        return


    print(
        "[Inside Investigation] "
        "Building retrieval concept datastore..."
    )


    embeddings = []


    for i, item in enumerate(
        retrieval_pool
    ):

        # ==============================================================
        # retrieval_pool_64_concepts.json에
        # concepts가 반드시 있어야 함
        # ==============================================================

        concepts = item.get(
            "concepts",
            []
        )


        if not concepts:

            raise ValueError(
                f"retrieval_id="
                f"{item.get('retrieval_id', i)} "
                "에 concepts가 없습니다.\n"
                "먼저 retrieval concept preprocessing을 "
                "실행해야 합니다."
            )


        # ==============================================================
        # 논문 방식:
        #
        # c1 + c2 + ... + cN
        # ==============================================================

        concept_text = " ".join(
            concepts[
                :TOP_CONCEPTS
            ]
        )


        embedding = get_embedding(
            concept_text
        )


        embeddings.append(
            embedding
        )


        print(
            f"[Embedding] "
            f"{i+1}/{len(retrieval_pool)} "
            f"{concepts[:TOP_CONCEPTS]}"
        )


    RETRIEVAL_EMBEDDINGS = np.stack(
        embeddings
    )


    print(
        "[Inside Investigation] "
        "Concept datastore ready."
    )

# ======================================================================================================================
# 2. INSIDE INVESTIGATION
#
# nearest:
#
# Real 5
# Fake 5
# ======================================================================================================================

def inside_investigation(
    index,
    concepts
):

    build_retrieval_embeddings()


    query = " ".join(
        concepts
    )


    target_embedding = get_embedding(
        query
    )


    distances = np.linalg.norm(
        RETRIEVAL_EMBEDDINGS
        - target_embedding,
        axis=1
    )


    real_candidates = []
    fake_candidates = []


    for i, distance in enumerate(
        distances
    ):

        item = retrieval_pool[i]


        record = {
    "distance":
        float(distance),

    "concepts":
        item.get(
            "concepts",
            []
        ),

    "txt":
        item["txt"],

    "label":
        item["label"],

    "path":
        item.get(
            "path",
            ""
        ),

    "retrieval_id":
        item.get(
            "retrieval_id",
            i
        )
}


        if item["label"] is True:

            real_candidates.append(
                record
            )

        else:

            fake_candidates.append(
                record
            )


    real_candidates.sort(
        key=lambda x: x["distance"]
    )

    fake_candidates.sort(
        key=lambda x: x["distance"]
    )


    real_top = real_candidates[
        :INSIDE_K
    ]

    fake_top = fake_candidates[
        :INSIDE_K
    ]


    return {
        "real":
            real_top,

        "fake":
            fake_top
    }


# ======================================================================================================================
# 3. INSIDE JUDGE
#
# Real 5 → Real 2
# Fake 5 → Fake 2
# ======================================================================================================================

def parse_inside_judge(
    response
):

    try:

        response = response.replace(
            "```json",
            ""
        )

        response = response.replace(
            "```",
            ""
        )

        response = response.strip()


        data = json.loads(
            response
        )


        indices = data.get(
            "index",
            []
        )


        return [
            int(x)
            for x in indices[
                :INSIDE_M
            ]
        ]


    except Exception as e:

        print(
            "[Warning] Inside Judge parse failed:",
            e
        )

        return [
            0,
            1
        ]


def inside_judge(
    inside_result,
    concepts
):

    real_passages = inside_result[
        "real"
    ]

    fake_passages = inside_result[
        "fake"
    ]


    # ------------------------------------------------------
    # Real 5 중 2개
    # ------------------------------------------------------

    real_response = Inside_Judge_actor(
        real_passages,
        concepts
    )


    real_indices = parse_inside_judge(
        real_response
    )


    # ------------------------------------------------------
    # Fake 5 중 2개
    # ------------------------------------------------------

    fake_response = Inside_Judge_actor(
        fake_passages,
        concepts
    )


    fake_indices = parse_inside_judge(
        fake_response
    )


    selected_real = []

    selected_fake = []


    for idx in real_indices:

        if (
            0 <= idx
            < len(real_passages)
        ):

            selected_real.append(
                real_passages[idx]
            )


    for idx in fake_indices:

        if (
            0 <= idx
            < len(fake_passages)
        ):

            selected_fake.append(
                fake_passages[idx]
            )


    # parse 문제로 2개 안 나오면
    # kNN 상위부터 채움

    while (
        len(selected_real)
        < INSIDE_M
    ):

        candidate = real_passages[
            len(selected_real)
        ]

        if (
            candidate
            not in selected_real
        ):

            selected_real.append(
                candidate
            )


    while (
        len(selected_fake)
        < INSIDE_M
    ):

        candidate = fake_passages[
            len(selected_fake)
        ]

        if (
            candidate
            not in selected_fake
        ):

            selected_fake.append(
                candidate
            )


    return {
        "selected_real":
            selected_real,

        "selected_fake":
            selected_fake,

        "real_judge_response":
            real_response,

        "fake_judge_response":
            fake_response
    }


# ======================================================================================================================
# OUTSIDE INVESTIGATION
#
# 이 부분은 검색 API가 필요함.
#
# 논문은 Google + SerpAPI 사용.
#
# 아래 함수는 SERPAPI_API_KEY가 환경변수에 있을 때 사용.
# ======================================================================================================================

import wikipedia

wikipedia.set_lang("en")


# ======================================================================================================================
# OUTSIDE INVESTIGATION
#
# DKFND 원 논문:
# Google Search + SerpAPI -> Top L=8
#
# 현재 구현:
# Wikipedia Search -> Top L=8
#
# ======================================================================================================================

def outside_investigation(concepts):

    # ------------------------------------------------------
    # Detection에서 선정한 top-5 concept를 query로 사용
    # ------------------------------------------------------

    query = " ".join(concepts)

    print(
        f"[Wikipedia Query] {query}"
    )


    # ------------------------------------------------------
    # Wikipedia 검색 결과 상위 8개
    # ------------------------------------------------------

    try:

        search_results = wikipedia.search(
            query,
            results=OUTSIDE_L
        )

    except Exception as e:

        print(
            "[Warning] Wikipedia search failed:",
            e
        )

        search_results = []


    documents = []


    # ------------------------------------------------------
    # 각각의 페이지 내용 가져오기
    # ------------------------------------------------------

    for title in search_results:

        try:

            page = wikipedia.page(
                title,
                auto_suggest=False
            )


            # ==================================================
            # 너무 긴 전체 문서를 넣으면 token이 커지므로
            # 첫 번째 의미 있는 문단 사용
            # ==================================================

            paragraphs = [
                p.strip()
                for p in page.content.split("\n\n")
                if p.strip()
            ]


            if paragraphs:

                content = paragraphs[0]

            else:

                content = page.summary


            documents.append({

                "title":
                    page.title,

                "text":
                    (
                        f"Title: {page.title}\n"
                        f"{content}"
                    ),

                "link":
                    page.url
            })


        except wikipedia.exceptions.DisambiguationError as e:

            print(
                f"[Skip] Disambiguation: "
                f"{title}"
            )

            continue


        except wikipedia.exceptions.PageError:

            print(
                f"[Skip] Page not found: "
                f"{title}"
            )

            continue


        except Exception as e:

            print(
                f"[Skip] {title}: {e}"
            )

            continue


        # 최대 8개
        if len(documents) >= OUTSIDE_L:

            break


    print(
        f"[Wikipedia Retrieved] "
        f"{len(documents)} documents"
    )


    return {

        "query":
            query,

        "documents":
            documents
    }

# ======================================================================================================================
# OUTSIDE JUDGE
#
# 논문에서는 여기서:
#
# LLM triple extraction
#   ↓
# ComplEx / Wikidata score
#   ↓
# Google top1 + KG score top n-1
#
# 현재 main.py에서는 triple extraction까지 수행하고,
# 우선 Google ranking을 사용해서 top 2 선택.
#
# ComplEx 부분은 다음 단계에서 붙일 수 있음.
# ======================================================================================================================

def outside_judge(
    outside_result,
    concepts
):

    documents = outside_result[
        "documents"
    ]


    judge_results = []


    for document in documents:

        response = Outside_Judge_actor(
            document[
                "text"
            ],
            concepts
        )


        judge_results.append({
            "document":
                document,

            "triple_response":
                response
        })


    # ------------------------------------------------------
    # TEMPORARY:
    #
    # Google ranking top 2
    #
    # 논문 완전 재현 시
    # top1 + ComplEx score highest 1
    # 로 변경
    # ------------------------------------------------------

    selected = documents[
        :OUTSIDE_N
    ]


    return {
        "judge_results":
            judge_results,

        "selected":
            selected
    }


# ======================================================================================================================
# 4. INSIDE DETERMINATION
# ======================================================================================================================

def run_inside_determination(
    index,
    inside_judge_result
):

    examples = []


    # Real 2
    for item in inside_judge_result[
        "selected_real"
    ]:

        examples.append({
            "txt":
                item["txt"],

            "label":
                True
        })


    # Fake 2
    for item in inside_judge_result[
        "selected_fake"
    ]:

        examples.append({
            "txt":
                item["txt"],

            "label":
                False
        })


    response = Inside_Determination_actor(
        index,
        examples
    )


    return response


# ======================================================================================================================
# 5. OUTSIDE DETERMINATION
# ======================================================================================================================

def run_outside_determination(
    index,
    outside_judge_result
):

    outside_information = []


    for item in outside_judge_result[
        "selected"
    ]:

        outside_information.append(
            item["text"]
        )


    response = Outside_Determination_actor(
        index,
        outside_information
    )


    return response


# ======================================================================================================================
# RESPONSE → prediction / reason
# ======================================================================================================================

def split_prediction_reason(
    response
):

    lower = response.lower()


    if "[this is fake news]" in lower:

        prediction = (
            "[This is fake news]"
        )


    elif "[this is real news]" in lower:

        prediction = (
            "[This is real news]"
        )


    else:

        prediction = (
            "[No idea]"
        )


    return (
        prediction,
        response
    )


# ======================================================================================================================
#
# RUN DKFND
#
# experiment.py에서 사용하는 핵심 함수
#
# ======================================================================================================================

def run_DKFND(
    index,
    use_gt=False,
    reflections=None,
    memory=None
):

    if reflections is None:

        reflections = []


    if memory is None:

        memory = []


    print()
    print(
        "=" * 100
    )

    print(
        f"[DKFND] index={index}"
    )

    print(
        "=" * 100
    )


    # ==================================================================
    # (1) DETECTION
    # ==================================================================

    print(
        "\n[1] Detection"
    )


    detection = run_detection(
        index
    )


    concepts = detection[
        "concepts"
    ]


    print(
        "Concepts:",
        concepts
    )


    # ==================================================================
    # (2.1) INSIDE INVESTIGATION
    # ==================================================================

    print(
        "\n[2.1] Inside Investigation"
    )


    inside_result = inside_investigation(
        index,
        concepts
    )


    print(
        "Retrieved Real:",
        len(
            inside_result[
                "real"
            ]
        )
    )

    print(
        "Retrieved Fake:",
        len(
            inside_result[
                "fake"
            ]
        )
    )


    # ==================================================================
    # (2.2) OUTSIDE INVESTIGATION
    # ==================================================================

    print(
        "\n[2.2] Outside Investigation"
    )


    outside_result = outside_investigation(
        concepts
    )


    print(
        "Outside Documents:",
        len(
            outside_result[
                "documents"
            ]
        )
    )


    # ==================================================================
    # (3.1) INSIDE JUDGE
    # ==================================================================

    print(
        "\n[3.1] Inside Judge"
    )


    inside_judge_result = inside_judge(
        inside_result,
        concepts
    )


    print(
        "Selected Real:",
        len(
            inside_judge_result[
                "selected_real"
            ]
        )
    )

    print(
        "Selected Fake:",
        len(
            inside_judge_result[
                "selected_fake"
            ]
        )
    )


    # ==================================================================
    # (3.2) OUTSIDE JUDGE
    # ==================================================================

    print(
        "\n[3.2] Outside Judge"
    )


    outside_judge_result = outside_judge(
        outside_result,
        concepts
    )


    print(
        "Selected Outside:",
        len(
            outside_judge_result[
                "selected"
            ]
        )
    )


    # ==================================================================
    # (4.1) INSIDE DETERMINATION
    # ==================================================================

    print(
        "\n[4.1] Inside Determination"
    )


    inside_response = (
        run_inside_determination(
            index,
            inside_judge_result
        )
    )


    print(
        inside_response
    )


    inside_prediction, inside_reason = (
        split_prediction_reason(
            inside_response
        )
    )


    # ==================================================================
    # (4.2) OUTSIDE DETERMINATION
    # ==================================================================

    print(
        "\n[4.2] Outside Determination"
    )


    outside_response = (
        run_outside_determination(
            index,
            outside_judge_result
        )
    )


    print(
        outside_response
    )


    outside_prediction, outside_reason = (
        split_prediction_reason(
            outside_response
        )
    )


    # ==================================================================
    # (4.3) INTEGRATED DETERMINATION
    #
    # 논문:
    #
    # Pi == Po
    # → 그대로 final
    #
    # Pi != Po
    # → Integrated Determination
    #
    #
    # 다만 우리 Reflexion / GT 실험에서는
    # GT / reflection / memory가 final decision에 영향을 줘야 하므로
    # extension이 하나라도 있으면 항상 Integrated actor 실행.
    # ==================================================================

    print(
        "\n[4.3] Integrated Determination"
    )


    has_extension = (
        use_gt
        or bool(reflections)
        or bool(memory)
    )


    if (
        inside_prediction
        == outside_prediction
        and not has_extension
    ):

        # --------------------------------------------------------------
        # 원본 DKFND
        #
        # 두 관점이 같으면 그대로 사용
        # --------------------------------------------------------------

        final_response = (
            inside_response
        )


        print(
            "[Same prediction -> direct final]"
        )


    else:

        final_response = (
            Integrated_Extended_actor(
                index=index,

                inside_prediction=
                    inside_prediction,

                inside_reason=
                    inside_reason,

                outside_prediction=
                    outside_prediction,

                outside_reason=
                    outside_reason,

                use_gt=
                    use_gt,

                reflections=
                    reflections,

                memory=
                    memory
            )
        )


    print(
        final_response
    )


    # ==================================================================
    # RETURN
    #
    # experiment.py가 사용하는 형태
    # ==================================================================

    return {

        "index":
            index,


        # --------------------------------------------------------------
        # Detection
        # --------------------------------------------------------------

        "concepts":
            concepts,

        "raw_concepts":
            detection[
                "raw_concepts"
            ],

        "detection_response":
            detection[
                "detection_response"
            ],


        # --------------------------------------------------------------
        # Inside
        # --------------------------------------------------------------

        "inside_retrieved":
            inside_result,

        "inside_selected":
            {
                "real":
                    inside_judge_result[
                        "selected_real"
                    ],

                "fake":
                    inside_judge_result[
                        "selected_fake"
                    ]
            },

        "inside_response":
            inside_response,

        "inside_prediction":
            inside_prediction,


        # --------------------------------------------------------------
        # Outside
        # --------------------------------------------------------------

        "outside_query":
            outside_result[
                "query"
            ],

        "outside_retrieved":
            outside_result[
                "documents"
            ],

        "outside_selected":
            outside_judge_result[
                "selected"
            ],

        "outside_response":
            outside_response,

        "outside_prediction":
            outside_prediction,


        # --------------------------------------------------------------
        # Final
        # --------------------------------------------------------------

        "final_response":
            final_response
    }


# ======================================================================================================================
# TEST
#
# python main.py 로 직접 실행했을 때 하나만 확인
# ======================================================================================================================

if __name__ == "__main__":

    result = run_DKFND(
        index=0,

        use_gt=False,

        reflections=[],

        memory=[]
    )


    print()
    print(
        "=" * 100
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 100
    )


    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )