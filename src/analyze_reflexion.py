import json
from pathlib import Path
import importlib


# ============================================================
# 설정
# ============================================================

ROOT = Path("/home/sscc/Humane")

ANALYSIS_DIR = ROOT / "outputs" / "reflexion_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# None -> 해당 조건의 모든 사례 저장
# 5 -> 각 조건에서 앞의 5개만 저장
MAX_CASES = None


CONFIGS = {
    "sprs_reflexion": {
        "path": ROOT / "outputs" / "propanews_sprs_mtcpsl_geminai" / "sprs_reflexion.json",
        "actor": "actor_mtcpsl",
        "method": "SPRS",
    },

    "pcot_reflexion": {
        "path": ROOT / "outputs" / "propanews_pcot_geminai" / "pcot_reflexion.json",
        "actor": "actor_pcot",
        "method": "PCoT",
    },
}


# ============================================================
# 유틸
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def divider(char="=", length=100):
    return char * length


def get_attempt_response(attempt):
    """
    SPRS:
        response

    PCoT:
        final_response
    """
    if "final_response" in attempt:
        return attempt["final_response"]

    return attempt.get("response", "")


# ============================================================
# 한 사례 출력
# ============================================================

def format_case(sample, actor, method):

    index = sample["index"]
    gold = sample["gold_label"]
    solved = sample["solved"]
    success_trial = sample["success_trial"]

    lines = []

    lines.append(divider("="))
    lines.append(
        f"INDEX = {index} | GOLD = {gold} | "
        f"SOLVED = {solved} | SUCCESS_TRIAL = {success_trial}"
    )
    lines.append(divider("="))
    lines.append("")

    # --------------------------------------------------------
    # 원문
    # --------------------------------------------------------

    article = actor.get_article(index)

    lines.append("[TARGET ARTICLE]")
    lines.append(article)
    lines.append("")

    # --------------------------------------------------------
    # Trial
    # --------------------------------------------------------

    for attempt in sample["attempts"]:

        trial = attempt["trial"]
        prediction = attempt.get(
            "predicted_label",
            attempt.get("prediction", "")
        )
        correct = attempt["correct"]

        lines.append(divider("-"))
        lines.append(
            f"TRIAL {trial} | "
            f"PRED = {prediction} | "
            f"GOLD = {gold} | "
            f"{'SUCCESS' if correct else 'FAIL'}"
        )
        lines.append(divider("-"))
        lines.append("")

        # ====================================================
        # 이번 Trial에 들어간 Reflection
        # ====================================================

        reflection_used = attempt.get("reflection_used", [])

        lines.append("[REFLECTION USED BEFORE THIS TRIAL]")

        if reflection_used:
            for i, ref in enumerate(reflection_used, start=1):
                lines.append("")
                lines.append(f"Reflection Memory {i}")
                lines.append(ref)
        else:
            lines.append("NONE")

        lines.append("")

        # ====================================================
        # PCoT Stage 1
        # ====================================================

        if method == "PCoT":

            persuasion = attempt.get(
                "persuasion_analysis",
                ""
            )

            lines.append("[PCoT STAGE 1 - PERSUASION ANALYSIS]")
            lines.append(persuasion)
            lines.append("")

        # ====================================================
        # 최종 추론
        # ====================================================

        response = get_attempt_response(attempt)

        if method == "PCoT":
            lines.append("[PCoT STAGE 2 - FINAL RESPONSE]")
        else:
            lines.append("[SPRS RESPONSE]")

        lines.append(response)
        lines.append("")

        # ====================================================
        # Reflection 생성 결과
        # ====================================================

        reflection_generated = attempt.get(
            "reflection_generated"
        )

        lines.append("[REFLECTION GENERATED AFTER THIS FAILURE]")

        if reflection_generated:
            lines.append(reflection_generated)
        else:
            lines.append("NONE")

        lines.append("")

    return "\n".join(lines)


# ============================================================
# 파이프라인 분석
# ============================================================

def analyze_pipeline(name, config):

    print()
    print(divider("="))
    print(name)
    print(divider("="))

    result = load_json(config["path"])

    actor = importlib.import_module(
        config["actor"]
    )

    samples = result["results"]

    # --------------------------------------------------------
    # 1. 처음 틀렸지만 Reflection 후 성공
    # --------------------------------------------------------

    recovered = [
        sample
        for sample in samples
        if sample["solved"]
        and sample["success_trial"] is not None
        and sample["success_trial"] > 1
    ]

    # --------------------------------------------------------
    # 2. Reflection을 끝까지 했지만 실패
    # --------------------------------------------------------

    persistent_failures = [
        sample
        for sample in samples
        if not sample["solved"]
    ]

    print()
    print("[Reflection 후 성공]")
    print(
        [
            (s["index"], s["success_trial"])
            for s in recovered
        ]
    )

    print()
    print("[끝까지 실패]")
    print(
        [s["index"] for s in persistent_failures]
    )

    print()
    print(
        f"Reflection 후 성공: {len(recovered)}개"
    )
    print(
        f"5회까지 실패: {len(persistent_failures)}개"
    )

    # 필요하면 개수 제한
    if MAX_CASES is not None:
        recovered_to_write = recovered[:MAX_CASES]
        failures_to_write = persistent_failures[:MAX_CASES]
    else:
        recovered_to_write = recovered
        failures_to_write = persistent_failures

    # --------------------------------------------------------
    # 성공 사례 저장
    # --------------------------------------------------------

    success_path = (
        ANALYSIS_DIR /
        f"{name}_recovered.txt"
    )

    with open(success_path, "w", encoding="utf-8") as f:

        f.write(
            f"{name}\n"
            f"FAIL -> REFLECTION -> SUCCESS\n"
            f"Total cases: {len(recovered)}\n\n"
        )

        for sample in recovered_to_write:

            f.write(
                format_case(
                    sample,
                    actor,
                    config["method"],
                )
            )

            f.write("\n\n\n")

    # --------------------------------------------------------
    # 실패 사례 저장
    # --------------------------------------------------------

    fail_path = (
        ANALYSIS_DIR /
        f"{name}_persistent_failures.txt"
    )

    with open(fail_path, "w", encoding="utf-8") as f:

        f.write(
            f"{name}\n"
            f"FAIL -> REFLECTION -> ... -> FAIL\n"
            f"Total cases: {len(persistent_failures)}\n\n"
        )

        for sample in failures_to_write:

            f.write(
                format_case(
                    sample,
                    actor,
                    config["method"],
                )
            )

            f.write("\n\n\n")

    print()
    print("[Saved]")
    print(success_path)
    print(fail_path)


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":

    for name, config in CONFIGS.items():
        analyze_pipeline(
            name,
            config
        )

    print()
    print(divider("="))
    print("분석 파일 생성 완료")
    print(ANALYSIS_DIR)
    print(divider("="))
