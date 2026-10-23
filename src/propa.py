from propa_main import run_DKFND

from propa_re import (
    make_reflection_DKFND,
    make_reflection_DKFND_GT
)

from propa_prom_DKFND import (
    get_news,
    get_gt_context,
    parse_prediction
)

import json
import os
import matplotlib.pyplot as plt


# ======================================================================================================================
# PATH
# ======================================================================================================================

DATA_PATH = (
    "/home/sscc/Humane/data/"
    "propanews_150_sample_final.json"
)

OUTPUT_DIR = (
    "/home/sscc/Humane/outputs/DKFND"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ======================================================================================================================
# DATA
# ======================================================================================================================

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    propanews_150 = json.load(f)


# ======================================================================================================================
# EXPERIMENT SETTING
#
# 100개
# 최대 5번 시도
#
# 성공한 문제:
#   이후 trial에서 실행하지 않음
#
# 실패한 문제:
#   다음 trial에서 다시 실행
# ======================================================================================================================

NUM = min(
    100,
    len(propanews_150)
)


TRIAL = 5


# Reflection은 최근 3개
MAX_REFLECTION = 3


# Episodic Memory
# 직전 trajectory 딱 1개
MAX_MEMORY = 1


print(
    f"[Dataset] {NUM} samples"
)

print(
    f"[Max Trial] {TRIAL}"
)


# ======================================================================================================================
# 사용할 100개 데이터
# ======================================================================================================================

experiment_data = propanews_150[
    :NUM
]


# ======================================================================================================================
# INDEX -> DATA
# ======================================================================================================================

data_by_index = {
    item["index"]: item
    for item in experiment_data
}


def get_label(index):

    return data_by_index[
        index
    ][
        "label"
    ]


# ======================================================================================================================
# SAVE
# ======================================================================================================================

def save_result(
    name,
    result
):

    path = os.path.join(
        OUTPUT_DIR,
        f"{name}.json"
    )


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )


    print(
        f"[Saved] {path}"
    )


# ======================================================================================================================
# LOAD OR RUN
#
# 이전 accuracy/F1 방식의 결과 파일은 사용하지 않음.
#
# 이번 실험:
# evaluation_mode = failure_retry_cumulative
#
# 이 형식의 100개 / 5 trial 결과만 load
# ======================================================================================================================

def load_or_run(
    name,
    func
):

    path = os.path.join(
        OUTPUT_DIR,
        f"{name}.json"
    )


    if os.path.exists(
        path
    ):

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                result = json.load(f)


            if (
                result.get(
                    "evaluation_mode"
                )
                == "failure_retry_cumulative"
                and
                result.get(
                    "num_samples"
                )
                == NUM
                and
                result.get(
                    "max_trials"
                )
                == TRIAL
                and
                len(
                    result.get(
                        "cumulative_success",
                        []
                    )
                )
                == TRIAL
            ):

                print(
                    f"[Load] {name}"
                )

                return result


            print(
                f"[Rerun] {name} "
                "(old or incomplete result)"
            )


        except Exception:

            print(
                f"[Rerun] {name} "
                "(failed to load old result)"
            )


    else:

        print(
            f"[Run] {name}"
        )


    result = func()


    save_result(
        name,
        result
    )


    return result


# ======================================================================================================================
# TRAJECTORY
#
# Episodic Memory에 저장할 "직전 실패 시도"
#
# 단순 final answer 하나만 넣는 것이 아니라
#
# Inside 판단
# Outside 판단
# Final 판단
#
# 을 하나의 previous trajectory로 저장.
#
# 다음 trial에서 이 trajectory 딱 1개만 memory로 전달됨.
#
# 정답 label이나 "틀렸다"는 내용 자체는 trajectory에 넣지 않음.
# ======================================================================================================================

def make_previous_trajectory(
    result
):

    inside_response = result.get(
        "inside_response",
        ""
    )


    outside_response = result.get(
        "outside_response",
        ""
    )


    final_response = result.get(
        "final_response",
        ""
    )


    trajectory = (
        "Previous trajectory:\n\n"

        "Inside determination:\n"
        + inside_response
        + "\n\n"

        "Outside determination:\n"
        + outside_response
        + "\n\n"

        "Final determination:\n"
        + final_response
    )


    return trajectory


# ======================================================================================================================
#
# COMMON EVALUATION
#
#
# 핵심:
#
# Trial 0
#   모든 100개 실행
#
#   맞음 -> solved
#   틀림 -> 다음 trial로
#
#
# Trial 1
#   Trial 0에서 실패한 문제만 실행
#
#
# Trial 2
#   Trial 1에서도 실패한 문제만 실행
#
# ...
#
#
# cumulative success:
#
# 지금까지 한 번이라도 성공한 문제 수 / 100
#
#
# 예:
#
# Trial 0 = 65 / 100
# Trial 1 = 78 / 100
# Trial 2 = 84 / 100
# Trial 3 = 88 / 100
# Trial 4 = 90 / 100
#
# ======================================================================================================================

def evaluate_method(
    method_name,
    use_gt=False,
    use_reflection=False,
    use_memory=False
):

    # ==================================================================
    # attempt별 새로 성공한 문제 수
    #
    # solved_at_attempt[0]
    #   = 첫 번째 시도에서 성공한 문제 수
    #
    # solved_at_attempt[1]
    #   = 첫 번째 실패 후 두 번째 시도에서 성공한 문제 수
    #
    # ...
    # ==================================================================

    solved_at_attempt = [
        0
        for _ in range(TRIAL)
    ]


    all_results = []


    # ==================================================================
    # 문제 하나씩 처리
    #
    # index 0을 최대 5번까지 끝낸 뒤
    # index 1로 넘어감
    # ==================================================================

    for i in range(NUM):

        index = experiment_data[
            i
        ][
            "index"
        ]

        label = get_label(
            index
        )


        print()
        print(
            "=" * 100
        )

        print(
            f"[{method_name}] "
            f"index={index} "
            f"({i+1}/{NUM})"
        )

        print(
            "=" * 100
        )


        # ==================================================================
        # 이 문제만의 Reflection
        # ==================================================================

        reflections = []


        # ==================================================================
        # 이 문제만의 trajectory memory
        #
        # 항상 직전 실패 trajectory 1개만 유지
        # ==================================================================

        trajectory_memory = []


        sample_attempts = []

        solved = False

        success_attempt = None


        # ==================================================================
        # 한 문제를 최대 5번 시도
        # ==================================================================

        for trial in range(TRIAL):

            print()
            print(
                "-" * 80
            )

            print(
                f"index={index} "
                f"Trial {trial}"
            )

            print(
                "-" * 80
            )


            # ==============================================================
            # 이번 trial에 실제로 전달되는 Reflection
            # ==============================================================

            if use_reflection:

                current_reflections = list(
                    reflections
                )

            else:

                current_reflections = []


            # ==============================================================
            # 이번 trial에 실제로 전달되는 trajectory
            # ==============================================================

            if use_memory:

                current_memory = list(
                    trajectory_memory
                )

            else:

                current_memory = []


            # ==============================================================
            # DKFND 실행
            # ==============================================================

            result = run_DKFND(

                index=index,

                use_gt=use_gt,

                reflections=
                    current_reflections,

                memory=
                    current_memory
            )


            inside_response = result.get(
                "inside_response",
                ""
            )


            outside_response = result.get(
                "outside_response",
                ""
            )


            final_response = result.get(
                "final_response",
                ""
            )


            prediction = parse_prediction(
                final_response
            )


            # ==============================================================
            # 성공 여부
            # ==============================================================

            is_correct = (
                prediction is not None
                and
                prediction == label
            )


            new_reflection = None


            # ==================================================================
            # SUCCESS
            # ==================================================================

            if is_correct:

                solved = True

                success_attempt = trial


                solved_at_attempt[
                    trial
                ] += 1


                print(
                    f"[{i+1}/{NUM}] "
                    f"index={index} "
                    f"trial={trial} "
                    f"label={label} "
                    f"pred={prediction} "
                    f"-> SUCCESS"
                )


            # ==================================================================
            # FAIL
            #
            # 다음 attempt가 남아 있을 경우
            # Reflection / trajectory 준비
            # ==================================================================

            else:

                print(
                    f"[{i+1}/{NUM}] "
                    f"index={index} "
                    f"trial={trial} "
                    f"label={label} "
                    f"pred={prediction} "
                    f"-> FAIL"
                )


                # ----------------------------------------------------------
                # 마지막 시도가 아니면 다음 시도를 위한 정보 생성
                # ----------------------------------------------------------

                if trial < TRIAL - 1:

                    news = get_news(
                        index
                    )


                    # ======================================================
                    # REFLEXION
                    #
                    # 실패했을 때만 생성
                    # ======================================================

                    if use_reflection:

                        if use_gt:

                            gt_context = get_gt_context(
                                index
                            )


                            new_reflection = (
                                make_reflection_DKFND_GT(

                                    news=
                                        news,

                                    final_response=
                                        final_response,

                                    gt_context=
                                        gt_context,

                                    inside_response=
                                        inside_response,

                                    outside_response=
                                        outside_response
                                )
                            )


                        else:

                            new_reflection = (
                                make_reflection_DKFND(

                                    news=
                                        news,

                                    final_response=
                                        final_response,

                                    inside_response=
                                        inside_response,

                                    outside_response=
                                        outside_response
                                )
                            )


                        reflections.append(
                            new_reflection
                        )


                        # 최근 3개만 유지
                        if (
                            len(reflections)
                            > MAX_REFLECTION
                        ):

                            reflections.pop(
                                0
                            )


                    # ======================================================
                    # TRAJECTORY MEMORY
                    #
                    # 직전 실패 시도의
                    #
                    # Inside
                    # Outside
                    # Final
                    #
                    # 전체 과정을 저장
                    #
                    # 이전 문제 trajectory는 절대 안 들어옴.
                    # 현재 index의 trajectory만 존재.
                    # ======================================================

                    if use_memory:

                        previous_trajectory = (
                            make_previous_trajectory(
                                result
                            )
                        )


                        # 누적 X
                        # 항상 바로 직전 실패 trajectory 1개로 교체
                        trajectory_memory = [
                            previous_trajectory
                        ]


            # ==================================================================
            # ATTEMPT 저장
            # ==================================================================

            sample_attempts.append({

                "trial":
                    trial,

                "label":
                    label,

                "prediction":
                    prediction,

                "correct":
                    is_correct,

                # 이번 시도에 실제 사용한 reflection
                "used_reflections":
                    current_reflections,

                # 이번 시도에 실제 사용한 trajectory
                "used_memory":
                    current_memory,

                "inside_response":
                    inside_response,

                "outside_response":
                    outside_response,

                "final_response":
                    final_response,

                # 실패 후 생성된 reflection
                "new_reflection":
                    new_reflection,

                # 다음 시도에서 사용할 reflection
                "next_reflections":
                    list(
                        reflections
                    ),

                # 다음 시도에서 사용할 trajectory
                "next_memory":
                    list(
                        trajectory_memory
                    )
            })


            # ==================================================================
            # 맞았으면 이 문제 종료
            #
            # 바로 다음 index로 이동
            # ==================================================================

            if solved:

                break


        # ==================================================================
        # 한 문제 결과 저장
        # ==================================================================

        all_results.append({

            "index":
                index,

            "label":
                label,

            "solved":
                solved,

            # 0-based
            # 첫 시도 성공 = 0
            "success_trial":
                success_attempt,

            # 사람이 보기 편하게
            # 첫 시도 성공 = 1
            "success_attempt":
                (
                    success_attempt + 1
                    if success_attempt is not None
                    else None
                ),

            "attempt_count":
                len(
                    sample_attempts
                ),

            "attempts":
                sample_attempts
        })


        if solved:

            print()
            print(
                f"[DONE] "
                f"index={index} "
                f"solved at attempt "
                f"{success_attempt + 1}"
            )

        else:

            print()
            print(
                f"[FAILED] "
                f"index={index} "
                f"failed after "
                f"{TRIAL} attempts"
            )


    # ==================================================================
    #
    # CUMULATIVE SUCCESS RATE
    #
    # Trial 0:
    #   첫 시도까지 성공한 문제
    #
    # Trial 1:
    #   두 번째 시도까지 성공한 문제
    #
    # ...
    #
    # ==================================================================

    cumulative_success = []

    cumulative_count = 0


    trial_summary = []


    for trial in range(TRIAL):

        newly_solved = solved_at_attempt[
            trial
        ]


        cumulative_count += (
            newly_solved
        )


        cumulative_rate = (
            cumulative_count
            /
            NUM
        )


        cumulative_success.append(
            cumulative_rate
        )


        trial_summary.append({

            "trial":
                trial,

            "attempt_number":
                trial + 1,

            "newly_solved":
                newly_solved,

            "total_solved":
                cumulative_count,

            "remaining":
                NUM - cumulative_count,

            "cumulative_success":
                cumulative_rate
        })


    # ==================================================================
    # PRINT SUMMARY
    # ==================================================================

    print()
    print(
        "=" * 100
    )

    print(
        f"[{method_name}] "
        f"CUMULATIVE SUCCESS"
    )

    print(
        "=" * 100
    )


    for item in trial_summary:

        print(

            f"Attempt "
            f"{item['attempt_number']} | "

            f"Newly Solved = "
            f"{item['newly_solved']} | "

            f"Total = "
            f"{item['total_solved']}/{NUM} | "

            f"Cumulative = "
            f"{item['cumulative_success'] * 100:.1f}%"
        )


    # ==================================================================
    # RETURN
    # ==================================================================

    return {

        "evaluation_mode":
            "failure_retry_cumulative",

        "method":
            method_name,

        "num_samples":
            NUM,

        "max_trials":
            TRIAL,

        "max_reflection":
            MAX_REFLECTION,

        "max_memory":
            MAX_MEMORY,

        "solved_at_attempt":
            solved_at_attempt,

        "cumulative_success":
            cumulative_success,

        "trial_summary":
            trial_summary,

        "results":
            all_results
    }

# ======================================================================================================================
#
# 1. WITHOUT GT
#
# DKFND
#
# 실패하면 그냥 DKFND를 다시 실행
#
# Reflection X
# Memory X
#
# ======================================================================================================================

def evaluate_DKFND():

    return evaluate_method(

        method_name=
            "DKFND",

        use_gt=False,

        use_reflection=False,

        use_memory=False
    )


# ======================================================================================================================
#
# 2. WITHOUT GT
#
# DKFND + REFLEXION
#
# 실패:
#   reflection 생성
#
# 다음 trial:
#   실패 문제만 reflection과 함께 재실행
#
# ======================================================================================================================

def evaluate_DKFND_Re():

    return evaluate_method(

        method_name=
            "DKFND + Reflexion",

        use_gt=False,

        use_reflection=True,

        use_memory=False
    )


# ======================================================================================================================
#
# 3. WITH GT
#
# GT + DKFND
#
# 실패하면 GT 조건 그대로 재실행
#
# ======================================================================================================================

def evaluate_GT_DKFND():

    return evaluate_method(

        method_name=
            "GT + DKFND",

        use_gt=True,

        use_reflection=False,

        use_memory=False
    )


# ======================================================================================================================
#
# 4. WITH GT
#
# GT + DKFND + REFLEXION
#
# 실패:
#   GT context 기반 reflection 생성
#
# 다음 trial:
#   실패 문제만 재실행
#
# ======================================================================================================================

def evaluate_GT_DKFND_Re():

    return evaluate_method(

        method_name=
            "GT + DKFND + Reflexion",

        use_gt=True,

        use_reflection=True,

        use_memory=False
    )


# ======================================================================================================================
#
# 5. WITH GT
#
# GT + DKFND + REFLEXION + MEMORY
#
#
# 실패:
#
#   Reflection 생성
#
#   +
#
#   직전 실패 trajectory 저장
#
#
# 다음 trial:
#
#   최근 Reflection 최대 3개
#
#   +
#
#   바로 직전 실패 trajectory 1개
#
#   를 함께 사용
#
#
# 예:
#
# Trial 0 실패
#
#   reflection = R0
#   trajectory = T0
#
#
# Trial 1
#
#   reflection = [R0]
#   memory     = [T0]
#
#
# Trial 1 또 실패
#
#   reflection = [R0, R1]
#   trajectory = [T1]
#
#
# Trial 2
#
#   reflection = [R0, R1]
#   memory     = [T1]
#
#
# trajectory는 T0 + T1로 누적되지 않음.
# 무조건 가장 최근 실패 시도 1개만 전달.
#
# ======================================================================================================================

def evaluate_GT_DKFND_Mem_Re():

    return evaluate_method(

        method_name=
            "GT + DKFND + Reflexion + Memory",

        use_gt=True,

        use_reflection=True,

        use_memory=True
    )


# ======================================================================================================================
#
# RUN
#
# ======================================================================================================================

dkfnd = load_or_run(

    "dkfnd",

    evaluate_DKFND
)


dkfnd_re = load_or_run(

    "dkfnd_re",

    evaluate_DKFND_Re
)


gt_dkfnd = load_or_run(

    "gt_dkfnd",

    evaluate_GT_DKFND
)


gt_dkfnd_re = load_or_run(

    "gt_dkfnd_re",

    evaluate_GT_DKFND_Re
)


gt_dkfnd_mem_re = load_or_run(

    "gt_dkfnd_mem_re",

    evaluate_GT_DKFND_Mem_Re
)


# ======================================================================================================================
#
# PRINT CUMULATIVE SUCCESS
#
# ======================================================================================================================

print()
print(
    "=" * 120
)

print(
    "CUMULATIVE SUCCESS RATE"
)

print(
    "=" * 120
)


methods = {

    "DKFND":
        dkfnd,

    "DKFND + Reflexion":
        dkfnd_re,

    "GT + DKFND":
        gt_dkfnd,

    "GT + DKFND + Reflexion":
        gt_dkfnd_re,

    "GT + DKFND + Reflexion + Mem":
        gt_dkfnd_mem_re
}


for name, result in methods.items():

    rates = result[
        "cumulative_success"
    ]


    percentages = [

        f"{rate * 100:.1f}%"

        for rate in rates
    ]


    print(
        f"{name:<40} "
        f"{percentages}"
    )


# ======================================================================================================================
#
# CUMULATIVE SUCCESS GRAPH
#
# ======================================================================================================================

def save_cumulative_success_graph():

    trials = list(
        range(
            TRIAL
        )
    )


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            15,
            4.5
        )
    )


    # ==============================================================================================================
    # (a) WITHOUT GT
    # ==============================================================================================================

    axes[
        0
    ].plot(

        trials,

        dkfnd[
            "cumulative_success"
        ],

        color=
            "lightgray",

        linestyle=
            "--",

        marker=
            "o",

        markersize=
            4,

        linewidth=
            1.2,

        label=
            "DKFND"
    )


    axes[
        0
    ].plot(

        trials,

        dkfnd_re[
            "cumulative_success"
        ],

        color=
            "firebrick",

        linestyle=
            "-",

        marker=
            "D",

        markersize=
            4,

        linewidth=
            1.3,

        label=
            "DKFND + Reflexion"
    )


    axes[
        0
    ].set_title(
        "(a) Without GT",
        fontweight="bold"
    )


    axes[
        0
    ].set_xlabel(
        "Trial Number",
        fontweight="bold"
    )


    axes[
        0
    ].set_ylabel(
        "Cumulative Success Rate",
        fontweight="bold"
    )


    axes[
        0
    ].set_xticks(
        trials
    )


    axes[
        0
    ].set_ylim(
        0.0,
        1.0
    )


    axes[
        0
    ].grid(
        True,
        alpha=0.5
    )


    axes[
        0
    ].legend(
        loc="lower right",
        fontsize=8,
        frameon=True
    )


    # ==============================================================================================================
    # (b) WITH GT
    # ==============================================================================================================

    axes[
        1
    ].plot(

        trials,

        gt_dkfnd[
            "cumulative_success"
        ],

        color=
            "lightgray",

        linestyle=
            "--",

        marker=
            "o",

        markersize=
            4,

        linewidth=
            1.2,

        label=
            "DKFND (GT)"
    )


    axes[
        1
    ].plot(

        trials,

        gt_dkfnd_re[
            "cumulative_success"
        ],

        color=
            "firebrick",

        linestyle=
            "-",

        marker=
            "D",

        markersize=
            4,

        linewidth=
            1.3,

        label=
            "DKFND (GT) + Reflexion"
    )


    axes[
        1
    ].set_title(
        "(b) With GT",
        fontweight="bold"
    )


    axes[
        1
    ].set_xlabel(
        "Trial Number",
        fontweight="bold"
    )


    axes[
        1
    ].set_ylabel(
        "Cumulative Success Rate",
        fontweight="bold"
    )


    axes[
        1
    ].set_xticks(
        trials
    )


    axes[
        1
    ].set_ylim(
        0.0,
        1.0
    )


    axes[
        1
    ].grid(
        True,
        alpha=0.5
    )


    axes[
        1
    ].legend(
        loc="lower right",
        fontsize=8,
        frameon=True
    )


    # ==============================================================================================================
    # (c) EPISODIC MEMORY
    # ==============================================================================================================

    axes[
        2
    ].plot(

        trials,

        gt_dkfnd[
            "cumulative_success"
        ],

        color=
            "lightgray",

        linestyle=
            "--",

        marker=
            "o",

        markersize=
            4,

        linewidth=
            1.2,

        label=
            "DKFND (GT)"
    )


    axes[
        2
    ].plot(

        trials,

        gt_dkfnd_re[
            "cumulative_success"
        ],

        color=
            "violet",

        linestyle=
            "--",

        marker=
            "D",

        markersize=
            4,

        linewidth=
            1.2,

        label=
            "DKFND (GT) + Reflexion"
    )


    axes[
        2
    ].plot(

        trials,

        gt_dkfnd_mem_re[
            "cumulative_success"
        ],

        color=
            "purple",

        linestyle=
            "-",

        marker=
            "D",

        markersize=
            4,

        linewidth=
            1.3,

        label=
            "DKFND (GT) + Reflexion + Mem"
    )


    axes[
        2
    ].set_title(
        "(c) Episodic Memory",
        fontweight="bold"
    )


    axes[
        2
    ].set_xlabel(
        "Trial Number",
        fontweight="bold"
    )


    axes[
        2
    ].set_ylabel(
        "Cumulative Success Rate",
        fontweight="bold"
    )


    axes[
        2
    ].set_xticks(
        trials
    )


    axes[
        2
    ].set_ylim(
        0.0,
        1.0
    )


    axes[
        2
    ].grid(
        True,
        alpha=0.5
    )


    axes[
        2
    ].legend(
        loc="lower right",
        fontsize=8,
        frameon=True
    )


    # ==============================================================================================================
    # SAVE
    # ==============================================================================================================

    plt.tight_layout()


    output_path = os.path.join(

        OUTPUT_DIR,

        "propanews_dkfnd_cumulative_success.png"
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print()
    print(
        f"Graph saved to "
        f"{output_path}"
    )


# ======================================================================================================================
# SAVE GRAPH
# ======================================================================================================================

save_cumulative_success_graph()