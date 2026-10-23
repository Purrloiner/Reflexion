from actor_propanews import (
    Vanilla_actor,
    CoT_actor,
)


def run_Vanilla(index):
    return Vanilla_actor(index)


def run_CoT(index):
    return CoT_actor(index)


def run_CoT_Re(index, reflection):
    return CoT_actor(
        index,
        reflection=reflection,
        use_gt=True,
    )


def run_CoT_GT_EPM_Re(index, trajectories, reflections):
    """Run the full GT + latest-trajectory EPM + Reflexion condition."""

    if not isinstance(trajectories, (list, tuple)):
        raise TypeError("trajectories must be a list or tuple")
    if len(trajectories) > 1:
        raise ValueError("CoT EPM accepts only the most recent trajectory")
    previous_trajectory = trajectories[0] if trajectories else None
    return CoT_actor(
        index,
        reflection=reflections,
        previous_trajectory=previous_trajectory,
        use_gt=True,
    )


__all__ = [
    "run_CoT",
    "run_CoT_GT_EPM_Re",
    "run_CoT_Re",
    "run_Vanilla",
]
