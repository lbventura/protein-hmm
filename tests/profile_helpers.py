"""Independent expected transition groups for profile probability tests."""

from protein_profile_hmm.traces import Transition


def allowed_transition_groups(K: int) -> list[tuple[int, list[Transition]]]:
    groups = [
        (0, [Transition.BM, Transition.MI, Transition.BD]),
        (0, [Transition.IM, Transition.II]),
    ]
    for k in range(1, K):
        groups.extend(
            [
                (k, [Transition.MM, Transition.MI, Transition.MD]),
                (k, [Transition.IM, Transition.II]),
                (k, [Transition.DM, Transition.DD]),
            ]
        )
    return [
        *groups,
        (K, [Transition.MI, Transition.ME]),
        (K, [Transition.II, Transition.IE]),
        (K, [Transition.DE]),
    ]
