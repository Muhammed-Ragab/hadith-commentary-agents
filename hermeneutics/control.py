"""Shuffled-pair control. Spec §4.1 - fixes validity threat V1.

The control runs the identical pipeline with commentary reassigned across bābs
within the same kitāb. It runs in a SEPARATE graph invocation with its own method
hypothesis, thread id, and checkpointer namespace. Sharing the hypothesis would let
genuine study contaminate the control - corrupting the one instrument built to
detect contamination.
"""
import random
from collections import defaultdict

DEFAULT_FRACTION = 0.10
DEFAULT_SEED = 20260821


def shuffle_within_kitab(babs: list[dict], seed: int = DEFAULT_SEED,
                         fraction: float = DEFAULT_FRACTION) -> dict[int, int]:
    """Map bāb index -> index whose commentary it should be paired with.

    Within-kitāb only: shuffling across kitābs would confound topic with pairing.
    A derangement, so no unit is paired with its own commentary.
    """
    rng = random.Random(seed)

    by_kitab: dict[str, list[int]] = defaultdict(list)
    for i, bab in enumerate(babs):
        by_kitab[bab["kitab"]].append(i)

    pairing: dict[int, int] = {}
    for indices in by_kitab.values():
        if len(indices) < 2:
            continue  # nothing to swap with
        n = max(2, round(fraction * len(indices)))
        n = min(n, len(indices))
        chosen = rng.sample(indices, n)
        rotated = chosen[1:] + chosen[:1]  # rotation is a derangement for n >= 2
        pairing.update(dict(zip(chosen, rotated)))

    return pairing


def control_indices(pairing: dict[int, int]) -> list[int]:
    """The bābs that belong to the control arm."""
    return sorted(pairing)


def assert_isolated(real_hypothesis: str, control_hypothesis: str) -> None:
    """Guard: the two arms must never converge on the same hypothesis text.

    Both start empty, which is the initial condition rather than a leak.
    """
    if not real_hypothesis and not control_hypothesis:
        return
    if real_hypothesis == control_hypothesis:
        raise ValueError(
            "control and real runs share a method hypothesis - the control is "
            "contaminated and its comparison is worthless (spec §4.1)"
        )
