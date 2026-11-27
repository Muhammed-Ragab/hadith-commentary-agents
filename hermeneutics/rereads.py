"""Bounded-agency re-read cap. Spec §3.1.

The cap is a CEILING, not a quota. The agent may always flag zero; a kitāb where
it declines to revisit is a finding, not a gap. The exercised count is reported.
"""
from typing import Iterable

MAX_PER_KITAB = 5
FRACTION = 0.10


def reread_cap(n_babs: int) -> int:
    """Max units the agent may flag for re-read after this kitāb.

    Original design used min(5, floor(0.10 * n)), which is 0 for any kitāb under
    10 bābs - that silenced 22 of Nawawi's 44 kitābs. max(1, round(...)) fixes it.
    """
    if n_babs <= 0:
        return 0
    return min(MAX_PER_KITAB, max(1, round(FRACTION * n_babs)))


def total_reread_budget(kitab_sizes: Iterable[int]) -> int:
    """Upper bound on re-reads across a commentator. Planning only."""
    return sum(reread_cap(n) for n in kitab_sizes)
