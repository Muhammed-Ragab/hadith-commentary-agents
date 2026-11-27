"""Scholar-review cadence. handoff.md §10.6.

The 2026-08-23 reparse took the corpus from 63 kitābs to 133. Interrupting at every
one is ~22-33 h of scholar time and does not fit the calendar, so ʿAynī groups
several kitābs per sitting. Grouping must change WHEN the scholar is interrupted,
never WHAT gets reviewed or how much re-read agency the agent has.
"""
from collections import Counter

from hermeneutics.rereads import reread_cap, total_reread_budget
from hermeneutics.study_graph import is_review_boundary, kitab_order
from scripts.check_corpus import load_babs
from scripts.run_study_graph import REVIEW_EVERY


def _babs(spec: list[tuple[str, int]]) -> list[dict]:
    return [{"kitab": k, "bab": f"b{i}", "text": "x"}
            for k, n in spec for i in range(n)]


def sittings(babs: list[dict], group: int) -> int:
    """Count interrupts a full pass would fire."""
    n = 0
    for cursor in range(1, len(babs) + 1):
        current = babs[cursor - 1]["kitab"]
        if is_review_boundary(babs, cursor, current, group):
            n += 1
    return n


def test_group_size_one_interrupts_at_every_kitab():
    babs = _babs([("k1", 3), ("k2", 2), ("k3", 4)])
    assert sittings(babs, 1) == 3


def test_grouping_reduces_sittings_without_dropping_the_tail():
    """Five kitābs, group of 2: sittings after k2 and k4, plus the final one."""
    babs = _babs([(f"k{i}", 2) for i in range(1, 6)])
    assert sittings(babs, 2) == 3


def test_last_kitab_is_always_reviewed():
    """A group that does not divide evenly must not leave the tail unreviewed."""
    for group in (2, 3, 4, 5, 7):
        babs = _babs([(f"k{i}", 2) for i in range(1, 10)])
        assert is_review_boundary(babs, len(babs), "k9", group), group


def test_configured_cadence_lands_near_the_pre_reparse_budget():
    """46 Nawawi sittings + ~18 ʿAynī = ~64, against 63 before the reparse."""
    total = 0
    for name in ("nawawi", "cayni"):
        babs = load_babs(name)
        total += sittings(babs, REVIEW_EVERY[name])
    assert 60 <= total <= 70, f"{total} sittings - handoff.md §10.6 expects ~64"


def test_grouping_does_not_shrink_the_reread_budget():
    """The cap is summed per kitāb inside a group, so bounded agency is unchanged."""
    for name in ("nawawi", "cayni"):
        sizes = Counter(b["kitab"] for b in load_babs(name)).values()
        grouped = sum(reread_cap(n) for n in sizes)
        assert grouped == total_reread_budget(sizes)
