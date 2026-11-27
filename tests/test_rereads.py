from collections import Counter
from pathlib import Path

import pytest

from hermeneutics.rereads import reread_cap, total_reread_budget
from scripts.check_corpus import load_babs

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "babs"
pytestmark = pytest.mark.skipif(
    not (_DATA_DIR / "nawawi_babs.json").exists(),
    reason="corpus babs data not present — skip integration tests",
)


def test_small_kitab_still_gets_one_reread():
    """The bug: floor(0.10 * 7) == 0 disabled bounded agency entirely."""
    assert reread_cap(7) == 1
    assert reread_cap(1) == 1
    # n=5: round(0.10 * 5) uses banker's rounding → 0; only max(1, …) saves it
    assert reread_cap(5) == 1


def test_cap_is_ten_percent_in_the_middle_range():
    assert reread_cap(30) == 3
    assert reread_cap(40) == 4


def test_cap_never_exceeds_five():
    assert reread_cap(200) == 5
    assert reread_cap(4000) == 5


def test_zero_babs_gets_no_rereads():
    assert reread_cap(0) == 0


def test_total_reread_budget_sums_correctly():
    assert total_reread_budget([7, 30, 200]) == 1 + 3 + 5


def test_total_budget_is_about_363_across_the_corpus():
    """Spec §3.1: 363 over the 2026-08-23 reparse (133 kitābs), vs 168 over the old 63."""
    total = 0
    for name in ("nawawi", "cayni"):
        sizes = Counter(b["kitab"] for b in load_babs(name)).values()
        total += total_reread_budget(sizes)
    assert 350 <= total <= 380, f"budget {total} is outside the spec §3.1 estimate"


def test_no_kitab_in_the_real_corpus_is_left_at_zero():
    for name in ("nawawi", "cayni"):
        sizes = Counter(b["kitab"] for b in load_babs(name)).values()
        assert all(reread_cap(n) >= 1 for n in sizes), (
            f"{name} has a kitāb with no re-read opportunity - the old bug is back"
        )
