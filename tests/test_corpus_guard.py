"""The whole compute and scholar-time budget in the spec derives from these numbers.
If a reparse changes them, every estimate in the design doc is stale.

Counts re-measured 2026-08-23 after the defect #7 parser fix (handoff.md §10.1) and
the addition of the Shia arm (§10.5).

Re-measured 2026-09-07: Māzandarānī 473→466 units, 9→5 kitābs, 9.91M→6.03M chars.
`load_babs` now returns bāb-level units only (see `check_corpus.is_bab`), dropping the
muqaddima and six whole-undivided-kitāb units. Every `--limit N` preflight had been
segmenting the muqaddima as the Shia arm's first "bāb".

The kitāb count falls by four because al-ʿAql wa-l-Jahl, al-Rawḍa and al-Kufr wa-l-Īmān
existed *only* as single undivided units and are now absent entirely, and the muqaddima's
empty kitāb string was counted as a ninth. That is a real content loss, not just a
re-index: al-Rawḍa alone was 1,692,380 characters. Splitting those kitābs into sub-units
in parse_openiti.py is the alternative, and would move these numbers again.

Re-measured 2026-09-17 after the title-wrap fix (`a21a41a`): ʿAynī 3959→3960 units. A
marker whose title was swallowed by its neighbour is now its own bāb, so `cayni/<n>` for
n >= 971 shifts by +1. Nawawī and Māzandarānī are unchanged.
"""
from scripts.check_corpus import BABS_FILES, corpus_stats

EXPECTED = {
    "nawawi": {"units": 1038, "kitabs": 46},
    "cayni": {"units": 3960, "kitabs": 87},
    "mazandarani": {"units": 466, "kitabs": 5},
}

TOTAL_UNITS = 5464
TOTAL_KITABS = 138
# Sunni arm only - the figure every pre-Māzandarānī estimate in the spec is built on.
SUNNI_UNITS = 4998
SUNNI_KITABS = 133


def test_every_configured_commentator_is_guarded():
    """A new commentator must land in EXPECTED, not slip past the guard unmeasured."""
    assert set(EXPECTED) == set(BABS_FILES)


def test_corpus_counts_match_spec():
    for name, expected in EXPECTED.items():
        got = corpus_stats(name)
        assert got["units"] == expected["units"], (
            f"{name}: {got['units']} units, spec §3.0 says {expected['units']}. "
            "Reparse changed the corpus - update spec §3.0, §5 and §6.1 before continuing."
        )
        assert got["kitabs"] == expected["kitabs"], (
            f"{name}: {got['kitabs']} kitabs, spec §3.0 says {expected['kitabs']}. "
            "Reparse changed the corpus - update spec accordingly."
        )


def test_totals():
    assert sum(corpus_stats(n)["units"] for n in EXPECTED) == TOTAL_UNITS
    assert sum(corpus_stats(n)["kitabs"] for n in EXPECTED) == TOTAL_KITABS


def test_sunni_arm_totals_still_hold():
    """Nawawi + ʿAynī alone, so the spec's pre-Māzandarānī figures stay checkable."""
    sunni = ("nawawi", "cayni")
    assert sum(corpus_stats(n)["units"] for n in sunni) == SUNNI_UNITS
    assert sum(corpus_stats(n)["kitabs"] for n in sunni) == SUNNI_KITABS


def test_no_arm_leaks_front_matter_or_a_whole_undivided_kitab():
    """The muqaddima was mazandarani/0 in every preflight through job 15775711."""
    from scripts.check_corpus import COMMENTATORS, MUQADDIMA, load_babs

    for name in COMMENTATORS:
        for unit in load_babs(name):
            assert unit["bab"] != MUQADDIMA, f"{name}: front matter is still a unit"
            assert not (unit["kitab"] and unit["bab"] == unit["kitab"]), (
                f"{name}: {unit['kitab']!r} is a whole undivided kitab, not a bab"
            )


def test_the_shia_arm_starts_on_a_real_bab():
    from scripts.check_corpus import load_babs

    first = load_babs("mazandarani")[0]
    assert first["bab"].startswith("باب")
    assert len(first["text"]) < 100_000, "mazandarani/0 must be bab-sized, not a whole kitab"
