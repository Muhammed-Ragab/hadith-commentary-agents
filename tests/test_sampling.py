"""S1 — stratified sampling, so a run is not "the first N bābs" of each work.

`--limit N` takes indices 0..N-1, which samples only the opening kitābs (īmān, ʿilm)
and invites the first question any hadith scholar asks: did you read anything but the
first chapter? §8.17 also showed the opening is *atypical* -- `cayni/0` is 57,117
characters of theological preamble with its first isnād at 34,815 -- so the pipeline
has been debugged against the least representative unit in the corpus.

The sample is consecutive WITHIN a kitāb (the kitāb is the scholar-review unit, and the
method hypothesis accumulates across a kitāb's bābs) and spread ACROSS kitābs. Three
phases -- segmentation, the naive pre-pass, and the study graph -- must select exactly
the same units or the arms silently disagree about which bāb `cayni/17` is.
"""
import pytest

from scripts.check_corpus import parse_sample, sample_indices

BABS = (
    [{"kitab": "K1", "bab": f"b{i}", "text": "x"} for i in range(10)]
    + [{"kitab": "K2", "bab": f"b{i}", "text": "x"} for i in range(10)]
    + [{"kitab": "K3", "bab": f"b{i}", "text": "x"} for i in range(3)]
)


def test_indices_are_global_positions_not_per_kitab_offsets():
    """`bab_ref` is `name/<position in load_babs>`; a per-kitāb offset would rename
    every unit and break the segmentation/naive/study join."""
    picked = sample_indices(BABS, n_kitabs=2, per_kitab=3, seed=1)
    assert all(0 <= i < len(BABS) for i in picked)
    assert picked == sorted(picked)


def test_the_same_seed_gives_the_same_sample():
    a = sample_indices(BABS, n_kitabs=2, per_kitab=3, seed=7)
    b = sample_indices(BABS, n_kitabs=2, per_kitab=3, seed=7)
    assert a == b


def test_a_different_seed_gives_a_different_sample():
    a = sample_indices(BABS, n_kitabs=2, per_kitab=3, seed=7)
    b = sample_indices(BABS, n_kitabs=2, per_kitab=3, seed=8)
    assert a != b


def test_bābs_are_consecutive_within_each_kitab():
    """The method hypothesis accumulates across a kitāb, and the review sheet is
    per kitāb. Scattering bābs inside one would break both."""
    picked = sample_indices(BABS, n_kitabs=2, per_kitab=3, seed=3)
    by_kitab = {}
    for i in picked:
        by_kitab.setdefault(BABS[i]["kitab"], []).append(i)
    for run in by_kitab.values():
        assert run == list(range(run[0], run[0] + len(run)))


def test_the_sample_spreads_across_kitabs():
    picked = sample_indices(BABS, n_kitabs=3, per_kitab=2, seed=3)
    assert len({BABS[i]["kitab"] for i in picked}) == 3


def test_the_window_does_not_always_start_at_the_top_of_a_kitab():
    """Always taking a kitāb's first bābs reproduces, one level down, exactly the
    opening-of-the-corpus bias this exists to remove."""
    starts = set()
    for seed in range(20):
        picked = sample_indices(BABS, n_kitabs=1, per_kitab=3, seed=seed)
        starts.add(BABS[picked[0]]["bab"])
    assert len(starts) > 1


def test_a_short_kitab_contributes_what_it_has():
    """K3 holds 3 bābs. Requiring per_kitab would silently restrict the sample to
    large kitābs, which is a bias, not a filter."""
    picked = sample_indices(BABS, n_kitabs=3, per_kitab=5, seed=2)
    k3 = [i for i in picked if BABS[i]["kitab"] == "K3"]
    assert len(k3) == 3


def test_asking_for_more_kitabs_than_exist_takes_them_all():
    picked = sample_indices(BABS, n_kitabs=99, per_kitab=2, seed=2)
    assert len({BABS[i]["kitab"] for i in picked}) == 3


# ------------------------------------------------------------------ the CLI spelling

def test_parse_sample_reads_kitabs_per_kitab_and_seed():
    assert parse_sample("12:5:20260914") == (12, 5, 20260914)


def test_parse_sample_rejects_a_malformed_spec():
    for bad in ("12:5", "12:5:x", "", "12:5:1:2", "0:5:1", "12:0:1"):
        with pytest.raises(ValueError):
            parse_sample(bad)


# ------------------------------------- the three phases must select the same units

def test_every_phase_selects_the_same_units_for_one_spec():
    """A disagreement here is silent and corrupting: segmentation writes `cayni/17`,
    the study graph asks for a different `cayni/17`, and the arms stop being matched.
    """
    from scripts.run_segmentation import select_units

    spec = "2:3:20260914"
    indexed = list(enumerate(BABS))
    seg = [i for i, _ in select_units(indexed, None, 0, 1, spec)]
    naive = [i for i, _ in select_units(indexed, None, 0, 1, spec)]
    study = sample_indices(BABS, *parse_sample(spec))
    assert seg == naive == study


def test_a_sample_ignores_limit_rather_than_truncating_the_strata():
    """Applying both would cut the later kitābs off the sample without saying so."""
    from scripts.run_segmentation import select_units

    picked = select_units(list(enumerate(BABS)), 2, 0, 1, "3:2:5")
    assert len(picked) == 6


def test_the_nameless_kitab_is_not_sampled():
    """Nawawī's units 0-4 are real bābs that precede the first kitāb header, so they
    group under an empty kitāb name. They make a poor stratum and an unnamed scholar
    review sheet, and §8.13 already fixed this class of thing on the Shia arm. Excluded
    from the sampling pool only -- filtering `load_babs` would renumber every bab_ref.
    """
    babs = [{"kitab": "", "bab": f"m{i}", "text": "x"} for i in range(5)] + BABS
    picked = sample_indices(babs, n_kitabs=99, per_kitab=2, seed=4)
    assert all(babs[i]["kitab"] for i in picked)


# ------------------------------------------------------- per-arm sample specs

def test_sample_specs_maps_a_bare_spec_to_every_arm():
    from scripts.validate_matched import sample_specs
    assert sample_specs(["12:5:1"])["cayni"] == "12:5:1"


def test_sample_specs_allows_a_per_arm_override():
    """The arms need different specs: Māzandarānī has 5 bāb-bearing kitābs against
    ʿAynī's 87, so one KITABS:PER_KITAB cannot give all three a comparable N."""
    from scripts.validate_matched import sample_specs
    specs = sample_specs(["12:5:1", "mazandarani=5:12:1", "nawawi=14:5:1"])
    assert specs["cayni"] == "12:5:1"
    assert specs["mazandarani"] == "5:12:1"
    assert specs["nawawi"] == "14:5:1"


def test_sample_specs_rejects_an_unknown_arm():
    from scripts.validate_matched import sample_specs
    with pytest.raises(ValueError, match="majlisi"):
        sample_specs(["majlisi=12:5:1"])


def test_sample_specs_rejects_an_arm_with_no_spec_at_all():
    """A silently unsampled arm would validate against the whole corpus."""
    from scripts.validate_matched import sample_specs
    with pytest.raises(ValueError, match="cayni"):
        sample_specs(["nawawi=14:5:1"])
