import pytest

from hermeneutics.control import assert_isolated, control_indices, shuffle_within_kitab

BABS = (
    [{"kitab": "K1", "bab": f"b{i}", "text": "x"} for i in range(20)]
    + [{"kitab": "K2", "bab": f"c{i}", "text": "y"} for i in range(20)]
)


def test_pairing_covers_about_ten_percent():
    pairing = shuffle_within_kitab(BABS, seed=42, fraction=0.10)
    assert 3 <= len(pairing) <= 5


def test_pairing_is_deterministic_for_a_seed():
    assert shuffle_within_kitab(BABS, seed=42) == shuffle_within_kitab(BABS, seed=42)


def test_different_seeds_give_different_pairings():
    assert shuffle_within_kitab(BABS, seed=1) != shuffle_within_kitab(BABS, seed=2)


def test_commentary_is_reassigned_within_the_same_kitab():
    """Cross-kitāb shuffling would confound topic with pairing."""
    for target, source in shuffle_within_kitab(BABS, seed=7, fraction=0.5).items():
        assert BABS[target]["kitab"] == BABS[source]["kitab"]


def test_no_bab_is_paired_with_its_own_commentary():
    for target, source in shuffle_within_kitab(BABS, seed=7, fraction=0.5).items():
        assert target != source


def test_control_indices_are_the_pairing_keys():
    pairing = shuffle_within_kitab(BABS, seed=3, fraction=0.25)
    assert control_indices(pairing) == sorted(pairing)


def test_single_bab_kitab_cannot_be_shuffled():
    """Nothing to swap with - the unit is excluded rather than self-paired."""
    assert shuffle_within_kitab([{"kitab": "K", "bab": "b", "text": "x"}],
                                seed=1, fraction=1.0) == {}


def test_assert_isolated_passes_for_distinct_hypotheses():
    assert_isolated("real hypothesis", "control hypothesis")


def test_assert_isolated_raises_when_hypotheses_coincide():
    with pytest.raises(ValueError, match="contaminat"):
        assert_isolated("same text", "same text")


def test_assert_isolated_ignores_two_empty_hypotheses():
    """Both start empty; that is the initial condition, not contamination."""
    assert_isolated("", "")


def test_assert_isolated_ignores_one_sided_empty():
    assert_isolated("some hypothesis", "")
    assert_isolated("", "some hypothesis")
