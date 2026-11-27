from hermeneutics.nodes import Deps, select_next_bab

BABS = [
    {"kitab": "K1", "bab": "b0", "text": "ا" * 500},
    {"kitab": "K1", "bab": "b1", "text": "short"},          # under the floor
    {"kitab": "K2", "bab": "b2", "text": "ب" * 500},
]


def _deps(**kw):
    base = dict(model=None, trace=None, manifest_id="m1", babs=BABS, commentator="nawawi",
                naive={}, seg={f"nawawi/{i}": 100 for i in range(50)},
                token_len=len, max_chunk_tokens=6000, pairing=None)
    base.update(kw)
    return Deps(**base)


def test_selects_the_bab_at_the_cursor():
    out = select_next_bab({"cursor": 0}, _deps())
    assert out["current_bab"]["bab"] == "b0"
    assert out["current_kitab"] == "K1"


def test_skips_babs_below_the_text_floor():
    """Empty/near-empty bābs are labelled, not silently dropped (defect #7)."""
    out = select_next_bab({"cursor": 1}, _deps())
    assert out["current_bab"]["bab"] == "b2"
    assert out["cursor"] == 3


def test_returns_none_past_the_end():
    assert select_next_bab({"cursor": 3}, _deps())["current_bab"] is None


def test_control_mode_pairs_commentary_from_another_bab():
    """Spec §4.1: shuffled control swaps the commentary, keeps the matn."""
    deps = _deps(pairing={0: 2})
    out = select_next_bab({"cursor": 0, "control_mode": True}, deps)
    assert out["current_bab"]["bab"] == "b0"           # matn is still bāb 0
    assert out["commentary_source_ref"] == "nawawi/2"  # commentary came from bāb 2


def test_non_control_mode_pairs_a_bab_with_its_own_commentary():
    out = select_next_bab({"cursor": 0, "control_mode": False}, _deps())
    assert out["commentary_source_ref"] == "nawawi/0"


def test_commentary_starts_at_the_segmented_boundary():
    out = select_next_bab({"cursor": 0}, _deps())
    assert out["commentary_text"] == BABS[0]["text"][100:]


def test_selection_fails_before_model_calls_when_segmentation_is_missing():
    import pytest
    with pytest.raises(ValueError, match="missing segmentation"):
        select_next_bab({"cursor": 0}, _deps(seg={}))


def test_bab_ref_uses_the_commentator_and_index():
    assert select_next_bab({"cursor": 2}, _deps())["current_bab_ref"] == "nawawi/2"


def test_global_index_map_keeps_refs_stable_for_a_subset_run():
    """The control arm runs over a ~10% subset. Its refs must still be the GLOBAL
    bāb ids, or score_control.py cannot look up the commentary sources."""
    deps = _deps(global_index=[7, 19, 44])
    assert select_next_bab({"cursor": 0}, deps)["current_bab_ref"] == "nawawi/7"
    assert select_next_bab({"cursor": 2}, deps)["current_bab_ref"] == "nawawi/44"


def test_control_source_ref_is_also_global():
    deps = _deps(global_index=[7, 19, 44], pairing={0: 2})
    out = select_next_bab({"cursor": 0, "control_mode": True}, deps)
    assert out["commentary_source_ref"] == "nawawi/44"


def test_control_mode_with_no_pairing_uses_own_commentary():
    """control_mode=True with pairing=None falls back to self-commentary."""
    out = select_next_bab({"cursor": 0, "control_mode": True}, _deps(pairing=None))
    assert out["commentary_source_ref"] == "nawawi/0"


def test_to_global_raises_on_out_of_bounds_pairing():
    """C1 regression: pairing that maps to an index beyond global_index must raise."""
    import pytest
    deps = _deps(global_index=[7, 19], pairing={0: 5})  # 5 >= len([7, 19])
    with pytest.raises(ValueError, match="out of range"):
        select_next_bab({"cursor": 0, "control_mode": True}, deps)


def test_skips_a_bab_with_a_declared_segmentation_failure():
    """B4: cayni/2232 and cayni/2872 carry an isnād the 20k window never reaches.
    segment_one refuses rather than fabricate a boundary and run_segmentation logs
    it instead of writing a seg record -- select_next_bab must skip it here too, or
    _segmentation_span raises at this same bāb instead (moving the crash, not
    removing it)."""
    deps = _deps(seg_failures=frozenset({"nawawi/1"}))
    out = select_next_bab({"cursor": 0}, deps)
    assert out["current_bab"]["bab"] == "b0"
    out = select_next_bab({"cursor": out["cursor"]}, deps)
    assert out["current_bab"]["bab"] == "b2"  # nawawi/1 skipped, not raised on


def test_a_declared_failure_does_not_relax_the_missing_segmentation_guard():
    """seg_failures names specific refs; it must not paper over deps.seg being
    empty for an unrelated reason (a real misconfiguration should still raise)."""
    import pytest
    deps = _deps(seg={}, seg_failures=frozenset({"nawawi/99"}))
    with pytest.raises(ValueError, match="missing segmentation"):
        select_next_bab({"cursor": 0}, deps)


def test_declared_failure_is_checked_against_the_control_source_ref():
    """Control mode swaps in commentary from a different bāb; the declared-failure
    check has to follow that swap, not the matn's own ref -- nawawi/0's own
    segmentation is fine, but its paired commentary source (bāb 2) failed."""
    babs = [
        {"kitab": "K", "bab": "b0", "text": "ا" * 500},
        {"kitab": "K", "bab": "b1", "text": "ب" * 500},
        {"kitab": "K", "bab": "b2", "text": "ج" * 500},
    ]
    deps = _deps(babs=babs, pairing={0: 2}, seg_failures=frozenset({"nawawi/2"}))
    out = select_next_bab({"cursor": 0, "control_mode": True}, deps)
    assert out["current_bab"]["bab"] == "b1"  # cursor 0 skipped: paired source bāb 2 failed


def test_matn_only_mode_feeds_the_matn_instead_of_the_sharh():
    """The no-sharh arm (2026-09-20, EXPLORATORY, not pre-registered).

    seg puts the matn at chars 0-100, so the sharh is everything after 100. Under
    matn_only_mode the arm must read the FIRST 100 chars and nothing else: that is
    what makes "no commentary" measurable at all, since withholding the text outright
    empties the evidence catalog and scores 0.0 by construction.
    """
    babs = [{"kitab": "K1", "bab": "b0", "text": "م" * 100 + "ش" * 400}]
    out = select_next_bab({"cursor": 0, "matn_only_mode": True}, _deps(babs=babs))
    assert out["commentary_text"] == "م" * 100
    assert "ش" not in out["commentary_text"], "the sharh must not leak into this arm"


def test_matn_only_mode_never_pairs_across_babs():
    """A shuffle here would confound "no sharh" with "wrong sharh", which is the
    control arm's job. The matn-only arm reads its OWN bāb."""
    babs = [{"kitab": "K1", "bab": "b0", "text": "م" * 100 + "ش" * 400},
            {"kitab": "K1", "bab": "b1", "text": "ن" * 100 + "خ" * 400}]
    out = select_next_bab({"cursor": 0, "matn_only_mode": True},
                          _deps(babs=babs, pairing={0: 1}))
    assert out["commentary_source_ref"] == "nawawi/0"
    assert out["commentary_text"] == "م" * 100
