import json

import pytest

from scripts.score_control import (
    CLAIM1_WITHDRAWAL_THRESHOLD,
    build_blind_sample,
    load_records,
    score,
)


def _rec(ref, evidence, control=False):
    return {"bab_ref": ref, "kitab": "K1", "gap": "g",
            "moves": [{"operation": "o", "arabic_evidence": evidence,
                       "closes_gap_by": "c",
                       "provenance": {"author": "compiler_own", "quoted_from": []},
                       "confidence": 0.8}],
            "revised_reading": "r", "hypothesis_hash": "h", "manifest_id": "m",
            "control": control}


def test_threshold_is_pre_registered_at_eighty_percent():
    assert CLAIM1_WITHDRAWAL_THRESHOLD == 0.80


def test_load_records_reads_jsonl(tmp_path):
    p = tmp_path / "notes.jsonl"
    p.write_text(json.dumps(_rec("nawawi/0", "ع")) + "\n", encoding="utf-8")
    assert len(load_records(p)) == 1


def test_score_reports_both_rates_and_the_ratio():
    real = [_rec("nawawi/0", "مخصوص")]
    control = [_rec("nawawi/1", "غير موجود", control=True)]
    sources = {"nawawi/0": "وهذا مخصوص بالمسافر", "nawawi/1": "نص اخر تماما"}

    out = score(real, control, sources)
    # Grounding is still reported -- it is the validation pass rate -- but the verdict
    # rides on yield now (see test_control_yield.py for why it had to).
    assert out["real_grounding"] == 1.0
    assert out["control_grounding"] == 0.0
    assert out["yield_ratio"] == 0.0


def test_claim1_survives_when_control_grounds_far_lower():
    real = [_rec("nawawi/0", "مخصوص")]
    control = [_rec("nawawi/1", "لاشيء", control=True)]
    sources = {"nawawi/0": "وهذا مخصوص", "nawawi/1": "نص مختلف"}
    assert score(real, control, sources)["claim1_withdrawn"] is False


def test_claim1_is_withdrawn_when_control_yields_as_well_as_real():
    """The falsification signal: the model closes gaps against arbitrary commentary."""
    real = [_rec("nawawi/0", "مخصوص")]
    control = [_rec("nawawi/1", "مخصوص", control=True)]
    sources = {"nawawi/0": "وهذا مخصوص", "nawawi/1": "وهذا مخصوص ايضا"}
    out = score(real, control, sources)
    assert out["yield_ratio"] == 1.0
    assert out["claim1_withdrawn"] is True


def test_zero_real_yield_does_not_divide_by_zero():
    real = [_rec("nawawi/0", "لاشيء")]
    control = [_rec("nawawi/1", "لاشيء", control=True)]
    out = score(real, control, {"nawawi/0": "x", "nawawi/1": "y"})
    assert out["yield_ratio"] == 0.0


def test_blind_sample_is_shuffled_unlabelled_and_sized():
    real = [_rec(f"nawawi/{i}", "ع") for i in range(40)]
    control = [_rec(f"nawawi/{i}", "ع", control=True) for i in range(40)]
    sample = build_blind_sample(real, control, n_per_arm=30, seed=1)

    assert len(sample) == 60
    assert all("control" not in item for item in sample)
    assert all("arm" not in item for item in sample)
    assert [s["id"] for s in sample] != sorted(s["id"] for s in sample)


def test_blind_sample_is_reproducible_for_a_seed():
    real = [_rec(f"nawawi/{i}", "ع") for i in range(40)]
    control = [_rec(f"nawawi/{i}", "ع", control=True) for i in range(40)]
    a = build_blind_sample(real, control, n_per_arm=10, seed=5)
    b = build_blind_sample(real, control, n_per_arm=10, seed=5)
    assert [x["id"] for x in a] == [x["id"] for x in b]


def test_commentary_sources_maps_control_units_to_paired_text():
    """pairing {0: 1} means bab 0 should be evaluated against bab 1's text."""
    babs = [
        {"kitab": "K1", "bab": "b0", "text": "text of bab 0"},
        {"kitab": "K1", "bab": "b1", "text": "text of bab 1"},
    ]
    # Patch load_babs to return our synthetic corpus
    import unittest.mock as mock
    from scripts.score_control import commentary_sources
    with mock.patch("scripts.score_control.load_babs", return_value=babs):
        real, control = commentary_sources("nawawi", {0: 1})
    assert control["nawawi/0"] == "text of bab 1"   # overridden by pairing {0: 1}
    assert control["nawawi/1"] == "text of bab 1"   # own text (not overridden)
    # The study arm read bāb 0's own text. Sharing one map is the 2026-09-20 defect:
    # it scored all three arms' real_grounding at exactly 0.0.
    assert real["nawawi/0"] == "text of bab 0"
    assert real["nawawi/1"] == "text of bab 1"


# ------------------------------- the pairing must come from the run (2026-09-19)

def test_pairing_is_read_from_the_control_manifest(tmp_path):
    from scripts.score_control import pairing_from_manifest
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"control_pairing": {"166": 171, "171": 166}}),
                 encoding="utf-8")
    assert pairing_from_manifest(m) == {166: 171, 171: 166}


def test_a_manifest_without_a_pairing_is_refused(tmp_path):
    """Scoring a control against a reconstructed pairing that disagrees with the run
    resolves every record to the wrong source text and decides a pre-registered rule
    on noise. Better to stop than to print a verdict."""
    from scripts.score_control import pairing_from_manifest
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"commentator": "nawawi"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="control_pairing"):
        pairing_from_manifest(m)
