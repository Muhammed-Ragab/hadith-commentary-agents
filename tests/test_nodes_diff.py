import json

from hermeneutics.nodes import Deps, horizon_diff

EVIDENCE_SPAN = "وهذا مخصوص بالمسافر"
SHORT_QUOTE = "مخصوص بالمسافر"  # a genuine substring of EVIDENCE_SPAN

GOOD_DIFF = json.dumps({
    "status": "moves",
    "gap": "I read it generally; he restricts it.",
    "moves": [{
        "operation": "scope_restriction",
        "evidence_id": "nawawi/0:c000:s000",
        "arabic_evidence": SHORT_QUOTE,
        "closes_gap_by": "restricts to travellers",
        "provenance": {"author": "compiler_own", "quoted_from": []},
        "confidence": 0.8,
    }],
    "revised_reading": "A restricted obligation.",
})

STATE = {
    "current_bab_ref": "nawawi/0",
    "current_kitab": "K1",
    "naive_reading": "my reading",
    "commentary_reading": {
        "status": "moves",
        "observations": [{
            "operation": "takhsis",
            "evidence_id": "nawawi/0:c000:s000",
            "provenance": {"author": "compiler_own", "quoted_from": []},
            "confidence": 0.8,
        }],
    },
    "evidence_catalog": {"nawawi/0:c000:s000": "وهذا مخصوص بالمسافر"},
    "commentary_text": "وهذا مخصوص بالمسافر",
    "method_hypothesis": "",
}


def _deps(fake_model, trace, notes_path=""):
    return Deps(model=fake_model, trace=trace, manifest_id="m1", babs=[],
                commentator="nawawi", naive={}, seg={}, token_len=len,
                notes_path=str(notes_path))


def test_valid_diff_passes_through(fake_model, trace):
    out = horizon_diff(STATE, _deps(fake_model([GOOD_DIFF]), trace))
    rec = out["diff_record"]
    assert rec["bab_ref"] == "nawawi/0"
    assert rec["moves"][0]["operation"] == "scope_restriction"


def test_diff_record_carries_the_hypothesis_hash(fake_model, trace):
    """Spec §3.4: replay needs the hypothesis in force at the call."""
    out = horizon_diff(STATE, _deps(fake_model([GOOD_DIFF]), trace))
    assert len(out["diff_record"]["hypothesis_hash"]) == 16


def test_control_flag_is_recorded_on_the_diff(fake_model, trace):
    state = {**STATE, "control_mode": True}
    out = horizon_diff(state, _deps(fake_model([GOOD_DIFF]), trace))
    assert out["diff_record"]["control"] is True


def test_valid_diff_does_not_call_a_same_model_self_check(fake_model, trace):
    out = horizon_diff(STATE, _deps(fake_model([GOOD_DIFF]), trace))
    assert out["diff_record"]["moves"][0]["operation"] == "scope_restriction"
    rows = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    assert [row["node"] for row in rows] == ["horizon_diff"]


def test_schema_violation_is_repaired_by_the_retry(fake_model, trace):
    """A 'transmitted' move with no source violates the V2 rule."""
    bad = json.dumps({
        "status": "moves",
        "gap": "g",
        "moves": [{
            "operation": "o", "evidence_id": "nawawi/0:c000:s000",
            "arabic_evidence": SHORT_QUOTE,
            "closes_gap_by": "c",
            "provenance": {"author": "transmitted", "quoted_from": []},
            "confidence": 0.5,
        }],
        "revised_reading": "r",
    })
    out = horizon_diff(STATE, _deps(fake_model([bad, GOOD_DIFF]), trace))
    assert out["diff_record"]["moves"][0]["provenance"]["author"] == "compiler_own"


def test_persistent_schema_violation_returns_no_record(fake_model, trace):
    bad = json.dumps({"status": "moves", "gap": "g", "moves": "not a list",
                      "revised_reading": "r"})
    out = horizon_diff(STATE, _deps(fake_model([bad, bad]), trace))
    assert out["diff_record"] is None


def test_both_attempts_with_unknown_evidence_return_no_record(fake_model, trace):
    bad = json.loads(GOOD_DIFF)
    bad["moves"][0]["evidence_id"] = "invented"
    out = horizon_diff(STATE, _deps(fake_model([json.dumps(bad), json.dumps(bad)]), trace))
    assert out["diff_record"] is None


def test_schema_retry_receives_validation_feedback(fake_model, trace):
    bad = json.dumps({
        "status": "moves",
        "gap": "g",
        "moves": [{"operation": "lexical analysis", "evidence_id": "invented"}],
        "revised_reading": "r",
    })

    horizon_diff(STATE, _deps(fake_model([bad, GOOD_DIFF]), trace))

    rows = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    prompts = [row["prompt"] for row in rows if row["node"] == "horizon_diff"]
    assert prompts[0] != prompts[1]
    assert "failed validation" in prompts[1]


def test_unknown_evidence_in_a_mixed_record_triggers_a_clean_retry(fake_model, trace):
    mixed = json.loads(GOOD_DIFF)
    mixed["moves"].append({
        "operation": "discussion of scholarly opinions",
        "evidence_id": "invented",
        "arabic_evidence": SHORT_QUOTE,
        "closes_gap_by": "presents competing views",
        "provenance": {"author": "compiler_own", "quoted_from": []},
        "confidence": 0.8,
    })

    out = horizon_diff(STATE, _deps(fake_model([json.dumps(mixed), GOOD_DIFF]), trace))

    assert [move["arabic_evidence"] for move in out["diff_record"]["moves"]] == [
        SHORT_QUOTE
    ]


def test_unknown_evidence_id_triggers_retry(fake_model, trace):
    bad = json.loads(GOOD_DIFF)
    bad["moves"][0]["evidence_id"] = "invented"

    out = horizon_diff(
        STATE,
        _deps(fake_model([json.dumps(bad), GOOD_DIFF]), trace),
    )

    assert out["diff_record"]["moves"][0]["arabic_evidence"] == SHORT_QUOTE
    assert out["diff_record"]["moves"][0]["evidence_chunk"] == EVIDENCE_SPAN
