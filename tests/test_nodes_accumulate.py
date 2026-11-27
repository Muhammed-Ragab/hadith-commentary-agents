import json

from hermeneutics.nodes import Deps, accumulate

RECORD = {
    "bab_ref": "nawawi/0", "kitab": "K1", "gap": "g",
    "moves": [{"operation": "takhsis", "arabic_evidence": "ع", "closes_gap_by": "c",
               "provenance": {"author": "compiler_own", "quoted_from": []},
               "confidence": 0.8}],
    "revised_reading": "r", "hypothesis_hash": "h", "manifest_id": "m1", "control": False,
}


def _deps(tmp_path, trace):
    return Deps(model=None, trace=trace, manifest_id="m1", babs=[], commentator="nawawi",
                naive={}, seg={}, token_len=len,
                notes_path=str(tmp_path / "nawawi_notes.jsonl"))


def test_appends_the_record_to_state(tmp_path, trace):
    out = accumulate({"diff_record": RECORD, "method_hypothesis": ""}, _deps(tmp_path, trace))
    assert out["notes"] == [RECORD]


def test_writes_one_jsonl_line_immediately(tmp_path, trace):
    accumulate({"diff_record": RECORD, "method_hypothesis": ""}, _deps(tmp_path, trace))
    line = (tmp_path / "nawawi_notes.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["bab_ref"] == "nawawi/0"


def test_appends_rather_than_overwrites(tmp_path, trace):
    deps = _deps(tmp_path, trace)
    accumulate({"diff_record": RECORD, "method_hypothesis": ""}, deps)
    accumulate({"diff_record": {**RECORD, "bab_ref": "nawawi/1"},
                "method_hypothesis": ""}, deps)
    lines = (tmp_path / "nawawi_notes.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_hypothesis_accumulates_the_operation_names(tmp_path, trace):
    out = accumulate({"diff_record": RECORD, "method_hypothesis": ""}, _deps(tmp_path, trace))
    assert "takhsis" in out["method_hypothesis"]


def test_hypothesis_stays_bounded_in_size(tmp_path, trace):
    """It goes into every read_commentary prompt, so it cannot grow without limit."""
    deps = _deps(tmp_path, trace)
    hypothesis = ""
    for i in range(500):
        rec = {**RECORD, "moves": [{**RECORD["moves"][0], "operation": f"op{i}"}]}
        hypothesis = accumulate({"diff_record": rec, "method_hypothesis": hypothesis},
                                deps)["method_hypothesis"]
    assert len(hypothesis) < 4000


def test_no_record_is_a_no_op(tmp_path, trace):
    assert accumulate({"diff_record": None, "method_hypothesis": "h"},
                      _deps(tmp_path, trace)) == {}


def test_hypothesis_parses_and_increments_existing_counts_in_legacy_mode(tmp_path, trace):
    """Roundtrip: existing count is parsed and incremented. legacy_menu=True is the
    pre-fix, counted/ranked render -- ablation condition A/B only (see
    Deps.legacy_hypothesis_menu); a normal run never takes this branch."""
    from hermeneutics.nodes import _update_hypothesis
    prior = "Operations observed so far, most frequent first:\n- takhsis (2x)"
    result = _update_hypothesis(prior, {"moves": [{"operation": "takhsis"}]}, legacy_menu=True)
    assert "takhsis (3x)" in result


def test_empty_moves_returns_current_hypothesis_unchanged(tmp_path, trace):
    """No-op on first call with empty moves — no bare header emitted."""
    from hermeneutics.nodes import _update_hypothesis
    result = _update_hypothesis("", {"moves": []})
    assert result == ""


# ------------------------------------------------- mechanism 2 fix: no menu, no counts


def test_fixed_hypothesis_has_no_counts_or_frequency_ordering(tmp_path, trace):
    """Saturation-curve run 15789363: the counted, frequency-ranked menu ('most
    frequent first', '(12x)') is what collapsed mazandarani to 2 distinct
    operation-sets across 24 bābs. The default (legacy_menu=False) render must not
    reproduce either the counts or the ranking that made it a menu."""
    from hermeneutics.nodes import _update_hypothesis
    result = _update_hypothesis("", {"moves": [{"operation": "takhsis"},
                                                {"operation": "lexical analysis"}]})
    assert "most frequent first" not in result
    assert "x)" not in result  # no "(Nx)" count anywhere
    assert "- takhsis" in result
    assert "- lexical analysis" in result


def test_fixed_hypothesis_round_trips_the_accumulated_operations_over_several_babs(
        tmp_path, trace):
    """update_hypothesis parses its own previous render back (§ the task's own
    constraint) -- a format change must keep render and parse symmetric or the
    accumulation silently resets every bāb. Read back over 3+ bābs, seeing the same
    operation repeatedly and a new one late: every operation ever seen must still be
    present at the end (the hermeneutic-circle accumulation this fix must not lose),
    with no operation dropped just because it was not the most recent or most frequent.
    """
    from hermeneutics.nodes import _update_hypothesis
    hypothesis = ""
    for op in ("takhsis", "takhsis", "grammatical analysis"):
        hypothesis = _update_hypothesis(hypothesis, {"moves": [{"operation": op}]})
    assert "- takhsis" in hypothesis
    assert "- grammatical analysis" in hypothesis
    # round trip: parsing back what was just rendered and adding nothing new must be
    # a no-op, i.e. the set of names survives unchanged.
    assert _update_hypothesis(hypothesis, {"moves": []}) == hypothesis


def test_accumulate_honours_the_legacy_hypothesis_menu_toggle(tmp_path, trace):
    """Deps.legacy_hypothesis_menu must actually switch accumulate()'s render --
    otherwise the ablation's condition A/B toggle is a no-op."""
    legacy_deps = Deps(model=None, trace=trace, manifest_id="m1", babs=[],
                       commentator="nawawi", naive={}, seg={}, token_len=len,
                       notes_path=str(tmp_path / "nawawi_notes.jsonl"),
                       legacy_hypothesis_menu=True)
    out = accumulate({"diff_record": RECORD, "method_hypothesis": ""}, legacy_deps)
    assert "most frequent first" in out["method_hypothesis"]
    assert "(1x)" in out["method_hypothesis"]

    fixed_out = accumulate({"diff_record": RECORD, "method_hypothesis": ""},
                           _deps(tmp_path, trace))
    assert "most frequent first" not in fixed_out["method_hypothesis"]
    assert "(1x)" not in fixed_out["method_hypothesis"]


def test_accumulate_without_notes_path_still_updates_state(tmp_path, trace):
    """notes_path='' skips file I/O but still returns notes and hypothesis."""
    deps = Deps(model=None, trace=trace, manifest_id="m1", babs=[], commentator="nawawi",
                naive={}, seg={}, token_len=len, notes_path="")
    out = accumulate({"diff_record": RECORD, "method_hypothesis": ""}, deps)
    assert out["notes"] == [RECORD]
    assert "takhsis" in out["method_hypothesis"]
