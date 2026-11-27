import pytest
from pydantic import ValidationError

import hermeneutics.schemas as schemas
from hermeneutics.schemas import (OPERATIONS, CommentaryObservation, CommentaryReading,
                                  DiffRecord,
                                  Intervention, Move, ReferencedMove, SegRecord)

VALID_MOVE = {
    "operation": "scope_restriction",
    "arabic_evidence": "وهذا مخصوص بالمسافر",
    "closes_gap_by": "restricts the general ruling to travellers",
    "provenance": {"author": "compiler_own", "quoted_from": []},
    "confidence": 0.8,
}


def test_move_parses():
    m = Move(**VALID_MOVE)
    assert m.provenance.author == "compiler_own"


def test_move_rejects_unknown_provenance_author():
    bad = {**VALID_MOVE, "provenance": {"author": "invented", "quoted_from": []}}
    with pytest.raises(ValidationError):
        Move(**bad)


def test_move_rejects_confidence_out_of_range():
    with pytest.raises(ValidationError):
        Move(**{**VALID_MOVE, "confidence": 1.5})


def test_transmitted_move_requires_a_named_source():
    """V2 attribution layering: 'transmitted' with no source names nobody."""
    with pytest.raises(ValidationError):
        Move(**{**VALID_MOVE, "provenance": {"author": "transmitted", "quoted_from": []}})


def test_diff_record_roundtrips_through_json():
    rec = DiffRecord(
        bab_ref="nawawi/12",
        kitab="كتاب الإيمان",
        gap="I read it as a general obligation; he restricts it.",
        moves=[Move(**VALID_MOVE)],
        revised_reading="A restricted obligation.",
        hypothesis_hash="abc123",
        manifest_id="m1",
        control=False,
    )
    restored = DiffRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec


def test_diff_record_accepts_explicit_no_move():
    rec = DiffRecord(
        bab_ref="nawawi/13",
        kitab="K1",
        status="no_move",
        gap="",
        moves=[],
        no_move_reason="Nawawi restates the matn and adds no gloss, isnad note, or lexical remark.",
        revised_reading="The naive reading is unchanged.",
        hypothesis_hash="abc123",
        manifest_id="m1",
    )
    assert rec.status == "no_move"
    assert rec.moves == []


def test_diff_record_rejects_moves_status_without_moves():
    with pytest.raises(ValidationError):
        DiffRecord(
            bab_ref="nawawi/13",
            kitab="K1",
            status="moves",
            gap="g",
            moves=[],
            revised_reading="r",
            hypothesis_hash="abc123",
            manifest_id="m1",
        )


def test_diff_record_rejects_no_move_with_a_move():
    with pytest.raises(ValidationError):
        DiffRecord(
            bab_ref="nawawi/13",
            kitab="K1",
            status="no_move",
            gap="",
            moves=[Move(**VALID_MOVE)],
            no_move_reason="No move.",
            revised_reading="r",
            hypothesis_hash="abc123",
            manifest_id="m1",
        )


def test_diff_record_rejects_no_move_without_a_reason():
    with pytest.raises(ValidationError):
        DiffRecord(
            bab_ref="nawawi/13",
            kitab="K1",
            status="no_move",
            gap="",
            moves=[],
            revised_reading="r",
            hypothesis_hash="abc123",
            manifest_id="m1",
        )


def test_commentary_reading_requires_observations_for_moves_status():
    with pytest.raises(ValidationError):
        schemas.CommentaryReading(status="moves", observations=[])


def test_commentary_reading_accepts_explicit_no_move():
    reading = schemas.CommentaryReading(
        status="no_move",
        observations=[],
        no_move_reason="The passage adds no identifiable operation.",
    )
    assert reading.status == "no_move"


def test_seg_record_and_intervention_parse():
    assert SegRecord(bab_ref="cayni/3", matn_end_char=812, confidence=0.9).matn_end_char == 812
    iv = Intervention(kind="flag_reread", kitab="كتاب الإيمان",
                      payload={"bab_refs": ["nawawi/1"]}, ts="2026-08-22T10:00:00")
    assert iv.kind == "flag_reread"


def test_intervention_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        Intervention(kind="delete_everything", kitab="k", payload={}, ts="2026-08-22T10:00:00")


def test_no_move_reason_copied_from_the_prompt_is_rejected():
    """8 of 9 no_move records in jobs 15701161-15781147 copied the prompt's example."""
    for copied in ("No identifiable interpretive operation.",
                   "no identifiable interpretive operation",
                   "<what you looked for and why it is absent>",
                   "   "):
        with pytest.raises(ValidationError):
            CommentaryReading(status="no_move", observations=[], no_move_reason=copied)


def test_no_move_reason_that_is_an_answer_is_accepted():
    reading = CommentaryReading(
        status="no_move", observations=[],
        no_move_reason="Nawawi restates the matn and adds no gloss or lexical remark.",
    )
    assert reading.status == "no_move"


def test_move_rejects_the_example_placeholder_as_an_operation():
    """Mechanism 1 (saturation_curve, run 15789363): the prompt's worked example now
    prints '<the interpretive operation, in your own words>' instead of a live,
    copyable value. Same defect class and same shape check as no_move_reason."""
    with pytest.raises(ValidationError):
        Move(**{**VALID_MOVE, "operation": "<the interpretive operation, in your own words>"})


def test_commentary_observation_rejects_the_example_placeholder_as_an_operation():
    with pytest.raises(ValidationError):
        CommentaryObservation(
            operation="<the interpretive operation, in your own words>",
            evidence_id="nawawi/0:c000:s000",
            provenance={"author": "compiler_own", "quoted_from": []},
            confidence=0.8,
        )


def test_bracketed_evidence_id_is_recovered():
    """cayni/10 in job 15789363 returned each id inside the brackets the prompt prints."""
    obs = CommentaryObservation.model_validate({
        "operation": "lexical analysis",
        "evidence_id": "[cayni/10:c000:s003]",
        "provenance": {"author": "compiler_own", "quoted_from": []},
        "confidence": 0.8,
    })
    assert obs.evidence_id == "cayni/10:c000:s003"


def test_collapsed_evidence_id_list_is_rejected():
    for bad in ("[a:c000:s001], [a:c000:s002]", ["a:c000:s001", "a:c000:s002"]):
        with pytest.raises(ValidationError):
            ReferencedMove.model_validate({
                "operation": "lexical analysis",
                "evidence_id": bad,
                "closes_gap_by": "x",
                "provenance": {"author": "compiler_own", "quoted_from": []},
                "confidence": 0.8,
            })


def test_plain_evidence_id_is_untouched():
    move = ReferencedMove.model_validate({
        "operation": "lexical analysis",
        "evidence_id": "nawawi/0:c000:s004",
        "arabic_evidence": "شاهد قصير",
        "closes_gap_by": "x",
        "provenance": {"author": "compiler_own", "quoted_from": []},
        "confidence": 0.8,
    })
    assert move.evidence_id == "nawawi/0:c000:s004"


def test_split_operation_separates_a_trailing_scope_note():
    from hermeneutics.schemas import split_operation
    assert split_operation("reconciliation of apparent contradiction (ẓāhir vs. ijmāʿ)") == (
        "reconciliation of apparent contradiction", "ẓāhir vs. ijmāʿ")


def test_split_operation_leaves_a_bare_label_alone():
    from hermeneutics.schemas import split_operation
    assert split_operation("grammatical analysis") == ("grammatical analysis", "")


def test_split_operation_keeps_a_label_that_is_only_a_parenthetical():
    from hermeneutics.schemas import split_operation
    assert split_operation("(lexical gloss)") == ("(lexical gloss)", "")


def _move(operation):
    return Move(**{**VALID_MOVE, "operation": operation})


def test_move_splits_the_qualifier_so_sprawl_counts_operations():
    """Job cayni2 scored six operations as twelve distinct, sprawl 1.00, because the
    second bab appended what each one applied to."""
    bare = _move("authority_citation")
    qualified = _move("authority_citation (Islam/Iman identity)")
    assert bare.operation == qualified.operation
    assert qualified.operation_qualifier == "Islam/Iman identity"
    assert bare.operation_qualifier == ""


def test_move_splitting_is_idempotent():
    once = _move("principle_extraction (exhaustive logical taxonomy)")
    twice = Move(**once.model_dump())
    assert (twice.operation, twice.operation_qualifier) == (
        once.operation, once.operation_qualifier)


def test_move_accepts_every_label_in_the_closed_set():
    """docs/2026-09-16-operation-vocabulary.md. If a label in OPERATIONS cannot be
    written into a Move, the prompt is teaching the model something the schema refuses."""
    for label in OPERATIONS:
        assert _move(label).operation == label


def test_move_rejects_a_free_text_operation():
    """The sprawl-1.00 shape: 135 sentence-length labels for 135 moves (§8.18). The
    rejection reaches the model as retry feedback, so it must name valid labels."""
    with pytest.raises(ValidationError) as excinfo:
        _move("draws a temporal-epistemic line: early deniers excused by proximity")
    assert "closed vocabulary" in str(excinfo.value)
    assert "operation_qualifier" in str(excinfo.value)


def test_move_normalizes_case_and_whitespace_on_a_label():
    assert _move("  Lexical_Gloss  ").operation == "lexical_gloss"


def test_move_keeps_the_qualifier_while_checking_the_closed_label():
    move = _move("tawil_imposition (the Khidr boy, counterfactual at bulugh)")
    assert move.operation == "tawil_imposition"
    assert move.operation_qualifier == "the Khidr boy, counterfactual at bulugh"


def test_other_is_accepted_so_a_missing_label_is_reported_not_forced():
    """The escape hatch is the measurement: a heavy `other` rate says the set is wrong.
    Forcing every move into the 17 would hide exactly that."""
    assert _move("other").operation == "other"
