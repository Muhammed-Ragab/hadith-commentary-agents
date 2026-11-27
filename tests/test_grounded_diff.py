import json

from hermeneutics.grounding import is_grounded
from hermeneutics.nodes import MAX_QUOTE_CHARS, Deps, horizon_diff

EVIDENCE_ID = "nawawi/0:c000:s000"
EVIDENCE = "وهذا مخصوص بالمسافر"
SHORT_QUOTE = "مخصوص بالمسافر"  # a genuine substring of EVIDENCE, not the whole span


def _draft(evidence_id=EVIDENCE_ID, arabic_evidence=SHORT_QUOTE):
    return json.dumps({
        "status": "moves",
        "gap": "I read it generally; he restricts it.",
        "moves": [{
            "operation": "scope_restriction",
            "evidence_id": evidence_id,
            "arabic_evidence": arabic_evidence,
            "closes_gap_by": "restricts to travellers",
            "provenance": {"author": "compiler_own", "quoted_from": []},
            "confidence": 0.8,
        }],
        "revised_reading": "A restricted obligation.",
    })


GOOD_DIFF = _draft()
NO_MOVE = json.dumps({
    "status": "no_move",
    "gap": "",
    "moves": [],
    "no_move_reason": "Nawawi restates the matn and adds no gloss, isnad note, or lexical remark.",
    "revised_reading": "The naive reading is unchanged.",
})

STATE = {
    "current_bab_ref": "nawawi/0",
    "current_kitab": "K1",
    "naive_reading": "my reading",
    "commentary_reading": {
        "status": "moves",
        "observations": [{
            "operation": "takhsis",
            "evidence_id": EVIDENCE_ID,
            "provenance": {"author": "compiler_own", "quoted_from": []},
            "confidence": 0.8,
        }],
    },
    "evidence_catalog": {EVIDENCE_ID: EVIDENCE},
    "commentary_text": f"قال النووي {EVIDENCE} رحمه الله",
    "method_hypothesis": "",
}


def _deps(fake_model, trace, notes_path=""):
    return Deps(model=fake_model, trace=trace, manifest_id="m1", babs=[],
                commentator="nawawi", naive={}, seg={}, token_len=len,
                notes_path=str(notes_path))


def test_valid_diff_materializes_a_verified_short_quote_without_self_check(fake_model, trace):
    out = horizon_diff(STATE, _deps(fake_model([GOOD_DIFF]), trace))
    rec = out["diff_record"]

    assert rec["bab_ref"] == "nawawi/0"
    assert rec["moves"][0]["arabic_evidence"] == SHORT_QUOTE
    # The full cited span stays available under its own key (e.g. for the
    # scholar-review sheet) even though arabic_evidence is now a short excerpt.
    assert rec["moves"][0]["evidence_chunk"] == EVIDENCE
    assert rec["moves"][0]["evidence_id"] == EVIDENCE_ID
    rows = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    assert [row["node"] for row in rows] == ["horizon_diff"]


def test_explicit_no_move_is_a_completed_record(fake_model, trace):
    state = {**STATE, "commentary_reading": {
        "status": "no_move",
        "observations": [],
        "no_move_reason": "Nawawi restates the matn and adds no gloss, isnad note, or lexical remark.",
    }}
    out = horizon_diff(state, _deps(fake_model([NO_MOVE]), trace))

    assert out["diff_record"]["status"] == "no_move"
    assert out["diff_record"]["moves"] == []


def test_unknown_evidence_id_is_repaired_by_the_retry(fake_model, trace):
    out = horizon_diff(STATE, _deps(fake_model([_draft("invented"), GOOD_DIFF]), trace))

    assert out["diff_record"]["moves"][0]["arabic_evidence"] == SHORT_QUOTE
    rows = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    prompts = [row["prompt"] for row in rows if row["node"] == "horizon_diff"]
    assert prompts[0] != prompts[1]
    assert "unknown evidence_id" in prompts[1].lower()


def test_fabricated_quote_not_in_the_cited_chunk_is_rejected_and_retried(fake_model, trace):
    """The load-bearing case: a span that does not occur in the chunk must be
    rejected and retried, not trusted and written through (the B5 defect)."""
    fabricated = _draft(arabic_evidence="لم يرد هذا في الشرح إطلاقا")
    out = horizon_diff(STATE, _deps(fake_model([fabricated, GOOD_DIFF]), trace))

    assert out["diff_record"]["moves"][0]["arabic_evidence"] == SHORT_QUOTE
    rows = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    prompts = [row["prompt"] for row in rows if row["node"] == "horizon_diff"]
    assert prompts[0] != prompts[1]
    assert "not occur verbatim" in prompts[1]


def test_persistently_fabricated_quote_returns_no_record(fake_model, trace):
    fabricated = _draft(arabic_evidence="لم يرد هذا في الشرح إطلاقا")
    out = horizon_diff(STATE, _deps(fake_model([fabricated, fabricated]), trace))

    assert out["diff_record"] is None


def test_overlong_quote_is_rejected_and_retried(fake_model, trace):
    """A quote as long as the whole chunk is the old defect wearing a new field
    name; MAX_QUOTE_CHARS must reject it even before checking occurrence."""
    whole_chunk_pretender = "غير موجود " * (MAX_QUOTE_CHARS // 10 + 2)
    assert len(whole_chunk_pretender) > MAX_QUOTE_CHARS
    overlong = _draft(arabic_evidence=whole_chunk_pretender)

    out = horizon_diff(STATE, _deps(fake_model([overlong, GOOD_DIFF]), trace))

    assert out["diff_record"]["moves"][0]["arabic_evidence"] == SHORT_QUOTE
    rows = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    prompts = [row["prompt"] for row in rows if row["node"] == "horizon_diff"]
    assert f"over the {MAX_QUOTE_CHARS}-char limit" in prompts[1]


def test_grounding_is_now_capable_of_failing_a_well_formed_record(fake_model, trace):
    """The whole point of B5: is_grounded must be able to return False for a
    genuinely schema-valid, code-verified record, once checked against a source
    the quote does not actually come from."""
    out = horizon_diff(STATE, _deps(fake_model([GOOD_DIFF]), trace))
    quote = out["diff_record"]["moves"][0]["arabic_evidence"]

    assert is_grounded(quote, STATE["commentary_text"]) is True
    assert is_grounded(quote, "نص شرح آخر تماما لا علاقة له بهذا الموضوع") is False


def test_horizon_prompt_includes_only_evidence_retained_by_the_reading(fake_model, trace):
    state = {
        **STATE,
        "evidence_catalog": {
            EVIDENCE_ID: EVIDENCE,
            "nawawi/0:c001:s000": "UNUSED SOURCE SPAN",
        },
    }

    horizon_diff(state, _deps(fake_model([GOOD_DIFF]), trace))

    row = json.loads(trace.path.read_text(encoding="utf-8").splitlines()[0])
    assert EVIDENCE in row["prompt"]
    assert "UNUSED SOURCE SPAN" not in row["prompt"]


def test_commentary_failure_is_logged_without_becoming_no_move(fake_model, trace, tmp_path):
    notes = tmp_path / "nawawi_notes.jsonl"
    state = {**STATE, "commentary_reading": None,
             "commentary_error": "unknown evidence_id: invented"}

    out = horizon_diff(state, _deps(fake_model([]), trace, notes))

    assert out["diff_record"] is None
    failure = json.loads((tmp_path / "nawawi_failures.jsonl").read_text(encoding="utf-8"))
    assert "unknown evidence_id" in failure["error"]


def test_a_written_record_always_grounds_against_its_own_source(fake_model, trace):
    """B5 moved the check from measurement time to write time -- it did not make the
    grounding RATE informative, and that distinction decides a pre-registered rule.

    `_ungrounded_quotes` enforces quote ⊂ catalog[evidence_id], and that chunk is a
    slice of `commentary_text`, which is exactly the source `grounding_rate` scores
    against. So every record that survives validation is grounded, and the rate is
    still 1.0 by construction -- what changed is that a fabricating model now fails
    to write a record at all, instead of writing one that scores as grounded.

    This matters for the shuffled control (handoff.md §10.3): a control record's
    evidence is verified against the REASSIGNED bāb's text, and score_control checks
    it against that same reassigned text, so control and real both score 1.0 and the
    pre-registered `ratio >= 0.80` rule would withdraw Claim 1 on an artifact. The
    control needs a metric that can separate; grounding rate cannot be it.
    """
    out = horizon_diff(STATE, _deps(fake_model([GOOD_DIFF]), trace))
    for move in out["diff_record"]["moves"]:
        assert is_grounded(move["arabic_evidence"], STATE["commentary_text"]) is True


def test_a_full_sentence_of_real_commentary_is_not_over_long():
    """The ceiling must clear the text the model is asked to quote from.

    Measured over the first 40 bābs of each arm, commentary sentences run to a median
    of 99 (ʿAynī) to 188 (Nawawī, Māzandarānī) characters, and only 32-58% fall under
    120. A ceiling below the median sentence rejects legitimate quotations, and under
    greedy decoding a rejected move is usually rejected twice (§8.10), so it becomes a
    declared failure -- which B4 caps at 10% of an arm. A mis-set constant could
    therefore fail a whole arm.
    """
    from hermeneutics.nodes import _ungrounded_quotes

    sentence = ("وقد ذهب جمهور العلماء إلى أن هذا الحديث محمول على من استحل ذلك، "
                "وهو قول الأكثرين من أهل العلم والفقه، وقد حكاه القاضي عياض عن "
                "جماعة من السلف، وردّ عليه الخطابي بأن الظاهر خلافه")
    chunk = "قال المصنف رحمه الله: " + sentence + " " + "ا" * 600

    class _M:
        evidence_id, arabic_evidence = "nawawi/0:c000:s001", sentence

    assert _ungrounded_quotes([_M()], {"nawawi/0:c000:s001": chunk}) is None


def test_a_quote_that_is_most_of_its_chunk_is_still_rejected():
    """The absolute ceiling alone would let a model copy most of a full-size span, which
    is the original defect at 40% scale. Spans run ~900 chars (256 tokens), so the share
    is bounded for anything above the absolute cap."""
    from hermeneutics.nodes import _ungrounded_quotes

    chunk = "ا" * 900

    class _M:
        evidence_id, arabic_evidence = "nawawi/0:c000:s001", "ا" * 390

    assert _ungrounded_quotes([_M()], {"nawawi/0:c000:s001": chunk}) is not None
