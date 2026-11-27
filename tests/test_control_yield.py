"""The control's primary metric, after grounding rate was found unable to vary.

§4.1 registered evidence-grounding rate as primary: "a control unit grounds only if
the model quotes the reassigned sharh". That assumed the model wrote the evidence. It
never did -- `nodes.py` substituted `catalog[evidence_id]`, the whole cited span -- and
since B5 the quote is verified against that span at write time, so an unanchorable
quote cannot be written at all. Measured on run 15789363: 88/88, 76/76, 84/84. The
rate is 1.0 by construction in both regimes and cannot falsify anything.

The signal moved rather than vanished. A model projecting priors onto mismatched sharh
now fails to produce a verifiable quote, so it retries, then fails, or returns
`no_move` -- and `grounding_rate` counts only written MOVES, discarding exactly those
outcomes. The denominator is where the information was being lost.

Primary metric is therefore YIELD: grounded moves per REQUESTED bāb, where a requested
bāb is one that produced a record (moves or no_move) or a declared failure.
"""
from scripts.score_control import score, yield_rate


def _rec(ref, n_moves, evidence="قال النووي"):
    return {"bab_ref": ref, "status": "moves" if n_moves else "no_move",
            "moves": [{"arabic_evidence": evidence} for _ in range(n_moves)]}


SOURCES = {f"nawawi/{i}": "وقد قال النووي رحمه الله في شرحه" for i in range(10)}


def test_yield_counts_moves_over_requested_babs():
    recs = [_rec("nawawi/0", 3), _rec("nawawi/1", 1)]
    assert yield_rate(recs, SOURCES, failures=0) == 2.0


def test_a_no_move_bab_counts_in_the_denominator_not_out_of_it():
    """A bāb the model declined is evidence about the model, not a missing sample."""
    recs = [_rec("nawawi/0", 4), _rec("nawawi/1", 0)]
    assert yield_rate(recs, SOURCES, failures=0) == 2.0


def test_declared_failures_count_in_the_denominator():
    """The control's expected signal IS failure: a model that cannot ground a quote on
    mismatched sharh gives up. Dropping those would score the survivors at 1.0."""
    recs = [_rec("nawawi/0", 4)]
    assert yield_rate(recs, SOURCES, failures=3) == 1.0


def test_an_ungrounded_move_does_not_count(): 
    recs = [_rec("nawawi/0", 2, evidence="نص لا وجود له في الشرح")]
    assert yield_rate(recs, SOURCES, failures=0) == 0.0


def test_claim1_is_withdrawn_when_the_control_yields_as_much_as_the_real_arm():
    real = [_rec("nawawi/0", 4), _rec("nawawi/1", 4)]
    control = [_rec("nawawi/0", 4), _rec("nawawi/1", 4)]
    out = score(real, control, SOURCES)
    assert out["claim1_withdrawn"] is True
    assert out["yield_ratio"] == 1.0


def test_claim1_stands_when_the_control_yields_far_less():
    real = [_rec("nawawi/0", 4), _rec("nawawi/1", 4)]
    control = [_rec("nawawi/0", 1), _rec("nawawi/1", 0)]
    out = score(real, control, SOURCES)
    assert out["claim1_withdrawn"] is False
    assert out["yield_ratio"] < 0.80


def test_the_verdict_no_longer_rests_on_grounding_rate():
    """Both arms ground at 1.0 -- every move they DID write is anchored, by
    construction -- while their yields differ by 16x. Under the old rule this pair
    withdrew Claim 1; under the new one it does not."""
    real = [_rec("nawawi/0", 4)]
    control = [_rec("nawawi/0", 1), _rec("nawawi/1", 0),
               _rec("nawawi/2", 0), _rec("nawawi/3", 0)]
    out = score(real, control, SOURCES)
    assert out["real_grounding"] == out["control_grounding"] == 1.0
    assert out["real_yield"] == 4.0 and out["control_yield"] == 0.25
    assert out["claim1_withdrawn"] is False


def test_the_registered_threshold_is_unchanged_at_080():
    """The metric was substituted; the bar was not moved (2026-09-14 re-registration)."""
    from scripts.score_control import CLAIM1_WITHDRAWAL_THRESHOLD
    assert CLAIM1_WITHDRAWAL_THRESHOLD == 0.80


def test_the_arms_are_scored_against_their_own_source_maps():
    """2026-09-20: score() took ONE source map and used it for both arms, while
    commentary_sources() had already overwritten every paired bāb with the control's
    reassigned text. On a --sample run the pairing is a derangement over the study's
    own bābs, so every real move was checked against commentary the study never read:
    real_grounding came back 0.0 on nawawi, cayni and mazandarani alike, yield_ratio
    fell out of 0/x, and all three printed claim1_withdrawn: false on no evidence.
    """
    real_sources = {"nawawi/0": "وقد قال النووي رحمه الله في شرحه"}
    control_sources = {"nawawi/0": "نص آخر لا صلة له"}
    real = [_rec("nawawi/0", 4)]
    control = [_rec("nawawi/0", 4)]

    out = score(real, control, real_sources, control_sources=control_sources)

    assert out["real_grounding"] == 1.0, "the study arm must score against its own text"
    assert out["control_grounding"] == 0.0
    assert out["real_yield"] == 4.0
    assert out["claim1_withdrawn"] is False


def test_matn_only_arm_feeds_the_matn_so_the_catalog_is_never_empty():
    """2026-09-20, EXPLORATORY. The obvious no-sharh arm -- withhold the commentary
    outright -- cannot be run: build_evidence_catalog would emit no spans, no
    evidence_id would exist to cite, _ungrounded_quotes would reject every move, and
    the arm would score 0.0 by construction. That measures the write gate, not the
    model. Feeding the bāb's own matn keeps the citation contract intact, so the
    yield stays comparable with the study and control arms.
    """
    from hermeneutics.nodes import build_evidence_catalog

    catalog, batches = build_evidence_catalog("", "nawawi/0", len, 100)
    assert catalog == {} and batches == [], "no sharh means nothing is citable"

    catalog, batches = build_evidence_catalog("متن الحديث هنا", "nawawi/0", len, 100)
    assert catalog, "the matn must yield a citable span"


def test_requested_overrides_the_inferred_denominator():
    """A bāb dropped upstream produces neither a note nor a failure, so notes+failures
    silently undercounts. The matched study asked for 65 nawawi bābs and scored on 59:
    4 inherited segmentation failures and 2 declared failures went missing from the
    denominator, overstating the yield by ~10%.
    """
    from scripts.score_control import yield_rate

    recs = [{"bab_ref": "n/0", "moves": [{"arabic_evidence": "قال"}]}]
    sources = {"n/0": "قال النووي"}

    assert yield_rate(recs, sources, failures=0) == 1.0          # inferred: 1/1
    assert yield_rate(recs, sources, failures=0, requested=5) == 0.2   # pinned: 1/5


def test_score_reports_babs_that_were_never_reached():
    from scripts.score_control import score

    recs = [{"bab_ref": "n/0", "moves": [{"arabic_evidence": "قال"}]}]
    sources = {"n/0": "قال النووي"}
    result = score(recs, recs, sources, requested=5)

    assert result["n_real_requested"] == 5
    assert result["n_real_unaccounted"] == 4
    assert result["n_control_unaccounted"] == 4


def test_failures_file_is_named_off_the_notes_stem():
    """score_control hardcoded <commentator>_control_failures.jsonl, so an arm running
    with --control-notes <x>_matnonly_notes.jsonl looked for a file that does not
    exist, counted 0 failures, and shortened its own denominator.
    """
    from pathlib import Path

    from scripts.score_control import failures_for

    assert failures_for(Path("d/nawawi_matnonly_notes.jsonl")).name == (
        "nawawi_matnonly_failures.jsonl")
    assert failures_for(Path("d/nawawi_control_notes.jsonl")).name == (
        "nawawi_control_failures.jsonl")
    assert failures_for(Path("d/nawawi_notes.jsonl")).name == "nawawi_failures.jsonl"
