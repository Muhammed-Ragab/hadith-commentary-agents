"""hermeneutics/answer_moves.py: unit splitting and the deterministic move rules."""
import json
from pathlib import Path

LEXICONS = json.loads(
    (Path(__file__).resolve().parent.parent / "scripts" / "compare_lexicons.json")
    .read_text(encoding="utf-8"))


def test_split_units_separates_paragraphs_on_blank_lines():
    from hermeneutics.answer_moves import split_units

    units = split_units("الفقرة الأولى هنا.\n\nالفقرة الثانية هنا.")

    assert len(units) == 2
    assert units[0].kind == "paragraph"
    assert units[0].text == "الفقرة الأولى هنا."
    assert units[1].text == "الفقرة الثانية هنا."


def test_split_units_treats_each_list_item_as_its_own_unit():
    from hermeneutics.answer_moves import split_units

    units = split_units("- البند الأول\n- البند الثاني\n- البند الثالث")

    assert len(units) == 3
    assert all(u.kind == "list_item" for u in units)
    assert units[1].text == "البند الثاني"


def test_split_units_merges_consecutive_blockquote_lines_into_one_unit():
    from hermeneutics.answer_moves import split_units

    units = split_units("> السطر الأول من الاقتباس\n> السطر الثاني من نفس الاقتباس")

    assert len(units) == 1
    assert units[0].kind == "blockquote"
    assert "السطر الأول" in units[0].text and "السطر الثاني" in units[0].text


def test_split_units_headings_are_not_units_but_carry_forward():
    from hermeneutics.answer_moves import split_units

    units = split_units("## مدخل الموضوع\nنص الفقرة هنا يتبع ما سبق مباشرة.")

    assert len(units) == 1
    assert units[0].heading == "مدخل الموضوع"
    assert "مدخل الموضوع" not in units[0].text


def test_tag_unit_blockquote_is_evidence_quote():
    from hermeneutics.answer_moves import Unit, tag_unit

    unit = Unit(index=0, text="قال رسول الله كذا وكذا", kind="blockquote")
    tagged = tag_unit(unit, LEXICONS)

    assert tagged.move == "evidence_quote"
    assert tagged.rule == "blockquote_or_quote_dominated"


def test_tag_unit_quote_dominated_paragraph_is_evidence_quote_even_with_grading_term():
    """Priority: a unit that is mostly a delimited quotation is evidence_quote even when
    it contains a grading-lexicon word like صحيح -- the quotation is the point."""
    from hermeneutics.answer_moves import Unit, tag_unit

    text = '«هذا الحديث صحيح الإسناد رواه البخاري عن أبي هريرة رضي الله عنه بلفظ طويل جدا هنا»'
    unit = Unit(index=0, text=text, kind="paragraph")
    tagged = tag_unit(unit, LEXICONS)

    assert tagged.move == "evidence_quote"


def test_tag_unit_stated_limit_lexicon():
    from hermeneutics.answer_moves import Unit, tag_unit

    unit = Unit(index=0, text="لم أجد هذا الحديث في المصادر المتوفرة لدي.", kind="paragraph")
    tagged = tag_unit(unit, LEXICONS)

    assert tagged.move == "limit"


def test_tag_unit_objection_then_reply_are_distinguished():
    from hermeneutics.answer_moves import Unit, tag_unit

    objection = tag_unit(Unit(index=0, text="فإن قيل إن هذا يتعارض مع كذا", kind="paragraph"),
                          LEXICONS)
    reply = tag_unit(Unit(index=1, text="قلنا: هذا الاعتراض مردود لأن كذا وكذا", kind="paragraph"),
                      LEXICONS)

    assert objection.move == "objection"
    assert reply.move == "reply"


def test_tag_unit_priority_reply_beats_isnad_grading_when_both_present():
    """A reply unit that also grades an isnad is still `reply`: it is answering an
    objection, and reply markers are checked before grading terms."""
    from hermeneutics.answer_moves import Unit, tag_unit

    text = "والجواب أن هذا الحديث صحيح الإسناد ولا يتعارض مع ما ذكر"
    tagged = tag_unit(Unit(index=0, text=text, kind="paragraph"), LEXICONS)

    assert tagged.move == "reply"


def test_tag_unit_isnad_grading_without_reply_markers():
    from hermeneutics.answer_moves import Unit, tag_unit

    tagged = tag_unit(Unit(index=0, text="هذا الحديث حسن الإسناد عند أهل الحديث", kind="paragraph"),
                       LEXICONS)

    assert tagged.move == "isnad_grading"


def test_tag_unit_proposal_markers():
    from hermeneutics.answer_moves import Unit, tag_unit

    tagged = tag_unit(Unit(index=0, text="وعليه، فالحاصل أن هذا هو الفهم الأقرب للسياق", kind="paragraph"),
                       LEXICONS)

    assert tagged.move == "proposal"


def test_tag_unit_default_is_other():
    from hermeneutics.answer_moves import Unit, tag_unit

    tagged = tag_unit(Unit(index=0, text="نص عادي بلا أي علامة خاصة به على الإطلاق هنا",
                            kind="paragraph"), LEXICONS)

    assert tagged.move == "other"


def test_tag_unit_list_item_with_no_marker_is_other_not_claim():
    """A list item is no longer tagged `claim` just for being a list item -- that
    conflated structure with rhetorical function. No marker fires -> other."""
    from hermeneutics.answer_moves import Unit, tag_unit

    tagged = tag_unit(Unit(index=0, text="نقطة عادية بلا أي علامة خاصة", kind="list_item"),
                       LEXICONS)

    assert tagged.move == "other"
    assert tagged.rule == "no_rule_fired"


def test_tag_unit_bare_attribution_verb_is_report_not_claim():
    """A unit that reports someone else's view (قال/روى/...) without being
    quote-dominated and without an explicit assertion marker is `report`, distinct
    from the answerer's own assertion (`claim`)."""
    from hermeneutics.answer_moves import Unit, tag_unit

    tagged = tag_unit(Unit(index=0, text="قال ابن حجر إن هذا الحديث مشهور بين أهل العلم",
                            kind="paragraph"), LEXICONS)

    assert tagged.move == "report"
    assert tagged.rule == "attribution_verb_or_report_marker"


def test_tag_unit_explicit_assertion_marker_is_claim():
    """claim now requires an explicit assertion marker (والصحيح أن، الراجح، we hold...),
    not just any attribution verb or list-item shape."""
    from hermeneutics.answer_moves import Unit, tag_unit

    tagged = tag_unit(Unit(index=0, text="والصحيح أن هذا الحديث يُحمل على هذا المعنى",
                            kind="paragraph"), LEXICONS)

    assert tagged.move == "claim"
    assert tagged.rule == "claim_markers"


def test_tag_unit_claim_marker_beats_report_marker_when_both_present():
    """Priority: claim_markers is checked before the attribution/report fallback, so a
    unit that both asserts (الراجح) and reports (قال) is tagged claim."""
    from hermeneutics.answer_moves import Unit, tag_unit

    text = "والراجح عندي، خلافاً لما قاله ابن حجر، أن هذا الحديث ضعيف"
    tagged = tag_unit(Unit(index=0, text=text, kind="paragraph"), LEXICONS)

    assert tagged.move == "claim"


def test_move_colors_and_letter_code_cover_report():
    """report needs its own colour (figures) and its own move_sequence letter, distinct
    from reply's "R"."""
    from hermeneutics.answer_moves import Unit, move_sequence

    report_unit = Unit(index=0, text="x", kind="paragraph", move="report", rule="x")
    reply_unit = Unit(index=1, text="x", kind="paragraph", move="reply", rule="x")
    seq = move_sequence([report_unit, reply_unit])

    assert seq[0] != seq[1]


def test_tag_answer_end_to_end_and_move_sequence():
    from hermeneutics.answer_moves import move_sequence, tag_answer

    answer = (
        "قال ابن حجر كذا وكذا.\n\n"
        "> نص الحديث المقتبس هنا بطوله الكافي للاعتبار كاقتباس\n\n"
        "لم أجد له أصلا آخر في المصادر المتوفرة."
    )
    units = tag_answer(answer, LEXICONS)
    seq = move_sequence(units)

    assert len(units) == 3
    assert units[1].move == "evidence_quote"
    assert units[2].move == "limit"
    assert len(seq) == 3


def test_edit_distance_and_normalized_edit_distance():
    from hermeneutics.answer_moves import edit_distance, normalized_edit_distance

    assert edit_distance("CQL", "CQL") == 0
    assert edit_distance("CQL", "CQ") == 1
    assert edit_distance("", "CQL") == 3
    assert normalized_edit_distance("CQL", "CQL") == 0.0
    assert normalized_edit_distance("CQL", "XXX") == 1.0
    assert normalized_edit_distance("", "") == 0.0


def test_move_shares_sums_to_one_and_covers_all_moves():
    from hermeneutics.answer_moves import MOVES, move_shares, tag_answer

    units = tag_answer("نص عادي هنا.\n\nلم أجد شيئا في المصادر المتوفرة.", LEXICONS)
    shares = move_shares(units)

    assert set(shares) == set(MOVES)
    assert abs(sum(shares.values()) - 1.0) < 1e-9


def test_llm_tag_prompt_contains_unit_text_and_all_labels():
    from hermeneutics.answer_moves import MOVES, Unit, llm_tag_prompt

    unit = Unit(index=0, text="نص تجريبي للاختبار", kind="paragraph")
    prompt = llm_tag_prompt(unit)

    assert unit.text in prompt
    assert all(m in prompt for m in MOVES)
