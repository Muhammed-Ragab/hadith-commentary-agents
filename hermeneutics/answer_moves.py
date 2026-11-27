"""Rule-based rhetorical-move tagging for hadith-answer comparison. compare_systems.py.

Splits a markdown answer into units (paragraphs, list items, blockquotes -- headings are
not units themselves but their text is attached to the unit that follows them) and tags
each unit with one deterministic move: claim, evidence_quote, isnad_grading,
lexical_gloss, objection, concession, reply, limit, proposal, other.

Markers live in scripts/compare_lexicons.json so the user can extend them without
touching code. Priority order is fixed and documented on `tag_unit` -- a unit is tagged
by the FIRST rule that fires, not the "best" one, so the tagging stays a function of the
lexicon and never needs a judgment call.

`llm_tag_prompt` builds a prompt for an optional later LLM pass (see
scripts/tag_moves_llm.py, --dry-run only, no API calls from this module).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?(.*)$")
_LIST_ITEM = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+(.*)$")
_QUOTE_CHARS = re.compile(r"[«»“”\"]")

MOVES = ("claim", "report", "evidence_quote", "isnad_grading", "lexical_gloss", "objection",
          "concession", "reply", "limit", "proposal", "other")


@dataclass
class Unit:
    index: int
    text: str
    kind: str  # paragraph | list_item | blockquote
    heading: str = ""
    move: str = "other"
    rule: str = "default"


def split_units(answer: str) -> list[Unit]:
    """Markdown blocks: paragraphs, list items, blockquotes. Headings inform, not units.

    A blank line separates paragraphs. Consecutive list-item lines are separate units
    (one per bullet); consecutive blockquote lines merge into one unit (a multi-line
    quotation is one quotation). A heading's text is carried forward as `heading` on the
    next unit so lexical rules can see "under a heading called X" context if extended
    later; it does not itself become a unit.
    """
    lines = (answer or "").replace("\r\n", "\n").split("\n")
    units: list[Unit] = []
    heading = ""
    buf: list[str] = []
    buf_kind = "paragraph"

    def flush():
        nonlocal buf, buf_kind, heading
        text = "\n".join(buf).strip()
        if text:
            units.append(Unit(index=len(units), text=text, kind=buf_kind, heading=heading))
        buf = []

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        hm = _HEADING.match(line)
        if hm:
            flush()
            heading = hm.group(1).strip()
            continue
        bm = _BLOCKQUOTE.match(line)
        if bm:
            if buf and buf_kind != "blockquote":
                flush()
            buf_kind = "blockquote"
            buf.append(bm.group(1))
            continue
        lm = _LIST_ITEM.match(line)
        if lm:
            flush()  # each list item is its own unit
            buf_kind = "list_item"
            buf.append(lm.group(1))
            flush()
            continue
        if buf_kind != "paragraph":
            flush()
        buf_kind = "paragraph"
        buf.append(line)
    flush()
    return units


def _contains_any(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(t.lower() in low for t in terms)


def _is_quote_dominated(text: str) -> bool:
    """True when the unit is mostly a delimited quotation (« » " " or > already stripped)."""
    quoted_chars = 0
    for m in re.finditer(r"[«“\"]([^«»“”\"]{8,3000})[»”\"]", text):
        quoted_chars += len(m.group(1))
    if not text.strip():
        return False
    return quoted_chars / max(len(text), 1) >= 0.35


def tag_unit(unit: Unit, lexicons: dict) -> Unit:
    """Tag one unit. Priority order (first match wins), the same for every answer:

    1. blockquote kind, or a unit dominated by a delimited quotation -> evidence_quote
    2. stated_limits lexicon present -> limit
    3. objection markers -> objection
    4. concession markers -> concession
    5. reply markers -> reply
    6. claim_markers present (explicit assertion: "والصحيح أن", "الراجح", "we hold"...)
       -> claim -- checked before isnad_grading/lexical_gloss/proposal so the answerer's
       own explicit assertion is not pre-empted by incidentally containing a grading
       word ("والراجح ... أن هذا الحديث ضعيف" is a claim, not an isnad_grading unit).
    7. isnad_grading_terms present -> isnad_grading
    8. lexical_gloss_markers present -> lexical_gloss
    9. proposal markers -> proposal
    10. an attribution verb or report marker present, not already quote-dominated
        -> report (a reported view: qala fulan, "according to" -- NOT the answerer's
        own assertion)
    11. otherwise -> other

    Being a list item or containing a bare attribution verb is NO LONGER enough on its
    own to be tagged `claim` -- that conflated "the answer asserts X" with "the answer
    reports that someone else said X", which is exactly the distinction `claim` vs
    `report` exists to keep. A unit with no explicit assertion marker and no
    attribution verb stays `other`, rule `no_rule_fired`, even if it is a list item.

    The order still matters: a reply that also grades an isnad is tagged `reply` (it is
    answering an objection), and a quotation that happens to contain a grading term is
    still `evidence_quote` (the quote is the point of the unit, not its vocabulary).
    """
    markers = lexicons.get("move_markers", {})
    limits = lexicons.get("stated_limits", [])
    attribution = lexicons.get("attribution_verbs", [])
    text = unit.text

    if unit.kind == "blockquote" or _is_quote_dominated(text):
        unit.move, unit.rule = "evidence_quote", "blockquote_or_quote_dominated"
    elif _contains_any(text, limits):
        unit.move, unit.rule = "limit", "stated_limits_lexicon"
    elif _contains_any(text, markers.get("objection", [])):
        unit.move, unit.rule = "objection", "objection_markers"
    elif _contains_any(text, markers.get("concession", [])):
        unit.move, unit.rule = "concession", "concession_markers"
    elif _contains_any(text, markers.get("reply", [])):
        unit.move, unit.rule = "reply", "reply_markers"
    elif _contains_any(text, markers.get("claim_markers", [])):
        unit.move, unit.rule = "claim", "claim_markers"
    elif _contains_any(text, markers.get("isnad_grading_terms", [])):
        unit.move, unit.rule = "isnad_grading", "grading_terms"
    elif _contains_any(text, markers.get("lexical_gloss_markers", [])):
        unit.move, unit.rule = "lexical_gloss", "gloss_markers"
    elif _contains_any(text, markers.get("proposal", [])):
        unit.move, unit.rule = "proposal", "proposal_markers"
    elif _contains_any(text, attribution) or _contains_any(text, markers.get("report_markers", [])):
        unit.move, unit.rule = "report", "attribution_verb_or_report_marker"
    else:
        unit.move, unit.rule = "other", "no_rule_fired"
    return unit


def tag_answer(answer: str, lexicons: dict) -> list[Unit]:
    """Split and tag every unit of one answer."""
    return [tag_unit(u, lexicons) for u in split_units(answer)]


def move_sequence(units: list[Unit]) -> str:
    """A compact string, one letter per unit, for edit-distance comparison."""
    code = {"claim": "C", "report": "T", "evidence_quote": "Q", "isnad_grading": "G",
            "lexical_gloss": "L", "objection": "O", "concession": "N",
            "reply": "R", "limit": "M", "proposal": "P", "other": "X"}
    return "".join(code.get(u.move, "X") for u in units)


def move_shares(units: list[Unit]) -> dict:
    """Fraction of units per move type."""
    if not units:
        return {m: 0.0 for m in MOVES}
    n = len(units)
    counts = {m: 0 for m in MOVES}
    for u in units:
        counts[u.move] = counts.get(u.move, 0) + 1
    return {m: counts[m] / n for m in MOVES}


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two move-sequence strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def normalized_edit_distance(a: str, b: str) -> float:
    """Edit distance normalized to [0, 1] by the longer sequence's length."""
    m = max(len(a), len(b))
    return edit_distance(a, b) / m if m else 0.0


def llm_tag_prompt(unit: Unit) -> str:
    """A prompt for an optional later LLM pass over units the rules tagged `other`.

    Not called anywhere in this module -- no API calls happen here. Consumed only by
    scripts/tag_moves_llm.py, which is itself --dry-run only until a later change.
    """
    return (
        "Classify the rhetorical move of the following passage from a hadith-question "
        "answer. Reply with exactly one label from: "
        + ", ".join(MOVES) +
        ".\n\nPassage:\n" + unit.text
    )
