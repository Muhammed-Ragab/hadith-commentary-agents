"""Evidence grounding. Spec §4.1 measure 1 - the primary control metric.

A move is grounded when its arabic_evidence appears verbatim in the commentary the
model actually read. Real pairings should ground high. A control unit grounds only
if the model quotes the REASSIGNED sharh; a move that cannot be anchored is a
fabricated or projected reading.

Costs zero model calls, so the control's verdict never depends on the automated
metric work deferred in spec §9.
"""
import re
import unicodedata

_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = re.compile(r"ـ")
_WHITESPACE = re.compile(r"\s+")
# A line-wrap artifact of the OpenITI source, and it lands MID-WORD: bukhari_raw.json
# carries "ما ~~نوى" 33,160 times, kafi_raw.json 68,067. Left in, a model quoting the
# hadith correctly FAILS verification while one reproducing the artifact passes -- the
# gate would censor honest quotations and reward corpus noise. It carries no meaning,
# so it folds here beside the diacritics and the tatweel.
# The *_babs.json commentary files contain none, so the frozen study numbers in
# RESULTS.md, computed over those, are untouched by this.
_LINE_WRAP = re.compile(r"~~")
# The OpenITI editions are largely UNPUNCTUATED; the model punctuates what it quotes.
# Folding neither way, a quotation identical to the commentary but for one "،" failed
# verification -- 7 of 26 strips in the six recorded Query-Agent sessions, including
# al-Nawawi's "لم يصح هذا الحديث عن النبي ... إلا من رواية عمر بن الخطاب" reproduced
# word for word. Every long quote traced broke at its first comma.
#
# Same defect and same direction as _LINE_WRAP above: the gate censored honest
# quotations while a model reproducing the edition's typography would have passed.
# Punctuation carries no semantic weight for "does this text occur here", so it folds
# here beside the diacritics, the tatweel and the line wrap.
#
# To a SPACE, not to nothing: the edition writes "القصد/الوجه" with no space around the
# slash, and gluing the words would invent a token that is in neither text.
_PUNCTUATION = re.compile(r"[،؛؟٪-٭۔"
                          r"!\"#%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~‐-‧"
                          r"‰-⁞]+")

_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي",
    "ؤ": "و",
    "ة": "ه",
})

MIN_EVIDENCE_CHARS = 3  # shorter than this matches by accident


def normalize_arabic(text: str) -> str:
    """Fold orthographic variation that carries no semantic weight here."""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _LINE_WRAP.sub("", text)
    text = _PUNCTUATION.sub(" ", text)
    text = text.translate(_FOLD)
    # Last, so that punctuation folded to a space collapses with the spaces beside it.
    return _WHITESPACE.sub(" ", text).strip()


def is_grounded(evidence: str, source: str) -> bool:
    """True when the quoted evidence really occurs in the source commentary."""
    ev = normalize_arabic(evidence)
    if len(ev) < MIN_EVIDENCE_CHARS:
        return False
    return ev in normalize_arabic(source)


def grounding_rate(records: list[dict], sources: dict[str, str]) -> float:
    """Fraction of moves across all records whose evidence is anchored.

    sources maps bab_ref -> the commentary text that was actually read. For control
    records that is the REASSIGNED bāb's text, not the unit's own.
    """
    total = grounded = 0
    for rec in records:
        source = sources.get(rec.get("bab_ref", ""), "")
        for move in rec.get("moves", []):
            total += 1
            if source and is_grounded(move.get("arabic_evidence", ""), source):
                grounded += 1
    return grounded / total if total else 0.0
