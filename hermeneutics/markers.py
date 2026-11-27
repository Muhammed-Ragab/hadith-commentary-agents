"""al-Kāfī's own matn/sharh markers. handoff.md §10.5, §8.15.

al-Māzandarānī's edition separates the two inline with `* الأصل:` (the text) and
`* الشرح:` (the commentary on it). §10.5 recorded this and left it unwired; §8.15 then
found a move whose `arabic_evidence` ended with an isnād fragment from a *later*
`الأصل:` block, because a single matn boundary leaves every subsequent matn sitting
inside what the agent reads as commentary.

Two functions, one marker pair:

* `matn_span` gives segmentation an exact boundary at no model call, on the 73% of the
  arm where the markers bound one in the right order.
* `sharh_only` removes the embedded matn blocks from the commentary the study graph
  reads, which is what actually closes the leak.

Both are no-ops on Nawawī and ʿAynī, which carry no such markers, so the pipeline keeps
one code path.
"""
import re

# The star is usual but not universal - `# الأصل:` occurs too, and requiring the star
# costs 25 of 350 real units. Anchored on a line or marker boundary so the bare words
# occurring in running prose are not mistaken for structure.
_ASL = re.compile(r"(?:^|[#*\n])\s*\*?\s*الأصل\s*:")
_SHARH = re.compile(r"(?:^|[#*\n])\s*\*?\s*الشرح\s*:")


def matn_span(text: str) -> tuple[int, int] | None:
    """(start, end) of the first matn, or None when the markers do not bound one.

    Returns None rather than guessing in the two cases the corpus actually presents:
    no markers at all (Nawawī, ʿAynī, and 116 Māzandarānī units), and a bāb that opens
    with a gloss on its own title so that `الشرح:` precedes any `الأصل:` (10 units).
    The second is ʿAynī's heading-gloss shape on the other arm, and §8.17 is the record
    of what guessing at it costs. Both fall through to the model.
    """
    asl = _ASL.search(text)
    if not asl:
        return None
    sharh = _SHARH.search(text, asl.end())
    if not sharh:
        return None
    if _SHARH.search(text, 0, asl.start()):
        return None
    return asl.end(), sharh.start()


def sharh_only(text: str) -> str:
    """The commentary with every embedded matn block removed.

    Text before the first marker is kept: `select_next_bab` slices from the end of the
    matn, so the commentary it passes here opens in the middle of a `الشرح:` block with
    no marker of its own.

    Segments are joined rather than spliced in place so each one stays a verbatim
    substring of the original bāb, which is what `validate_matched.py` grounds evidence
    against. A quotation spanning a join is two non-adjacent passages welded together,
    and failing its grounding check is the correct outcome.
    """
    marks = sorted(
        [(m.start(), m.end(), True) for m in _SHARH.finditer(text)]
        + [(m.start(), m.end(), False) for m in _ASL.finditer(text)]
    )
    if not marks:
        return text

    kept: list[str] = []
    pos, keeping = 0, True
    for start, end, is_sharh in marks:
        if keeping and start > pos:
            kept.append(text[pos:start])
        keeping, pos = is_sharh, end
    if keeping and pos < len(text):
        kept.append(text[pos:])

    return "\n".join(s.strip() for s in kept if s.strip())
