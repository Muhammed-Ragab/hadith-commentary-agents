"""OpenITI mARkdown parser for sharḥ works.

Structural markers look like ``# | 1 ( باب ... )`` (Nawawi, one pipe level) or
``### || 12 - (باب ... )`` (ʿAynī, kitāb at one pipe, bāb at two). Three things about the
real files that the first version of this parser got wrong, all measured against the raw
texts on 2026-08-23 — together they were defect #7 (handoff §10.1):

1. **Titles wrap.** 257 of Nawawi's 1041 bāb markers and 192 of ʿAynī's 4207 markers carry
   no closing paren on the marker line; the title continues on the next line. A regex
   anchored to a same-line ``)`` missed them and merged those units into the previous bāb.
   This also hid 65 of ʿAynī's 84 kitāb boundaries, which is why the corpus looked like it
   had 19 kitābs of 197 bābs each.
2. **The ordinal prefix.** ʿAynī writes ``### || 12 - (باب``; the ordinal has to be
   stripped before the title starts.
3. **فصل / فرع are sub-bāb, not bāb.** Their content belongs to the enclosing bāb, so the
   marker line is dropped and the body keeps accumulating. Deliberate policy: the research
   unit is "one hadith + its commentary", which is the bāb; sub-sections are part of that
   unit, not peers of it.

Level semantics. In a file that uses more than one pipe level, level 1 is the kitāb tier
and level 2+ is the bāb tier — a level-2 marker is a bāb whatever its wording, which is
what keeps ʿAynī's ``باب كتاب النبي ...`` from being read as a new kitāb. In a single-level
file the tier comes from the leading word instead.
"""
import re
from pathlib import Path

_MARKER = re.compile(r"^\s*#+\s*(\|+)\s*(.*)$")
_HEAD = re.compile(r"^\s*#+\s*\|")
# Most markers parenthesise the title: "( باب ... )". A few do not (17 of Nawawi's 1139,
# 126 of Ayni's 4207), and for those there is no closing paren to wrap toward -- scanning
# for one ate the commentary that follows. A bare title continues only on "~~" lines.
_OPENS = re.compile(r"^[\d\s\-]*\(")
_WRAP = re.compile(r"^\s*~~")
# Level digit ("1"), ordinal ("12 - ") and the opening paren, in any combination.
_PREFIX = re.compile(r"^[\d\s\-]*\(?\s*")

_PAGE = re.compile(r"PageV\d+P\d+")
_MS = re.compile(r"ms\d+")
_CONT = re.compile(r"~~")

# ponytail: 5 is well past the 2-line wrap the corpora actually use; it only exists so a
# malformed marker cannot swallow a whole kitāb.
_MAX_WRAP = 5

KITAB, BAB, SKIP = "kitab", "bab", "skip"
_SUB_WORDS = ("فصل", "فرع")


def clean_text(text: str) -> str:
    text = _PAGE.sub("", text)
    text = _MS.sub("", text)
    text = _CONT.sub("", text)
    return re.sub(r" +", " ", text).strip()


def _title_at(lines: list[str], i: int) -> tuple[int, str, int] | None:
    """Return (pipe_level, title, lines_consumed) for the marker starting at lines[i]."""
    m = _MARKER.match(lines[i])
    if not m:
        return None
    body = m.group(2).strip()
    payload = _PREFIX.sub("", body)
    consumed = 1
    if not _OPENS.match(body):
        while consumed < _MAX_WRAP and i + consumed < len(lines) and _WRAP.match(lines[i + consumed]):
            payload = f"{payload} {lines[i + consumed].strip()}"
            consumed += 1
        return len(m.group(1)), clean_text(payload), consumed
    while ")" not in payload and consumed < _MAX_WRAP and i + consumed < len(lines):
        nxt = lines[i + consumed]
        if _HEAD.match(nxt):  # next marker: this title simply has no closing paren
            break
        # A genuine wrap can cross a "#" paragraph line (ʿAynī 42); the marker is not
        # part of the title, so it is dropped rather than kept inside the label.
        payload = f"{payload} {nxt.strip().lstrip('#').strip()}"
        consumed += 1
    return len(m.group(1)), clean_text(payload.split(")")[0]), consumed


def _classify(level: int, title: str, multilevel: bool) -> str:
    first = title.split()[0] if title.split() else ""
    if not first or first.startswith("#") or first.startswith(_SUB_WORDS):
        return SKIP
    if multilevel:
        return BAB if level > 1 else (KITAB if first.startswith("كتاب") else SKIP)
    if first.startswith("كتاب"):
        return KITAB
    return BAB if first.startswith("باب") else SKIP


# --- plain-header format (Māzandarānī, Shia001174) -----------------------------
# This version carries no "#|" structural markers at all: every line is a bare "# "
# paragraph, and kitāb/bāb headers are simply lines that open with the word. Matn and
# commentary are separated inline by "* الأصل:" / "* الشرح:" — segmentation the other
# two corpora need a model pass to recover.
_PLAIN_KITAB = re.compile(r"^#\s*(كتاب\s+.*)$")
_PLAIN_BAB = re.compile(r"^#\s*(باب\s+.*)$")
_PLAIN_BODY = re.compile(r"\s\*\s")           # "* الأصل" / "* الشرح" begins the body
_PLAIN_STOP = ("باب", "بسم", "*")             # a header runs straight into its content
_PLAIN_TITLE_WORDS = 4                        # al-Kāfī book names are 1-3 words plus "كتاب"


def _plain_title(line: str) -> tuple[str, str]:
    """Split a header line into (title, remainder). Headers are not delimited here,
    so the title is cut at the first structural word or after a few words."""
    words = line.split()
    title = [words[0]]
    for word in words[1:_PLAIN_TITLE_WORDS]:
        if word.startswith(_PLAIN_STOP):
            break
        title.append(word)
    return " ".join(title), " ".join(words[len(title):]).strip()


def _plain_body(rest: str) -> str:
    """Drop the '* الأصل' marker that separates a header from its content."""
    parts = _PLAIN_BODY.split(rest, 1)
    return (parts[1] if len(parts) > 1 else parts[0]).strip()


def _parse_plain(lines: list[str]) -> list[dict]:
    babs: list[dict] = []
    current_kitab = current_bab = ""
    buf: list[str] = []

    def flush() -> None:
        """Emit the buffered unit. A kitāb with no bāb markers of its own — al-ʿAql
        wa-l-Jahl and al-Rawḍa are both undivided in al-Kāfī — becomes one unit rather
        than being dropped."""
        nonlocal buf
        text = clean_text(" ".join(buf))
        label = current_bab or current_kitab or "مقدمة"
        if text:
            babs.append({"kitab": current_kitab, "bab": label, "text": text})
        buf = []

    for line in lines:
        km = _PLAIN_KITAB.match(line)
        bm = _PLAIN_BAB.match(line)
        if km:
            flush()
            current_kitab, rest = _plain_title(km.group(1))
            current_bab = ""
            if rest.startswith("باب"):
                current_bab, rest = _plain_title(rest)
            if rest:
                buf.append(_plain_body(rest))
        elif bm:
            flush()
            current_bab, rest = _plain_title(bm.group(1))
            if rest:
                buf.append(_plain_body(rest))
        else:
            buf.append(line)

    flush()
    return babs


def parse_babs(raw: str) -> list[dict]:
    lines = raw.splitlines()
    if not any(_HEAD.match(ln) for ln in lines):
        return _parse_plain(lines)
    multilevel = any(len(m.group(1)) > 1 for m in filter(None, map(_MARKER.match, lines)))
    babs: list[dict] = []
    current_kitab = ""
    current_bab = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if current_bab and buf:
            text = clean_text(" ".join(buf))
            if text:
                babs.append({"kitab": current_kitab, "bab": current_bab, "text": text})
        buf = []

    i = 0
    while i < len(lines):
        marker = _title_at(lines, i) if _HEAD.match(lines[i]) else None
        if marker is None:
            buf.append(lines[i])
            i += 1
            continue
        level, title, consumed = marker
        kind = _classify(level, title, multilevel)
        if kind is not SKIP:
            flush()
            if kind is KITAB:
                current_kitab, current_bab = title, ""
            else:
                current_bab = title
        i += consumed

    flush()
    return babs


def parse_file(path: str | Path) -> list[dict]:
    return parse_babs(Path(path).read_text(encoding="utf-8"))


# The following parsers preserve the small, source-specific structures needed by the
# wave corpora. They deliberately do not pretend that OpenITI's editorial markers are
# a universal schema.
_DOLLAR = re.compile(r"^\s*#+\s*\$\s*(.*)$")
_NUMBER = re.compile(r"(?<!\d)(\d+)(?!\d)")
_QAL = "\u0642\u0627\u0644"


def _body_line(line: str) -> str:
    return clean_text(re.sub(r"^\s*(?:~~|#)\s*", "", line))


def _heading(line: str) -> tuple[int, str] | None:
    match = _MARKER.match(line)
    return (len(match.group(1)), clean_text(match.group(2))) if match else None


def _flush_text(parts: list[str]) -> str:
    return clean_text(" ".join(part for part in parts if part))


def _split_hadith(text: str) -> tuple[str, str]:
    """Split on the first explicit قال marker when one exists.

    ponytail: OpenITI Muslim has no reliable isnad/matn delimiter; this first-marker
    heuristic is the smallest useful fallback. Replace with editorial alignment if
    chain-level precision becomes a requirement.
    """
    text = clean_text(text)
    match = re.search(rf"\s{_QAL}(?:\u0627|\u0648\u0627)?\s*:", text)
    if not match:
        return "", text
    return text[: match.start()].strip(), text[match.start() :].strip()


def parse_hadith_file(path: str | Path, book_name: str = "") -> dict:
    """Parse an OpenITI hadith text into the Bukhari-compatible nested shape."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    chapters: list[dict] = []
    chapter: dict | None = None
    subchapter: dict | None = None
    current: dict | None = None
    next_number = 1

    def flush() -> None:
        nonlocal current
        if current is None or chapter is None or subchapter is None:
            current = None
            return
        text = _flush_text(current["parts"])
        if not text:
            current = None
            return
        isnad, matn = _split_hadith(text)
        subchapter["hadiths"].append(
            {"hadith_number": current["number"], "isnad": isnad, "matn": matn}
        )
        current = None

    for line in lines:
        dollar = _DOLLAR.match(line)
        marker = _heading(line)
        if dollar:
            payload = clean_text(dollar.group(1))
            number = _NUMBER.search(payload)
            # Muslim uses a numbered `$` line followed by a bare `$` line for
            # the actual hadith text. The bare marker is a continuation.
            if current is not None and number is None:
                current["parts"].append(payload)
                continue
            flush()
            if chapter is None:
                chapter = {"chapter_name": "", "sub_chapters": []}
                chapters.append(chapter)
            if subchapter is None:
                subchapter = {"sub_chapter_name": "", "hadiths": []}
                chapter["sub_chapters"].append(subchapter)
            hadith_number = number.group(1) if number else str(next_number)
            next_number += 1
            if number:
                payload = re.sub(r"^\s*\[?\d+\]?\s*[-.:]?\s*", "", payload)
            current = {"number": hadith_number, "parts": [payload]}
        elif marker:
            flush()
            level, payload = marker
            payload = re.sub(r"^\s*\[?\d+\]?\s*[-.:]?\s*", "", payload)
            if level == 1:
                chapter = {"chapter_name": payload, "sub_chapters": []}
                chapters.append(chapter)
                subchapter = None
            else:
                if chapter is None:
                    chapter = {"chapter_name": "", "sub_chapters": []}
                    chapters.append(chapter)
                subchapter = {"sub_chapter_name": payload, "hadiths": []}
                chapter["sub_chapters"].append(subchapter)
        elif current is not None:
            current["parts"].append(_body_line(line))

    flush()
    chapters = [c for c in chapters if any(s["hadiths"] for s in c["sub_chapters"])]
    return {"book_name": book_name, "chapters": chapters}


def _entry_header(line: str, marker: str) -> str | None:
    if not line.lstrip().startswith("###"):
        return None
    match = re.match(rf"^\s*#+\s*{re.escape(marker)}\s*(.*)$", line)
    return clean_text(match.group(1)) if match else None


def parse_lexicon_file(path: str | Path, style: str) -> list[dict]:
    """Segment Nihaya (pipe entries) or Lisan (dollar entries) into entries."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    entries: list[dict] = []
    section = ""
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            current["text"] = _flush_text(current.pop("parts"))
            if current["text"]:
                entries.append(current)
        current = None

    for number, line in enumerate(lines, 1):
        pipe = _heading(line)
        dollar = _entry_header(line, "$")
        if style == "nihaya" and pipe:
            flush()
            if pipe[1].startswith(("\u062d\u0631\u0641 ", "\u0628\u0627\u0628 ")):
                section = pipe[1]
            elif pipe[1] and not pipe[1].startswith("PARATEXT"):
                label = pipe[1].strip("() ")
                current = {"entry": label, "section": section, "line_start": number, "parts": []}
        elif style == "lisan" and dollar is not None:
            flush()
            label = dollar.split(":", 1)[0].strip()
            current = {"entry": label, "section": section, "line_start": number, "parts": [dollar]}
        elif pipe and style == "lisan":
            section = pipe[1]
        elif current is not None:
            current["parts"].append(_body_line(line))

    flush()
    return entries


def parse_rijal_file(path: str | Path, source_id: str) -> list[dict]:
    """Segment one rijal source without interpreting its grading vocabulary."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    entries: list[dict] = []
    section = ""
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            current["text"] = _flush_text(current.pop("parts"))
            if current["text"]:
                entries.append(current)
        current = None

    for number, line in enumerate(lines, 1):
        pipe = _heading(line)
        dollar = _entry_header(line, "$")
        if dollar is not None:
            flush()
            ordinal = _NUMBER.search(dollar)
            current = {
                "source": source_id,
                "entry_number": ordinal.group(1) if ordinal else "",
                "label": dollar,
                "section": section,
                "line_start": number,
                "parts": [],
            }
        elif pipe:
            section = pipe[1]
        elif current is not None:
            current["parts"].append(_body_line(line))

    flush()
    return entries
