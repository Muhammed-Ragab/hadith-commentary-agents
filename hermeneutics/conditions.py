"""The experiment's conditions, and the metric that separates them.

The claim is not "the agent answers better" -- that is answered with "you gave one of
them the book". It is that the chatbot hands you a matn with no isnad: an answer whose
sources cannot be checked. So the conditions vary ONE thing, what the model may draw on,
and the metric counts what can be verified afterwards.

    no_sources    the same model, no tools, NO retrieval of any kind. The clean
                  control: the only difference from `agent` is access, not capability.
                  "mode" is explicit because supports_tools=False used to fall back to
                  retrieve-and-inject, which scored this cell 3/3 verified while it was
                  citing commentary files -- a RAG condition wearing the baseline's label.
    passages      no tools; commentary passages the researcher chose by hand. This is
                  the abstract's second setting, "researcher-selected commentary
                  passages directly, without prior study".
    agent         tools on: search, quote verification, lexicon, rijal.
    agent_method  tools on, plus the derived method file as instructions. The abstract's
                  first setting -- the knowledge base as "structured set of instructions
                  and evaluative criteria", not as a topic to be asked about.

A cell is a condition paired with a model. B1 is no_sources on the study model; B2 is
no_sources on a general chatbot model. The human-in-the-loop cells (A2, A4) are the same
conditions run through the review interface and are not automated here -- that is the
point of them.
"""
import json
import re
from pathlib import Path

BASE = ("You are a careful hadith research assistant. Answer in the language of the "
        "question.")

GROUNDED = (
    " Every quotation must come from the local sources and be checked with verify_quote "
    "before you present it. Cite each one by its reference. Use lookup_word for what a "
    "word means and lookup_narrator for who a narrator was. If the sources do not settle "
    "the question, say so plainly -- that is a complete answer, not a failure."
)

CONDITIONS = {
    "no_sources": {
        "tools": False,
        "mode": "none",
        "system": BASE,
        "needs": None,
        "label": "no sources (chatbot baseline)",
        # The clean control must stay a model with no pipeline help at all -- not even
        # the question rewritten into classical Arabic before it answers.
        "canonicalise": False,
    },
    "passages": {
        "tools": False,
        "mode": "none",
        "system": BASE + " Answer only from the passages given below. If they do not "
                         "settle the question, say so.",
        "needs": "passages",
        "label": "researcher-selected passages",
        "canonicalise": True,
    },
    "agent": {
        "tools": True,
        "mode": "tools",
        "system": BASE + GROUNDED,
        "needs": None,
        "label": "agent with tools",
        "canonicalise": True,
    },
    "agent_method": {
        "tools": True,
        "mode": "tools",
        "system": BASE + GROUNDED +
                  " Follow the commentator's method below: perform ITS operations, in ITS "
                  "order, and judge competing readings by ITS criteria. It is a procedure "
                  "to apply, not a text to describe.",
        "needs": "method",
        "label": "agent with tools + derived method",
        "canonicalise": True,
    },
}

# Quotations as they are actually written in Arabic answers. Long runs of Arabic without
# any quotation mark are NOT collected: a paraphrase is not a quotation, and counting one
# as an unverified quote would understate every condition equally but honestly measure
# nothing.
#
# The first version also took straight ' and " as delimiters, which in a markdown answer
# matched citation markers (<sup>lexica:lisan_entries.json</sup>), attributions
# (لابن حجر العسقلاني), bare punctuation and ordinary prose. Those counted as failed
# quotations and buried the real signal: the agent scored 1/7 on a question where six of
# the seven "quotations" were never quotations at all.
_QUOTED = re.compile(r"[«“]([^«»“”]{12,2000})[»”]|\"([^\"\n]{20,2000})\"")
_ARABIC_CHAR = re.compile(r"[\u0600-\u06ff]")
# <sup>commentary:cayni_babs.json</sup> is a citation marker, not a quotation. But **bold**
# INSIDE a quotation is just the model emphasising a word -- rejecting on it discarded
# genuine quotations like "**ظلمه وغشمه**", so markdown is stripped, not refused.
_NOT_A_QUOTE = re.compile(r"[<>|]|^\s*\(\d")
_MARKDOWN = re.compile(r"\*+|`+|_{2,}")
# Scholars elide. "كان رجال... فجاءت عشائرهم... فقالوا" is one quotation with two cuts,
# and substring matching fails on the whole while every piece of it is genuine.
_ELLIPSIS = re.compile(r"\.{2,}|…|\[\s*\.{2,}\s*\]")
MIN_FRAGMENT = 15


def _clean(text: str) -> str:
    return _MARKDOWN.sub("", text).strip()


def _is_quotation(text: str) -> bool:
    """Arabic enough, long enough, and not markdown furniture."""
    stripped = text.strip()
    if len(stripped) < 12 or _NOT_A_QUOTE.search(stripped):
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    return sum(bool(_ARABIC_CHAR.match(c)) for c in letters) / len(letters) >= 0.8


def load_passages(path: Path | None) -> dict:
    """question -> [passage, ...], chosen by the researcher. Absent means skip the cell."""
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_method(path: Path | None) -> str:
    """The derived method file, e.g. outputs/.../nawawi/nawawi_method.md."""
    return Path(path).read_text(encoding="utf-8").strip() if path else ""


def build_messages(condition: str, question: str, passages=None, method: str = "") -> list[dict]:
    """Messages for one cell, or raise if the condition's material is missing.

    Raising rather than degrading is deliberate: a `passages` run with no passages is
    just `no_sources` under a different name, and would quietly become a third copy of
    the baseline in the results table.
    """
    spec = CONDITIONS.get(condition)
    if spec is None:
        raise ValueError(f"Unknown condition: {condition}. Use one of {', '.join(CONDITIONS)}")
    if spec["needs"] == "passages" and not passages:
        raise ValueError(f"condition '{condition}' needs researcher-selected passages for "
                         f"this question; supply --passages")
    if spec["needs"] == "method" and not method:
        raise ValueError(f"condition '{condition}' needs a derived method file; supply --method")

    system = spec["system"]
    if spec["needs"] == "method":
        system += "\n\n--- COMMENTATOR'S METHOD ---\n" + method
    user = question
    if spec["needs"] == "passages":
        body = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        user = f"{question}\n\n--- PASSAGES ---\n{body}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def quotations(answer: str) -> list[str]:
    """The quotations an answer actually presents as quotations."""
    out = []
    for match in _QUOTED.finditer(answer or ""):
        text = _clean(match.group(1) or match.group(2) or "")
        if _is_quotation(text):
            out.append(text)
    return out


def verify_quotation(quote: str, corpus) -> dict:
    """Verify a quotation, honouring elisions.

    An elided quotation is verified when EVERY fragment either occurs in the sources or
    is too short to check. Requiring the whole string verbatim would fail honest
    quotations; accepting the longest fragment would pass invented ones welded to a real
    opening. Fragments under MIN_FRAGMENT characters are not evidence either way.

    Also returns "sources": the provenance `verify_quote` already computes and that the
    caller used to discard. It is the INTERSECTION of the source sets that carry every
    checkable fragment, not their union -- a quotation whose fragments each occur but in
    *different* works is not a quotation from any one work, and intersecting says so.
    This is additive: `verified`, `elided` and `checked` keep exactly their old meaning.
    """
    fragments = [f.strip() for f in _ELLIPSIS.split(quote) if f.strip()]
    checkable = [f for f in fragments if len(f) >= MIN_FRAGMENT]
    if not checkable:
        return {"verified": False, "elided": len(fragments) > 1, "checked": 0, "sources": []}
    results = [corpus.verify_quote(f) for f in checkable]
    ok = all(r.get("verified") for r in results)
    fragment_sources = [
        {o.get("source") for o in r.get("occurrences", []) if o.get("source")}
        for r in results
    ]
    sources = sorted(set.intersection(*fragment_sources))
    return {"verified": ok, "elided": len(fragments) > 1, "checked": len(checkable),
            "sources": sources}


def score_answer(answer: str, corpus) -> dict:
    """The paper's first number: how much of this answer can be checked?

    Reported as a fraction with the count of quotations found, so a zero can be read
    correctly -- no quotations at all is a different result from quotations that fail.
    """
    quotes = quotations(answer)
    results = [(q, verify_quotation(q, corpus)) for q in quotes]
    verified = [q for q, r in results if r["verified"]]
    return {
        "quotations": len(quotes),
        "verified": len(verified),
        "elided": sum(1 for _, r in results if r["elided"]),
        "unverified": [q[:120] for q, r in results if not r["verified"]][:5],
        "verifiable_rate": round(len(verified) / len(quotes), 3) if quotes else None,
    }
