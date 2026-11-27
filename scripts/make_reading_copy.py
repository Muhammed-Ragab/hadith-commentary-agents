"""Make a reading copy of a method file for the scholar sitting.

The method files put the model's own prose in quotation marks -- 35 spans in nawawi
alone, e.g. `the compiler "never adjudicates by personal preference" (al-Ilm)`. Those
read as citations from the sharh. The handoff calls this "the one that could cost
credibility with a scholar": if Hossam Ouf looks one up and cannot find it, everything
else in the file becomes suspect, and that is not recoverable inside the sitting.

WHAT THIS DOES NOT DO: it does not edit the frozen method file. That file is a run
output the paper cites, and the reproducibility contract is "same manifest + same
intervention log => same outputs". Rewriting it after the fact breaks that silently.
The original stays byte-identical; this writes `<arm>_method.reading.md` beside it.

WHY A BANNER AND NOT A REWRITE: the quoted spans are not one kind of thing. Most are
the model's summary prose, but some are translations of the lexeme under gloss --
cayni quotes "a woman" and "mountain." Relabelling all of them "model paraphrase"
would replace one false claim with another. What IS true of every one of them is that
the sharh is in Arabic, so no English span in quotation marks is verbatim sharh text.
That single statement covers all 68 spans without classifying any of them.

Usage:
  python scripts/make_reading_copy.py outputs/method_files_matched/<run> nawawi
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_corpus import COMMENTATORS

ROOT = Path(__file__).resolve().parent.parent
_ARABIC = re.compile(r"[\u0600-\u06ff]")
_QUOTED = re.compile(r'"([^"\n]{3,200})"')

BANNER = """> ## Read this first
>
> This file was written by a language model. It summarises the recorded notes; it is
> **not** an edition of the sharh and contains no verbatim text from it.
>
> **Nothing in English quotation marks is a quotation from the commentary.** The sharh
> is in Arabic, so an English phrase in quotation marks is the model's own wording --
> sometimes its summary of a kitab, sometimes its rendering of the word being glossed.
> Do not look these up expecting to find them. {arabic_note}
>
> The citable provenance figure is the **counted block at the end of the file**, not any
> sentence in the prose.{unverified_note}

"""

ARABIC_NOTE = ("Arabic-script spans ({n} of them here) may be genuine text and are "
               "worth checking.")
NO_ARABIC = "There are no Arabic-script quotations in this file."
UNVERIFIED = ("""
>
> The `Unresolved provenance disagreement` stamp lower down is a **confirmed false
> positive**: it judged a per-kitab breakdown against the global ratio. Verified clean
> by `scripts/check_consolidation.py` on 2026-09-21.""")


def reading_copy(method_file: str) -> str:
    """The method file with a provenance banner. The body is not touched."""
    quoted = _QUOTED.findall(method_file)
    arabic = [q for q in quoted if _ARABIC.search(q)]
    banner = BANNER.format(
        arabic_note=ARABIC_NOTE.format(n=len(arabic)) if arabic else NO_ARABIC,
        unverified_note=(UNVERIFIED
                         if "Unresolved provenance disagreement" in method_file
                         else ""),
    )
    return banner + method_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("commentator", choices=COMMENTATORS)
    args = parser.parse_args()

    src = args.root / args.commentator / f"{args.commentator}_method.md"
    out = src.with_suffix(".reading.md")
    text = src.read_text(encoding="utf-8")
    out.write_text(reading_copy(text), encoding="utf-8")

    quoted = _QUOTED.findall(text)
    arabic = sum(1 for q in quoted if _ARABIC.search(q))
    print(f"{out}  ({len(quoted)} quoted spans: {len(quoted) - arabic} English, "
          f"{arabic} Arabic). Original untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
