"""Build the scholar's review sheets from a FINISHED run's notes. No model calls.

The H1-H3 design assumed the scholar interrupts a live graph at each kitab boundary
(`kitab_review` in hermeneutics/study_graph.py). That is no longer how the sitting can
work: all four arms are finished, scored and committed, and re-running nawawi at
6.6 min/bab is ~7 hours against a 1-1.5 h sitting and a frozen dataset. So the one
Hossam Ouf sitting is a POST-HOC review of frozen output.

This renders exactly what the agent produced, per kitab, from the notes on disk --
the same sheet `kitab_review` would have shown, built from the same fields -- so the
review covers the agent's actual work rather than 600 KB of JSONL. It also carries the
two agenda items the handoff names for the sitting: normalising the cross-script
predecessor roster, and the advisory list of authorities the prose names that no note
records.

Because nothing is running, `intervention_log` cannot record the sitting. Each sheet
therefore ends with a decision block to fill in by hand; that log is now the only
qualitative evidence the abstract has, so completeness matters more than it did when
six sittings would have averaged out a thin one.

Usage:
  python scripts/build_review_sheets.py outputs/method_files_matched/<run> nawawi
  python scripts/build_review_sheets.py <run> nawawi -o outputs/reviews/sitting.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermeneutics.nodes import provenance_ratio, transmitted_sources
from hermeneutics.rereads import reread_cap
from hermeneutics.study_graph import LOW_CONFIDENCE, MAX_REVIEW_MOVES
from scripts.check_consolidation import unrecorded_names
from scripts.check_corpus import COMMENTATORS
from scripts.run_study_graph import render_review_sheet

ROOT = Path(__file__).resolve().parent.parent


def sheet_for(kitab: str, records: list[dict]) -> dict:
    """The same fields kitab_review passes to interrupt(), from frozen notes.

    One deliberate difference: moves are selected RELATIVELY, lowest confidence first,
    rather than by `LOW_CONFIDENCE`. On this run the model never reports below 0.70 --
    278 nawawi moves span 0.70 to 0.95 -- so the 0.5 threshold selects nothing and
    every sheet's review section comes out empty. A sheet with nothing on it wastes
    the one sitting there is. The absolute threshold is left untouched in
    `study_graph.kitab_review`, because changing it there would change the pipeline;
    the compression is a property of this model's self-reporting and is worth stating
    in the paper rather than tuning away.
    """
    moves = sorted(
        (
            {"bab_ref": r["bab_ref"], "operation": m.get("operation", ""),
             "confidence": m.get("confidence", 1.0),
             "evidence": m.get("arabic_evidence", "")}
            for r in records for m in r.get("moves", [])
        ),
        key=lambda m: m["confidence"],
    )
    low_confidence = moves[:MAX_REVIEW_MOVES]

    return {
        "kitab": kitab,
        "kitabs": [kitab],
        "n_records": len(records),
        "provenance": provenance_ratio(records),
        "low_confidence_moves": low_confidence,
        "reread_cap": reread_cap(len(records)),
        "n_moves": len(moves),
        "confidence_range": (
            (moves[0]["confidence"], moves[-1]["confidence"]) if moves else None),
        # The per-kitab mini-consolidation was a model call made during finalize and
        # was never persisted separately -- only the merged method file survives. Say
        # so rather than printing an empty heading the scholar will wonder about.
        "summary": "(not persisted by the run; review the method file separately)",
    }


DECISION_BLOCK = """
### Decision -- fill in during the sitting

- [ ] approve  [ ] correct  [ ] mark for re-read
- Verdict on this kitab's moves:
- What the agent got RIGHT:
- What it got WRONG:
- What it MISSED:

"""


def roster_section(records: list[dict], commentator: str, method_file: str) -> str:
    """Agenda item: normalise the roster, and look at the unrecorded names."""
    recorded = transmitted_sources(records)
    lines = [
        "## Predecessor roster -- normalise this",
        "",
        f"{len(recorded)} names, deduplicated by diacritics, case and punctuation "
        "only. Cross-script duplicates survive that (Malik / مالك, al-Khatibi / "
        "الخطابي / أبو سليمان الخطابي) and normalisation cannot reach them: the model "
        "emits names in two scripts. Tick the ones that are the SAME man.",
        "",
        "Second question, and it bears on the headline ratio: some entries are not "
        "individuals at all -- collectives (ahl al-lugha, al-ulama, al-huffaz) and "
        "book titles (al-Mashariq). Every move attributed to one of those counts "
        "toward `transmitted from NAMED predecessors`, which may overstate it. Mark "
        "any entry that is not a person.",
        "",
    ]
    lines += [f"- [ ] {name}" for name in recorded]

    stray = unrecorded_names(method_file, recorded, commentator) if method_file else []
    if stray:
        lines += [
            "",
            "## Named in the prose, recorded in no note -- plausible or invented?",
            "",
            "These appear in the method file but in no `quoted_from` of any note. "
            "Kitab titles and Arabic-script recorded names can appear here by "
            "accident, so the question for each is simply: would this commentator "
            "relay from this man, here?",
            "",
        ]
        lines += [f"- [ ] {name}" for name in stray]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="a run directory holding <arm>/")
    parser.add_argument("commentator", choices=COMMENTATORS)
    parser.add_argument("-o", "--out", type=Path, default=None)
    args = parser.parse_args()

    name = args.commentator
    arm = args.root / name
    records = [json.loads(line) for line
               in (arm / f"{name}_notes.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    method_path = arm / f"{name}_method.md"
    method_file = method_path.read_text(encoding="utf-8") if method_path.is_file() else ""

    by_kitab: dict[str, list[dict]] = {}
    for record in records:
        by_kitab.setdefault(record.get("kitab", ""), []).append(record)

    out = [
        f"# Scholar review -- {name}",
        "",
        f"Run: `{args.root}`   Kitabs: {len(by_kitab)}   Notes: {len(records)}",
        "",
        "Post-hoc review of a finished run. Nothing is running, so nothing here is "
        "fed back into the pipeline -- this is the qualitative record.",
        "",
        "> The method file's section 6 puts the MODEL'S OWN summary prose in quotation "
        "marks. Those are not the commentator's words and will not be found in the "
        "sharh. Read them as paraphrase.",
        "",
        "---",
        "",
    ]
    for kitab, group in by_kitab.items():
        sheet = sheet_for(kitab, group)
        note = ""
        if sheet["confidence_range"]:
            lo, hi = sheet["confidence_range"]
            note = (f"\n(The heading counts the {len(sheet['low_confidence_moves'])} "
                    f"LOWEST-confidence moves of {sheet['n_moves']} in this kitab, not "
                    f"moves under {LOW_CONFIDENCE}. This model never reports below "
                    f"{lo}; the range here is {lo}-{hi}.)\n")
        out += [render_review_sheet(sheet), note, DECISION_BLOCK, "---", ""]
    out.append(roster_section(records, name, method_file))

    text = "\n".join(out) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {len(by_kitab)} kitab sheets to {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
