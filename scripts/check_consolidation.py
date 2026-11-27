"""Check a finished method file against the notes it summarises.

The handoff's highest-risk open item. `consolidate_global` guards its own output as it
writes it, but nothing has ever re-checked a method file that is already on disk -- and
that file is the layer whose text reaches the paper. Until this existed the method files
were asserted, not checked.

Two things are recomputed from the notes and compared with what the file says:

  provenance  every figure in a provenance sentence must match a ratio the notes
              actually counted -- the global one, or one kitab's. A figure matching
              nothing is a fabrication and fails the run.
  roster      every predecessor the notes record, and whether the file's prose names
              anyone outside that set. Advisory: 8.15 caught Ayni naming al-Karmani,
              al-Jawhari and al-Nawawi with `quoted_from` empty on all 76 moves, so a
              name the notes never recorded is worth a human look even when the
              ratio is clean.

Usage:
  python scripts/check_consolidation.py outputs/method_files_matched/<run>
  python scripts/check_consolidation.py <run> --arm nawawi
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermeneutics.nodes import (_source_key, check_provenance_claim, kitab_provenance,
                                provenance_line, provenance_ratio, transmitted_sources)
from scripts.check_corpus import COMMENTATORS

ROOT = Path(__file__).resolve().parent.parent

# A transliterated authority as these files write them: al-Nawawi, Ibn Battal,
# Qadi Iyad, Abu Hanifa. Deliberately narrow -- this feeds an ADVISORY list, and a
# pattern loose enough to catch every phrasing would report half the prose.
_NAME = re.compile(
    r"\b(?:al-[A-Z\u00c0-\u024f][\w\u00c0-\u024f\u02bf\u02be]+"
    r"|(?:Ibn|Abu|Ab\u016b|Qadi|Q\u0101d\u012b|Imam|Im\u0101m)"
    r"\s+[A-Z\u00c0-\u024f][\w\u00c0-\u024f\u02bf\u02be]*)"
)

# "al-Itq: 100% own" is a kitab label in a breakdown, not an authority. The model
# writes kitab names transliterated while the notes carry them in Arabic, so they
# cannot be matched against the notes directly -- but the colon-then-figure context
# identifies them exactly, and without this filter every kitab in the breakdown was
# reported as an unrecorded scholar.
_BREAKDOWN_LABEL = re.compile(r"([^,;(]+?)\s*:\s*\d")


def load_notes(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def unrecorded_names(method_file: str, recorded: list[str],
                     commentator: str) -> list[str]:
    """Authorities the prose names that no note records in a `quoted_from`.

    Advisory only, and deliberately left imprecise in one direction. The commentator's
    own name is excluded -- a method file describing al-Nawawi's practice names
    al-Nawawi constantly, and that is not a relayed source.

    ponytail: two classes of false positive are ACCEPTED rather than engineered away.
    (1) A kitab title the prose mentions outside a "label: figure" breakdown reads as a
    name -- al-Jihad, al-Qadar, al-Adab. (2) The notes may record an authority in
    Arabic script while the prose transliterates it, so nawawi's recorded `بن بطال`
    does not match the prose's `Ibn Battal`. Both need a transliteration matcher to
    fix, which is a bigger and more failure-prone thing than the check it serves. The
    list is short enough to scan by eye, and the caller is told what may be in it.
    Upgrade path if it ever matters: fold Arabic and Latin to a shared consonant
    skeleton in `_source_key`.
    """
    known = {_source_key(name) for name in recorded}
    known.add(_source_key(commentator))
    known.add(_source_key("al-" + commentator))
    known.discard("")

    # Containment, not equality. The notes record a full name ("Abu Nuʿaym
    # al-Isbahani", "al-Shahid al-Awwal") while the prose uses a short form, and
    # equality reported both halves of the same man as unrecorded. Matching either
    # way round cut the advisory list from 45-57 names to something a human will
    # actually read -- an advisory nobody reads is worse than no advisory.
    labels = {_source_key(m) for m in _BREAKDOWN_LABEL.findall(method_file)}
    labels.discard("")

    found: dict[str, str] = {}
    for match in _NAME.findall(method_file):
        key = _source_key(match)
        if not key or key in known:
            continue
        if any(key in k or k in key for k in known):
            continue
        if any(key in label for label in labels):
            continue
        found.setdefault(key, match)
    return sorted(found.values())


def check_arm(arm_dir: Path, name: str) -> tuple[list[str], list[str]]:
    """(errors, notes_for_the_reader) for one arm."""
    method_path = arm_dir / f"{name}_method.md"
    notes_path = arm_dir / f"{name}_notes.jsonl"
    if not method_path.is_file() or not notes_path.is_file():
        return ([f"{name}: missing {method_path.name} or {notes_path.name}"], [])

    records = load_notes(notes_path)
    method_file = method_path.read_text(encoding="utf-8")
    global_prov = provenance_ratio(records)
    per_kitab = kitab_provenance(records)
    recorded = transmitted_sources(records)

    errors, remarks = [], []
    remarks.append(f"counted: {provenance_line(global_prov)}")
    remarks.append(f"kitabs counted: {len(per_kitab)}; "
                   f"predecessors recorded: {len(recorded)}")

    disagreement = check_provenance_claim(method_file, global_prov, per_kitab)
    if disagreement:
        errors.append(f"{name}: {disagreement}")

    # The counted block is what the paper cites, so its absence is a real defect even
    # when every stated figure happens to be right.
    if "## Provenance (counted" not in method_file:
        errors.append(f"{name}: method file carries no counted provenance block")

    if "Unresolved provenance disagreement" in method_file:
        remarks.append("carries an UNVERIFIED stamp from the inline check at write "
                       "time; re-judged above against per-kitab counts")

    if not recorded:
        remarks.append("NO predecessor recorded in any note -- every name in the "
                       "prose is unsupported")
    stray = unrecorded_names(method_file, recorded, name)
    if stray:
        remarks.append(
            f"{len(stray)} name(s) in the prose that no note records -- ADVISORY, "
            f"check by hand; kitab titles and Arabic-script recorded names can "
            f"appear here: {', '.join(stray[:12])}"
            + (" ..." if len(stray) > 12 else ""))
    return errors, remarks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="a run directory holding <arm>/ dirs")
    parser.add_argument("--arm", action="append", choices=COMMENTATORS, default=None,
                        help="check only these arms (repeatable)")
    args = parser.parse_args()

    errors = []
    for name in (args.arm or COMMENTATORS):
        arm_dir = args.root / name
        if not arm_dir.is_dir():
            print(f"{name}: no directory in {args.root} -- skipped")
            continue
        arm_errors, remarks = check_arm(arm_dir, name)
        print(f"\n== {name}")
        for remark in remarks:
            print(f"   {remark}")
        errors.extend(arm_errors)
        if not arm_errors:
            print("   OK: every stated figure matches a counted ratio")

    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
