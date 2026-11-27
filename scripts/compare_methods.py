"""Compare induced operation profiles across commentators.

    python scripts/compare_methods.py outputs/method_files_matched

Reads each <commentator>/<commentator>_notes.jsonl under the given directory and
reports the operation vocabulary per arm.

Counts are over raw operation strings. Spec §1 and open item #2: the induced
vocabulary is uncontrolled, so "lexical gloss" and "explains gharib word" count
separately. The sprawl ratio printed below quantifies that problem rather than
hiding it -- do not read these counts as a method comparison until the controlled
vocabulary exists.

Raw, but no longer decorated: schemas.Move splits a trailing scope note into
operation_qualifier, so "reconciliation of apparent contradiction (zahir vs.
ijma')" counts as the same operation as the bare label and keeps the qualifier.
Sprawl is meant to measure divergent vocabulary; before that split a parenthesis
alone scored cayni2's six operations as twelve, at a sprawl of 1.00.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The same defect as handoff.md §8.5, in a script that had not been given the fix: batch
# stdout is a redirected file, Python then uses the locale encoding, and a C/POSIX locale
# on a compute node raises UnicodeEncodeError on the first Arabic or "bāb" it prints.
# study_matched.sbatch runs this before validate_matched, so it would take the job's
# summary with it after the arms had already finished.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from hermeneutics.schemas import split_operation


def load(notes_path: Path) -> list[dict]:
    if not notes_path.is_file():
        return []
    with open(notes_path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _operation(move: dict) -> str:
    """The bare operation label, however the record happened to be written.

    `4767842` moved the trailing scope note into `operation_qualifier`, but only for
    records written after it: every note file from 15789363 and earlier still carries
    the decorated string, and reading `operation` raw scored those runs the old way.
    Folding here means one scorer gives one answer for both shapes, so a rescored old
    run is comparable with a new one. `split_operation` is a no-op on an already-bare
    label, so this costs nothing on current records.
    """
    return split_operation(move.get("operation", ""))[0].strip().lower()


def operation_set(record: dict) -> frozenset:
    """The distinct operations used within one bāb, folded like `_operation` does."""
    return frozenset(_operation(move) for move in record.get("moves", []))


def distinct_operation_sets(records: list[dict]) -> tuple[int, int]:
    """(distinct operation-SETS, bābs with at least one move).

    The saturation-curve report (run 15789363) reads a collapsed vocabulary off two
    numbers together: how many distinct labels appear at all (`profile`'s `distinct`
    column), and how many distinct *combinations* of labels a bāb actually uses. A
    3-label vocabulary that still recombines into a dozen different sets per bāb is
    a live repertoire; mazandarani's 2 of 24 said the opposite -- almost every bāb
    used the same one or two labels together, which `distinct` alone does not show.
    """
    sets = [operation_set(r) for r in records if r.get("moves")]
    return len(set(sets)), len(sets)


def saturation_curve(records: list[dict]) -> list[int]:
    """Distinct operations seen after each bāb, in the order the arm read them.

    This is what says whether the sample was big enough: a curve that flattens is the
    evidence that reading more bābs would not have added vocabulary, and one still
    climbing at the end is a finding to report rather than a reason to run more.

    A `no_move` bāb still advances the x-axis. It was read; dropping it would shorten
    the axis and make the vocabulary look like it saturated sooner than it did.
    """
    seen: set[str] = set()
    curve: list[int] = []
    for record in records:
        for move in record.get("moves", []):
            seen.add(_operation(move))
        curve.append(len(seen))
    return curve


def profile(records: list[dict]) -> tuple[Counter, int, int, int]:
    operations = Counter()
    verified = 0
    total = 0
    no_moves = 0
    for record in records:
        if record.get("status") == "no_move":
            no_moves += 1
        for move in record.get("moves", []):
            operations[_operation(move)] += 1
            total += 1
            if move.get("arabic_evidence"):
                verified += 1
    return operations, verified, total, no_moves


def count_failures(root: Path, name: str) -> int:
    """B4: declared failures for this arm -- horizon_diff giving up twice (logged
    to <arm>/<name>_failures.jsonl) plus segment_one giving up (logged under
    segmentation/ instead, since it runs before the arm directory even exists).
    A run that finished with fewer notes than requested must say so here, not only
    in the JSONL a human is not otherwise reading.
    """
    graph = load(root / name / f"{name}_failures.jsonl")
    seg = [rec for path in sorted((root / "segmentation").glob(f"{name}_seg_failures.*.jsonl"))
           for rec in load(path)]
    return len(graph) + len(seg)


def collect_arms(roots: list[Path]) -> dict[str, tuple[Path, str, list[dict]]]:
    """arm label -> (directory the failures file lives in, notes-file stem, records).

    The bake-off layout is one root with a subdirectory per arm:
    <root>/<commentator>/<commentator>_notes.jsonl (and _failures.jsonl alongside it,
    stem == label). The ablation (scripts/run_ablation.sh) runs one arm (nawawi) three
    times, each under its own --notes-dir, so there is no arm-named subdirectory to
    descend into -- the notes file sits directly at <condition-root>/nawawi_notes.jsonl,
    labeled here by the condition (the root's own directory name) while the stem stays
    "nawawi", since that is what run_study_graph.py actually names the failures file.
    Passing multiple roots on the command line selects this second shape, so the report
    can put mechanism-toggled runs of the SAME commentator side by side.
    """
    arms: dict[str, tuple[Path, str, list[dict]]] = {}
    if len(roots) == 1 and roots[0].is_dir() and any(p.is_dir() for p in roots[0].iterdir()):
        root = roots[0]
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            name = directory.name
            records = load(directory / f"{name}_notes.jsonl")
            if records:
                arms[name] = (directory, name, records)
        return arms

    for root in roots:
        if not root.is_dir():
            continue
        notes_files = sorted(root.glob("*_notes.jsonl"))
        if len(notes_files) == 1:
            records = load(notes_files[0])
            if records:
                stem = notes_files[0].name.removesuffix("_notes.jsonl")
                arms[root.name] = (root, stem, records)
    return arms


def main() -> int:
    roots = [Path(a) for a in sys.argv[1:]] or [Path("outputs/method_files_matched")]
    located = collect_arms(roots)

    if not located:
        print(f"No notes found under {', '.join(str(r) for r in roots)}")
        return 1

    arms = {name: records for name, (_, _, records) in located.items()}

    print(f"{'arm':<14}{'babs':>6}{'moves':>7}{'no move':>9}{'distinct':>10}"
          f"{'op-sets':>10}{'sprawl':>8}{'w/ evidence':>13}{'failed':>8}")
    profiles = {}
    for name, records in arms.items():
        arm_dir, stem, _ = located[name]
        operations, verified, total, no_moves = profile(records)
        profiles[name] = operations
        n_sets, n_babs_with_moves = distinct_operation_sets(records)
        op_sets = f"{n_sets}/{n_babs_with_moves}"
        sprawl = len(operations) / total if total else 0
        share = f"{verified}/{total}"
        seg_dir = arm_dir.parent / "segmentation"
        failed = (len(load(arm_dir / f"{stem}_failures.jsonl"))
                  + sum(len(load(p)) for p in sorted(seg_dir.glob(f"{stem}_seg_failures.*.jsonl"))))
        print(f"{name:<14}{len(records):>6}{total:>7}{no_moves:>9}{len(operations):>10}"
              f"{op_sets:>10}{sprawl:>8.2f}{share:>13}{failed:>8}")
        if failed:
            print(f"    ** {name}: {len(records)} notes + {failed} declared failure(s) "
                  f"-- this arm is SHORT of a full run, see {stem}_failures.jsonl")

    print("\nTop operations per arm (raw strings):")
    for name, operations in profiles.items():
        print(f"\n  {name}")
        for operation, count in operations.most_common(8):
            print(f"    {count:>3}  {operation[:70]}")

    print("\nVocabulary saturation -- distinct operations after each bāb read:")
    curves = {}
    for name, records in arms.items():
        curve = saturation_curve(records)
        curves[name] = curve
        # The last quarter is where a flattening curve shows it: if the vocabulary is
        # still growing there, the sample did not cover the arm's repertoire and the
        # honest report is that it did not.
        tail = curve[max(0, len(curve) * 3 // 4):]
        growth = (tail[-1] - tail[0]) if tail else 0
        spark = " ".join(str(v) for v in curve[:: max(1, len(curve) // 12)])
        print(f"  {name:<14} {spark}")
        verdict = "still climbing" if growth else "flat"
        print(f"  {'':<14} final {curve[-1] if curve else 0} distinct; "
              f"last quarter +{growth} -> {verdict}")

    out_path = roots[0] / "saturation.json"
    out_path.write_text(json.dumps(curves, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n  curve data for plotting: {out_path}")

    shared = set.intersection(*(set(p) for p in profiles.values())) if len(profiles) > 1 else set()
    print(f"\nOperation strings shared by all {len(profiles)} arms: {len(shared)}")
    for operation in sorted(shared)[:10]:
        print(f"    {operation[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
