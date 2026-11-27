"""Dump every induced operation label from one or more notes directories.

    python scripts/dump_operations.py outputs/ablation_cluster2/A outputs/ablation_cluster2/B outputs/ablation_cluster2/C

Input for building the controlled vocabulary (open item #2). §8.18 measured sprawl
1.00 on all three ablation arms -- 135 free-text labels for 135 moves, with zero
strings shared between arms -- so the label set has to be derived by reading what the
arms actually produced, not invented from theory.

Labels are folded through `split_operation` exactly as `compare_methods.py` folds
them, so what is printed here is what that scorer would count. The qualifier is kept
on the same line: it is the evidence for whether two differently-worded labels are
the same operation or not.

ponytail: no clustering, no embeddings. 135 short phrases is a reading task, and the
naming carries the paper's analytic claims -- that is Muhammed's call, not a
k-means run's. stdlib only, so it runs in the light client venv (no torch).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# §8.5: a redirected stdout under a C/POSIX locale raises UnicodeEncodeError on the
# first "bāb" printed. Same one-line fix compare_methods.py carries.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from hermeneutics.schemas import split_operation


def main() -> int:
    roots = [Path(a) for a in sys.argv[1:]]
    if not roots:
        print(__doc__.strip().splitlines()[2].strip())
        return 1

    rows = []
    for root in roots:
        for notes in sorted(root.glob("*_notes.jsonl")):
            with open(notes, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    for move in record.get("moves", []):
                        label, qualifier = split_operation(move.get("operation", ""))
                        rows.append((root.name, record.get("bab_ref", ""),
                                     label.strip(),
                                     (qualifier or move.get("operation_qualifier", "")).strip()))

    if not rows:
        print(f"No moves found under {', '.join(str(r) for r in roots)}")
        return 1

    print(f"{len(rows)} operations from {len(roots)} arm(s)\n")
    for arm, bab_ref, label, qualifier in sorted(rows, key=lambda r: (r[2].lower(), r[0])):
        print(f"[{arm}] {bab_ref}")
        print(f"  {label}")
        if qualifier:
            print(f"      ({qualifier})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
