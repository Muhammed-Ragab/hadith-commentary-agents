"""Score the model bake-off. Handoff §8 q2 / §10.4.

Two numbers per candidate, both free — no extra model calls, no scholar time:

  grounding_rate  the primary selection criterion. Fraction of moves whose
                  arabic_evidence really occurs in the commentary that was read
                  (hermeneutics.grounding). Blind rating is the tiebreak, not the test.
  s_per_call      replaces the unverified 2.5 s/call assumption in spec §6.1 that the
                  whole "9-12 h per commentator" schedule rests on.

Usage:
  python scripts/score_bakeoff.py nawawi --bakeoff-dir outputs/bakeoff
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermeneutics.grounding import grounding_rate
from scripts.check_corpus import COMMENTATORS, load_babs

ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def score_candidate(cand_dir: Path, commentator: str, sources: dict[str, str]) -> dict:
    notes = read_jsonl(cand_dir / f"{commentator}_notes.jsonl")
    # bakeoff.sbatch passes --run-id <slug>, so the traces for this candidate are here.
    trace_dir = ROOT / "outputs" / "runs" / cand_dir.name
    calls = [c for f in sorted(trace_dir.glob("*.jsonl"))
             for c in read_jsonl(f) if "latency_s" in c]
    lat = [c["latency_s"] for c in calls]
    moves = sum(len(r.get("moves", [])) for r in notes)
    return {
        "candidate": cand_dir.name,
        "units": len(notes),
        "moves": moves,
        "grounding_rate": round(grounding_rate(notes, sources), 4),
        "calls": len(lat),
        "s_per_call_median": round(statistics.median(lat), 3) if lat else None,
        "s_per_call_mean": round(statistics.fmean(lat), 3) if lat else None,
        "json_repair_rate": (
            round(sum(1 for c in calls if c.get("error")) / len(calls), 4) if calls else None
        ),
    }


def projected_hours(s_per_call: float, sequential_calls: int) -> float:
    """Sequential-phase wall clock. Batch calls parallelise; these do not."""
    return round(s_per_call * sequential_calls / 3600, 1)


def main() -> int:
    p = argparse.ArgumentParser(description="Score the model bake-off.")
    p.add_argument("commentator", choices=COMMENTATORS)
    p.add_argument("--bakeoff-dir", default="outputs/bakeoff")
    p.add_argument("--sequential-calls", type=int, default=14783,
                   help="ʿAynī sequential-phase calls at a 6K chunk; 3969 for Nawawi")
    args = p.parse_args()

    babs = load_babs(args.commentator)
    sources = {f"{args.commentator}/{i}": b["text"] for i, b in enumerate(babs)}

    root = Path(args.bakeoff_dir)
    if not root.is_absolute():
        root = ROOT / root
    rows = [score_candidate(d, args.commentator, sources)
            for d in sorted(root.iterdir()) if d.is_dir()]
    if not rows:
        print(f"No candidate directories under {root}")
        return 1

    rows.sort(key=lambda r: r["grounding_rate"], reverse=True)
    print(f"{'candidate':44} {'ground':>7} {'moves':>6} {'s/call':>7} {'repair':>7} {'proj h':>7}")
    for r in rows:
        proj = projected_hours(r["s_per_call_median"], args.sequential_calls) \
            if r["s_per_call_median"] else None
        print(f"{r['candidate'][:44]:44} {r['grounding_rate']:>7.3f} {r['moves']:>6} "
              f"{r['s_per_call_median'] or 0:>7.2f} {r['json_repair_rate'] or 0:>7.3f} "
              f"{proj if proj is not None else 0:>7}")

    out = root / "bakeoff_scores.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWinner by grounding: {rows[0]['candidate']}")
    print(f"Pin it in the manifest (VLLM_MODEL_ID) and record s/call in spec §6.1.\n{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
