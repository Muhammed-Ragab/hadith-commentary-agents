"""Score a filled-in hand_check.csv against its hand_check_key.csv.

scripts/tag_moves_llm.py writes both files together into an outputs/comparison/<stamp>/
dir: hand_check.csv has an empty `user_move` column for a human to fill in (one of the
move labels in hermeneutics.answer_moves.MOVES), and hand_check_key.csv holds the
LLM's final_move for the same sample_id, kept separate so filling in hand_check.csv
does not show the model's own answer.

    python scripts/score_hand_check.py outputs/comparison/<stamp>/hand_check.csv
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score(hand_check_path: Path, key_path: Path | None = None) -> dict:
    """Percent agreement and Cohen's kappa between user_move (hand_check.csv) and
    final_move (hand_check_key.csv), joined on sample_id. Rows with an empty
    user_move are skipped and counted separately -- they are not yet scoreable."""
    key_path = key_path or hand_check_path.with_name("hand_check_key.csv")
    checked = _read_csv(hand_check_path)
    key_by_id = {r["sample_id"]: r["final_move"] for r in _read_csv(key_path)}

    pairs = []
    unfilled = 0
    unmatched = []
    for row in checked:
        user_move = (row.get("user_move") or "").strip()
        sample_id = row["sample_id"]
        if not user_move:
            unfilled += 1
            continue
        if sample_id not in key_by_id:
            unmatched.append(sample_id)
            continue
        pairs.append((user_move, key_by_id[sample_id]))

    n = len(pairs)
    if n == 0:
        return {"n_scored": 0, "n_unfilled": unfilled, "unmatched_ids": unmatched,
               "percent_agreement": None, "cohen_kappa": None}

    agree = sum(1 for a, b in pairs if a == b)
    percent_agreement = agree / n

    from sklearn.metrics import cohen_kappa_score
    user_labels = [a for a, _ in pairs]
    key_labels = [b for _, b in pairs]
    kappa = (cohen_kappa_score(user_labels, key_labels)
            if len(set(user_labels) | set(key_labels)) > 1 else 1.0 if percent_agreement == 1.0 else 0.0)

    return {"n_scored": n, "n_unfilled": unfilled, "unmatched_ids": unmatched,
           "percent_agreement": round(percent_agreement, 4), "cohen_kappa": round(float(kappa), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hand_check_csv", type=Path)
    ap.add_argument("--key", type=Path, default=None,
                   help="Default: hand_check_key.csv next to hand_check_csv.")
    args = ap.parse_args()

    result = score(args.hand_check_csv, args.key)
    print(f"scored: {result['n_scored']}, unfilled user_move: {result['n_unfilled']}",
          flush=True)
    if result["unmatched_ids"]:
        print(f"WARNING: {len(result['unmatched_ids'])} sample_id(s) in hand_check.csv "
              f"have no match in the key: {result['unmatched_ids']}", flush=True)
    if result["n_scored"] == 0:
        print("Nothing scoreable yet -- fill in user_move and rerun.", flush=True)
        return 0
    print(f"percent agreement: {result['percent_agreement']:.1%}", flush=True)
    print(f"Cohen's kappa: {result['cohen_kappa']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
