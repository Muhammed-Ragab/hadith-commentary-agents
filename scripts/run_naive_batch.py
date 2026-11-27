"""Naive-reading batch pass. Spec §4 call 1, §6.2 phase 1b.

4,522 calls with no cross-unit state. Running these in batch is a correctness
property: the naive reading must not see the method hypothesis, and a batch pass
makes that structurally impossible rather than merely intended.

Usage:
  python scripts/run_naive_batch.py nawawi --shard 0 --n-shards 8
  python scripts/run_naive_batch.py nawawi --merge
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from hermeneutics.llm import call_model, get_chat_model
from hermeneutics.manifest import build_manifest, manifest_id, write_manifest
from hermeneutics.prompts import render
from hermeneutics.trace import TraceWriter
from scripts.check_corpus import load_babs
from scripts.run_segmentation import select_units

from scripts.check_corpus import COMMENTATORS

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "naive"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute naive readings.")
    p.add_argument("commentator", choices=COMMENTATORS)
    p.add_argument("--limit", type=int, default=None,
                   help="precompute only the first N bābs, matching run_study_graph --limit")
    p.add_argument("--sample", default=None, help="stratified sample KITABS:PER_KITAB:SEED (e.g. 12:5:20260914), instead of --limit's first N babs. Every phase of a run must be given the SAME spec.")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--merge", action="store_true", help="merge shard files and exit")
    p.add_argument("--run-id", default=None)
    return p


def naive_one(bab: dict, ref: str, model, trace, mid: str) -> dict:
    text = call_model(
        model, "You interpret hadith on their own terms.",
        render("naive_reading", matn=bab["text"]),
        trace=trace, node="naive_reading", bab_ref=ref,
        model_id=getattr(model, "model_name", "fake"), manifest_id=mid,
    )
    return {"bab_ref": ref, "naive_reading": text}


def resolve_run_id(commentator: str, run_id: str | None) -> str:
    """--run-id keys the output directory (outputs/naive/<run_id>/); default to the
    same manifest id the write path already falls back to, so a bare invocation is
    still per-run instead of landing in a directory every run shares."""
    if run_id:
        return run_id
    model_id = os.environ["VLLM_MODEL_ID"]
    manifest = build_manifest(commentator=commentator, model_id=model_id)
    return manifest_id(manifest)


def merge_shards(directory: Path, commentator: str) -> Path:
    """Merge shard files into one, in bāb order. Later duplicates win."""
    directory = Path(directory)
    merged: dict[int, dict] = {}

    for path in sorted(directory.glob(f"{commentator}_naive.*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                merged[int(rec["bab_ref"].split("/")[1])] = rec

    out = directory / f"{commentator}_naive.jsonl"
    out.write_text(
        "".join(json.dumps(merged[k], ensure_ascii=False) + "\n" for k in sorted(merged)),
        encoding="utf-8",
    )
    return out


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    name = args.commentator

    if args.merge:
        # Keyed by run id so a merge can only ever glob its own run's shards -- two
        # arms of the matched study now run six concurrent shards each, across two
        # models, and a shared outputs/naive/ let one run's --merge absorb another's
        # in-flight shard files.
        run_out_dir = OUT_DIR / resolve_run_id(name, args.run_id)
        out = merge_shards(run_out_dir, name)
        n = len(out.read_text(encoding="utf-8").strip().split("\n"))
        print(f"Merged {n} naive readings into {out}")
        return 0

    babs = load_babs(name)
    # select_units applies --limit BEFORE sharding, exactly as run_segmentation does,
    # so a limited study run and its naive pre-pass cover the same bābs.
    mine = select_units(list(enumerate(babs)), args.limit, args.shard,
                        args.n_shards, args.sample)

    model_id = os.environ["VLLM_MODEL_ID"]
    manifest = build_manifest(commentator=name, model_id=model_id)
    run_id = args.run_id or manifest_id(manifest)
    run_dir = ROOT / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, run_dir)

    mid = manifest_id(manifest)

    # 600 was a cap on the answer; on a thinking model it is a cap on the
    # reasoning, and job 15849569 wrote 75 empty readings because of it.
    model = get_chat_model(max_tokens=int(os.environ.get("STUDY_NAIVE_MAX_TOKENS", "4096")))
    trace = TraceWriter(run_dir=run_dir, phase=f"naive_{name}_{args.shard:03d}")

    run_out_dir = OUT_DIR / run_id
    run_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_out_dir / f"{name}_naive.{args.shard:03d}-of-{args.n_shards:03d}.jsonl"
    # A skipped bāb has to be DECLARED, not merely not written. A silent `continue`
    # left the study graph with no naive reading for that ref, so it dropped the bāb
    # with neither a note nor a failure -- nawawi/166 vanished from a 65-bāb arm that
    # then scored on 64, and validate_matched could not tell a skip from a crash.
    # Same rule as run_segmentation's seg_failures: declare it and stay accounted.
    skips_path = run_out_dir / f"{name}_naive_skips.{args.shard:03d}-of-{args.n_shards:03d}.jsonl"
    written = skipped = 0
    with open(out_path, "w", encoding="utf-8", buffering=1) as fh, \
            open(skips_path, "w", encoding="utf-8", buffering=1) as skip_fh:
        for i, bab in mine:
            ref = f"{name}/{i}"
            n_chars = len(bab["text"].strip())
            if n_chars < 100:
                skip_fh.write(json.dumps(
                    {"bab_ref": ref, "reason": "bab text under 100 characters",
                     "n_chars": n_chars, "manifest_id": mid},
                    ensure_ascii=False) + "\n")
                skipped += 1
                continue
            fh.write(json.dumps(naive_one(bab, ref, model, trace, mid),
                                ensure_ascii=False) + "\n")
            written += 1

    # Count what was WRITTEN, not what was selected. The old line printed len(mine),
    # so it reported 65 for an arm that had produced 64 readings.
    print(f"Wrote {written} naive readings to {out_path}"
          + (f" ({skipped} declared skip(s) in {skips_path})" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
