"""Segmentation batch pass. Spec §3.6, §6.2 phase 1a.

Marks where commentary begins in each unit. Carries NO cross-unit state, so it is
embarrassingly parallel and runs as a SLURM array before any graph starts.

Interleaved sharh passages are a documented limitation: those units get a low
confidence and the scholar audit protocol (30-unit sample) checks them.

Usage:
  python scripts/run_segmentation.py nawawi
  python scripts/run_segmentation.py cayni --shard 3 --n-shards 8
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from hermeneutics.llm import call_json, get_chat_model
from hermeneutics.manifest import build_manifest, manifest_id, write_manifest
from hermeneutics.markers import matn_span
from hermeneutics.prompts import render
from hermeneutics.schemas import SegRecord
from hermeneutics.trace import TraceWriter
from scripts.check_corpus import load_babs, parse_sample, sample_indices

from scripts.check_corpus import COMMENTATORS

ROOT = Path(__file__).resolve().parent.parent
MAX_SEG_CHARS = 20_000  # the window handed to the model, measured from the anchor below
MIN_MATN_ARABIC = 10    # a matn fragment always carries more Arabic than this.
                        # A char floor cannot be used: Nawawi quotes short lemmas and
                        # segments legitimately at 79 chars.
MIN_CONFIDENCE = 0.1    # the prompt tells the model to use <0.5 for interleaved text;
                        # 0.0 is a copied default, not an answer

ARABIC = re.compile(r"[؀-ۿ]")
# The OpenITI mARkdown hadith marker. Present in 94.1% of ʿAynī's babs and in none of
# Nawawi's or Mazandarani's, which carry no such marker at all.
HADITH_MARK = re.compile(r"###\s*\$")
ISNAD = re.compile(r"حدثنا|حدثني|أخبرنا|أنبأنا")


def anchor_offset(text: str, max_chars: int = MAX_SEG_CHARS) -> int:
    """Where to start the window handed to the model.

    "The boundary is always near the start" held for Nawawi and Mazandarani and was
    false for ʿAynī: cayni/0 is 57,117 characters of theological preamble and its
    first isnād sits at 34,815, so the model was shown a window with no matn in it
    and correctly answered 0-0 (job cayni1). Anchoring on the marker also shrinks
    the prompt -- ʿAynī's median marker offset is 431, so most babs send less, not
    more.

    The isnād fallback is deliberately gated on the text being longer than the
    window: inside it, truncation cannot hide the matn, and an isnād formula
    appearing deep in the sharh is a worse anchor than the top of the bab.
    """
    mark = HADITH_MARK.search(text)
    if mark:
        return mark.start()
    if len(text) > max_chars:
        isnad = ISNAD.search(text)
        if isnad:
            # Exactly at the chain, no lead-in. Backing off 200 characters "so the
            # model sees what precedes it" cost cayni/0 its matn: the model took the
            # 200 characters of ʿAynī quoting Sufyān b. ʿUyayna as the hadith and
            # stopped where the isnād actually began (job cayni2, confidence 0.3).
            return isnad.start()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute matn/sharh boundaries.")
    p.add_argument("commentator", choices=COMMENTATORS)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None,
                   help="segment only the first N babs (matched smoke/study runs)")
    p.add_argument("--sample", default=None, help="stratified sample KITABS:PER_KITAB:SEED (e.g. 12:5:20260914), instead of --limit's first N babs. Every phase of a run must be given the SAME spec.")
    p.add_argument("--output-dir", default=None,
                   help="directory for segmentation JSONL; defaults to outputs/segmentation")
    p.add_argument("--run-id", default=None)
    return p


def shard(items: list, index: int, n_shards: int) -> list:
    """Contiguous slice for one array task. Stateless, so any split is valid."""
    if n_shards <= 1:
        return list(items)
    size = (len(items) + n_shards - 1) // n_shards
    return list(items)[index * size : (index + 1) * size]


def select_units(items: list, limit: int | None, index: int, n_shards: int,
                 sample: str | None = None) -> list:
    """Apply the study selection before splitting work across shards.

    `--sample` and `--limit` are alternatives: the sample already fixes which bābs the
    run covers, so applying a limit on top of it would silently truncate the strata
    and leave the arms covering different kitābs.
    """
    items = list(items)
    if sample:
        wanted = set(sample_indices([bab for _, bab in items], *parse_sample(sample)))
        selected = [pair for position, pair in enumerate(items) if position in wanted]
    else:
        selected = items[:limit] if limit is not None else items
    return shard(selected, index, n_shards)


def reject_reason(offset: int, confidence: float, text: str,
                  min_matn_arabic: int = MIN_MATN_ARABIC,
                  min_confidence: float = MIN_CONFIDENCE,
                  start: int = 0) -> str | None:
    """Why this answer is not a usable matn span, or None if it is.

    A structurally valid offset is not enough. Qwen job 15775708 answered `1` for
    Nawawi and Cayni: that passes `0 < offset < len(text)` but hands the naive
    reader only the `#` heading marker, so every move derived from it is an
    artifact of empty input rather than a result.

    The span exists because that guard was also rejecting the truth. 93% of ʿAynī's
    3,959 bābs open with a gloss on Bukhārī's chapter heading (أي, يعني, قوله) and
    quote the hadith only afterwards -- 86% carry an isnād within the first 2,000
    characters. Asked where the matn *ended* in such a bāb, the model answered 1,
    which was correct and unusable, and the arm died on it (jobs 15849569, saia2,
    saia3). The matn is bounded, not prefixed.
    """
    if not 0 <= start < offset < len(text):
        return (f"The previous span {start}-{offset} was invalid. Return "
                f"matn_start_char >= 0 and matn_end_char greater than it, both "
                f"below {len(text)}.")
    # ponytail: half the unit's own Arabic is the floor, so the handful of "#"-only
    # and 2-character corpus babs cannot abort a whole arm on an impossible demand.
    need = min(min_matn_arabic, len(ARABIC.findall(text)) // 2)
    got = len(ARABIC.findall(text[start:offset]))
    if got < need:
        return (f"The previous span {start}-{offset} holds a matn of only {got} Arabic "
                f"characters, which cannot hold a hadith. Return a span holding "
                f"at least {need}.")
    if confidence < min_confidence:
        return (f"The previous confidence {confidence} is not an answer. Report your real "
                f"confidence; use a value below 0.5 only if matn and sharh are interleaved.")
    return None


def segment_one(bab: dict, ref: str, model, trace, mid: str,
                max_chars: int = MAX_SEG_CHARS,
                min_matn_arabic: int = MIN_MATN_ARABIC,
                min_confidence: float = MIN_CONFIDENCE) -> dict:
    """Locate one credible matn span, with one corrective retry.

    The model sees a window, not the whole bab, and answers in window coordinates.
    The record carries absolute offsets into bab["text"], because that is what the
    study graph slices.
    """
    text = bab["text"]
    # al-Kāfī marks its own boundary. Where the markers bound a matn, the corpus has
    # already answered and a model call can only be worse (handoff.md §10.5, §8.15).
    # 73% of the Shia arm; confidence 1.0 marks a record no model produced.
    marked = matn_span(text)
    if marked:
        return SegRecord(bab_ref=ref, matn_start_char=marked[0],
                         matn_end_char=marked[1], confidence=1.0).model_dump()

    anchor = anchor_offset(text, max_chars)
    window = text[anchor:anchor + max_chars]
    retry_feedback = ""
    for _ in range(2):
        raw = call_json(
            model, "You locate where commentary begins.",
            render("segmentation", text=window, retry_feedback=retry_feedback),
            trace=trace, node="segmentation", bab_ref=ref,
            model_id=getattr(model, "model_name", "fake"), manifest_id=mid,
        )
        start = int(raw.get("matn_start_char", 0))
        offset = int(raw.get("matn_end_char", 0))
        confidence = float(raw.get("confidence", 0.0))
        # Judged against the window, so the retry feedback quotes the offsets the
        # model itself reported rather than absolute ones it has never seen.
        retry_feedback = reject_reason(
            offset, confidence, window, min_matn_arabic, min_confidence, start
        )
        if retry_feedback is None:
            return SegRecord(
                bab_ref=ref, matn_start_char=anchor + start,
                matn_end_char=anchor + offset, confidence=confidence,
            ).model_dump()
    raise ValueError(f"segmentation returned no credible boundary for {ref}: {retry_feedback}")


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    name = args.commentator

    babs = load_babs(name)
    indexed = list(enumerate(babs))
    mine = select_units(indexed, args.limit, args.shard, args.n_shards, args.sample)

    model_id = os.environ["VLLM_MODEL_ID"]
    manifest = build_manifest(commentator=name, model_id=model_id)
    run_id = args.run_id or manifest_id(manifest)
    run_dir = ROOT / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, run_dir)

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_tag = f"{args.shard:03d}-of-{args.n_shards:03d}"
    out_path = out_dir / f"{name}_seg.{shard_tag}.jsonl"
    # A distinct stem (not "<out_path stem>_failures.jsonl" as elsewhere) so this
    # never collides with the "{name}_seg.*.jsonl" glob validate_matched.py and
    # load_segmentation() use to find the boundary records themselves.
    failures_path = out_dir / f"{name}_seg_failures.{shard_tag}.jsonl"

    model = get_chat_model(max_tokens=200, enable_thinking=False)
    trace = TraceWriter(run_dir=run_dir, phase=f"segmentation_{name}_{args.shard:03d}")
    max_chars = int(os.environ.get("SEGMENTATION_MAX_CHARS", MAX_SEG_CHARS))

    n_failed = 0
    with open(out_path, "w", encoding="utf-8", buffering=1) as fh:
        for i, bab in mine:
            ref = f"{name}/{i}"
            try:
                rec = segment_one(bab, ref, model, trace, manifest_id(manifest), max_chars=max_chars)
            except ValueError as exc:
                # B4: cayni/2232 and cayni/2872 carry an isnād the 20,000-char window
                # never reaches; segment_one correctly refuses rather than fabricate a
                # boundary. One bad bāb used to abort the whole arm (raise here), which
                # threw away every record already written for it. Log it and move on --
                # validate_matched.py treats this as a declared shortfall, never a
                # silent pass, because the ref is then also missing from deps.seg and
                # select_next_bab (hermeneutics/nodes.py) skips it in step with this.
                failure = {"bab_ref": ref, "attempts": 2, "error": str(exc)}
                with open(failures_path, "a", encoding="utf-8", buffering=1) as ffh:
                    ffh.write(json.dumps(failure, ensure_ascii=False) + "\n")
                n_failed += 1
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    msg = f"Wrote {len(mine) - n_failed} segmentation records to {out_path}"
    if n_failed:
        msg += f" ({n_failed} declared failure(s) logged to {failures_path})"
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
