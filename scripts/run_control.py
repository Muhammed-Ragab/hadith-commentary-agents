"""Control phase. Spec §6.2 phase 3, §4.1.

Runs the identical study graph over the shuffled subset, in a fully isolated
namespace: own thread id, own checkpointer file, own notes file, own trace, and a
method hypothesis that starts empty and never touches the real run's.

Usage:
  python scripts/run_control.py nawawi
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from hermeneutics.chunking import DEFAULT_MAX_TOKENS, token_len_for
from hermeneutics.control import (DEFAULT_FRACTION, assert_isolated,
                                  shuffle_within_kitab)
from hermeneutics.llm import get_chat_model
from hermeneutics.manifest import build_manifest, manifest_id, write_manifest
from hermeneutics.nodes import Deps
from hermeneutics.state import initial_study_state
from hermeneutics.study_graph import build_study_graph
from hermeneutics.trace import TraceWriter
from scripts.check_corpus import (COMMENTATORS, load_babs, parse_sample,
                                  sample_indices)
from scripts.run_study_graph import (load_segmentation, load_segmentation_failures,
                                     model_output_budgets, require_naive)

ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the shuffled-pair control arm.")
    p.add_argument("commentator", choices=COMMENTATORS)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--sample", default=None,
                   help="KITABS:PER_KITAB:SEED, the SAME spec the study arm ran. The "
                        "control then shuffles within the study's own bābs, so the two "
                        "arms differ only in the pairing -- and the naive readings and "
                        "segmentation already on disk cover exactly these units.")
    p.add_argument("--fraction", type=float, default=None,
                   help="share of each kitāb to control. Default 1.0 with --sample "
                        "(the sample is already the selection) and 0.10 without.")
    p.add_argument("--chunk-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--run-id", default=None)
    p.add_argument("--naive-dir", default=None,
                   help="naive-reading JSONL directory, as written by "
                        "run_naive_batch.py for the run being controlled. "
                        "This run's own --run-id cannot locate it: the control "
                        "manifest carries control_pairing, so its hash differs.")
    p.add_argument("--segmentation-dir", default=None,
                   help="the study run's segmentation. The paired bāb's boundary is "
                        "what splits matn from sharh (`source_ref` in _select_bab), so "
                        "omitting it runs the control on a different footing from the "
                        "arm it is compared against.")
    p.add_argument("--notes-dir", default=None,
                   help="where to write control notes. Keep this OUT of the study arm's "
                        "directory: compare_methods falls back to glob('*_notes.jsonl') "
                        "and <name>_control_notes.jsonl both matches it and sorts first.")
    p.add_argument("--matn-only", action="store_true",
                   help="EXPLORATORY no-sharh arm (2026-09-20, NOT pre-registered). "
                        "Feeds each bāb's own MATN in place of the sharh, with no "
                        "pairing. Answers whether the pipeline can manufacture "
                        "operations from the hadith alone. Dropping the commentary "
                        "entirely would instead score 0.0 by construction: no evidence "
                        "spans means no citable id means every move rejected.")
    p.add_argument("--real-method", default=None,
                   help="the study arm's <name>_method.md, for the isolation check. "
                        "Separate from --notes-dir because the control's notes must not "
                        "live beside the study's.")
    return p


def resolve_fraction(fraction: float | None, sample: str | None) -> float:
    """With --sample the selection has already happened, so control all of it.

    Left at 0.10, `--sample 14:5:...` would control `round(0.1 * 5)` -> the floor of 2
    bābs per kitāb, making the control arm a third of the study arm it is compared
    against, on a rule that divides one by the other.
    """
    if fraction is not None:
        return fraction
    return 1.0 if sample else DEFAULT_FRACTION


def build_control_deps_args(commentator: str, control: bool,
                            matn_only: bool = False) -> dict:
    """Namespacing for one arm. Kept separate so the isolation is testable.

    The matn-only arm gets its own namespace for the same reason the control does:
    compare_methods falls back to glob("*_notes.jsonl"), which every arm's filename
    matches. Sharing a directory silently mixes arms.
    """
    arm = "matnonly" if matn_only else ("control" if control else "study")
    return {
        "thread_id": f"{arm}_{commentator}",
        "db_name": f"{arm}_{commentator}.sqlite",
        "notes_name": (f"{commentator}_matnonly_notes.jsonl" if matn_only
                       else f"{commentator}_control_notes.jsonl" if control
                       else f"{commentator}_notes.jsonl"),
        "trace_phase": f"{arm}_{commentator}",
    }


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    name = args.commentator

    babs = load_babs(name)
    # The study arm walks a stratified sample, so a control drawn from the whole corpus
    # compares different kitābs -- and at 10% of ʿAynī that is 396 bābs, ~66 h. With
    # --sample the control shuffles inside the study's own units.
    sample_global: list[int] | None = None
    if args.sample:
        sample_global = sample_indices(babs, *parse_sample(args.sample))
        babs = [babs[i] for i in sample_global]

    fraction = resolve_fraction(args.fraction, args.sample)
    if args.matn_only:
        # Every sampled bāb, each reading its OWN matn. Shuffling here would confound
        # "no sharh" with "wrong sharh" -- the control arm already measures the latter.
        pairing = {i: i for i in range(len(babs))}
        print(f"Matn-only arm (EXPLORATORY, not pre-registered): {len(pairing)} bābs, "
              "no pairing, sharh withheld")
    else:
        pairing = shuffle_within_kitab(babs, seed=args.seed, fraction=fraction)
        print(f"Control arm: {len(pairing)} of {len(babs)} bābs, seed {args.seed}, "
              f"fraction {fraction}")

    model_id = os.environ["VLLM_MODEL_ID"]
    manifest = build_manifest(commentator=name, model_id=model_id)
    manifest["control_seed"] = args.seed
    manifest["control_fraction"] = fraction

    ns = build_control_deps_args(name, control=True, matn_only=args.matn_only)
    notes_dir = Path(args.notes_dir) if args.notes_dir else ROOT / "outputs" / "method_files"
    notes_path = notes_dir / ns["notes_name"]
    notes_path.parent.mkdir(parents=True, exist_ok=True)

    # Only the shuffled units run, in bāb order. Positions in control_babs are LOCAL,
    # so translate the pairing into local coordinates and keep global_index around so
    # every emitted bab_ref stays global (score_control.py needs that, and so does the
    # segmentation lookup, which is keyed on the global ref of the PAIRED bāb).
    selected = sorted(pairing)
    control_babs = [babs[i] for i in selected]
    local_of = {g: p for p, g in enumerate(selected)}
    local_pairing = {local_of[t]: local_of[pairing[t]] for t in selected}

    def to_global(i: int) -> int:
        return sample_global[i] if sample_global else i

    arm_global = [to_global(i) for i in selected]
    manifest["control_sample"] = args.sample or ""
    manifest["control_pairing"] = {str(to_global(k)): to_global(v)
                                   for k, v in pairing.items()}
    # Compute run_id after all manifest mutations so that run_dir and Deps.manifest_id
    # both derive from the same fully-populated manifest hash.
    run_id = args.run_id or manifest_id(manifest)

    run_dir = ROOT / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, run_dir)

    segmentation_dir = (Path(args.segmentation_dir) if args.segmentation_dir
                        else ROOT / "outputs" / "segmentation")
    # 1500 and 4000 were hard-coded here and never learned §8.17: read_commentary needs
    # up to 5,281 output tokens against SAIA, and 1500 raises LengthFinishReasonError
    # or returns a successful call with empty content on a thinking model.
    output_tokens, consolidate_tokens = model_output_budgets()
    deps = Deps(
        model=get_chat_model(max_tokens=output_tokens),
        trace=TraceWriter(run_dir=run_dir, phase=ns["trace_phase"]),
        manifest_id=manifest_id(manifest),
        babs=control_babs,
        commentator=name,
        naive=require_naive(
            (Path(args.naive_dir) if args.naive_dir
             else ROOT / "outputs" / "naive" / run_id) / f"{name}_naive.jsonl"),
        seg=load_segmentation(segmentation_dir, name),
        seg_failures=load_segmentation_failures(segmentation_dir, name),
        token_len=token_len_for(model_id),
        max_chunk_tokens=args.chunk_tokens,
        pairing=local_pairing,
        global_index=arm_global,
        consolidate_model=get_chat_model(max_tokens=consolidate_tokens),
        notes_path=str(notes_path),
    )

    db = run_dir / ns["db_name"]
    with SqliteSaver.from_conn_string(str(db)) as checkpointer:
        graph = build_study_graph(deps, checkpointer=checkpointer)
        cfg = {"configurable": {"thread_id": ns["thread_id"]}}

        state = initial_study_state(name, manifest, str(notes_path),
                                    control_mode=not args.matn_only,
                                    matn_only_mode=args.matn_only)
        result = graph.invoke(state, cfg)

        while "__interrupt__" in result:
            # Control interrupts are auto-approved. Scholar review of the control arm
            # would let the reviewer's knowledge steer it, which is the contamination
            # the control exists to measure. Spec §4.1.
            result = graph.invoke(Command(resume={"action": "approve"}), cfg)

    # Isolation check: the two arms evolved their hypotheses independently, so an
    # identical final hypothesis means they were not actually isolated.
    real_method = (Path(args.real_method) if args.real_method
                   else ROOT / "outputs" / "method_files" / f"{name}_method.md")
    if real_method.exists():
        assert_isolated(real_method.read_text(encoding="utf-8"),
                        result.get("method_hypothesis", ""))
    else:
        print(f"Warning: {real_method} not found; isolation check skipped. "
              "Run run_study_graph.py first for a complete validation.")

    arm_label = "Matn-only notes" if args.matn_only else "Control notes"
    score_hint = (f"python scripts/score_control.py {name} --no-pairing"
                  if args.matn_only else f"python scripts/score_control.py {name}")
    print(f"{arm_label}: {notes_path}")
    print(f"Now run: {score_hint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
