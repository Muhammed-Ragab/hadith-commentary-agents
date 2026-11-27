"""Sequential study phase. Spec §6.2 phase 2.

One process per commentator. Each owns its own SQLite checkpointer file - a shared
checkpointer across concurrent writers is the wrong persistence model and is exactly
what the SLURM bāb-sharding plan got wrong.

Usage:
  python scripts/run_study_graph.py nawawi
  python scripts/run_study_graph.py cayni --limit 10   # smoke run
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from hermeneutics.chunking import DEFAULT_MAX_TOKENS, token_len_for
from hermeneutics.llm import get_chat_model
from hermeneutics.manifest import build_manifest, manifest_id, write_manifest
from hermeneutics.nodes import Deps
from hermeneutics.state import initial_study_state
from hermeneutics.study_graph import build_study_graph
from hermeneutics.trace import TraceWriter
from scripts.check_corpus import (COMMENTATORS, load_babs, parse_sample,
                                  sample_indices)

ROOT = Path(__file__).resolve().parent.parent

# The review sheet prints kitāb names in Arabic. A Windows console defaults to cp1252
# and raises UnicodeEncodeError at the first interrupt, killing the run mid-study.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# handoff.md §10.6: the 2026-08-23 reparse took the corpus to 133 kitābs. Nawawi is
# the closely-read case and keeps per-kitāb review; ʿAynī groups 5 kitābs per sitting,
# bringing the total from 133 to ~64.
REVIEW_EVERY = {"nawawi": 1, "cayni": 5, "mazandarani": 1}


def model_output_budgets() -> tuple[int, int]:
    """Output caps, raised from 1500/4000 after job 15849569.

    On a thinking model the cap bounds reasoning + answer, not the answer, so
    1500 was spent before read_commentary emitted a token (LengthFinishReasonError
    at completion_tokens=1500, prompt_tokens=6542). The server runs at
    --max-model-len 32768, so 6.5k of prompt plus 8k of output leaves headroom.
    """
    return (
        int(os.environ.get("STUDY_MAX_OUTPUT_TOKENS", "8192")),
        int(os.environ.get("STUDY_CONSOLIDATE_MAX_TOKENS", "8192")),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the sequential study phase.")
    p.add_argument("commentator", choices=COMMENTATORS)
    p.add_argument("--limit", type=int, default=None, help="process only the first N bābs")
    p.add_argument("--sample", default=None, help="stratified sample KITABS:PER_KITAB:SEED (e.g. 12:5:20260914), instead of --limit's first N babs. Every phase of a run must be given the SAME spec.")
    p.add_argument("--run-id", default=None, help="defaults to the manifest id")
    p.add_argument(
        "--chunk-tokens",
        type=int,
        default=int(os.environ.get("STUDY_CHUNK_TOKENS", DEFAULT_MAX_TOKENS)),
    )
    p.add_argument("--segmentation-dir", default=None,
                   help="segmentation JSONL directory; defaults to outputs/segmentation")
    p.add_argument("--naive-dir", default=None,
                   help="naive-reading JSONL directory; defaults to outputs/naive/<run-id>, "
                        "matching where run_naive_batch.py writes for the same --run-id")
    p.add_argument("--live-naive", action="store_true",
                   help="ignore cached naïve readings and compute them with the active model")
    p.add_argument("--review-every", type=int, default=None,
                   help="interrupt the scholar every Nth kitāb. Defaults to 1 for "
                        "nawawi (46 kitābs) and 5 for cayni (87), which keeps total "
                        "sittings near the pre-reparse budget. See handoff.md §10.6.")
    p.add_argument("--control", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--notes-dir", default=None,
                   help="where to write notes/method files; defaults to outputs/method_files. "
                        "Point candidates at separate dirs during the bake-off so they do not "
                        "overwrite each other.")
    p.add_argument("--auto-approve", action="store_true",
                   help="approve every kitāb interrupt without asking. For the bake-off and "
                        "smoke runs only - a real study run needs the scholar (spec §5).")
    # Ablation toggles for the two saturation-curve mechanisms (run 15789363, see
    # hermeneutics/nodes.py Deps.legacy_example_prompt / legacy_hypothesis_menu). Both
    # default off, i.e. a normal run gets the fix without asking. scripts/run_ablation.sh
    # flips these on one at a time over the same arm to measure each mechanism's effect.
    p.add_argument("--legacy-example-prompt", action="store_true",
                   help="ablation only: use the pre-fix read_commentary prompt, whose "
                        "worked example prints a live 'lexical analysis' value instead "
                        "of a placeholder (mechanism 1). Off by default.")
    p.add_argument("--legacy-hypothesis-menu", action="store_true",
                   help="ablation only: render the method hypothesis as the pre-fix "
                        "counted, frequency-ranked menu instead of an alphabetical, "
                        "uncounted list (mechanism 2). Off by default.")
    return p


def require_naive(path: Path) -> dict[str, str]:
    """Read the batch-phase naive readings, or say which file is missing.

    B3 keys these by run, and a wrong or absent run id used to degrade twice over in
    silence: first to another run's readings at the old shared path, then -- if that
    was gone too -- to an empty dict, which the study graph reads as "compute the
    naive reading inline". Inline means with the method hypothesis already in play,
    which is a different experimental condition (spec §6.2 phase 1b) arrived at by
    accident. `--live-naive` is how that is asked for on purpose.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"no batch naive readings at {path}. Run run_naive_batch.py with the same "
            "--run-id, point --naive-dir at the directory it wrote, or pass "
            "--live-naive to compute them inline on purpose."
        )
    return load_naive(path)


def load_naive(path: Path) -> dict[str, str]:
    """Read the batch-phase naive readings. Empty dict means 'compute live'."""
    if not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["bab_ref"]] = rec["naive_reading"]
    return out


def load_segmentation(directory: Path, commentator: str) -> dict[str, tuple[int, int]]:
    """Merge this commentator's segmentation shards; later duplicates win.

    Values are (matn_start_char, matn_end_char). A file written before the span
    existed has no start, and 0 is what it meant.

    The glob has a literal dot after "_seg": B4's "{commentator}_seg_failures.*"
    shards live in this same directory, and `_seg*.jsonl` (no dot) used to catch
    those too -- reading one as a boundary record raised KeyError on the missing
    matn_end_char.
    """
    out = {}
    for path in sorted(Path(directory).glob(f"{commentator}_seg.*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["bab_ref"]] = (int(rec.get("matn_start_char", 0)),
                                       int(rec["matn_end_char"]))
    return out


def load_segmentation_failures(directory: Path, commentator: str) -> frozenset[str]:
    """Refs segment_one gave up on (B4) -- run_segmentation.py logs these instead of
    writing a seg record, so they never appear in load_segmentation() either. Passed
    to Deps.seg_failures so select_next_bab (hermeneutics/nodes.py) skips them
    instead of letting _segmentation_span raise on a ref that was never written.
    """
    out = set()
    for path in sorted(Path(directory).glob(f"{commentator}_seg_failures.*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.add(json.loads(line)["bab_ref"])
    return frozenset(out)


def render_review_sheet(sheet: dict) -> str:
    """Terminal review sheet. Spec §5."""
    prov = sheet["provenance"]
    covered = sheet.get("kitabs") or [sheet["kitab"]]
    title = covered[0] if len(covered) == 1 else f"{len(covered)} kitābs: " + " · ".join(covered)
    lines = [
        f"# Kitāb review: {title}",
        "",
        f"Records: {sheet['n_records']}   Re-read cap: {sheet['reread_cap']}",
        f"Provenance: {prov['compiler_own']} own / {prov['transmitted']} transmitted "
        f"({prov['compiler_own_pct']}% own)",
        "",
        "## Mini-consolidation",
        sheet.get("summary", "(deferred to method file)"),
        "",
        f"## Low-confidence moves ({len(sheet['low_confidence_moves'])})",
    ]
    for m in sheet["low_confidence_moves"]:
        lines.append(f"- [{m['bab_ref']}] {m['operation']} "
                     f"(conf {m['confidence']}) — {m['evidence'][:80]}")
    return "\n".join(lines)


def prompt_scholar(sheet: dict) -> dict:
    """Read the scholar's decision from the terminal."""
    print("\n" + render_review_sheet(sheet) + "\n")
    action = input("approve / edit / flag_reread [approve]: ").strip() or "approve"
    if action == "edit":
        return {"action": "edit", "note": input("note: ").strip()}
    if action == "flag_reread":
        raw = input(f"bāb indices to re-read (max {sheet['reread_cap']}, comma-separated): ")
        indices = [int(x) for x in raw.replace(" ", "").split(",") if x]
        return {"action": "flag_reread", "bab_indices": indices}
    return {"action": "approve"}


def has_checkpoint(checkpointer, cfg) -> bool:
    """True when this thread_id already carries saved history.

    Handing graph.invoke() a fresh initial_study_state on such a thread restarts the
    graph from the first bāb and, because notes open in append mode, writes duplicates
    of everything the earlier attempt already recorded. Measured on the ʿAynī arm
    2026-09-18: the rerun's first write was a bāb already in the file. `None` resumes.
    """
    return checkpointer.get(cfg) is not None


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    name = args.commentator

    babs = load_babs(name)
    # The graph walks `babs` positionally but `bab_ref` must stay the GLOBAL index, or
    # the study's refs stop matching the segmentation and naive records keyed on them.
    # Deps.global_index already exists for exactly this (the control arm uses it).
    global_index = None
    if args.sample:
        picked = sample_indices(babs, *parse_sample(args.sample))
        babs = [babs[i] for i in picked]
        global_index = picked
    elif args.limit:
        babs = babs[: args.limit]

    model_id = os.environ["VLLM_MODEL_ID"]
    manifest = build_manifest(commentator=name, model_id=model_id)
    run_id = args.run_id or manifest_id(manifest)
    phase = "control" if args.control else "study"

    run_dir = ROOT / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, run_dir)

    notes_name = f"{name}_control_notes.jsonl" if args.control else f"{name}_notes.jsonl"
    notes_dir = Path(args.notes_dir) if args.notes_dir else ROOT / "outputs" / "method_files"
    notes_path = notes_dir / notes_name
    notes_path.parent.mkdir(parents=True, exist_ok=True)

    segmentation_dir = (Path(args.segmentation_dir) if args.segmentation_dir
                        else ROOT / "outputs" / "segmentation")
    naive_dir = Path(args.naive_dir) if args.naive_dir else ROOT / "outputs" / "naive" / run_id
    # No fallback to the old shared outputs/naive/<name>_naive.jsonl: those files are
    # still on disk from earlier runs, so falling back to them is cross-run
    # contamination wearing backwards compatibility's clothes -- the defect B3 exists
    # to remove.
    naive_path = naive_dir / f"{name}_naive.jsonl"
    output_tokens, consolidate_tokens = model_output_budgets()
    deps = Deps(
        model=get_chat_model(max_tokens=output_tokens),
        trace=TraceWriter(run_dir=run_dir, phase=f"{phase}_{name}"),
        manifest_id=manifest_id(manifest),
        babs=babs,
        commentator=name,
        naive={} if args.live_naive else require_naive(naive_path),
        seg=load_segmentation(segmentation_dir, name),
        seg_failures=load_segmentation_failures(segmentation_dir, name),
        token_len=token_len_for(model_id),
        max_chunk_tokens=args.chunk_tokens,
        global_index=global_index,
        review_group_size=args.review_every or REVIEW_EVERY[name],
        consolidate_model=get_chat_model(max_tokens=consolidate_tokens),
        notes_path=str(notes_path),
        legacy_example_prompt=args.legacy_example_prompt,
        legacy_hypothesis_menu=args.legacy_hypothesis_menu,
    )

    db = run_dir / f"{phase}_{name}.sqlite"
    with SqliteSaver.from_conn_string(str(db)) as checkpointer:
        graph = build_study_graph(deps, checkpointer=checkpointer)
        cfg = {"configurable": {"thread_id": f"{phase}_{name}"}}

        if has_checkpoint(checkpointer, cfg):
            print(f"[resume] {cfg['configurable']['thread_id']} has a checkpoint in {db.name}")
            state = None
        else:
            state = initial_study_state(name, manifest, str(notes_path),
                                        control_mode=args.control)
        result = graph.invoke(state, cfg)

        while "__interrupt__" in result:
            sheet = result["__interrupt__"][0].value
            if args.auto_approve:
                print(f"[auto-approve] kitāb {sheet['kitab']} ({sheet['n_records']} records)")
                decision = {"action": "approve"}
            else:
                decision = prompt_scholar(sheet)
            result = graph.invoke(Command(resume=decision), cfg)

    method_path = notes_dir / f"{name}_method.md"
    method_path.parent.mkdir(parents=True, exist_ok=True)
    method_path.write_text(result.get("method_file", ""), encoding="utf-8")
    print(f"\nMethod file: {method_path}\nNotes: {notes_path}\nTraces: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
