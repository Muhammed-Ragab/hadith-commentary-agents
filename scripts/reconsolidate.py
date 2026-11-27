"""Re-run only the global consolidation, from a finished arm's trace.

`consolidate_global` is a single model call at the end of an arm. When it overruns
its output budget the method file loses its tail: matched_qwen3_20260917 wrote
Nawawī's file at 8192 tokens and lost §6 "Limits Observed" entirely, because
STUDY_MAX_OUTPUT_TOKENS raises the reading budget and STUDY_CONSOLIDATE_MAX_TOKENS
is a separate knob that was left at its default.

Re-running the arm to recover one call costs hours. The per-kitāb summaries that
call consumes live on `Deps`, not in the checkpointer, so they do not survive the
process -- but every `consolidate_kitab` call was traced with its response, so they
can be read back out. This redoes the last step alone against the same inputs.

  STUDY_CONSOLIDATE_MAX_TOKENS=16384 python scripts/reconsolidate.py nawawi \
      --trace outputs/runs/<tag>_nawawi/study_nawawi.jsonl \
      --notes outputs/method_files_matched/<tag>/nawawi/nawawi_notes.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from hermeneutics.llm import get_chat_model
from hermeneutics.nodes import (consolidate_global, provenance_ratio,
                                transmitted_sources)
from hermeneutics.trace import TraceWriter


def kitab_summaries_from_trace(trace_path: Path, commentator: str) -> dict[str, str]:
    """Rebuild deps.kitab_summaries from the traced consolidate_kitab calls.

    A kitāb whose notes did not fit one chunk was summarised in several calls and the
    node joined the partials with a blank line, so every successful response for a
    kitāb is kept, in file order. Failed attempts carry `error` and are dropped, which
    is what keeps a retry's discarded first answer out of the summary.
    """
    summaries: dict[str, list[str]] = {}
    with open(trace_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("node") != "consolidate_kitab" or rec.get("error"):
                continue
            response = rec.get("response") or ""
            if not response.strip():
                continue
            kitab = rec.get("bab_ref", "").removeprefix(f"{commentator}/")
            summaries.setdefault(kitab, []).append(response)
    return {k: "\n\n".join(v) for k, v in summaries.items()}


def load_records(notes_path: Path) -> list[dict]:
    with open(notes_path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("commentator")
    p.add_argument("--trace", required=True, help="study_<arm>.jsonl from the finished run")
    p.add_argument("--notes", required=True, help="<arm>_notes.jsonl from the finished run")
    p.add_argument("--out", help="method file to write (default: alongside --notes)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what was recovered and make no model call")
    return p


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    name = args.commentator

    summaries = kitab_summaries_from_trace(Path(args.trace), name)
    if not summaries:
        raise SystemExit(f"no consolidate_kitab responses for {name} in {args.trace}")

    records = load_records(Path(args.notes))
    provenance = provenance_ratio(records)
    sources = transmitted_sources(records)

    print(f"{name}: {len(summaries)} kitāb summaries recovered, {len(records)} notes, "
          f"{provenance['compiler_own']}/{provenance['compiler_own'] + provenance['transmitted']} "
          f"moves in the compiler's own voice")
    for kitab in sorted(summaries):
        print(f"  {len(summaries[kitab]):>6} chars  {kitab}")
    if args.dry_run:
        return 0

    out = Path(args.out) if args.out else Path(args.notes).with_name(f"{name}_method.md")
    budget = int(os.environ.get("STUDY_CONSOLIDATE_MAX_TOKENS", "16384"))
    deps = SimpleNamespace(
        model=None,
        consolidate_model=get_chat_model(max_tokens=budget),
        commentator=name,
        trace=TraceWriter(run_dir=Path(args.trace).parent, phase=f"reconsolidate_{name}"),
        manifest_id=records[0].get("manifest_id", "") if records else "",
    )

    method_file = consolidate_global(summaries, deps, provenance, sources)
    out.write_text(method_file, encoding="utf-8")
    print(f"\nWrote {out} ({len(method_file)} chars, budget {budget} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
