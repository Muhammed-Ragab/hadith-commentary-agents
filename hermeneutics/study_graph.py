"""Study graph assembly. Spec §3.1, §5.

Defect #4: position lives in the checkpointer, not in a count of '## Bāb ' headings
in a markdown file. Resume is exact.

Design note: consolidate_kitab is called in finalize (after all bābs), NOT in
kitab_review.  This keeps the interrupt sheet cheap (no model call before the user
can answer) and ensures response ordering is deterministic for tests:
    PER_BAB * n_babs model calls, then kitab-summary calls, then global.
"""
from datetime import datetime, timezone
from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from hermeneutics.nodes import (
    Deps,
    accumulate,
    consolidate_global,
    consolidate_kitab,
    horizon_diff,
    naive_reading,
    provenance_ratio,
    read_commentary,
    select_next_bab,
    transmitted_sources,
)
from hermeneutics.rereads import reread_cap
from hermeneutics.state import StudyState

LOW_CONFIDENCE = 0.5
MAX_REVIEW_MOVES = 20  # an uncapped sheet is unreviewable in one sitting


def is_kitab_boundary(babs: list[dict], cursor: int, current_kitab: str) -> bool:
    """True when the next bāb starts a new kitāb, or the corpus is exhausted."""
    if cursor >= len(babs):
        return True
    return babs[cursor]["kitab"] != current_kitab


def kitab_order(babs: list[dict]) -> list[str]:
    """Distinct kitābs in corpus order."""
    seen: list[str] = []
    for b in babs:
        if not seen or seen[-1] != b["kitab"]:
            if b["kitab"] not in seen:
                seen.append(b["kitab"])
    return seen


def is_review_boundary(babs: list[dict], cursor: int, current_kitab: str,
                       group_size: int = 1) -> bool:
    """True when the scholar should be interrupted. Spec §5, handoff.md §10.6.

    Every kitāb boundary at group_size=1. At group_size=N, every Nth boundary --
    plus always the last, so no kitāb ever goes unreviewed. Grouping exists because
    the 2026-08-23 reparse took the corpus to 133 kitābs, and 133 sittings does not
    fit the calendar; it does not change what is reviewed, only when.
    """
    if not is_kitab_boundary(babs, cursor, current_kitab):
        return False
    if cursor >= len(babs) or group_size <= 1:
        return True
    order = kitab_order(babs)
    if current_kitab not in order:
        return True
    return (order.index(current_kitab) + 1) % group_size == 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def kitab_review(state: dict, deps: Deps) -> dict:
    """interrupt() at a kitāb boundary. Spec §5.

    Everything written before this point is already on disk (accumulate flushes per
    record), so aborting at the prompt loses nothing.

    No model call happens here: the interrupt sheet is built from already-computed
    state (notes, provenance ratio). consolidate_kitab is deferred to finalize so
    the full response sequence is: all bāb calls, then all consolidate calls.
    """
    kitab = state.get("current_kitab", "")
    # With grouped cadence a sitting covers every kitāb finished since the last
    # interrupt, not just the current one -- otherwise skipped kitābs are never seen.
    already = set(state.get("reviewed_kitabs", []))
    records = [r for r in state.get("notes", []) if r.get("kitab") not in already]
    pending = []
    for r in records:
        if r.get("kitab", "") not in pending:
            pending.append(r.get("kitab", ""))

    low_confidence = sorted(
        (
            {"bab_ref": r["bab_ref"], "operation": m["operation"],
             "confidence": m["confidence"], "evidence": m["arabic_evidence"]}
            for r in records for m in r.get("moves", [])
            if m.get("confidence", 1.0) < LOW_CONFIDENCE
        ),
        key=lambda m: m["confidence"],
    )[:MAX_REVIEW_MOVES]

    # The cap stays per kitāb and is summed over the group, so grouping changes the
    # sitting count without weakening bounded agency (spec §3.1).
    per_kitab = {k: sum(1 for r in records if r.get("kitab") == k) for k in pending}
    cap = sum(reread_cap(n) for n in per_kitab.values()) or reread_cap(len(records))

    decision = interrupt({
        "kitab": kitab,
        "kitabs": pending,
        "n_records": len(records),
        "provenance": provenance_ratio(records),
        "low_confidence_moves": low_confidence,
        "reread_cap": cap,
    })

    action = (decision or {}).get("action", "approve")
    intervention = {"kind": action if action in
                    ("approve", "edit", "flag_reread") else "approve",
                    "kitab": kitab, "kitabs": pending,
                    "payload": dict(decision or {}), "ts": _now()}

    flagged: list[int] = []
    if action == "flag_reread":
        flagged = list((decision or {}).get("bab_indices", []))[:cap]

    return {
        "notes": [],
        "reviewed_kitabs": pending,
        "re_read_queue": flagged,
        "intervention_log": [intervention],
    }


def finalize(state: dict, deps: Deps) -> dict:
    """Consolidate per-kitāb then globally. Defect #3.

    consolidate_kitab is called here (not in kitab_review) so the model-call
    ordering matches the test fixture: all bāb processing, then all consolidations.
    """
    notes = state.get("notes", [])
    by_kitab: dict[str, list[dict]] = {}
    for record in notes:
        k = record.get("kitab", "")
        by_kitab.setdefault(k, []).append(record)

    for kitab in sorted(by_kitab.keys()):
        summary = consolidate_kitab(kitab, by_kitab[kitab], deps)
        deps.kitab_summaries[kitab] = summary["summary"]

    # Counted over every move, not per kitāb: §5 of the method file is a claim about
    # the whole work. Discarding this was how §8.15's method files came to assert
    # "frequently references named predecessors" from 6 of 88 moves.
    return {"method_file": consolidate_global(deps.kitab_summaries, deps,
                                              provenance_ratio(notes),
                                              transmitted_sources(notes))}


def _route_after_select(state: dict) -> str:
    return "naive_reading" if state.get("current_bab") is not None else "finalize"


def _route_after_accumulate(state: dict, deps: Deps) -> str:
    if is_review_boundary(deps.babs, state.get("cursor", 0),
                          state.get("current_kitab", ""), deps.review_group_size):
        return "kitab_review"
    return "select_next_bab"


def _route_after_review(state: dict, deps: Deps) -> str:
    """Serve any pending re-read before moving on; otherwise continue or finish."""
    queue = state.get("re_read_queue", [])
    if state.get("re_read_cursor", 0) < len(queue):
        return "revisit_flagged"
    return "select_next_bab" if state.get("cursor", 0) < len(deps.babs) else "finalize"


def revisit_flagged(state: dict, deps: Deps) -> dict:
    """Re-read one flagged unit in light of the evolved hypothesis. Spec §3.1."""
    cursor = state.get("re_read_cursor", 0)
    index = state["re_read_queue"][cursor]
    bab = deps.babs[index]
    return {
        "re_read_cursor": cursor + 1,
        "current_bab": bab,
        "current_bab_ref": f"{deps.commentator}/{index}",
        "current_kitab": bab["kitab"],
        "commentary_text": bab["text"],
        "intervention_log": [{"kind": "agent_reread", "kitab": bab["kitab"],
                              "payload": {"bab_index": index}, "ts": _now()}],
    }


def build_study_graph(deps: Deps, checkpointer):
    """Assemble the graph. One checkpointer per process - never shared (spec §6.2)."""
    g = StateGraph(StudyState)

    g.add_node("select_next_bab", partial(select_next_bab, deps=deps))
    g.add_node("naive_reading", partial(naive_reading, deps=deps))
    g.add_node("read_commentary", partial(read_commentary, deps=deps))
    g.add_node("horizon_diff", partial(horizon_diff, deps=deps))
    g.add_node("accumulate", partial(accumulate, deps=deps))
    g.add_node("kitab_review", partial(kitab_review, deps=deps))
    g.add_node("revisit_flagged", partial(revisit_flagged, deps=deps))
    g.add_node("finalize", partial(finalize, deps=deps))

    g.add_edge(START, "select_next_bab")
    g.add_conditional_edges("select_next_bab", _route_after_select,
                            ["naive_reading", "finalize"])
    g.add_edge("naive_reading", "read_commentary")
    g.add_edge("read_commentary", "horizon_diff")
    g.add_edge("horizon_diff", "accumulate")
    g.add_conditional_edges("accumulate", partial(_route_after_accumulate, deps=deps),
                            ["kitab_review", "select_next_bab"])
    g.add_conditional_edges("kitab_review", partial(_route_after_review, deps=deps),
                            ["revisit_flagged", "select_next_bab", "finalize"])
    g.add_edge("revisit_flagged", "read_commentary")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
