import json

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from hermeneutics.nodes import Deps
from hermeneutics.state import initial_study_state
from hermeneutics.study_graph import build_study_graph, is_kitab_boundary

def _reading(index):
    return json.dumps({
        "status": "moves",
        "observations": [{
            "operation": "takhsis",
            "evidence_id": f"nawawi/{index}:c000:s000",
            "provenance": {"author": "compiler_own", "quoted_from": []},
            "confidence": 0.8,
        }],
    })


def _diff(index):
    return json.dumps({
        "status": "moves",
        "gap": "g",
        "moves": [{
            "operation": "scope_restriction",
            "evidence_id": f"nawawi/{index}:c000:s000",
            "arabic_evidence": "ع" * 10,
            "closes_gap_by": "c",
            "provenance": {"author": "compiler_own", "quoted_from": []},
            "confidence": 0.8,
        }],
        "revised_reading": "r",
    })


def _per_bab(index):
    return [_reading(index), _diff(index)]

BABS = [
    {"kitab": "K1", "bab": "b0", "text": "ا" * 100 + "ع" * 300},
    {"kitab": "K1", "bab": "b1", "text": "ب" * 100 + "ع" * 300},
    {"kitab": "K2", "bab": "b2", "text": "ج" * 100 + "ع" * 300},
]

def _deps(fake_model, trace, tmp_path, n_babs=3, extra=()):
    responses = [response for i in range(n_babs) for response in _per_bab(i)] + list(extra)
    return Deps(model=fake_model(responses), trace=trace, manifest_id="m1", babs=BABS,
                commentator="nawawi",
                naive={f"nawawi/{i}": "naive" for i in range(len(BABS))},
                seg={f"nawawi/{i}": 100 for i in range(len(BABS))},
                token_len=lambda s: max(1, len(s) // 4),
                notes_path=str(tmp_path / "notes.jsonl"))


def test_is_kitab_boundary_detects_the_change():
    assert is_kitab_boundary(BABS, cursor=2, current_kitab="K1") is True
    assert is_kitab_boundary(BABS, cursor=1, current_kitab="K1") is False


def test_end_of_corpus_counts_as_a_boundary():
    assert is_kitab_boundary(BABS, cursor=3, current_kitab="K2") is True


def test_graph_interrupts_at_the_first_kitab_boundary(fake_model, trace, tmp_path):
    deps = _deps(fake_model, trace, tmp_path)
    graph = build_study_graph(deps, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}

    result = graph.invoke(initial_study_state("nawawi", {}, str(tmp_path / "notes.jsonl")), cfg)
    assert "__interrupt__" in result

    sheet = result["__interrupt__"][0].value
    assert sheet["kitab"] == "K1"
    assert sheet["n_records"] == 2
    assert sheet["reread_cap"] == 1          # 2 bābs -> max(1, round(0.2)) == 1
    assert "provenance" in sheet


def test_approving_resumes_and_finishes_the_corpus(fake_model, trace, tmp_path):
    deps = _deps(fake_model, trace, tmp_path,
                 extra=["kitab summary", "kitab summary", "THE METHOD"])
    graph = build_study_graph(deps, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t2"}}

    graph.invoke(initial_study_state("nawawi", {}, str(tmp_path / "notes.jsonl")), cfg)
    graph.invoke(Command(resume={"action": "approve"}), cfg)   # end of K1
    final = graph.invoke(Command(resume={"action": "approve"}), cfg)  # end of K2

    assert final["method_file"].startswith("THE METHOD")
    assert len(final["notes"]) == 3


def test_scholar_edits_are_recorded_as_interventions(fake_model, trace, tmp_path):
    deps = _deps(fake_model, trace, tmp_path)
    graph = build_study_graph(deps, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t3"}}

    graph.invoke(initial_study_state("nawawi", {}, str(tmp_path / "notes.jsonl")), cfg)
    result = graph.invoke(
        Command(resume={"action": "edit", "note": "takhsis is over-reported here"}), cfg
    )
    kinds = [i["kind"] for i in result["intervention_log"]]
    assert "edit" in kinds


def test_flagged_rereads_land_in_the_queue_and_are_capped(fake_model, trace, tmp_path):
    deps = _deps(fake_model, trace, tmp_path)
    graph = build_study_graph(deps, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t4"}}

    graph.invoke(initial_study_state("nawawi", {}, str(tmp_path / "notes.jsonl")), cfg)
    result = graph.invoke(
        Command(resume={"action": "flag_reread", "bab_indices": [0, 1, 0, 1, 0, 1]}), cfg
    )
    assert len(result["re_read_queue"]) == 1  # cap for a 2-bāb kitāb


def test_flagged_reread_is_executed_on_resume(fake_model, trace, tmp_path):
    """The re-read loop actually runs: revisit_flagged → read_commentary path.

    After flagging bab 0 for re-read, the resumed graph should:
    1. Enter revisit_flagged (sets re_read_cursor=1)
    2. Execute read_commentary + horizon_diff + accumulate for bab 0 again
    3. Then continue to K2 babs
    4. Interrupt again at end of K2
    """
    responses = _per_bab(0) + _per_bab(1) + _per_bab(0) + _per_bab(2)
    deps = _deps(fake_model, trace, tmp_path, n_babs=0, extra=responses)
    graph = build_study_graph(deps, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t6"}}

    # First run: processes K1 (babs 0,1), interrupts at K1 boundary
    graph.invoke(initial_study_state("nawawi", {}, str(tmp_path / "notes.jsonl")), cfg)

    # Flag bab 0 for re-read (cap=1 for 2-bāb kitāb)
    result = graph.invoke(
        Command(resume={"action": "flag_reread", "bab_indices": [0]}), cfg
    )

    # The intervention_log should contain an "agent_reread" entry from revisit_flagged
    kinds = [i["kind"] for i in result["intervention_log"]]
    assert "agent_reread" in kinds

    # re_read_cursor should have advanced past the queued item
    assert result.get("re_read_cursor", 0) >= 1


def test_notes_are_on_disk_before_the_interrupt(fake_model, trace, tmp_path):
    """Spec §5: state is checkpointed before each interrupt - abort never loses work."""
    deps = _deps(fake_model, trace, tmp_path)
    graph = build_study_graph(deps, checkpointer=MemorySaver())
    graph.invoke(initial_study_state("nawawi", {}, str(tmp_path / "notes.jsonl")),
                 {"configurable": {"thread_id": "t5"}})
    lines = (tmp_path / "notes.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_failed_bab_is_logged_and_the_graph_continues(fake_model, trace, tmp_path):
    bad = json.dumps({"status": "moves", "gap": "g", "moves": "not a list",
                      "revised_reading": "r"})
    responses = [_reading(0), bad, bad] + _per_bab(1)
    deps = _deps(fake_model, trace, tmp_path, n_babs=0, extra=responses)
    graph = build_study_graph(deps, checkpointer=MemorySaver())

    result = graph.invoke(
        initial_study_state("nawawi", {}, str(tmp_path / "notes.jsonl")),
        {"configurable": {"thread_id": "failed-bab"}},
    )

    assert "__interrupt__" in result
    assert [record["bab_ref"] for record in result["notes"]] == ["nawawi/1"]
    failures = [
        json.loads(line)
        for line in (tmp_path / "notes_failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert failures[0]["bab_ref"] == "nawawi/0"
