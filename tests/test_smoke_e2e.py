"""End-to-end on a 10-bāb fixture with a fake model. No network, no GPU."""
import json

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from hermeneutics.grounding import grounding_rate
from hermeneutics.nodes import Deps
from hermeneutics.state import initial_study_state
from hermeneutics.study_graph import build_study_graph

EVIDENCE = "وهذا مخصوص بالمسافر"
COMMENTARY = f"قال الشارح {EVIDENCE} في السفر الطويل"

BABS = [{"kitab": f"K{i // 5}", "bab": f"b{i}", "text": COMMENTARY * 5} for i in range(10)]


def _responses():
    responses = []
    for i in range(10):
        evidence_id = f"nawawi/{i}:c000:s000"
        responses.extend([
            json.dumps({
                "status": "moves",
                "observations": [{
                    "operation": "takhsis", "evidence_id": evidence_id,
                    "provenance": {"author": "compiler_own", "quoted_from": []},
                    "confidence": 0.8,
                }],
            }),
            json.dumps({
                "status": "moves",
                "gap": "general vs restricted",
                "moves": [{
                    "operation": "scope_restriction", "evidence_id": evidence_id,
                    "arabic_evidence": "مخصوص بالمسافر",
                    "closes_gap_by": "restricts to travellers",
                    "provenance": {"author": "compiler_own", "quoted_from": []},
                    "confidence": 0.8,
                }],
                "revised_reading": "a restricted obligation",
            }),
        ])
    return responses + ["kitab summary"] * 2 + ["METHOD"]


def test_ten_bab_run_completes_and_produces_grounded_notes(fake_model, trace, tmp_path):
    deps = Deps(
        model=fake_model(_responses()), trace=trace, manifest_id="m1", babs=BABS,
        commentator="nawawi", naive={f"nawawi/{i}": "naive" for i in range(10)},
        seg={f"nawawi/{i}": 100 for i in range(10)},
        token_len=lambda s: max(1, len(s) // 4),
        notes_path=str(tmp_path / "notes.jsonl"),
    )
    graph = build_study_graph(deps, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "smoke"}}

    result = graph.invoke(initial_study_state("nawawi", {}, str(tmp_path / "notes.jsonl")), cfg)
    while "__interrupt__" in result:
        result = graph.invoke(Command(resume={"action": "approve"}), cfg)

    assert result["method_file"].startswith("METHOD")
    assert len(result["notes"]) == 10

    lines = (tmp_path / "notes.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 10

    records = [json.loads(x) for x in lines]
    sources = {f"nawawi/{i}": BABS[i]["text"] for i in range(10)}
    assert grounding_rate(records, sources) == 1.0


def test_every_call_reached_the_trace(fake_model, trace, tmp_path):
    """Spec §6.3: the record must be complete, not sampled."""
    deps = Deps(
        model=fake_model(_responses()), trace=trace, manifest_id="m1", babs=BABS,
        commentator="nawawi", naive={f"nawawi/{i}": "naive" for i in range(10)},
        seg={f"nawawi/{i}": 100 for i in range(10)},
        token_len=lambda s: max(1, len(s) // 4),
        notes_path=str(tmp_path / "notes.jsonl"),
    )
    graph = build_study_graph(deps, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "smoke2"}}
    result = graph.invoke(initial_study_state("nawawi", {}, str(tmp_path / "notes.jsonl")), cfg)
    while "__interrupt__" in result:
        result = graph.invoke(Command(resume={"action": "approve"}), cfg)

    traced = (tmp_path / "test.jsonl").read_text(encoding="utf-8").strip().split("\n")
    nodes = [json.loads(t)["node"] for t in traced]
    assert nodes.count("read_commentary") == 10
    assert nodes.count("horizon_diff") == 10
    assert nodes.count("self_check") == 0
    assert "consolidate_global" in nodes
