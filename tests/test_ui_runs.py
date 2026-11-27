"""Background runs for the UI, driven directly -- no HTTP."""
import json
import shutil
import time
from pathlib import Path

import pytest

from hermeneutics.inquiry_session import RunParams
from scripts.ui_runs import RunBusy, RunManager, list_sessions, read_trace, summarise_trace

POC = Path(__file__).resolve().parent.parent


def _root(tmp_path):
    nets = tmp_path / "outputs" / "networks"
    nets.mkdir(parents=True)
    shutil.copy(POC / "outputs" / "networks" / "muslim_metrics.json", nets)
    return tmp_path


def _params(**kw):
    base = {"question": "ما معنى النية", "condition": "no_sources", "model_id": "fake-model"}
    base.update(kw)
    return RunParams(**base)


def wait_for(run, statuses, timeout=15):
    deadline = time.monotonic() + timeout
    while run.status not in statuses:
        if time.monotonic() > deadline:
            raise AssertionError(f"run stuck at {run.status}: {run.error}")
        time.sleep(0.02)
    return run


def _session(run):
    return json.loads(Path(run.session_path).read_text(encoding="utf-8"))


def test_a_gated_run_waits_at_each_stage_and_records_the_note(tmp_path, fake_model):
    mgr = RunManager(lambda p: fake_model(["isnad out", "matn out", "synthesis out"]),
                      root=_root(tmp_path), out_root=tmp_path)
    run = mgr.start(_params())

    wait_for(run, {"awaiting_review"})
    assert (run.stage, run.output) == ("isnad", "isnad out")
    mgr.decide(run.run_id, "approve")
    wait_for(run, {"awaiting_review"})
    assert run.stage == "matn"
    mgr.decide(run.run_id, "edit", "check the chain")
    wait_for(run, {"awaiting_review"})
    assert run.stage == "synthesis"
    mgr.decide(run.run_id, "approve")
    wait_for(run, {"done"})

    s = _session(run)
    assert [e["kind"] for e in s["intervention_log"]] == ["approve", "edit", "approve"]
    assert s["intervention_log"][1]["payload"]["note"] == "check the chain"
    saved = json.loads((run.run_dir / "ui_run.json").read_text(encoding="utf-8"))
    assert saved["status"] == "done"


def test_auto_approve_runs_straight_through(tmp_path, fake_model):
    mgr = RunManager(lambda p: fake_model(["i", "m", "s"]),
                      root=_root(tmp_path), out_root=tmp_path)
    run = wait_for(mgr.start(_params(auto_approve=True)), {"done"})
    assert [e["kind"] for e in _session(run)["intervention_log"]] == ["auto_approve"] * 3


def test_one_run_at_a_time(tmp_path, fake_model):
    mgr = RunManager(lambda p: fake_model(["i", "m", "s"]),
                      root=_root(tmp_path), out_root=tmp_path)
    run = mgr.start(_params())
    wait_for(run, {"awaiting_review"})
    with pytest.raises(RunBusy):
        mgr.start(_params())


def test_a_failed_run_resumes_from_its_checkpoint(tmp_path, fake_model, failing_model):
    models = iter([failing_model(["isnad out"]), fake_model(["matn out", "synthesis out"])])
    mgr = RunManager(lambda p: next(models), root=_root(tmp_path), out_root=tmp_path)

    run = wait_for(mgr.start(_params(auto_approve=True)), {"failed"})
    assert "ConnectionError" in run.error
    assert "does not support tools" in run.error

    mgr.resume(run.run_id)
    wait_for(run, {"done"})

    s = _session(run)
    assert (s["isnad_output"], s["synthesis_output"]) == ("isnad out", "synthesis out")
    assert len(s["resumes"]) == 1


def test_resume_refuses_a_run_that_did_not_stop(tmp_path, fake_model):
    mgr = RunManager(lambda p: fake_model(["i", "m", "s"]),
                      root=_root(tmp_path), out_root=tmp_path)
    run = wait_for(mgr.start(_params(auto_approve=True)), {"done"})
    with pytest.raises(ValueError):
        mgr.resume(run.run_id)


def test_a_run_left_active_by_a_dead_server_comes_back_interrupted(tmp_path, fake_model):
    run_dir = tmp_path / "outputs" / "runs" / "ui_old"
    run_dir.mkdir(parents=True)
    (run_dir / "ui_run.json").write_text(json.dumps({
        "run_id": "ui_old", "params": _params(run_id="ui_old").to_dict(),
        "stamp": "20260922T100000", "manifest": None, "status": "awaiting_review",
        "stage": "matn"}), encoding="utf-8")

    mgr = RunManager(lambda p: fake_model([]), root=_root(tmp_path), out_root=tmp_path)

    assert mgr.get("ui_old").status == "interrupted"
    assert mgr.active() is None


def test_cancel_at_a_gate_writes_no_session(tmp_path, fake_model):
    mgr = RunManager(lambda p: fake_model(["i", "m", "s"]),
                      root=_root(tmp_path), out_root=tmp_path)
    run = mgr.start(_params())
    wait_for(run, {"awaiting_review"})
    mgr.cancel(run.run_id)
    wait_for(run, {"cancelled"})
    assert not list((tmp_path / "outputs" / "sessions").glob("*.json"))


def test_read_trace_returns_only_complete_lines_after_the_offset(tmp_path):
    p = tmp_path / "t.jsonl"

    def line(node):
        return json.dumps({"node": node, "tool_calls": []}) + "\n"

    p.write_text(line("a") + line("b") + '{"node": "c"', encoding="utf-8")

    rows, off = read_trace(p, 0)
    assert [r["node"] for r in rows] == ["a", "b"]

    with open(p, "a", encoding="utf-8") as fh:
        fh.write(', "tool_calls": []}\n')
    rows, off2 = read_trace(p, off)
    assert [r["node"] for r in rows] == ["c"]
    assert read_trace(p, off2) == ([], off2)
    assert read_trace(None, 0) == ([], 0)


def test_a_tool_round_shows_its_tool_and_arguments():
    msg = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "1", "function": {"name": "search_sources",
                                  "arguments": json.dumps({"query": "النية"}, ensure_ascii=False)}}]}
    row = summarise_trace({"node": "inquiry_matn", "tool_calls": ["search_sources"],
                            "response": json.dumps(msg, ensure_ascii=False),
                            "tokens_in": 10, "tokens_out": 5, "latency_s": 1.2})
    assert row["tools"] == [{"name": "search_sources", "args": '{"query": "النية"}'}]
    assert row["preview"] == ""


def test_list_sessions_reads_newest_first(tmp_path):
    d = tmp_path / "outputs" / "sessions"
    d.mkdir(parents=True)
    old = d / "bukhari_agent_20260921T201359.json"
    new = d / "kafi_agent_method_20260922T101500.json"
    old.write_text(json.dumps({"question": "q1", "commentator": "cayni",
                                "condition": "agent"}), encoding="utf-8")
    time.sleep(0.05)
    new.write_text(json.dumps({"question": "q2", "commentator": "mazandarani",
                                "condition": "agent_method", "model_id": "m"}),
                   encoding="utf-8")

    rows = list_sessions(tmp_path)

    assert [r["stem"] for r in rows] == [new.stem, old.stem]
    assert rows[0]["date"] == "2026-09-22 10:15"
    assert rows[1]["model_id"] == ""
