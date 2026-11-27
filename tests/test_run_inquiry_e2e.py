"""run_inquiry.main() end to end on a fake model.

Written BEFORE the setup/save code moved into hermeneutics/inquiry_session.py, so the
move is checked against what the CLI actually wrote, not against what we remember it
wrote. no_sources keeps it offline: no method file, no corpus tools.
"""
import json
import shutil
import sys
from pathlib import Path

from scripts import run_inquiry

POC = Path(__file__).resolve().parent.parent

SESSION_KEYS = {"question", "condition", "corpus", "commentator", "manifest_id",
                "method_run", "isnad_output", "matn_output", "synthesis_output",
                "intervention_log", "sources", "tool_calls_log", "grounding_log"}


def _fake_root(tmp_path):
    nets = tmp_path / "outputs" / "networks"
    nets.mkdir(parents=True)
    shutil.copy(POC / "outputs" / "networks" / "muslim_metrics.json", nets)
    return tmp_path


def test_cli_writes_the_session_json_it_always_wrote(tmp_path, monkeypatch, fake_model):
    root = _fake_root(tmp_path)
    monkeypatch.setattr(run_inquiry, "ROOT", root)
    monkeypatch.setattr(run_inquiry, "get_chat_model",
                        lambda **kw: fake_model(["isnad out", "matn out", "synthesis out"]))
    monkeypatch.setenv("VLLM_MODEL_ID", "fake-model")
    monkeypatch.setattr(sys, "argv", ["run_inquiry.py", "-q", "ما معنى النية",
                                      "--condition", "no_sources", "--auto-approve",
                                      "--run-id", "t1"])

    assert run_inquiry.main() == 0

    sessions = list((root / "outputs" / "sessions").glob("muslim_no_sources_*.json"))
    assert len(sessions) == 1
    s = json.loads(sessions[0].read_text(encoding="utf-8"))
    assert SESSION_KEYS <= set(s)
    assert s["question"] == "ما معنى النية"
    assert (s["commentator"], s["corpus"], s["condition"]) == ("nawawi", "muslim", "no_sources")
    assert (s["isnad_output"], s["matn_output"], s["synthesis_output"]) == (
        "isnad out", "matn out", "synthesis out")
    assert [e["kind"] for e in s["intervention_log"]] == ["auto_approve"] * 3

    manifest = json.loads((root / "outputs" / "runs" / "t1" / "manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["model_id"] == "fake-model"
    assert manifest["condition"] == "no_sources"
    assert manifest["corpus"] == "muslim"
    assert manifest["max_tokens"] == 4000
    assert s["manifest_id"] == run_inquiry.manifest_id(manifest)
    assert s["model_id"] == "fake-model"
    assert s["resumes"] == []
