"""The HTTP layer over RunManager, through FastAPI's TestClient."""
import shutil
import time
from pathlib import Path

from fastapi.testclient import TestClient

from hermeneutics.inquiry_session import ROOT
from scripts.serve_ui import build_manager, create_app
from scripts.ui_runs import RunManager

POC = Path(__file__).resolve().parent.parent


def _client(tmp_path, fake_model, models=(["llama", "qwen3.8-27b"], None)):
    nets = tmp_path / "outputs" / "networks"
    nets.mkdir(parents=True)
    shutil.copy(POC / "outputs" / "networks" / "muslim_metrics.json", nets)
    calls = {"models": 0}

    def models_fn():
        calls["models"] += 1
        return models

    mgr = RunManager(lambda p: fake_model(["isnad out", "matn out", "synthesis out"]),
                     root=tmp_path, out_root=tmp_path)
    return TestClient(create_app(mgr, models_fn)), calls


def _poll(client, run_id, statuses, timeout=15):
    deadline = time.monotonic() + timeout
    while True:
        r = client.get(f"/api/runs/{run_id}").json()
        if r["status"] in statuses:
            return r
        assert time.monotonic() < deadline, f"stuck at {r['status']}: {r['error']}"
        time.sleep(0.05)


START = {"question": "ما معنى النية", "condition": "no_sources", "model_id": "fake-model"}


def test_index_is_served(tmp_path, fake_model):
    client, _ = _client(tmp_path, fake_model)
    r = client.get("/")
    assert r.status_code == 200
    assert "Query-Agent" in r.text


def test_models_are_cached_until_refreshed(tmp_path, fake_model):
    client, calls = _client(tmp_path, fake_model)
    assert client.get("/api/models").json() == {
        "models": ["llama", "qwen3.8-27b"], "default": "qwen3.8-27b", "warning": None}
    client.get("/api/models")
    assert calls["models"] == 1
    client.get("/api/models?refresh=1")
    assert calls["models"] == 2


def test_models_warning_is_passed_through(tmp_path, fake_model):
    client, _ = _client(tmp_path, fake_model, models=(["qwen3.8-27b"], "SAIA down"))
    assert client.get("/api/models").json()["warning"] == "SAIA down"


def test_a_full_gated_run_over_http(tmp_path, fake_model):
    client, _ = _client(tmp_path, fake_model)
    run = client.post("/api/runs", json=START).json()

    r = _poll(client, run["run_id"], {"awaiting_review"})
    assert (r["stage"], r["output"]) == ("isnad", "isnad out")
    assert any(row["node"] == "inquiry_isnad" for row in r["trace"])
    again = client.get(f"/api/runs/{run['run_id']}?trace_offset={r['trace_offset']}").json()
    assert again["trace"] == []

    client.post(f"/api/runs/{run['run_id']}/decision", json={"action": "approve"})
    _poll(client, run["run_id"], {"awaiting_review"})
    client.post(f"/api/runs/{run['run_id']}/decision",
                json={"action": "edit", "note": "check the chain"})
    _poll(client, run["run_id"], {"awaiting_review"})
    client.post(f"/api/runs/{run['run_id']}/decision", json={"action": "approve"})
    done = _poll(client, run["run_id"], {"done"})

    sessions = client.get("/api/sessions").json()["sessions"]
    assert sessions[0]["stem"] == done["session_stem"]
    assert sessions[0]["model_id"] == "fake-model"
    page = client.get(f"/sessions/{done['session_stem']}.html")
    assert page.status_code == 200
    assert "<html" in page.text.lower()


def test_second_start_is_refused_while_a_run_is_active(tmp_path, fake_model):
    client, _ = _client(tmp_path, fake_model)
    run = client.post("/api/runs", json=START).json()
    _poll(client, run["run_id"], {"awaiting_review"})
    assert client.post("/api/runs", json=START).status_code == 409
    assert client.get("/api/runs/active").json()["run"]["run_id"] == run["run_id"]


def test_bad_input_is_a_400(tmp_path, fake_model):
    client, _ = _client(tmp_path, fake_model)
    assert client.post("/api/runs", json={**START, "question": "  "}).status_code == 400
    assert client.post("/api/runs", json={**START, "condition": "B"}).status_code == 400
    assert client.post("/api/runs", json={**START, "commentator": "x"}).status_code == 400


def test_decision_on_a_run_that_is_not_waiting_is_a_409(tmp_path, fake_model):
    client, _ = _client(tmp_path, fake_model)
    run = client.post("/api/runs", json={**START, "auto_approve": True}).json()
    _poll(client, run["run_id"], {"done"})
    r = client.post(f"/api/runs/{run['run_id']}/decision", json={"action": "approve"})
    assert r.status_code == 409


def test_unknown_run_and_path_tricks_are_404(tmp_path, fake_model):
    client, _ = _client(tmp_path, fake_model)
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/sessions/..%2F..%2Fsecret.html").status_code == 404
    assert client.get("/sessions/missing.html").status_code == 404


def test_fake_manager_writes_outside_research_sessions():
    """--fake must never write into outputs/runs or outputs/sessions beside real runs."""
    manager, _ = build_manager(True)
    assert manager.out_root == ROOT / "outputs" / "ui_fake"


def test_real_manager_writes_under_root():
    manager, _ = build_manager(False)
    assert manager.out_root == ROOT


def test_the_key_never_reaches_the_page(tmp_path, fake_model, monkeypatch):
    monkeypatch.setenv("SAIA_API_KEY", "SECRET-KEY")
    client, _ = _client(tmp_path, fake_model)
    run = client.post("/api/runs", json={**START, "auto_approve": True}).json()
    done = _poll(client, run["run_id"], {"done"})
    bodies = [client.get(u).text for u in (
        "/api/options", "/api/models", "/api/runs", "/api/sessions",
        f"/api/runs/{run['run_id']}", f"/sessions/{done['session_stem']}.html")]
    assert not any("SECRET-KEY" in b for b in bodies)
