"""The setup/save code the CLI and the UI share."""
import json
import shutil
from pathlib import Path

import pytest
from langgraph.types import Command

from hermeneutics.inquiry_session import RunParams, open_graph, prepare, write_session

POC = Path(__file__).resolve().parent.parent


def _root(tmp_path):
    nets = tmp_path / "outputs" / "networks"
    nets.mkdir(parents=True)
    shutil.copy(POC / "outputs" / "networks" / "muslim_metrics.json", nets)
    return tmp_path


def _params(**kw):
    base = {"question": "ما معنى النية", "condition": "no_sources",
            "model_id": "fake-model", "run_id": "r1"}
    base.update(kw)
    return RunParams(**base)


def _run_through(prepared):
    with open_graph(prepared) as (graph, cfg):
        result = graph.invoke(prepared.initial_state, cfg)
        while "__interrupt__" in result:
            result = graph.invoke(Command(resume={"action": "auto_approve"}), cfg)
    return result


def test_params_round_trip_and_ignore_unknown_keys():
    p = _params(auto_approve=True)
    assert RunParams.from_dict({**p.to_dict(), "not_a_field": 1}) == p


def test_params_default_to_the_saia_backend_with_no_route_pinned():
    """provider_route defaults to None, not DEFAULT_PROVIDER_ROUTE: serve_ui.py
    builds a RunParams for a SAIA run without ever setting this field, and a
    non-None default would record a stack that never served the run."""
    p = _params()
    assert p.provider == "saia"
    assert p.provider_route is None
    assert p.quantization is None


def test_a_default_run_params_manifest_records_no_provider_route(tmp_path, fake_model):
    root = _root(tmp_path)
    prepared = prepare(_params(), model=fake_model([]), root=root, out_root=tmp_path)
    assert prepared.manifest["provider"] == "saia"
    assert prepared.manifest["provider_route"] is None
    assert prepared.manifest["quantization"] is None


def test_an_old_ui_run_json_without_provider_fields_still_loads():
    """Before Part 1, ui_run.json never had provider/provider_route/quantization at
    all -- a resume of a run saved before this change must still load, defaulting to
    the backend that was the only one that ever existed and no pinned route."""
    old = _params().to_dict()
    del old["provider"]
    del old["provider_route"]
    del old["quantization"]
    p = RunParams.from_dict(old)
    assert p.provider == "saia"
    assert p.provider_route is None
    assert p.quantization is None


def test_the_provider_and_route_reach_the_manifest_and_session(tmp_path, fake_model):
    root = _root(tmp_path)
    prepared = prepare(_params(provider="openrouter", provider_route="deepinfra",
                               quantization="bf16"),
                       model=fake_model(["i", "m", "s"]), root=root, out_root=tmp_path)
    assert prepared.manifest["provider"] == "openrouter"
    assert prepared.manifest["provider_route"] == "deepinfra"
    assert prepared.manifest["quantization"] == "bf16"

    result = _run_through(prepared)
    path = write_session(prepared, result)
    s = json.loads(path.read_text(encoding="utf-8"))
    assert s["provider"] == "openrouter"
    assert s["provider_route"] == "deepinfra"
    assert s["quantization"] == "bf16"


def test_the_chosen_model_reaches_the_manifest_and_its_id(tmp_path, fake_model):
    root = _root(tmp_path)
    a = prepare(_params(), model=fake_model([]), root=root, out_root=tmp_path)
    b = prepare(_params(model_id="other-model", run_id="r2"), model=fake_model([]),
                root=root, out_root=tmp_path)

    assert a.manifest["model_id"] == "fake-model"
    assert b.manifest["model_id"] == "other-model"
    assert a.manifest_id != b.manifest_id


def test_corpus_follows_the_commentator_unless_given(tmp_path, fake_model):
    root = _root(tmp_path)
    p = prepare(_params(), model=fake_model([]), root=root, out_root=tmp_path)
    assert p.corpus == "muslim"
    assert p.manifest["corpus_explicit"] is False


def test_session_json_carries_model_and_resumes(tmp_path, fake_model):
    root = _root(tmp_path)
    prepared = prepare(_params(), model=fake_model(["i", "m", "s"]),
                       root=root, out_root=tmp_path)
    result = _run_through(prepared)

    path = write_session(prepared, result, resumes=["2026-09-22T10:00:00+00:00"])

    s = json.loads(path.read_text(encoding="utf-8"))
    assert path.parent == tmp_path / "outputs" / "sessions"
    assert path.name == f"muslim_no_sources_{prepared.stamp}.json"
    assert s["model_id"] == "fake-model"
    assert s["resumes"] == ["2026-09-22T10:00:00+00:00"]
    assert s["synthesis_output"] == "s"


def test_resume_after_a_failure_restarts_the_failed_stage_not_the_run(
        tmp_path, fake_model, failing_model):
    root = _root(tmp_path)
    first = prepare(_params(), model=failing_model(["isnad out"]),
                    root=root, out_root=tmp_path)
    with open_graph(first) as (graph, cfg):
        graph.invoke(first.initial_state, cfg)                          # isnad gate
        with pytest.raises(ConnectionError):
            graph.invoke(Command(resume={"action": "approve"}), cfg)    # matn dies

    again = prepare(_params(), model=fake_model(["matn out", "synthesis out"]),
                    root=root, out_root=tmp_path,
                    manifest=first.manifest, stamp=first.stamp)
    assert again.manifest_id == first.manifest_id
    with open_graph(again) as (graph, cfg):
        assert graph.get_state(cfg).next == ("matn_stage",)
        result = graph.invoke(None, cfg)
        assert result["__interrupt__"][0].value["stage"] == "matn"
        result = graph.invoke(Command(resume={"action": "approve"}), cfg)
        result = graph.invoke(Command(resume={"action": "approve"}), cfg)

    assert result["isnad_output"] == "isnad out"
    assert result["synthesis_output"] == "synthesis out"
    assert [e["kind"] for e in result["intervention_log"]] == ["approve"] * 3
