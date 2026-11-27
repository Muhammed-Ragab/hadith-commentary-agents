import json

import networkx as nx
import pytest

from scripts.export_network_metrics import build_and_export, compute_metrics, write_metrics


def _graph():
    g = nx.Graph()
    g.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e")])
    return g


def test_metrics_have_the_documented_top_level_keys():
    m = compute_metrics(_graph(), corpus="bukhari")
    for key in ("corpus", "graph", "nodes", "schema_version"):
        assert key in m


def test_graph_level_stats_are_present():
    g = compute_metrics(_graph(), corpus="bukhari")["graph"]
    for key in ("n_nodes", "n_edges", "modularity", "n_components", "component_sizes"):
        assert key in g
    assert g["n_nodes"] == 5
    assert g["n_edges"] == 5


def test_every_node_has_all_four_measures():
    for node in compute_metrics(_graph(), corpus="kafi")["nodes"]:
        for key in ("id", "degree", "betweenness", "eigenvector", "community"):
            assert key in node


def test_nodes_are_sorted_by_id_for_stable_hashing():
    nodes = compute_metrics(_graph(), corpus="bukhari")["nodes"]
    assert [n["id"] for n in nodes] == sorted(n["id"] for n in nodes)


def test_export_is_byte_identical_across_runs(tmp_path):
    """Spec §3.5: the hash enters the run manifest, so it must be deterministic."""
    a = write_metrics(compute_metrics(_graph(), "bukhari"), tmp_path / "a.json")
    b = write_metrics(compute_metrics(_graph(), "bukhari"), tmp_path / "b.json")
    assert a.read_bytes() == b.read_bytes()


def test_floats_are_rounded_to_a_fixed_precision():
    """Unrounded floats differ in the last bits across runs and break the hash."""
    for node in compute_metrics(_graph(), corpus="bukhari")["nodes"]:
        assert len(str(node["betweenness"]).split(".")[-1]) <= 6


def test_no_timestamp_leaks_into_the_export(tmp_path):
    path = write_metrics(compute_metrics(_graph(), "bukhari"), tmp_path / "m.json")
    blob = path.read_text(encoding="utf-8").lower()
    assert "timestamp" not in blob
    assert "generated" not in blob


def test_empty_graph_produces_a_valid_empty_export():
    m = compute_metrics(nx.Graph(), corpus="bukhari")
    assert m["nodes"] == []
    assert m["graph"]["n_nodes"] == 0


def test_written_file_parses_as_json(tmp_path):
    path = write_metrics(compute_metrics(_graph(), "kafi"), tmp_path / "m.json")
    assert json.loads(path.read_text(encoding="utf-8"))["corpus"] == "kafi"


def test_directed_graph_is_handled_gracefully():
    g = nx.DiGraph()
    g.add_edges_from([("a", "b"), ("b", "c")])
    m = compute_metrics(g, corpus="bukhari")
    assert m["graph"]["n_nodes"] == 3
    assert m["nodes"][0]["id"] == "a"


def test_eigenvector_convergence_failure_emits_warning(monkeypatch):
    import warnings
    import networkx as nx
    from scripts.export_network_metrics import compute_metrics

    def fail_convergence(g, **kwargs):
        raise nx.PowerIterationFailedConvergence(100)

    monkeypatch.setattr(nx, "eigenvector_centrality", fail_convergence)

    with pytest.warns(RuntimeWarning, match="eigenvector"):
        m = compute_metrics(_graph(), corpus="bukhari")

    # values zeroed but record is still valid
    assert all(n["eigenvector"] == 0.0 for n in m["nodes"])


def test_provenance_records_which_input_built_the_graph():
    """A 500-item sample and the full corpus give very different centrality
    numbers, so a reader of the artifact must be able to tell which one this is."""
    m = compute_metrics(_graph(), corpus="bukhari", source="bukhari_sample.json", n_hadiths=500)
    assert m["provenance"] == {"source": "bukhari_sample.json", "n_hadiths": 500}


def test_provenance_is_present_even_when_the_caller_omits_it():
    """Schema stability: every export has a provenance key, even a bare compute_metrics call."""
    m = compute_metrics(_graph(), corpus="bukhari")
    assert "provenance" in m


def test_build_and_export_writes_a_real_isnad_graph_from_the_sample(tmp_path, monkeypatch):
    import scripts.export_network_metrics as export_mod

    monkeypatch.setattr(export_mod, "OUT_DIR", tmp_path)
    path = build_and_export("bukhari")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert path.parent == tmp_path
    assert data["corpus"] == "bukhari"
    assert data["provenance"] == {"source": "bukhari_sample.json", "n_hadiths": 500}
    assert data["graph"]["n_nodes"] > 0
    assert data["graph"]["n_edges"] > 0


def test_build_and_export_is_deterministic_across_two_runs(tmp_path, monkeypatch):
    """Spec §3.5: the hash enters the run manifest, so the whole pipeline -- not
    just compute_metrics -- must be byte-identical run to run on the same input."""
    import scripts.export_network_metrics as export_mod

    monkeypatch.setattr(export_mod, "OUT_DIR", tmp_path / "a")
    path_a = build_and_export("kafi")
    monkeypatch.setattr(export_mod, "OUT_DIR", tmp_path / "b")
    path_b = build_and_export("kafi")

    assert path_a.read_bytes() == path_b.read_bytes()


def test_a_known_narrator_appears_with_real_metrics(tmp_path, monkeypatch):
    """شعبة is a documented top-5 narrator by degree centrality in notebook 01's
    Bukhārī run (validation checklist, T016) -- it must show up with real numbers,
    not be dropped by the extraction."""
    import scripts.export_network_metrics as export_mod

    monkeypatch.setattr(export_mod, "OUT_DIR", tmp_path)
    path = build_and_export("bukhari")
    data = json.loads(path.read_text(encoding="utf-8"))

    ids = {n["id"] for n in data["nodes"]}
    matches = [i for i in ids if "شعبة" in i]
    assert matches, "شعبة should appear as (part of) some narrator node"
    node = next(n for n in data["nodes"] if n["id"] in matches)
    assert node["degree"] > 0


def test_an_invented_name_never_appears_in_the_graph(tmp_path, monkeypatch):
    import scripts.export_network_metrics as export_mod

    monkeypatch.setattr(export_mod, "OUT_DIR", tmp_path)
    path = build_and_export("bukhari")
    data = json.loads(path.read_text(encoding="utf-8"))

    ids = {n["id"] for n in data["nodes"]}
    assert not any("لا يوجد راو بهذا الاسم اختلقته" in i for i in ids)


def test_every_sampled_corpus_is_reachable_from_the_cli():
    """build_and_export handles muslim, but the choices list did not, so the Muslim
    metrics could only be produced by calling the function by hand."""
    from scripts.export_network_metrics import build_parser

    for corpus in ("bukhari", "kafi", "muslim"):
        assert build_parser().parse_args([corpus]).corpus == corpus
