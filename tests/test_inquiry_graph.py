import json

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from hermeneutics.inquiry_graph import InquiryDeps, build_inquiry_graph, load_metrics

METRICS = {
    "schema_version": 1, "corpus": "bukhari",
    "graph": {"n_nodes": 3, "n_edges": 2, "modularity": 0.4,
              "n_components": 1, "component_sizes": [3]},
    "nodes": [{"id": "a", "degree": 1.0, "betweenness": 0.5,
               "eigenvector": 0.7, "community": 0}],
}


def _deps(fake_model, trace, framework="THE METHOD"):
    return InquiryDeps(model=fake_model(["isnad out", "matn out", "synthesis out"]),
                       trace=trace, manifest_id="m1", metrics=METRICS,
                       framework=framework, commentator="nawawi", corpus="bukhari",
                       corpus_tools=None, condition="no_sources")


def test_load_metrics_reads_the_export(tmp_path):
    p = tmp_path / "bukhari_metrics.json"
    p.write_text(json.dumps(METRICS), encoding="utf-8")
    assert load_metrics(p)["graph"]["n_nodes"] == 3


def test_load_metrics_fails_loudly_when_the_export_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_network_metrics"):
        load_metrics(tmp_path / "missing.json")


def test_graph_interrupts_after_the_isnad_stage(fake_model, trace):
    graph = build_inquiry_graph(_deps(fake_model, trace), checkpointer=MemorySaver())
    result = graph.invoke({"condition": "no_sources", "question": "q", "corpus": "bukhari",
                           "intervention_log": []},
                          {"configurable": {"thread_id": "i1"}})
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["stage"] == "isnad"


def test_all_three_stages_run_with_approvals(fake_model, trace):
    graph = build_inquiry_graph(_deps(fake_model, trace), checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "i2"}}

    graph.invoke({"condition": "no_sources", "question": "q", "corpus": "bukhari",
                  "intervention_log": []}, cfg)
    graph.invoke(Command(resume={"action": "approve"}), cfg)
    graph.invoke(Command(resume={"action": "approve"}), cfg)
    final = graph.invoke(Command(resume={"action": "approve"}), cfg)

    assert final["isnad_output"] == "isnad out"
    assert final["synthesis_output"] == "synthesis out"


def test_stage_one_receives_computed_metrics_not_raw_json(fake_model, trace, tmp_path):
    """Defect #9: no more hallucinated topology."""
    graph = build_inquiry_graph(_deps(fake_model, trace), checkpointer=MemorySaver())
    graph.invoke({"condition": "no_sources", "question": "q", "corpus": "bukhari",
                  "intervention_log": []}, {"configurable": {"thread_id": "i3"}})

    prompt = json.loads((tmp_path / "test.jsonl").read_text(
        encoding="utf-8").strip().split("\n")[0])["prompt"]
    assert "modularity" in prompt
    assert "betweenness" in prompt


def test_no_sources_condition_carries_no_framework(fake_model, trace, tmp_path):
    """no_sources is the no-method-file control."""
    deps = _deps(fake_model, trace, framework="")
    graph = build_inquiry_graph(deps, checkpointer=MemorySaver())
    graph.invoke({"condition": "no_sources", "question": "q", "corpus": "bukhari",
                  "intervention_log": []}, {"configurable": {"thread_id": "i4"}})

    prompt = json.loads((tmp_path / "test.jsonl").read_text(
        encoding="utf-8").strip().split("\n")[0])["prompt"]
    assert "THE METHOD" not in prompt


def test_edit_intervention_is_logged_at_isnad_checkpoint(fake_model, trace):
    graph = build_inquiry_graph(_deps(fake_model, trace), checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "i5"}}
    graph.invoke({"condition": "no_sources", "question": "q", "corpus": "bukhari",
                  "intervention_log": []}, cfg)
    result = graph.invoke(Command(resume={"action": "edit", "note": "n"}), cfg)
    assert result["intervention_log"][0]["kind"] == "edit"


def test_an_unwatched_run_is_not_logged_as_an_approval(fake_model, trace):
    """The intervention log is the only evidence the paper has that a human saw a stage.
    --auto-approve exists so a comparison can be run unattended; folding it into
    "approve" would make an unwatched run indistinguishable from a scholar's sign-off."""
    graph = build_inquiry_graph(_deps(fake_model, trace), checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "i6"}}
    graph.invoke({"condition": "no_sources", "question": "q", "corpus": "bukhari",
                  "intervention_log": []}, cfg)
    result = graph.invoke(Command(resume={"action": "auto_approve"}), cfg)
    assert result["intervention_log"][0]["kind"] == "auto_approve"


def test_an_unrecognised_action_still_falls_back_to_approve(fake_model, trace):
    """The fallback is what keeps a malformed decision from inventing a new kind."""
    graph = build_inquiry_graph(_deps(fake_model, trace), checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "i7"}}
    graph.invoke({"condition": "no_sources", "question": "q", "corpus": "bukhari",
                  "intervention_log": []}, cfg)
    result = graph.invoke(Command(resume={"action": "banana"}), cfg)
    assert result["intervention_log"][0]["kind"] == "approve"


def test_the_rendered_source_names_the_author_in_arabic():
    from hermeneutics.inquiry_graph import _render
    out = _render([{"source": "commentary:mazandarani_babs.json", "author": "المازندراني شرح الكافي",
                    "reference": "كتاب العلم", "excerpt": "نص"}])
    assert out.startswith("[المازندراني شرح الكافي — commentary:mazandarani_babs.json")


def test_the_rendered_source_labels_commentary_versus_hadith_when_kind_is_set():
    """_prefetch_matn tags its two searches with `kind`; _render must show which
    result set is which, not silently merge them the way it always did before."""
    from hermeneutics.inquiry_graph import _render
    out = _render([{"source": "commentary:nawawi_babs.json", "kind": "commentary",
                    "reference": "r1", "excerpt": "e1"},
                   {"source": "muslim_matn:muslim_raw.json", "kind": "hadith",
                    "reference": "r2", "excerpt": "e2"}])
    assert "COMMENTARY —" in out
    assert "HADITH —" in out


# --- Part 5: retrieval scoping ------------------------------------------------------

def test_the_isnad_and_synthesis_prompts_carry_the_network_metrics_note():
    """Real sessions have argued from centrality figures straight to a doctrinal
    conclusion ('the imam is the central node, therefore...'). One sentence, in both
    prompts that see network metrics, says not to. Lives in inquiry_graph.py itself,
    not hermeneutics.prompts -- that module's TEMPLATES back the FROZEN study
    pipeline's published prompt_hashes, and this sentence is the Query-Agent's."""
    from hermeneutics.inquiry_graph import (
        NETWORK_METRICS_NOTE,
        InquiryDeps,
        _isnad_user,
        _synthesis_user,
    )

    deps = InquiryDeps(model=None, trace=None, manifest_id="m1", metrics=METRICS,
                       framework="", commentator="cayni", corpus="bukhari",
                       corpus_tools=None, condition="agent_method")
    state = {"question": "q", "isnad_output": "", "matn_output": ""}

    assert NETWORK_METRICS_NOTE in _isnad_user(state, deps, [])
    assert NETWORK_METRICS_NOTE in _synthesis_user(state, deps, [])


def test_the_network_metrics_note_does_not_touch_the_frozen_study_prompts():
    """The study pipeline's prompt_hashes() (hermeneutics.prompts.TEMPLATES) backs
    already-published results; this sentence must not be added there."""
    from hermeneutics.prompts import TEMPLATES

    assert "network_metrics_note" not in TEMPLATES
    assert not any("doctrinal or legal position" in text for text in TEMPLATES.values())


def test_the_gate_flags_outside_scope_for_a_verified_quotation_found_only_in_kafi():
    """A cayni run producing genuine al-Kafi text from memory: verify_quotation still
    says it is real (whole-corpus), foreign_commentary stays False (quoting the hadith
    corpus itself is legitimate), and outside_scope is the flag that catches it."""
    from pathlib import Path

    from hermeneutics.corpus_tools import CorpusTools, scope_for
    from hermeneutics.inquiry_graph import InquiryDeps, _synthesis_gate

    root = Path(__file__).resolve().parents[1]
    kafi_text = "ورأيت العظيم من المال ينفق في سخط الله عز وجل"
    deps = InquiryDeps(model=None, trace=None, manifest_id="m1", metrics=METRICS,
                       framework="", commentator="cayni", corpus="bukhari",
                       corpus_tools=CorpusTools(root, scope=scope_for("cayni")),
                       condition="agent_method")
    state = {"synthesis_output": f"كما جاء «{kafi_text}»", "sources": [],
            "question": "q", "question_ar": ""}

    out = _synthesis_gate(state, deps)

    entry = out["grounding_log"][0]
    assert entry["verified"] is True
    assert entry["foreign_commentary"] is False
    assert entry["outside_scope"] is True
    assert kafi_text in out["synthesis_output"]  # verified quotations are never stripped


def test_the_gate_does_not_flag_outside_scope_for_the_runs_own_commentary():
    from pathlib import Path

    from hermeneutics.corpus_tools import CorpusTools, scope_for
    from hermeneutics.inquiry_graph import InquiryDeps, _synthesis_gate

    root = Path(__file__).resolve().parents[1]
    corpus = CorpusTools(root, scope=scope_for("cayni"))
    own_hit = corpus.search("العيني عمدة القاري", "commentary")[0]
    quote = own_hit["excerpt"][:60].strip()
    deps = InquiryDeps(model=None, trace=None, manifest_id="m1", metrics=METRICS,
                       framework="", commentator="cayni", corpus="bukhari",
                       corpus_tools=corpus, condition="agent_method")
    state = {"synthesis_output": f"كما جاء «{quote}»", "sources": [],
            "question": "q", "question_ar": ""}

    out = _synthesis_gate(state, deps)

    entry = out["grounding_log"][0]
    assert entry["verified"] is True
    assert entry["outside_scope"] is False
