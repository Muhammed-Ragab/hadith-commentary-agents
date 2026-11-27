"""The canonicalise node (Part 2): turn any question into one classical-Arabic
research question plus Arabic search terms, before any stage runs.

Without this, an English question against a literal-term Arabic search retrieves
nothing (CorpusTools.search is literal term counting over Arabic text), and a
colloquial Arabic question misses the commentators' own chapter-heading vocabulary
almost as often. See hermeneutics/corpus_tools.py and inquiry_graph.py:_prefetch_matn.
"""
import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from hermeneutics.inquiry_graph import (
    InquiryDeps,
    _canonicalise,
    _synthesis_gate,
    build_inquiry_graph,
)

METRICS = {
    "schema_version": 1, "corpus": "bukhari",
    "graph": {"n_nodes": 1, "n_edges": 0, "modularity": 0.0,
              "n_components": 1, "component_sizes": [1]},
    "nodes": [],
}


class _JsonModel:
    """Behaves like get_chat_model's structured-output path: .bind(response_format=...)
    returns a model whose .invoke() replies come from a scripted queue, in order."""

    model_name = "fake"

    def __init__(self, replies):
        self._replies = list(replies)

    def bind(self, **kwargs):
        return self

    def invoke(self, _messages):
        return AIMessage(content=self._replies.pop(0))

    def bind_tools(self, _tools):
        return self


def _deps(trace, model, condition="agent"):
    return InquiryDeps(model=model, trace=trace, manifest_id="m1", metrics=METRICS,
                       framework="", commentator="nawawi", corpus="bukhari",
                       corpus_tools=None, condition=condition)


VALID = json.dumps({"question_ar": "ما حكم الجبر والقدر", "terms": ["الجبر", "القدر", "المشيئة"],
                    "lang": "english"}, ensure_ascii=False)


def test_canonicalise_returns_question_ar_terms_and_lang(trace):
    deps = _deps(trace, _JsonModel([VALID]))
    out = _canonicalise({"question": "is it free will or predestination?"}, deps)
    assert out["question_ar"] == "ما حكم الجبر والقدر"
    assert out["search_terms"] == ["الجبر", "القدر", "المشيئة"]
    assert out["question_lang"] == "english"


def test_canonicalise_lands_in_the_trace(trace, tmp_path):
    deps = _deps(trace, _JsonModel([VALID]))
    _canonicalise({"question": "q"}, deps)
    lines = (tmp_path / "test.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert any(json.loads(line)["node"] == "inquiry_canonicalise" for line in lines)


def test_canonicalise_retries_once_on_a_bad_schema_then_succeeds(trace):
    """3 terms is the floor; 1 term must be rejected and retried, not accepted."""
    bad = json.dumps({"question_ar": "س", "terms": ["فقط"], "lang": "arabic"})
    deps = _deps(trace, _JsonModel([bad, VALID]))
    out = _canonicalise({"question": "q"}, deps)
    assert out["question_ar"] == "ما حكم الجبر والقدر"


def test_canonicalise_raises_after_the_retry_rather_than_falling_back_to_the_raw_question(trace):
    """Silently falling back to state['question'] is exactly the bug being fixed --
    an empty question_ar must fail loudly, not degrade."""
    bad = json.dumps({"question_ar": "", "terms": ["a", "b", "c"], "lang": "en"})
    deps = _deps(trace, _JsonModel([bad, bad]))
    with pytest.raises(ValueError, match="canonicalise"):
        _canonicalise({"question": "q"}, deps)


def test_canonicalise_is_skipped_under_no_sources(trace):
    """The baseline must stay a model with no pipeline help -- not even a rewritten
    question. A model that would raise on ANY call proves the node never touched it."""
    class _ExplodingModel:
        model_name = "fake"

        def bind(self, **kwargs):
            raise AssertionError("no_sources must never call the model to canonicalise")

    deps = _deps(trace, _ExplodingModel(), condition="no_sources")
    assert _canonicalise({"question": "q"}, deps) == {}


def test_canonicalise_applies_to_an_arabic_question_too(trace):
    """Colloquial/MSA Arabic gets the same treatment as English -- not just a
    language switch, a rewrite into the chapter-heading vocabulary."""
    reply = json.dumps({"question_ar": "الجبر والقدر", "terms": ["الجبر", "القدر", "الاختيار"],
                        "lang": "colloquial arabic"}, ensure_ascii=False)
    deps = _deps(trace, _JsonModel([reply]))
    out = _canonicalise({"question": "هل الإنسان مسيّر أم مخيّر؟"}, deps)
    assert out["search_terms"] == ["الجبر", "القدر", "الاختيار"]


def test_canonicalise_node_runs_before_isnad_in_the_full_graph(trace, tmp_path):
    deps = _deps(trace, _JsonModel([VALID, "isnad out"]), condition="no_sources")
    # no_sources skips canonicalise, so use "agent" with corpus_tools=None to isolate
    # the graph-order check without needing a real corpus for the isnad stage's own
    # (tools-off) call. agent needs corpus_tools for retrieves-gate purposes; with
    # corpus_tools=None the stage node falls back to the no-retrieval call_model path.
    deps.condition = "agent"
    graph = build_inquiry_graph(deps, checkpointer=MemorySaver())
    result = graph.invoke(
        {"condition": "agent", "question": "q", "corpus": "bukhari", "intervention_log": [],
         "sources": [], "tool_calls_log": [], "grounding_log": []},
        {"configurable": {"thread_id": "canon1"}})
    assert "__interrupt__" in result
    lines = [json.loads(line) for line in
             (tmp_path / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    nodes_in_order = [line["node"] for line in lines]
    assert nodes_in_order.index("inquiry_canonicalise") < nodes_in_order.index("inquiry_isnad")


class _NeverVerifiesCorpus:
    """Nothing this corpus is asked about is ever found -- isolates echoes_question
    from verification, which is a separate flag."""

    def verify_quote(self, quote, source="all"):
        return {"verified": False, "occurrences": []}


def test_echoes_question_catches_an_echo_of_the_canonical_arabic_question():
    """An answer that scare-quotes question_ar (the model's OWN classical-Arabic
    rendering, not the asker's raw words) must still be flagged as an echo, not
    misread as a genuine citation from the corpus."""
    canonical_fragment = "الجبر والقدر في المشيئة الإلهية جدا"
    deps = InquiryDeps(model=None, trace=None, manifest_id="m1", metrics={}, framework="",
                       commentator="nawawi", corpus="bukhari",
                       corpus_tools=_NeverVerifiesCorpus(), condition="agent")
    state = {
        "question": "is it free will or predestination?",
        "question_ar": f"ما حكم {canonical_fragment} عند العلماء",
        "synthesis_output": f'as the sources discuss, «{canonical_fragment}» therefore',
    }
    out = _synthesis_gate(state, deps)
    assert out["grounding_log"][0]["echoes_question"] is True


def test_echoes_question_is_false_for_a_real_quotation_absent_from_either_question():
    deps = InquiryDeps(model=None, trace=None, manifest_id="m1", metrics={}, framework="",
                       commentator="nawawi", corpus="bukhari",
                       corpus_tools=_NeverVerifiesCorpus(), condition="agent")
    state = {
        "question": "is it free will or predestination?",
        "question_ar": "ما حكم الجبر والقدر عند العلماء",
        "synthesis_output": 'as he says «إنما الأعمال بالنيات وإنما لكل امرئ ما نوى» therefore',
    }
    out = _synthesis_gate(state, deps)
    assert out["grounding_log"][0]["echoes_question"] is False
