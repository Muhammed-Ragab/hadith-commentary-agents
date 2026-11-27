"""The defect this closes: inquiry_graph.py said 'read the matn' and passed none."""
import json

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from hermeneutics.inquiry_graph import InquiryDeps, build_inquiry_graph

METRICS = {
    "schema_version": 1, "corpus": "bukhari",
    "graph": {"n_nodes": 3, "n_edges": 2, "modularity": 0.4,
              "n_components": 1, "component_sizes": [3]},
    "nodes": [{"id": "a", "degree": 1.0, "betweenness": 0.5,
               "eigenvector": 0.7, "community": 0}],
}

HADITH = "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى"


# canonicalise now runs before every tool-bearing stage (Part 2). Echoing the
# question straight back as question_ar keeps every pre-existing assertion below
# about "what the matn/isnad/synthesis stages saw" unchanged; CANON_TERMS is fixed
# so the one test that checks the exact search query (below) can compute it.
CANON_TERMS = ["النية", "الأعمال", "النوايا"]


def _canon_reply(question: str) -> str:
    return json.dumps({"question_ar": question, "terms": CANON_TERMS, "lang": "arabic"},
                      ensure_ascii=False)


class _CanonicaliseJsonModel:
    """What model.bind(response_format=...) returns -- call_json's JSON-mode path.
    Answers directly from the ORIGINAL question in the outer messages, so it never
    touches FakeToolModel's own scripted reply queue."""

    def invoke(self, messages):
        question = messages[-1].content.removeprefix("QUESTION: ").split("\n", 1)[0]
        return AIMessage(content=_canon_reply(question))


class FakeToolModel:
    model_name = "fake"

    def __init__(self, messages):
        self._messages = list(messages)

    def bind_tools(self, tools):
        return self

    def bind(self, **kwargs):
        return _CanonicaliseJsonModel()

    def invoke(self, messages):
        return self._messages.pop(0)


class FakeCorpus:
    def __init__(self, occurrences=None):
        self.searches = []
        # quote-substring -> occurrences, so tests can exercise provenance
        # (mis-attribution / foreign-commentary) without a real corpus. Anything not
        # matched here falls back to the original "quote is the hadith itself"
        # behaviour, so every pre-existing test is untouched.
        self._occurrences = occurrences or {}

    @staticmethod
    def tool_specs():
        return [{"type": "function", "function": {"name": "search_sources",
                                                  "parameters": {"type": "object",
                                                                 "properties": {}}}}]

    def search(self, query, source="all", **kw):
        self.searches.append((query, source))
        return [{"source": f"hadith:{source}", "reference": "الإيمان 1",
                 "excerpt": HADITH}]

    def verify_quote(self, quote, source="all"):
        for needle, occurrences in self._occurrences.items():
            if needle in quote:
                return {"verified": True, "occurrences": occurrences}
        verified = HADITH.startswith(quote[:10])
        occurrences = [{"source": "hadith:bukhari_raw.json",
                        "reference": "كتاب الإيمان 1"}] if verified else []
        return {"verified": verified, "occurrences": occurrences}

    def lookup_word(self, word, lexicon="all"):
        return {"word": word, "articles": []}

    def lookup_narrator(self, name, school="all"):
        return {"name": name, "matches": []}

    def lookup_isnad_network(self, name, corpus="all"):
        return {"name": name, "matches": []}


def _deps(trace, corpus, replies):
    return InquiryDeps(model=FakeToolModel([AIMessage(content=r) for r in replies]),
                       trace=trace, manifest_id="m1", metrics=METRICS,
                       framework="THE METHOD", commentator="nawawi", corpus="muslim",
                       corpus_tools=corpus, condition="agent_method")


def _run_to_end(graph, cfg, question="ما معنى النية"):
    result = graph.invoke({"condition": "agent_method", "question": question,
                           "commentator": "nawawi", "corpus": "muslim",
                           "intervention_log": [], "sources": [],
                           "tool_calls_log": [], "grounding_log": []}, cfg)
    while "__interrupt__" in result:
        result = graph.invoke(Command(resume={"action": "approve"}), cfg)
    return result


def test_the_matn_stage_receives_actual_hadith_text(trace, tmp_path):
    """The whole point. Before this task the matn prompt contained no matn."""
    corpus = FakeCorpus()
    graph = build_inquiry_graph(_deps(trace, corpus, ["isnad", "matn", "synth"]),
                                checkpointer=MemorySaver())
    _run_to_end(graph, {"configurable": {"thread_id": "t1"}})

    lines = [json.loads(line) for line
             in (tmp_path / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    matn_prompt = next(line["prompt"] for line in lines if line["node"] == "inquiry_matn")
    assert HADITH in matn_prompt


def test_the_matn_prefetch_reads_the_commentators_own_corpus(trace):
    """al-Nawawi commentates Muslim. Searching Bukhari under his method file pairs a
    method with a corpus it was never derived from (cf. d042bf8, 526f90c)."""
    corpus = FakeCorpus()
    graph = build_inquiry_graph(_deps(trace, corpus, ["isnad", "matn", "synth"]),
                                checkpointer=MemorySaver())
    _run_to_end(graph, {"configurable": {"thread_id": "t2"}})

    # Searches with the CANONICAL question + terms (Part 2), not the raw question.
    expected_query = " ".join(["ما معنى النية", *CANON_TERMS])
    assert (expected_query, "muslim_matn") in corpus.searches


def test_the_matn_prefetch_also_searches_the_commentators_own_commentary(trace):
    """Part 5 item 2: two of four recorded mazandarani runs retrieved zero passages
    from his own commentary because _prefetch_matn only ever searched the hadith
    collection. It must now search "commentary" too, with the same canonical query."""
    corpus = FakeCorpus()
    graph = build_inquiry_graph(_deps(trace, corpus, ["isnad", "matn", "synth"]),
                                checkpointer=MemorySaver())
    _run_to_end(graph, {"configurable": {"thread_id": "t18"}})

    expected_query = " ".join(["ما معنى النية", *CANON_TERMS])
    assert (expected_query, "commentary") in corpus.searches


def test_the_isnad_stage_still_receives_computed_metrics(trace, tmp_path):
    """Defect #9 must not regress while the stage gains tools."""
    graph = build_inquiry_graph(_deps(trace, FakeCorpus(), ["isnad", "matn", "synth"]),
                                checkpointer=MemorySaver())
    _run_to_end(graph, {"configurable": {"thread_id": "t3"}})

    lines = [json.loads(line) for line
             in (tmp_path / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    prompt = next(line["prompt"] for line in lines if line["node"] == "inquiry_isnad")
    assert "modularity" in prompt and "betweenness" in prompt


def test_sources_accumulate_across_stages(trace):
    graph = build_inquiry_graph(_deps(trace, FakeCorpus(), ["isnad", "matn", "synth"]),
                                checkpointer=MemorySaver())
    final = _run_to_end(graph, {"configurable": {"thread_id": "t4"}})

    assert len(final["sources"]) >= 1


def test_a_no_retrieval_condition_never_touches_the_corpus(trace):
    """no_sources is the clean baseline: the only difference from `agent` is access,
    not capability. A baseline that retrieves is a RAG condition wearing the wrong
    label, and it invalidates every comparison against it (compare.py:92)."""
    corpus = FakeCorpus()
    deps = InquiryDeps(
        model=FakeToolModel([AIMessage(content=r) for r in ["isnad", "matn", "synth"]]),
        trace=trace, manifest_id="m1", metrics=METRICS, framework="",
        commentator="nawawi", corpus="muslim", corpus_tools=None,
        condition="no_sources")
    graph = build_inquiry_graph(deps, checkpointer=MemorySaver())

    final = graph.invoke({"condition": "no_sources", "question": "q",
                          "commentator": "nawawi", "corpus": "muslim",
                          "intervention_log": [], "sources": [],
                          "tool_calls_log": [], "grounding_log": []},
                         {"configurable": {"thread_id": "t5"}})
    while "__interrupt__" in final:
        final = graph.invoke(Command(resume={"action": "approve"}),
                             {"configurable": {"thread_id": "t5"}})

    assert corpus.searches == []
    assert final.get("sources", []) == []
    assert final.get("tool_calls_log", []) == []


FABRICATED = "«هذا كلام لم يقله أحد قط في أي كتاب من الكتب»"
GENUINE = f"«{HADITH}»"


def test_a_fabricated_quotation_is_stripped_from_the_synthesis(trace):
    """A fabricated quotation reaching a human review is worse than a refusal:
    the review launders it."""
    corpus = FakeCorpus()
    graph = build_inquiry_graph(
        _deps(trace, corpus, ["isnad", "matn", f"as he says {FABRICATED} therefore"]),
        checkpointer=MemorySaver())
    final = _run_to_end(graph, {"configurable": {"thread_id": "t5"}})

    assert "لم يقله أحد" not in final["synthesis_output"]
    stripped = [r for r in final["grounding_log"] if not r["verified"]]
    assert len(stripped) == 1
    assert "لم يقله أحد" in stripped[0]["quote"]


def test_a_genuine_quotation_survives_the_gate(trace):
    corpus = FakeCorpus()
    graph = build_inquiry_graph(
        _deps(trace, corpus, ["isnad", "matn", f"the text reads {GENUINE} and so"]),
        checkpointer=MemorySaver())
    final = _run_to_end(graph, {"configurable": {"thread_id": "t6"}})

    assert HADITH in final["synthesis_output"]
    assert all(r["verified"] for r in final["grounding_log"])


def test_the_gate_records_whether_the_quote_was_in_this_sessions_sources(trace):
    """Recorded, not enforced: a tool result is a truncated window onto a file, so a
    genuine quotation can be absent from it. Both numbers, only one censors."""
    corpus = FakeCorpus()
    graph = build_inquiry_graph(
        _deps(trace, corpus, ["isnad", "matn", f"the text reads {GENUINE}"]),
        checkpointer=MemorySaver())
    final = _run_to_end(graph, {"configurable": {"thread_id": "t7"}})

    assert final["grounding_log"][0]["in_session"] is True


def test_a_synthesis_with_no_quotations_passes_through_untouched(trace):
    graph = build_inquiry_graph(
        _deps(trace, FakeCorpus(), ["isnad", "matn", "a paraphrase with no quotes"]),
        checkpointer=MemorySaver())
    final = _run_to_end(graph, {"configurable": {"thread_id": "t8"}})

    assert final["synthesis_output"] == "a paraphrase with no quotes"
    assert final["grounding_log"] == []


def test_a_no_retrieval_condition_is_not_gated(trace):
    """no_sources has no corpus to verify against. Gating it would either strip every
    quotation or require a corpus the condition must not touch."""
    from hermeneutics.inquiry_graph import InquiryDeps
    from langchain_core.messages import AIMessage

    deps = InquiryDeps(
        model=FakeToolModel([AIMessage(content=r) for r in
                             ["isnad", "matn", f"it says {FABRICATED}"]]),
        trace=trace, manifest_id="m1", metrics=METRICS, framework="",
        commentator="nawawi", corpus="muslim", corpus_tools=None,
        condition="no_sources")
    graph = build_inquiry_graph(deps, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t9"}}
    final = graph.invoke({"condition": "no_sources", "question": "q",
                          "commentator": "nawawi", "corpus": "muslim",
                          "intervention_log": [], "sources": [],
                          "tool_calls_log": [], "grounding_log": []}, cfg)
    while "__interrupt__" in final:
        final = graph.invoke(Command(resume={"action": "approve"}), cfg)

    assert "لم يقله أحد" in final["synthesis_output"]
    assert final["grounding_log"] == []


# --- Defect 1: mis-attribution -- provenance recorded, not stripped ---------------

FOREIGN_QUOTE = "وهذا نص كامل من شرح العيني على صحيح البخاري تحديدا هنا"
OWN_QUOTE = "وهذا نص كامل من شرح النووي على صحيح مسلم تحديدا هنا كذلك"
LEXICON_QUOTE = "وهذا نص كامل موجود في لسان العرب فقط تحديدا هنا أيضا"


def test_a_quotation_found_only_in_a_rival_commentary_is_flagged_and_not_stripped(trace):
    """al-`Ayni's text passing under al-Nawawi's name is worse than a fabrication that
    gets blocked -- a human review would launder it. The gate cannot strip it (the text
    really occurs) but must make the mis-attribution visible on the review sheet."""
    corpus = FakeCorpus({FOREIGN_QUOTE: [{"source": "commentary:cayni_babs.json",
                                          "reference": "بدء الوحي"}]})
    graph = build_inquiry_graph(
        _deps(trace, corpus, ["isnad", "matn", f"as he says «{FOREIGN_QUOTE}» therefore"]),
        checkpointer=MemorySaver())
    final = _run_to_end(graph, {"configurable": {"thread_id": "t10"}})

    assert FOREIGN_QUOTE in final["synthesis_output"]
    entry = final["grounding_log"][0]
    assert entry["verified"] is True
    assert entry["foreign_commentary"] is True
    assert entry["found_in"] == ["commentary:cayni_babs.json"]


def test_a_quotation_found_in_the_commentators_own_commentary_is_not_flagged(trace):
    """`_deps` studies "nawawi" -- a quotation found in nawawi_babs.json is the honest
    case and must not be confused with mis-attribution."""
    corpus = FakeCorpus({OWN_QUOTE: [{"source": "commentary:nawawi_babs.json",
                                      "reference": "بدء الوحي"}]})
    graph = build_inquiry_graph(
        _deps(trace, corpus, ["isnad", "matn", f"as he says «{OWN_QUOTE}» therefore"]),
        checkpointer=MemorySaver())
    final = _run_to_end(graph, {"configurable": {"thread_id": "t11"}})

    entry = final["grounding_log"][0]
    assert entry["verified"] is True
    assert entry["foreign_commentary"] is False


def test_a_quotation_found_only_in_the_lexica_is_not_flagged_foreign(trace):
    """Quoting a dictionary is legitimate -- commentators do it constantly. Only a
    RIVAL commentary is the mis-attribution signal."""
    corpus = FakeCorpus({LEXICON_QUOTE: [{"source": "lexica:lisan_entries.json",
                                          "reference": "بوق"}]})
    graph = build_inquiry_graph(
        _deps(trace, corpus, ["isnad", "matn", f"as he says «{LEXICON_QUOTE}» therefore"]),
        checkpointer=MemorySaver())
    final = _run_to_end(graph, {"configurable": {"thread_id": "t12"}})

    entry = final["grounding_log"][0]
    assert entry["verified"] is True
    assert entry["foreign_commentary"] is False


def test_found_in_is_populated_for_verified_and_empty_for_unverified(trace):
    corpus = FakeCorpus()
    graph = build_inquiry_graph(
        _deps(trace, corpus, ["isnad", "matn", f"real: {GENUINE} fake: {FABRICATED}"]),
        checkpointer=MemorySaver())
    final = _run_to_end(graph, {"configurable": {"thread_id": "t13"}})

    by_verified = {r["verified"]: r for r in final["grounding_log"]}
    assert by_verified[True]["found_in"] == ["hadith:bukhari_raw.json"]
    assert by_verified[False]["found_in"] == []


# --- Defect 2: the detector counting the question back at itself ------------------

# --- Synthesis prefetch: the stage that must quote sees what was retrieved --------

def test_the_synthesis_prompt_contains_text_retrieved_by_an_earlier_stage(trace, tmp_path):
    """The whole point, one stage later: before this task synthesis was told to check
    and cite quotations while being shown none of the retrieved text."""
    corpus = FakeCorpus()
    graph = build_inquiry_graph(_deps(trace, corpus, ["isnad", "matn", "synth"]),
                                checkpointer=MemorySaver())
    _run_to_end(graph, {"configurable": {"thread_id": "t15"}})

    lines = [json.loads(line) for line
             in (tmp_path / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    synth_prompt = next(line["prompt"] for line in lines
                        if line["node"] == "inquiry_synthesis")
    assert HADITH in synth_prompt
    assert "RETRIEVED SOURCES" in synth_prompt


def test_final_sources_have_no_duplicates_after_a_full_run(trace):
    """synthesis prefetches state["sources"] wholesale (it must see what earlier stages
    read). If that prefetch were written back into the "sources" channel the way the
    matn stage's genuinely-new fetch is, every entry from the matn stage would appear
    twice in the final state -- corrupting the session record and inflating the source
    count. Without the fix this run would end with 4 entries, not 2."""
    corpus = FakeCorpus()
    graph = build_inquiry_graph(_deps(trace, corpus, ["isnad", "matn", "synth"]),
                                checkpointer=MemorySaver())
    final = _run_to_end(graph, {"configurable": {"thread_id": "t16"}})

    sources = final["sources"]
    seen = set()
    for item in sources:
        key = (item.get("source"), item.get("reference"), item.get("excerpt"))
        assert key not in seen, f"duplicate source entry: {item}"
        seen.add(key)
    # The matn stage's two prefetch hits (hadith + the commentator's own commentary,
    # Part 5 item 2); synthesis added none.
    assert len(sources) == 2


def test_the_rendered_synthesis_source_block_is_capped(trace):
    """129 sources at ~1500 chars each is far beyond a sane prompt. Build a state with
    more sources than SYNTHESIS_SOURCE_CAP and confirm they are not all rendered."""
    from hermeneutics.inquiry_graph import SYNTHESIS_SOURCE_CAP, _synthesis_user

    many = [{"source": "hadith:bukhari", "reference": f"ref-{i}",
             "excerpt": f"excerpt-marker-{i}"} for i in range(SYNTHESIS_SOURCE_CAP + 10)]
    state = {"question": "q", "isnad_output": "", "matn_output": ""}
    deps = InquiryDeps(model=None, trace=trace, manifest_id="m1", metrics=METRICS,
                       framework="", commentator="nawawi", corpus="muslim",
                       corpus_tools=None, condition="agent_method")

    prompt = _synthesis_user(state, deps, many)
    rendered = sum(1 for i in range(len(many)) if f"excerpt-marker-{i}" in prompt)

    assert rendered == SYNTHESIS_SOURCE_CAP
    assert rendered < len(many)
    print(f"rendered synthesis prompt size: {len(prompt)} chars, {rendered} sources")


def test_a_no_retrieval_condition_renders_no_sources_and_writes_none(trace, tmp_path):
    """no_sources retrieves nothing anywhere: the synthesis prefetch must not run either,
    and the source block must render empty, not fabricate content."""
    deps = InquiryDeps(
        model=FakeToolModel([AIMessage(content=r) for r in ["isnad", "matn", "synth"]]),
        trace=trace, manifest_id="m1", metrics=METRICS, framework="",
        commentator="nawawi", corpus="muslim", corpus_tools=None,
        condition="no_sources")
    graph = build_inquiry_graph(deps, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t17"}}
    final = graph.invoke({"condition": "no_sources", "question": "q",
                          "commentator": "nawawi", "corpus": "muslim",
                          "intervention_log": [], "sources": [],
                          "tool_calls_log": [], "grounding_log": []}, cfg)
    while "__interrupt__" in final:
        final = graph.invoke(Command(resume={"action": "approve"}), cfg)

    assert final.get("sources", []) == []

    lines = [json.loads(line) for line
             in (tmp_path / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    synth_prompt = next(line["prompt"] for line in lines
                        if line["node"] == "inquiry_synthesis")
    assert "(nothing retrieved)" in synth_prompt


def test_a_quotation_echoing_the_question_is_flagged_and_still_logged(trace):
    """Both smoke runs put fragments of the QUESTION in quotation marks. That is
    scare-quoting, not citation, and must still appear in the log (not silently
    dropped) so both the raw and the corrected rate stay computable."""
    question = "ما دلالة اختلاف الرواة في حديث النية؟"
    echo = "اختلاف الرواة في حديث النية"
    corpus = FakeCorpus()
    graph = build_inquiry_graph(
        _deps(trace, corpus, ["isnad", "matn", f"as he asks «{echo}» therefore"]),
        checkpointer=MemorySaver())
    final = _run_to_end(graph, {"configurable": {"thread_id": "t14"}}, question=question)

    assert len(final["grounding_log"]) == 1
    assert final["grounding_log"][0]["echoes_question"] is True
