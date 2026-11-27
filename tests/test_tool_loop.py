"""The loop: call tools until the model answers, then stop. Never return markup."""
import pytest
from langchain_core.messages import AIMessage

from hermeneutics.tool_loop import run_tools


class FakeToolModel:
    model_name = "fake"

    def __init__(self, messages):
        self._messages = list(messages)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self._messages.pop(0)


class FakeCorpus:
    """Two tools are enough to exercise dispatch, results and the sources channel."""

    def __init__(self, hits=None):
        self.queries = []
        self._hits = hits if hits is not None else [
            {"source": "hadith:bukhari_raw.json", "reference": "الإيمان 1", "excerpt": "نص"}]

    @staticmethod
    def tool_specs():
        return [{"type": "function", "function": {"name": "search_sources",
                                                  "parameters": {"type": "object",
                                                                 "properties": {}}}}]

    def search(self, query, source="all", **kw):
        self.queries.append((query, source))
        return list(self._hits)

    def verify_quote(self, quote, source="all"):
        return {"verified": True, "occurrences": []}

    def lookup_word(self, word, lexicon="all"):
        return {"word": word, "articles": []}

    def lookup_narrator(self, name, school="all"):
        return {"name": name, "matches": []}

    def lookup_isnad_network(self, name, corpus="all"):
        return {"name": name, "matches": []}


def _call(name, args, cid="c1"):
    return {"name": name, "args": args, "id": cid}


def test_the_loop_runs_a_tool_and_returns_the_answer_after_it(trace):
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[_call("search_sources", {"query": "النية"})]),
        AIMessage(content="the answer"),
    ])
    corpus = FakeCorpus()

    result = run_tools(model, "sys", "user", corpus, trace=trace, node="matn",
                       bab_ref="bukhari/matn", model_id="fake")

    assert result.text == "the answer"
    assert corpus.queries == [("النية", "all")]
    assert result.sources[0]["reference"] == "الإيمان 1"
    assert result.rounds == 1


def test_an_answer_with_no_tool_call_returns_immediately(trace):
    model = FakeToolModel([AIMessage(content="straight answer")])

    result = run_tools(model, "sys", "user", FakeCorpus(), trace=trace, node="matn",
                       bab_ref="bukhari/matn", model_id="fake")

    assert result.text == "straight answer"
    assert result.rounds == 0


def test_the_round_budget_forces_an_answer_instead_of_a_refusal(trace):
    """Four tools need more rounds than one did; running out is not a reason to
    discard the research (the 'I reached the research-tool limit' cell)."""
    calling = [AIMessage(content="", tool_calls=[_call("search_sources", {"query": "x"})])
               for _ in range(3)]
    model = FakeToolModel([*calling, AIMessage(content="answer from what I have")])

    result = run_tools(model, "sys", "user", FakeCorpus(), trace=trace, node="matn",
                       bab_ref="bukhari/matn", model_id="fake", max_rounds=2)

    assert result.text == "answer from what I have"
    assert result.rounds == 2
    assert result.wrapped_up is True


def test_an_unknown_tool_is_reported_to_the_model_not_raised(trace):
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[_call("no_such_tool", {})]),
        AIMessage(content="recovered"),
    ])

    result = run_tools(model, "sys", "user", FakeCorpus(), trace=trace, node="matn",
                       bab_ref="bukhari/matn", model_id="fake")

    assert result.text == "recovered"


def test_the_calls_log_records_tool_and_arguments(trace):
    model = FakeToolModel([
        AIMessage(content="", tool_calls=[_call("lookup_word", {"word": "بوائقه"})]),
        AIMessage(content="done"),
    ])

    result = run_tools(model, "sys", "user", FakeCorpus(), trace=trace, node="matn",
                       bab_ref="bukhari/matn", model_id="fake")

    assert result.calls == [{"tool": "lookup_word", "args": {"word": "بوائقه"}}]


def test_an_empty_answer_raises_rather_than_returning_nothing(trace):
    """call_model raises rather than let a thinking model's budget-swallowed reply pass
    as a real answer (see llm.call_model). run_tools must not be the place that quietly
    lets that same failure through as an empty string."""
    model = FakeToolModel([AIMessage(content="")])

    with pytest.raises(ValueError, match="matn"):
        run_tools(model, "sys", "user", FakeCorpus(), trace=trace, node="matn",
                 bab_ref="bukhari/matn", model_id="fake")


def test_an_empty_answer_after_the_round_budget_also_raises(trace):
    """The wrap-up call is deliberately offered no tools (Correction 1) so it cannot
    return another tool call -- but if it still comes back with no text, that is the
    same empty-answer failure and must not pass silently either (Correction 2)."""
    calling = [AIMessage(content="", tool_calls=[_call("search_sources", {"query": "x"})])
               for _ in range(3)]
    model = FakeToolModel([*calling, AIMessage(content="")])

    with pytest.raises(ValueError, match="matn"):
        run_tools(model, "sys", "user", FakeCorpus(), trace=trace, node="matn",
                 bab_ref="bukhari/matn", model_id="fake", max_rounds=2)


def test_an_empty_answer_error_names_the_model(trace):
    """Fix 3: the old message told every caller to lower STUDY_REASONING_EFFORT,
    which does nothing for a model that get_chat_model never caps -- name the
    model and give a hint that is actually true for it."""
    model = FakeToolModel([AIMessage(content="")])

    with pytest.raises(ValueError, match="glm-5.3-flash"):
        run_tools(model, "sys", "user", FakeCorpus(), trace=trace, node="matn",
                 bab_ref="bukhari/matn", model_id="glm-5.3-flash")


def test_an_empty_answer_from_openrouter_gets_openrouters_hint_not_saias(trace):
    """provider_of recovers the backend from the model object's base URL, with no
    new plumbing through run_tools's own signature -- an OpenRouter empty answer
    must not be told to fiddle with STUDY_REASONING_EFFORT the SAIA way."""
    model = FakeToolModel([AIMessage(content="")])
    model.openai_api_base = "https://openrouter.ai/api/v1"

    with pytest.raises(ValueError, match="OpenRouter's unified"):
        run_tools(model, "sys", "user", FakeCorpus(), trace=trace, node="matn",
                 bab_ref="bukhari/matn", model_id="qwen/qwen3.8-27b")
