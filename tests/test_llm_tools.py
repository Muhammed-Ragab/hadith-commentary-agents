"""A tool round is a model call, so it obeys llm.py's rule: everything lands in the trace."""
import json

import pytest
from langchain_core.messages import AIMessage

from hermeneutics.llm import call_with_tools, recover_text_tool_calls

SPECS = [{"type": "function", "function": {"name": "search_sources",
                                           "parameters": {"type": "object",
                                                          "properties": {}}}}]


class FakeToolModel:
    """Returns one scripted AIMessage per invoke; records what it was bound with."""

    model_name = "fake"

    def __init__(self, messages):
        self._messages = list(messages)
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        return self._messages.pop(0)


def test_a_tool_call_is_returned_structured(trace):
    model = FakeToolModel([AIMessage(content="", tool_calls=[
        {"name": "search_sources", "args": {"query": "x"}, "id": "c1"}])])

    message = call_with_tools(model, [{"role": "user", "content": "q"}], SPECS,
                              trace=trace, node="matn", bab_ref="bukhari/matn",
                              model_id="fake")

    assert message["tool_calls"][0]["function"]["name"] == "search_sources"


def test_every_tool_round_lands_in_the_trace(trace, tmp_path):
    model = FakeToolModel([AIMessage(content="an answer")])

    call_with_tools(model, [{"role": "user", "content": "q"}], SPECS,
                    trace=trace, node="matn", bab_ref="bukhari/matn", model_id="fake")

    line = json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8").strip())
    assert line["node"] == "matn"
    assert line["response"] == "an answer"


def test_a_tool_round_records_which_upstream_served_it(trace, tmp_path):
    model = FakeToolModel([AIMessage(content="an answer",
                                     response_metadata={"provider": "DeepInfra"})])

    call_with_tools(model, [{"role": "user", "content": "q"}], SPECS,
                    trace=trace, node="matn", bab_ref="bukhari/matn", model_id="fake")

    line = json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8").strip())
    assert line["served_by"] == "DeepInfra"


def test_a_tool_call_written_as_text_is_recovered():
    """Qwen writes the call as markup when the server does not parse it (08feea5)."""
    message = recover_text_tool_calls({
        "role": "assistant",
        "content": ("first, the word: <tool_call><function=lookup_word>"
                    "<parameter=word>بوائقه</parameter></function></tool_call>"),
    })

    assert message["tool_calls"][0]["function"]["name"] == "lookup_word"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"])["word"] == "بوائقه"
    assert message["content"] == "first, the word:"


def test_a_message_with_no_markup_is_returned_unchanged():
    message = {"role": "assistant", "content": "just prose"}
    assert recover_text_tool_calls(message) == message


def test_no_specs_omits_the_tools_field_instead_of_sending_an_empty_array(trace):
    """bind_tools([]) serialises to "tools": [], and SAIA rejects that with a 400:
    "must not be an empty array ... or omit the field entirely". The only caller that
    passes no specs is the wrap-up in tool_loop.run_tools, whose entire purpose is to
    stop the model calling more tools -- so every stage that exhausted its tool rounds
    died at the one call meant to rescue it. It killed the cayni arm three times."""
    model = FakeToolModel([AIMessage(content="the answer")])

    message = call_with_tools(model, [{"role": "user", "content": "q"}], [],
                              trace=trace, node="matn_wrap_up", bab_ref="bukhari/matn",
                              model_id="m")

    assert model.bound_tools is None, "bind_tools must not be called with no specs"
    assert message["content"] == "the answer"


def test_specs_are_still_bound_when_there_are_any(trace):
    model = FakeToolModel([AIMessage(content="ok")])

    call_with_tools(model, [{"role": "user", "content": "q"}], SPECS,
                    trace=trace, node="matn", bab_ref="bukhari/matn", model_id="m")

    assert model.bound_tools == SPECS


def _capped(content=""):
    return AIMessage(content=content, response_metadata={"finish_reason": "length"})


class _RunnableBindingLike:
    """Stands in for langchain's RunnableBinding: the real model (carrying
    max_tokens) lives at .bound, exactly where Fix 2's budget lookup checks."""

    def __init__(self, bound):
        self.bound = bound

    def invoke(self, messages, **kwargs):
        return self.bound.invoke(messages, **kwargs)


class _ScriptedCappedToolModel:
    """bind_tools returns a RunnableBinding-shaped wrapper; invoke records the
    kwargs (max_tokens, in particular) it received so Fix 2's doubled retry
    budget can be checked."""

    model_name = "fake"

    def __init__(self, replies, max_tokens=4000):
        self._replies = list(replies)
        self.max_tokens = max_tokens
        self.invocations = []

    def bind_tools(self, tools):
        return _RunnableBindingLike(self)

    def invoke(self, _messages, **kwargs):
        self.invocations.append(kwargs)
        return self._replies.pop(0)


def test_call_with_tools_retries_once_with_doubled_budget_on_an_empty_capped_reply(trace):
    model = _ScriptedCappedToolModel([_capped(""), AIMessage(content="the answer")],
                                     max_tokens=4000)

    message = call_with_tools(model, [{"role": "user", "content": "q"}], SPECS,
                              trace=trace, node="matn", bab_ref="bukhari/matn",
                              model_id="glm-5.3-flash")

    assert message["content"] == "the answer"
    assert len(model.invocations) == 2
    assert model.invocations[0] == {}
    assert model.invocations[1]["max_tokens"] == 8000


def test_call_with_tools_caps_the_retry_budget_at_32768(trace):
    model = _ScriptedCappedToolModel([_capped(""), AIMessage(content="ok")], max_tokens=20000)

    call_with_tools(model, [{"role": "user", "content": "q"}], SPECS, trace=trace,
                    node="matn", bab_ref="bukhari/matn", model_id="glm-5.3-flash")

    assert model.invocations[1]["max_tokens"] == 32768


def test_call_with_tools_traces_both_attempts(trace, tmp_path):
    model = _ScriptedCappedToolModel([_capped(""), AIMessage(content="the answer")])

    call_with_tools(model, [{"role": "user", "content": "q"}], SPECS, trace=trace,
                    node="matn", bab_ref="bukhari/matn", model_id="glm-5.3-flash")

    lines = (tmp_path / "test.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["attempt"] == 1 and first["error"] == "length"
    assert second["attempt"] == 2


def test_call_with_tools_returns_the_message_unretried_when_the_retry_is_also_empty(trace):
    """Fall-through per Fix 2: call_with_tools returns the (still empty) message
    rather than raising -- run_tools is the layer that raises on an empty answer."""
    model = _ScriptedCappedToolModel([_capped(""), _capped("")])

    message = call_with_tools(model, [{"role": "user", "content": "q"}], SPECS,
                              trace=trace, node="matn", bab_ref="bukhari/matn",
                              model_id="glm-5.3-flash")

    assert message["content"] == ""
    assert len(model.invocations) == 2


def test_call_with_tools_does_not_retry_an_empty_reply_that_did_not_hit_the_cap(trace):
    model = _ScriptedCappedToolModel(
        [AIMessage(content="", response_metadata={"finish_reason": "stop"})])

    message = call_with_tools(model, [{"role": "user", "content": "q"}], SPECS,
                              trace=trace, node="matn", bab_ref="bukhari/matn",
                              model_id="fake")

    assert message["content"] == ""
    assert len(model.invocations) == 1


class _UpstreamFlakyToolModel:
    """Raises the OpenRouter-shaped ValueError a scripted number of times, then
    returns a real reply -- the tool-loop path's version of the same live
    failure (a 504 in an HTTP-200 body) that call_model/call_json retry."""

    model_name = "fake"

    def __init__(self, fail_times, error=None, reply=None):
        self.fail_times = fail_times
        self.calls = 0
        self.error = error if error is not None else {
            "message": "Provider timed out after 353032ms", "code": 504,
            "metadata": {"error_type": "timeout"},
        }
        self.reply = reply or AIMessage(content="the answer")

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages, **_kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ValueError(self.error)
        return self.reply


def test_call_with_tools_retries_a_transient_upstream_error_then_succeeds(trace, tmp_path):
    model = _UpstreamFlakyToolModel(fail_times=1)
    sleeps = []

    message = call_with_tools(model, [{"role": "user", "content": "q"}], SPECS,
                              trace=trace, node="matn", bab_ref="bukhari/matn",
                              model_id="fake", sleep=sleeps.append)

    assert message["content"] == "the answer"
    assert model.calls == 2
    assert sleeps == [10]
    lines = (tmp_path / "test.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["attempt"] == 1 and first["error"] == "upstream_504"
    assert second["attempt"] == 2 and second["error"] is None


def test_call_with_tools_reraises_the_transient_error_after_exhausting_retries(trace):
    model = _UpstreamFlakyToolModel(fail_times=99)
    sleeps = []

    with pytest.raises(ValueError) as exc:
        call_with_tools(model, [{"role": "user", "content": "q"}], SPECS, trace=trace,
                        node="matn", bab_ref="bukhari/matn", model_id="fake",
                        sleep=sleeps.append)

    assert exc.value.args[0]["code"] == 504
    assert model.calls == 3
    assert sleeps == [10, 30]


def test_call_with_tools_does_not_retry_a_non_transient_error(trace):
    model = _UpstreamFlakyToolModel(fail_times=99, error={"message": "bad request", "code": 400})
    sleeps = []

    with pytest.raises(ValueError) as exc:
        call_with_tools(model, [{"role": "user", "content": "q"}], SPECS, trace=trace,
                        node="matn", bab_ref="bukhari/matn", model_id="fake",
                        sleep=sleeps.append)

    assert exc.value.args[0]["code"] == 400
    assert model.calls == 1
    assert sleeps == []


def test_call_with_tools_does_not_retry_a_plain_string_value_error(trace):
    model = _UpstreamFlakyToolModel(fail_times=99, error="something else went wrong")
    sleeps = []

    with pytest.raises(ValueError, match="something else went wrong"):
        call_with_tools(model, [{"role": "user", "content": "q"}], SPECS, trace=trace,
                        node="matn", bab_ref="bukhari/matn", model_id="fake",
                        sleep=sleeps.append)

    assert model.calls == 1
    assert sleeps == []


def test_call_with_tools_does_not_retry_a_tool_call_even_if_capped(trace):
    """A tool call is a normal round, not an empty-answer failure -- it must not
    be retried just because finish_reason happened to be 'length' too."""
    model = _ScriptedCappedToolModel([
        AIMessage(content="", tool_calls=[{"name": "search_sources", "args": {"query": "x"},
                                          "id": "c1"}],
                 response_metadata={"finish_reason": "length"}),
    ])

    message = call_with_tools(model, [{"role": "user", "content": "q"}], SPECS,
                              trace=trace, node="matn", bab_ref="bukhari/matn",
                              model_id="fake")

    assert message["tool_calls"][0]["function"]["name"] == "search_sources"
    assert len(model.invocations) == 1
