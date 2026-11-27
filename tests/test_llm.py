import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from hermeneutics.llm import call_json, call_model, empty_answer_hint, get_chat_model, served_by


class _JsonOnlyWhenConstrained:
    """Behaves like the failed Falcon call unless JSON mode is requested."""

    model_name = "fake"

    def invoke(self, _messages):
        return AIMessage(content="not valid json")

    def bind(self, **kwargs):
        if kwargs.get("response_format") == {"type": "json_object"}:
            return _BoundJsonModel()
        return self


class _BoundJsonModel:
    def invoke(self, _messages):
        return AIMessage(content='{"ok": true}')


class _HumainPromptJsonModel:
    model_name = "humain-m3-preview"

    def invoke(self, _messages):
        return AIMessage(content='{"ok": true}')

    def bind(self, **_kwargs):
        raise AssertionError("HUMAIN must use prompt-only JSON")


def test_call_model_returns_text_and_traces_it(fake_model, trace, tmp_path):
    model = fake_model(["the answer"])
    out = call_model(model, "system text", "user text", trace=trace,
                     node="naive_reading", bab_ref="nawawi/1", model_id="fake")
    assert out == "the answer"

    rec = json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8"))
    assert rec["response"] == "the answer"
    assert "user text" in rec["prompt"]
    assert rec["node"] == "naive_reading"


def test_call_model_records_which_upstream_served_it(fake_model, trace, tmp_path):
    """With price-sorted (or any unpinned) OpenRouter routing, the manifest's
    provider_route cannot say who answered -- only the response's own "provider"
    field can. Since this survives only when response_metadata carries it (see
    served_by's docstring: the installed langchain_openai does not currently copy
    it there), this fakes that field directly to prove the trace plumbing works."""
    model = fake_model([AIMessage(content="the answer",
                                  response_metadata={"provider": "DeepInfra"})])
    call_model(model, "system text", "user text", trace=trace,
              node="naive_reading", bab_ref="nawawi/1", model_id="fake")

    rec = json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8"))
    assert rec["served_by"] == "DeepInfra"


def test_call_model_records_none_when_the_response_says_nothing(fake_model, trace, tmp_path):
    model = fake_model(["the answer"])
    call_model(model, "system text", "user text", trace=trace,
              node="naive_reading", bab_ref="nawawi/1", model_id="fake")

    rec = json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8"))
    assert rec["served_by"] is None


def test_served_by_reads_the_response_metadata_field():
    assert served_by(AIMessage(content="x", response_metadata={"provider": "OpenAI"})) == "OpenAI"


def test_served_by_is_none_without_a_provider_field():
    assert served_by(AIMessage(content="x")) is None
    assert served_by(None) is None


def test_call_json_retries_once_on_invalid_json(fake_model, trace):
    model = fake_model(["not json at all", '{"ok": true}'])
    out = call_json(model, "sys", "usr", trace=trace, node="horizon_diff",
                    bab_ref="nawawi/1", model_id="fake")
    assert out == {"ok": True}


def test_call_json_requests_server_side_json_mode(trace):
    out = call_json(_JsonOnlyWhenConstrained(), "sys", "usr", trace=trace,
                    node="horizon_diff", bab_ref="nawawi/1", model_id="fake")
    assert out == {"ok": True}


def test_call_json_uses_prompt_only_json_for_humain(trace):
    out = call_json(_HumainPromptJsonModel(), "sys", "usr", trace=trace,
                    node="horizon_diff", bab_ref="nawawi/1", model_id="humain-m3-preview")
    assert out == {"ok": True}


class _JsonModelWithProvider:
    model_name = "fake"

    def invoke(self, _messages):
        return AIMessage(content='{"ok": true}', response_metadata={"provider": "Groq"})

    def bind(self, **_kwargs):
        return self


def test_call_json_records_which_upstream_served_it(trace, tmp_path):
    call_json(_JsonModelWithProvider(), "sys", "usr", trace=trace,
             node="horizon_diff", bab_ref="nawawi/1", model_id="fake")
    rec = json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8"))
    assert rec["served_by"] == "Groq"


def test_call_json_raises_after_the_single_retry(fake_model, trace):
    model = fake_model(["garbage", "still garbage"])
    with pytest.raises(ValueError, match="valid JSON"):
        call_json(model, "sys", "usr", trace=trace, node="horizon_diff",
                  bab_ref="nawawi/1", model_id="fake")


def test_call_json_strips_markdown_fences(fake_model, trace):
    model = fake_model(['```json\n{"a": 1}\n```'])
    assert call_json(model, "s", "u", trace=trace, node="n",
                     bab_ref="x/0", model_id="fake") == {"a": 1}


def test_both_attempts_are_traced(fake_model, trace, tmp_path):
    model = fake_model(["bad", '{"ok": 1}'])
    call_json(model, "s", "u", trace=trace, node="horizon_diff",
              bab_ref="nawawi/1", model_id="fake")
    lines = (tmp_path / "test.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["error"] is not None
    assert json.loads(lines[1])["attempt"] == 2


class _UpstreamFlakyModel:
    """Raises the OpenRouter-shaped ValueError a scripted number of times, then
    returns a real reply -- the 'Provider timed out after 353032ms' shape from
    the live failure this fix targets."""

    model_name = "fake"

    def __init__(self, fail_times, error=None, reply=None):
        self.fail_times = fail_times
        self.calls = 0
        self.error = error if error is not None else {
            "message": "Provider timed out after 353032ms", "code": 504,
            "metadata": {"error_type": "timeout"},
        }
        self.reply = reply or AIMessage(content="the answer")

    def invoke(self, _messages, **_kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ValueError(self.error)
        return self.reply

    def bind(self, **_kwargs):
        return self

    def bind_tools(self, _tools):
        return self


def test_call_model_retries_a_transient_upstream_error_then_succeeds(trace, tmp_path):
    model = _UpstreamFlakyModel(fail_times=1)
    sleeps = []

    out = call_model(model, "sys", "usr", trace=trace, node="naive_reading",
                     bab_ref="nawawi/1", model_id="fake", sleep=sleeps.append)

    assert out == "the answer"
    assert model.calls == 2
    assert sleeps == [10]
    lines = (tmp_path / "test.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["attempt"] == 1 and first["error"] == "upstream_504"
    assert second["attempt"] == 2 and second["error"] is None and second["response"] == "the answer"


def test_call_model_reraises_the_transient_error_after_exhausting_retries(trace):
    model = _UpstreamFlakyModel(fail_times=99)
    sleeps = []

    with pytest.raises(ValueError) as exc:
        call_model(model, "sys", "usr", trace=trace, node="naive_reading",
                  bab_ref="nawawi/1", model_id="fake", sleep=sleeps.append)

    assert exc.value.args[0]["code"] == 504
    assert model.calls == 3  # 1 original + 2 retries, then give up
    assert sleeps == [10, 30]


def test_call_model_does_not_retry_a_non_transient_dict_error(trace):
    model = _UpstreamFlakyModel(fail_times=99, error={"message": "bad request", "code": 400})
    sleeps = []

    with pytest.raises(ValueError) as exc:
        call_model(model, "sys", "usr", trace=trace, node="naive_reading",
                  bab_ref="nawawi/1", model_id="fake", sleep=sleeps.append)

    assert exc.value.args[0]["code"] == 400
    assert model.calls == 1
    assert sleeps == []


def test_call_model_does_not_retry_a_plain_string_value_error(trace):
    model = _UpstreamFlakyModel(fail_times=99, error="something else went wrong")
    sleeps = []

    with pytest.raises(ValueError, match="something else went wrong"):
        call_model(model, "sys", "usr", trace=trace, node="naive_reading",
                  bab_ref="nawawi/1", model_id="fake", sleep=sleeps.append)

    assert model.calls == 1
    assert sleeps == []


def test_call_json_retries_a_transient_upstream_error_then_succeeds(trace, tmp_path):
    model = _UpstreamFlakyModel(fail_times=1, reply=AIMessage(content='{"ok": true}'))
    sleeps = []

    out = call_json(model, "sys", "usr", trace=trace, node="horizon_diff",
                    bab_ref="nawawi/1", model_id="fake", sleep=sleeps.append)

    assert out == {"ok": True}
    assert model.calls == 2
    assert sleeps == [10]
    lines = (tmp_path / "test.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["attempt"] == 1 and first["error"] == "upstream_504"
    assert second["attempt"] == 2 and second["error"] is None


def test_call_json_reraises_the_transient_error_after_exhausting_retries(trace):
    model = _UpstreamFlakyModel(fail_times=99, reply=AIMessage(content='{"ok": true}'))
    sleeps = []

    with pytest.raises(ValueError) as exc:
        call_json(model, "sys", "usr", trace=trace, node="horizon_diff",
                 bab_ref="nawawi/1", model_id="fake", sleep=sleeps.append)

    assert exc.value.args[0]["code"] == 504
    assert model.calls == 3
    assert sleeps == [10, 30]


def test_call_json_does_not_retry_a_non_transient_error(trace):
    model = _UpstreamFlakyModel(fail_times=99, error={"message": "bad request", "code": 400},
                                reply=AIMessage(content='{"ok": true}'))
    sleeps = []

    with pytest.raises(ValueError) as exc:
        call_json(model, "sys", "usr", trace=trace, node="horizon_diff",
                 bab_ref="nawawi/1", model_id="fake", sleep=sleeps.append)

    assert exc.value.args[0]["code"] == 400
    assert model.calls == 1
    assert sleeps == []


def test_get_chat_model_uses_greedy_decoding(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "Qwen/Qwen2.5-32B-Instruct")
    monkeypatch.setenv("VLLM_API_KEY", "dummy")
    m = get_chat_model(max_tokens=1500)
    assert m.temperature == 0
    assert m.max_tokens == 1500


def test_get_chat_model_passes_reasoning_effort_to_chat_template(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "Qwen/Qwen3.8-27B")
    monkeypatch.setenv("VLLM_API_KEY", "dummy")
    monkeypatch.setenv("STUDY_REASONING_EFFORT", "medium")

    model = get_chat_model(max_tokens=8192)

    assert model.extra_body == {
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "medium",
        }
    }


def test_get_chat_model_can_disable_thinking(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "Qwen/Qwen3.8-27B")
    monkeypatch.setenv("VLLM_API_KEY", "dummy")
    monkeypatch.setenv("STUDY_REASONING_EFFORT", "medium")

    model = get_chat_model(max_tokens=200, enable_thinking=False)

    assert model.extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_get_chat_model_does_not_send_qwen_options_to_other_models(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "Falcon-H1-Arabic-34B-Instruct")
    monkeypatch.setenv("VLLM_API_KEY", "dummy")

    model = get_chat_model(max_tokens=200, enable_thinking=False)

    assert model.extra_body is None


def test_get_chat_model_uses_configured_request_timeout(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "Qwen/Qwen3.8-27B")
    monkeypatch.setenv("VLLM_API_KEY", "dummy")
    monkeypatch.setenv("STUDY_LLM_TIMEOUT", "3600")

    model = get_chat_model(max_tokens=8192)

    assert model.request_timeout == 3600


class _CappedModel:
    """vLLM stopped at max_tokens. On a thinking model the reasoning ate the
    budget, so content is empty and finish_reason is 'length' — job 15849569."""

    model_name = "fake"

    def __init__(self, content=""):
        self._content = content

    def invoke(self, _messages):
        return AIMessage(content=self._content,
                         response_metadata={"finish_reason": "length"})

    def bind(self, **_kwargs):
        return self


def test_call_model_raises_when_the_budget_went_to_reasoning(trace):
    with pytest.raises(ValueError, match="empty answer"):
        call_model(_CappedModel(), "sys", "usr", trace=trace,
                   node="naive_reading", bab_ref="nawawi/1", model_id="fake")


def test_call_model_traces_the_empty_answer_before_raising(trace, tmp_path):
    with pytest.raises(ValueError):
        call_model(_CappedModel(), "sys", "usr", trace=trace,
                   node="naive_reading", bab_ref="nawawi/1", model_id="fake")
    rec = json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8"))
    assert rec["error"] == "length"


def test_call_model_flags_truncation_but_keeps_a_partial_answer(trace, tmp_path):
    out = call_model(_CappedModel("half a rea"), "sys", "usr", trace=trace,
                     node="naive_reading", bab_ref="nawawi/1", model_id="fake")
    assert out == "half a rea"
    assert json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8"))["error"] == "length"


class _ScriptedCappedModel:
    """Returns scripted AIMessages in order and records the kwargs (max_tokens,
    in particular) each invoke received, so Fix 2's doubled retry budget can be
    checked. GenericFakeChatModel does not expose invoke-time kwargs, hence a
    tiny fake here instead."""

    model_name = "fake"

    def __init__(self, replies, max_tokens=4000):
        self._replies = list(replies)
        self.max_tokens = max_tokens
        self.invocations = []

    def invoke(self, _messages, **kwargs):
        self.invocations.append(kwargs)
        return self._replies.pop(0)

    def bind(self, **_kwargs):
        return self


def _capped(content=""):
    return AIMessage(content=content, response_metadata={"finish_reason": "length"})


def test_call_model_retries_once_with_doubled_budget_on_an_empty_capped_reply(trace):
    """job trace: glm-5.3-flash, tokens_in 9037, tokens_out 4000, error "length",
    response "" -- the thinking budget ate all of max_tokens=4000. Retrying once
    with double the budget is Fix 2."""
    model = _ScriptedCappedModel([_capped(""), AIMessage(content="the answer")],
                                 max_tokens=4000)

    out = call_model(model, "sys", "usr", trace=trace, node="naive_reading",
                     bab_ref="nawawi/1", model_id="glm-5.3-flash")

    assert out == "the answer"
    assert len(model.invocations) == 2
    assert model.invocations[0] == {}
    assert model.invocations[1]["max_tokens"] == 8000


def test_call_model_caps_the_retry_budget_at_32768(trace):
    model = _ScriptedCappedModel([_capped(""), AIMessage(content="ok")], max_tokens=20000)

    call_model(model, "sys", "usr", trace=trace, node="naive_reading",
              bab_ref="nawawi/1", model_id="glm-5.3-flash")

    assert model.invocations[1]["max_tokens"] == 32768


def test_call_model_traces_both_attempts(trace, tmp_path):
    model = _ScriptedCappedModel([_capped(""), AIMessage(content="the answer")])

    call_model(model, "sys", "usr", trace=trace, node="naive_reading",
              bab_ref="nawawi/1", model_id="glm-5.3-flash")

    lines = (tmp_path / "test.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["attempt"] == 1 and first["error"] == "length"
    assert second["attempt"] == 2


def test_call_model_still_raises_when_the_retry_is_also_empty(trace):
    model = _ScriptedCappedModel([_capped(""), _capped("")])

    with pytest.raises(ValueError, match="empty answer"):
        call_model(model, "sys", "usr", trace=trace, node="naive_reading",
                  bab_ref="nawawi/1", model_id="glm-5.3-flash")

    assert len(model.invocations) == 2


def test_call_model_does_not_retry_an_empty_reply_that_did_not_hit_the_cap(trace):
    """finish_reason 'stop' with empty content is a different failure, not a
    truncated thinking budget, so it must not trigger Fix 2's retry."""
    model = _ScriptedCappedModel(
        [AIMessage(content="", response_metadata={"finish_reason": "stop"})])

    with pytest.raises(ValueError, match="empty answer"):
        call_model(model, "sys", "usr", trace=trace, node="naive_reading",
                  bab_ref="nawawi/1", model_id="fake")

    assert len(model.invocations) == 1


def test_empty_answer_hint_for_qwen3_names_the_model_and_the_env_var(monkeypatch):
    monkeypatch.delenv("STUDY_REASONING_EFFORT", raising=False)
    hint = empty_answer_hint("Qwen/Qwen3.8-27B")
    assert "Qwen/Qwen3.8-27B" in hint
    assert "STUDY_REASONING_EFFORT" in hint
    assert "medium" in hint  # get_chat_model's own qwen3 default


def test_empty_answer_hint_for_qwen3_reflects_an_explicit_env_value(monkeypatch):
    monkeypatch.setenv("STUDY_REASONING_EFFORT", "xhigh")
    hint = empty_answer_hint("Qwen/Qwen3.8-27B")
    assert "xhigh" in hint


def test_empty_answer_hint_for_glm_names_the_model_and_the_env_var(monkeypatch):
    monkeypatch.delenv("STUDY_REASONING_EFFORT", raising=False)
    hint = empty_answer_hint("glm-5.3-flash")
    assert "glm-5.3-flash" in hint
    assert "STUDY_REASONING_EFFORT" in hint
    assert "low" in hint  # get_chat_model's own glm default


def test_empty_answer_hint_for_an_uncapped_model_names_it_but_not_the_env_var():
    """Fix 3: the old message told every caller to lower STUDY_REASONING_EFFORT,
    which does nothing for a model get_chat_model never applies a cap to."""
    hint = empty_answer_hint("llama-3.3-70b-instruct")
    assert "llama-3.3-70b-instruct" in hint
    assert "STUDY_REASONING_EFFORT" not in hint
    assert "qwen3" in hint.lower()
    assert "glm" in hint.lower()


def test_call_model_error_names_the_model(trace):
    with pytest.raises(ValueError, match="glm-5.3-flash"):
        call_model(_CappedModel(), "sys", "usr", trace=trace,
                  node="naive_reading", bab_ref="nawawi/1", model_id="glm-5.3-flash")


def test_call_model_error_for_an_uncapped_model_does_not_blame_the_env_var(trace):
    with pytest.raises(ValueError) as exc:
        call_model(_CappedModel(), "sys", "usr", trace=trace,
                  node="naive_reading", bab_ref="nawawi/1",
                  model_id="llama-3.3-70b-instruct")
    assert "STUDY_REASONING_EFFORT" not in str(exc.value)


def test_call_model_does_not_retry_when_no_budget_can_be_found(trace, tmp_path):
    """_CappedModel carries no max_tokens attribute, so _next_budget must return
    None and the call falls through to the existing failure path unretried: one
    trace line, not two."""
    with pytest.raises(ValueError, match="empty answer"):
        call_model(_CappedModel(), "sys", "usr", trace=trace,
                  node="naive_reading", bab_ref="nawawi/1", model_id="fake")
    lines = (tmp_path / "test.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1


class _SdkLengthThenJson:
    """The SDK's structured-output parser raises before langchain builds a response,
    so _hit_the_cap never sees the truncation -- mazandarani died on this, 24 min in."""

    model_name = "fake"

    def __init__(self):
        self.calls = 0
        self.prompts = []

    def invoke(self, messages):
        from openai import LengthFinishReasonError
        self.calls += 1
        self.prompts.append(messages[-1].content)
        if self.calls == 1:
            # The real usage from the mazandarani failure: the whole 8192-token
            # budget went out before the JSON closed.
            usage = SimpleNamespace(completion_tokens=8192, prompt_tokens=6974,
                                    total_tokens=15166)
            raise LengthFinishReasonError(
                completion=SimpleNamespace(usage=usage))
        return AIMessage(content='{"ok": true}')

    def bind(self, **_kwargs):
        return self


def test_call_json_retries_when_the_sdk_raises_on_a_truncated_reply(trace):
    model = _SdkLengthThenJson()
    assert call_json(model, "sys", "usr", trace=trace,
                     node="read_commentary", bab_ref="mazandarani/1",
                     model_id="fake") == {"ok": True}
    assert model.calls == 2


def test_the_length_retry_asks_for_brevity_not_for_valid_json(trace):
    """Telling a model that overran the cap its JSON was invalid only makes it
    write more. The retry prompt has to ask for less."""
    model = _SdkLengthThenJson()
    call_json(model, "sys", "usr", trace=trace, node="read_commentary",
              bab_ref="mazandarani/1", model_id="fake")
    assert "more briefly" in model.prompts[1]
    assert "not valid JSON" not in model.prompts[1]


def test_call_json_traces_the_sdk_length_failure(trace, tmp_path):
    call_json(_SdkLengthThenJson(), "sys", "usr", trace=trace,
              node="read_commentary", bab_ref="mazandarani/1", model_id="fake")
    first = json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["error"] == "length"


def test_call_json_rejects_json_that_stopped_at_the_cap(trace):
    with pytest.raises(ValueError, match="valid JSON"):
        call_json(_CappedModel('{"ok": true}'), "sys", "usr", trace=trace,
                  node="horizon_diff", bab_ref="nawawi/1", model_id="fake")


def test_get_chat_model_defaults_qwen3_to_a_bounded_thinking_budget(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "Qwen/Qwen3.8-27B")
    monkeypatch.setenv("VLLM_API_KEY", "dummy")
    monkeypatch.delenv("STUDY_REASONING_EFFORT", raising=False)

    model = get_chat_model(max_tokens=8192)

    assert model.extra_body == {
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"}
    }


def test_get_chat_model_takes_model_endpoint_and_key_per_call(monkeypatch):
    """The UI picks a model per run; a process-wide VLLM_MODEL_ID would race between
    worker threads."""
    for var in ("VLLM_MODEL_ID", "VLLM_BASE_URL", "VLLM_API_KEY", "STUDY_BRIDGE_DIR",
                "STUDY_REASONING_EFFORT"):
        monkeypatch.delenv(var, raising=False)

    m = get_chat_model(max_tokens=100, model_id="qwen3.8-27b",
                       base_url="https://example.invalid/v1", api_key="k")

    assert m.model_name == "qwen3.8-27b"
    assert str(m.openai_api_base) == "https://example.invalid/v1"
    assert m.extra_body["chat_template_kwargs"]["enable_thinking"] is True


def test_a_non_qwen3_model_gets_no_thinking_kwargs(monkeypatch):
    for var in ("VLLM_MODEL_ID", "VLLM_BASE_URL", "STUDY_BRIDGE_DIR"):
        monkeypatch.delenv(var, raising=False)

    m = get_chat_model(max_tokens=100, model_id="llama-3.3-70b-instruct",
                       base_url="https://example.invalid/v1", api_key="k")

    assert m.extra_body is None


def test_env_still_supplies_the_model_when_none_is_passed(monkeypatch):
    monkeypatch.delenv("STUDY_BRIDGE_DIR", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "Qwen/Qwen2.5-32B-Instruct")
    assert get_chat_model(max_tokens=100).model_name == "Qwen/Qwen2.5-32B-Instruct"


# Fix 1: GLM-5.3 is a thinking model too, and reasoning control in get_chat_model
# used to fire only for qwen3.* ids. A glm-5.3-flash run died on its very first
# call: tokens_in 9037, tokens_out 4000, error "length", response "" -- the whole
# max_tokens budget went to reasoning and left content empty.
def test_get_chat_model_caps_glm_reasoning(monkeypatch):
    monkeypatch.delenv("STUDY_BRIDGE_DIR", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "glm-5.3-flash")
    monkeypatch.setenv("VLLM_API_KEY", "dummy")
    monkeypatch.delenv("STUDY_REASONING_EFFORT", raising=False)

    model = get_chat_model(max_tokens=4000)

    assert model.extra_body == {"chat_template_kwargs": {"reasoning_effort": "low"}}


def test_get_chat_model_glm_honours_an_explicit_reasoning_effort(monkeypatch):
    monkeypatch.delenv("STUDY_BRIDGE_DIR", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "glm-5.3-flash")
    monkeypatch.setenv("VLLM_API_KEY", "dummy")
    monkeypatch.setenv("STUDY_REASONING_EFFORT", "medium")

    model = get_chat_model(max_tokens=4000)

    assert model.extra_body == {"chat_template_kwargs": {"reasoning_effort": "medium"}}


def test_get_chat_model_never_sends_enable_thinking_to_glm(monkeypatch):
    """Verified on SAIA, 2026-09-22: chat_template_kwargs.enable_thinking=False gives
    0 reasoning tokens but GLM then writes its reasoning into `content` instead
    ("The user is asking about...") -- unusable, so the caller's enable_thinking
    argument (used to disable Qwen3's thinking) must be ignored for GLM."""
    monkeypatch.delenv("STUDY_BRIDGE_DIR", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "glm-5.3-flash")
    monkeypatch.setenv("VLLM_API_KEY", "dummy")
    monkeypatch.delenv("STUDY_REASONING_EFFORT", raising=False)

    model = get_chat_model(max_tokens=4000, enable_thinking=False)

    assert model.extra_body == {"chat_template_kwargs": {"reasoning_effort": "low"}}
