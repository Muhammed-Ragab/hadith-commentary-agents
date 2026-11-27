"""OpenRouter as a second backend for get_chat_model (spec Part 1).

SAIA's extra_body must stay byte-identical -- pinned here before anything about
get_chat_model changes, so a regression in either branch fails loudly. OpenRouter's
extra_body must never carry chat_template_kwargs (that is SAIA/vLLM's knob only);
it carries OpenRouter's own unified `reasoning` parameter and a `provider` block
that pins the upstream serving stack, with allow_fallbacks always False so a silent
fallback can never mix serving stacks within one run.
"""
import pytest
from langchain_core.messages import AIMessage

from hermeneutics.llm import call_model, empty_answer_hint, get_chat_model, provider_of
from hermeneutics.openrouter import (
    DEFAULT_PROVIDER_ROUTE,
    DEFAULT_QUANTIZATION,
    OPENROUTER_BASE_URL,
    openrouter_key,
)


def _clean_env(monkeypatch):
    for var in ("VLLM_MODEL_ID", "VLLM_BASE_URL", "VLLM_API_KEY", "STUDY_BRIDGE_DIR",
                "STUDY_REASONING_EFFORT", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# --- pin SAIA's extra_body exactly, both for qwen3 and for GLM, before anything
# about get_chat_model is touched for OpenRouter's sake. ------------------------

def test_saia_qwen3_extra_body_is_pinned(monkeypatch):
    _clean_env(monkeypatch)
    m = get_chat_model(max_tokens=100, model_id="qwen3.8-27b",
                       base_url="https://example.invalid/v1", api_key="k")
    assert m.extra_body == {
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"}
    }


def test_saia_glm_extra_body_is_pinned(monkeypatch):
    _clean_env(monkeypatch)
    m = get_chat_model(max_tokens=100, model_id="glm-5.3-flash",
                       base_url="https://example.invalid/v1", api_key="k")
    assert m.extra_body == {"chat_template_kwargs": {"reasoning_effort": "low"}}


def test_saia_is_still_the_default_provider(monkeypatch):
    _clean_env(monkeypatch)
    m = get_chat_model(max_tokens=100, model_id="qwen3.8-27b",
                       base_url="https://example.invalid/v1", api_key="k")
    assert "reasoning" not in (m.extra_body or {})
    assert "provider" not in (m.extra_body or {})


# --- OpenRouter body -----------------------------------------------------------

def test_openrouter_pins_the_provider_route_with_no_fallback(monkeypatch):
    _clean_env(monkeypatch)
    m = get_chat_model(max_tokens=100, model_id="qwen/qwen3.8-27b",
                       base_url=OPENROUTER_BASE_URL, api_key="k",
                       provider="openrouter", provider_route="deepinfra",
                       quantization="bf16")
    assert m.extra_body["provider"] == {
        "order": ["deepinfra"], "allow_fallbacks": False,
        "quantizations": ["bf16"], "data_collection": "deny",
    }


def test_openrouter_never_sends_chat_template_kwargs_even_for_a_qwen3_id(monkeypatch):
    """qwen/qwen3.8-27b matches the same "qwen3." substring SAIA's detection uses,
    but OpenRouter must never receive chat_template_kwargs -- that knob means
    nothing to OpenRouter's own router."""
    _clean_env(monkeypatch)
    m = get_chat_model(max_tokens=100, model_id="qwen/qwen3.8-27b",
                       base_url=OPENROUTER_BASE_URL, api_key="k",
                       provider="openrouter", provider_route="deepinfra")
    assert "chat_template_kwargs" not in m.extra_body


def test_openrouter_reasoning_defaults_to_medium(monkeypatch):
    _clean_env(monkeypatch)
    m = get_chat_model(max_tokens=100, model_id="qwen/qwen3.8-27b",
                       base_url=OPENROUTER_BASE_URL, api_key="k",
                       provider="openrouter", provider_route="deepinfra")
    assert m.extra_body["reasoning"] == {"effort": "medium"}


def test_openrouter_reasoning_effort_is_overridden_by_the_env_var(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("STUDY_REASONING_EFFORT", "high")
    m = get_chat_model(max_tokens=100, model_id="qwen/qwen3.8-27b",
                       base_url=OPENROUTER_BASE_URL, api_key="k",
                       provider="openrouter", provider_route="deepinfra")
    assert m.extra_body["reasoning"] == {"effort": "high"}


def test_openrouter_frontier_model_with_no_route_sends_only_data_collection(monkeypatch):
    """A frontier model (condition no_sources) is not pinned to any upstream stack --
    only the privacy flag goes in the provider block."""
    _clean_env(monkeypatch)
    m = get_chat_model(max_tokens=100, model_id="openai/placeholder-frontier",
                       base_url=OPENROUTER_BASE_URL, api_key="k",
                       provider="openrouter", provider_route=None)
    assert m.extra_body["provider"] == {"data_collection": "deny"}


def test_openrouter_price_route_sorts_by_price_with_no_pin(monkeypatch):
    """--provider-route price (reserved): cheapest available upstream, no pin and no
    quantization filter -- meant for the frontier model in condition no_sources."""
    _clean_env(monkeypatch)
    m = get_chat_model(max_tokens=100, model_id="openai/gpt-5.6-luna",
                       base_url=OPENROUTER_BASE_URL, api_key="k",
                       provider="openrouter", provider_route="price",
                       quantization="bf16")
    assert m.extra_body["provider"] == {"sort": "price", "data_collection": "deny"}


# --- empty_answer_hint -----------------------------------------------------

def test_empty_answer_hint_names_openrouters_unified_reasoning_param(monkeypatch):
    monkeypatch.delenv("STUDY_REASONING_EFFORT", raising=False)
    hint = empty_answer_hint("qwen/qwen3.8-27b", provider="openrouter")
    assert "reasoning" in hint
    assert "medium" in hint


def test_empty_answer_hint_still_talks_about_saia_by_default():
    hint = empty_answer_hint("llama-3.3-70b-instruct")
    assert "reasoning cap from this code" in hint


# --- openrouter_key: same registry fallback as saia_key ------------------------

def test_openrouter_key_reads_from_environ(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert openrouter_key() == "sk-or-test"


def test_openrouter_key_falls_back_to_empty_string_off_windows(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # winreg only exists on Windows; on any platform this must not raise.
    assert isinstance(openrouter_key(), str)


def test_openrouter_defaults():
    assert DEFAULT_PROVIDER_ROUTE == "deepinfra"
    assert DEFAULT_QUANTIZATION == "bf16"
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


# --- provider_of: recovering which backend served a call from the model object
# alone, since get_chat_model's own knowledge of "openrouter" does not otherwise
# survive past the ChatOpenAI it returns. -----------------------------------------

class _WithBaseUrl:
    def __init__(self, base_url):
        self.openai_api_base = base_url


def test_provider_of_recognises_openrouters_base_url():
    assert provider_of(_WithBaseUrl(OPENROUTER_BASE_URL)) == "openrouter"


def test_provider_of_defaults_to_saia_for_anything_else():
    assert provider_of(_WithBaseUrl("https://chat-ai.academiccloud.de/v1")) == "saia"
    assert provider_of(object()) == "saia"   # no openai_api_base at all


def test_provider_of_unwraps_bind_tools_runnable_binding():
    """bind_tools wraps the real ChatOpenAI in a RunnableBinding under .bound --
    every tool-calling model would otherwise read back as saia."""
    class _Bound:
        def __init__(self, inner):
            self.bound = inner

    assert provider_of(_Bound(_WithBaseUrl(OPENROUTER_BASE_URL))) == "openrouter"


class _OpenRouterCappedModel:
    """An OpenRouter-backed model whose reply is empty and length-capped, with no
    max_tokens to retry against -- call_model must raise immediately, with a hint
    that names OpenRouter's own reasoning knob, not SAIA's."""

    model_name = "openai/placeholder-frontier"
    openai_api_base = OPENROUTER_BASE_URL

    def invoke(self, _messages, **_kwargs):
        return AIMessage(content="", response_metadata={"finish_reason": "length"})


def test_an_empty_answer_from_openrouter_gets_openrouters_own_hint(trace, monkeypatch):
    monkeypatch.delenv("STUDY_REASONING_EFFORT", raising=False)
    with pytest.raises(ValueError) as exc_info:
        call_model(_OpenRouterCappedModel(), "sys", "usr", trace=trace,
                  node="naive_reading", bab_ref="nawawi/1",
                  model_id="openai/placeholder-frontier")
    assert "OpenRouter's unified" in str(exc_info.value)
