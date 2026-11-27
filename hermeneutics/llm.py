"""Chat model access. Spec §6 - vLLM OpenAI-compatible endpoint, greedy decoding.

Every model call in the project goes through call_model, call_json or
call_with_tools so that every call lands in the trace. Do not call
model.invoke() directly anywhere else.
"""
import json
import os
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def get_chat_model(max_tokens: int = 1500, *, enable_thinking: bool | None = None,
                   model_id: str | None = None, base_url: str | None = None,
                   api_key: str | None = None, provider: str = "saia",
                   provider_route: str | None = None, quantization: str = "bf16"):
    """Build a ChatOpenAI pointed at the vLLM server. Greedy decoding, always.

    model_id / base_url / api_key default to VLLM_MODEL_ID / VLLM_BASE_URL /
    VLLM_API_KEY. The UI passes them per run: it serves several runs from one process,
    and a model chosen by mutating the environment would leak between them.

    provider selects the backend: "saia" (default, the vLLM endpoint above,
    unchanged) or "openrouter". OpenRouter controls reasoning through its own
    unified `reasoning` parameter and pins the upstream serving stack through a
    `provider` body block -- neither is the SAIA/qwen3-detection path below, so
    they are kept in their own branch rather than folded into it. See
    hermeneutics/openrouter.py for provider_body().
    """
    from langchain_openai import ChatOpenAI

    extra_body = None
    model_id = model_id or os.environ["VLLM_MODEL_ID"]

    # STUDY_BRIDGE_DIR parks every call on disk for an operator to answer by hand.
    # No endpoint, no key, no temperature control -- see hermeneutics/bridge.py for
    # why a run driven this way is not a matched arm.
    from hermeneutics.bridge import BridgeChatModel, bridge_dir_from_env

    bridge_dir = bridge_dir_from_env()
    if bridge_dir is not None:
        return BridgeChatModel(bridge_dir=bridge_dir, model_name=model_id)

    if provider == "openrouter":
        from hermeneutics.openrouter import provider_body

        effort = os.environ.get("STUDY_REASONING_EFFORT") or "medium"
        extra_body = {"reasoning": {"effort": effort},
                      "provider": provider_body(provider_route, quantization)}
        return ChatOpenAI(
            base_url=base_url or os.environ["VLLM_BASE_URL"],
            api_key=api_key or os.environ.get("VLLM_API_KEY", "dummy"),
            model=model_id,
            temperature=0,          # spec §3.4 reproducibility contract
            max_tokens=max_tokens,  # defect #2: bounded output
            timeout=int(os.environ.get("STUDY_LLM_TIMEOUT", "600")),
            max_retries=3,          # defect #6: transport-level backoff
            extra_body=extra_body,
        )

    is_qwen3 = "qwen3." in model_id.lower()
    is_glm = "glm" in model_id.lower()
    if is_glm:
        # GLM-5.3 is a thinking model too: a run died on its first call with
        # tokens_in 9037, tokens_out 4000, error "length", response "" -- the whole
        # max_tokens budget went to reasoning and left content empty. Verified live
        # against SAIA, 2026-09-22: chat_template_kwargs.reasoning_effort="low" is
        # the only level that gave 0 reasoning tokens with a clean answer.
        # enable_thinking is NEVER sent to GLM, even when the caller passes
        # enable_thinking=False (the knob that disables Qwen3's thinking): on GLM
        # that setting gives 0 reasoning tokens too, but the model then writes its
        # reasoning straight into `content` ("The user is asking about...") --
        # unusable, it would pollute stage outputs and the quotation gate.
        effort = os.environ.get("STUDY_REASONING_EFFORT") or "low"
        extra_body = {"chat_template_kwargs": {"reasoning_effort": effort}}
    else:
        # Qwen3.8's chat template accepts only xhigh/medium/low -- "high" raises a
        # jinja2 TemplateError. Default to medium rather than leaving it unset: with
        # nothing sent, the template's own default thinking budget is unbounded, and
        # in job 15849569 it swallowed every one of the 600 tokens the naive pass had.
        reasoning_effort = os.environ.get("STUDY_REASONING_EFFORT", "medium")
        if is_qwen3 and enable_thinking is not None:
            extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        elif is_qwen3 and reasoning_effort:
            extra_body = {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_effort": reasoning_effort,
                }
            }

    return ChatOpenAI(
        base_url=base_url or os.environ["VLLM_BASE_URL"],
        api_key=api_key or os.environ.get("VLLM_API_KEY", "dummy"),
        model=model_id,
        temperature=0,          # spec §3.4 reproducibility contract
        max_tokens=max_tokens,  # defect #2: bounded output
        timeout=int(os.environ.get("STUDY_LLM_TIMEOUT", "600")),
        max_retries=3,          # defect #6: transport-level backoff
        extra_body=extra_body,
    )


def served_by(response) -> str | None:
    """Which upstream actually answered, when the response says so.

    With price-sorted or otherwise unpinned OpenRouter routing, the manifest's
    provider_route cannot say who served a given call -- only OpenRouter's response
    itself can, via its top-level "provider" field (e.g. "DeepInfra", "OpenAI").

    Checked against the installed langchain-openai (1.6.0): ChatOpenAI's
    _create_chat_result builds response_metadata's source dict (llm_output) from a
    fixed set of keys -- token_usage, a hardcoded model_provider="openai",
    model_name, system_fingerprint, id, service_tier -- not a generic copy of every
    extra top-level response field. OpenRouter's "provider" key survives into the
    OpenAI SDK's own parsed object (its ChatCompletion model allows extra fields),
    but langchain_openai never reads it back out, so response_metadata does not
    currently carry it. This still reads response_metadata for it, defensively
    and for whenever the field does survive (e.g. a differently-shaped response
    object, or a future langchain_openai) -- not a live guarantee. No custom HTTP
    client was written to force the point.
    """
    meta = getattr(response, "response_metadata", None) or {}
    return meta.get("provider")


def call_model(model, system: str, user: str, *, trace, node: str, bab_ref: str,
               model_id: str, attempt: int = 1, hypothesis_hash: str | None = None,
               manifest_id: str | None = None, error: str | None = None,
               sleep=time.sleep) -> str:
    """One traced model call. Returns the response text.

    Fix 2 (glm-5.3-flash, SAIA: tokens_in 9037, tokens_out 4000, error "length",
    response ""): an empty reply that stopped at max_tokens is a thinking model's
    budget going entirely to reasoning, not a real failure -- retried once with
    double the budget before giving up. Both attempts are traced, per the
    project's rule that every model call lands in the trace.

    A separate mechanism (_invoke_or_retry_upstream) retries a *transient upstream
    error* -- OpenRouter's timeout/rate-limit/5xx reported inside an HTTP-200 body
    -- with the same call unchanged, not a doubled budget. See that function's
    docstring. `sleep` is injectable so tests do not wait out the real backoff.
    """
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    prompt = f"<<SYSTEM>>\n{system}\n<<USER>>\n{user}"
    invoke_kwargs: dict = {}
    current_attempt = attempt
    current_error = error
    retried = False

    while True:
        started = time.monotonic()
        response, current_attempt = _invoke_or_retry_upstream(
            lambda kw=invoke_kwargs: model.invoke(messages, **kw),
            trace=trace, node=node, bab_ref=bab_ref, model_id=model_id, prompt=prompt,
            attempt=current_attempt, hypothesis_hash=hypothesis_hash,
            manifest_id=manifest_id, sleep=sleep,
        )
        text = response.content if isinstance(response.content, str) else str(response.content)
        usage = getattr(response, "usage_metadata", None) or {}
        capped = _hit_the_cap(response)
        if capped:
            current_error = current_error or "length"

        trace.log(
            node=node, bab_ref=bab_ref, prompt=prompt, response=text, model_id=model_id,
            tokens_in=usage.get("input_tokens", 0), tokens_out=usage.get("output_tokens", 0),
            latency_s=time.monotonic() - started, attempt=current_attempt, error=current_error,
            hypothesis_hash=hypothesis_hash, manifest_id=manifest_id,
            served_by=served_by(response),
        )

        if text.strip() or not capped or retried:
            break
        budget = _next_budget(model)
        if budget is None:
            break
        invoke_kwargs = {"max_tokens": budget}
        current_attempt += 1
        current_error = None
        retried = True

    if not text.strip():
        raise ValueError(
            f"{node} on {bab_ref} returned an empty answer from {model_id} after "
            f"{usage.get('output_tokens', 0)} output tokens. "
            f"{empty_answer_hint(model_id, provider=provider_of(model))}"
        )
    return text


def provider_of(model) -> str:
    """Recover which backend actually served a call, from the model object alone.

    No new plumbing through call_model/run_tools's signatures: get_chat_model's
    OpenRouter branch already knows it built an OpenRouter model, but that fact
    does not otherwise survive past the ChatOpenAI it returns. The base URL does.
    bind_tools wraps the real ChatOpenAI in a RunnableBinding (see _next_budget),
    so unwrap .bound first or every tool-calling model would read as "saia".
    """
    target = getattr(model, "bound", model)
    base_url = str(getattr(target, "openai_api_base", "") or "")
    return "openrouter" if "openrouter.ai" in base_url else "saia"


def empty_answer_hint(model_id: str, provider: str = "saia") -> str:
    """What an operator should do about an empty answer -- named per model.

    Fix 3: the old message told every caller to lower STUDY_REASONING_EFFORT, but
    that env var only ever reaches qwen3.*/glm-* (see get_chat_model) -- a
    glm-5.3-flash run that died with tokens_in 9037, tokens_out 4000, error
    "length", response "" got a hint that happened to work, but any other model
    (e.g. llama-3.3-70b-instruct) would be sent chasing a knob that does nothing.

    On OpenRouter the knob is always the same one (the unified `reasoning`
    parameter), unlike SAIA where only qwen3.*/glm-* respond to a cap at all.
    """
    if provider == "openrouter":
        effort = os.environ.get("STUDY_REASONING_EFFORT") or "medium"
        hint = (f"{model_id}'s reasoning is capped by OpenRouter's unified `reasoning` "
                f"parameter (STUDY_REASONING_EFFORT, now {effort}); lower it or raise "
                "max_tokens.")
    else:
        lower = model_id.lower()
        if "qwen3." in lower or "glm" in lower:
            default_effort = "low" if "glm" in lower else "medium"
            effort = os.environ.get("STUDY_REASONING_EFFORT") or default_effort
            hint = (f"{model_id}'s reasoning is capped by STUDY_REASONING_EFFORT "
                    f"(now {effort}); lower it or raise max_tokens.")
        else:
            hint = (f"{model_id} gets no reasoning cap from this code; raise max_tokens, "
                    "or use a model whose reasoning can be capped (qwen3.*, glm-*).")
    return (f"{hint} If this was a cutoff at max_tokens, it was already retried "
            "once with double the budget before this error.")


def _hit_the_cap(response) -> bool:
    """True when the server stopped at max_tokens rather than at an end token."""
    meta = getattr(response, "response_metadata", None) or {}
    return meta.get("finish_reason") == "length"


MAX_RETRY_MAX_TOKENS = 32768


def _next_budget(model) -> int | None:
    """Double the model's current max_tokens budget, capped at 32768.

    Fix 2: bind_tools returns a RunnableBinding wrapping the real ChatOpenAI at
    .bound, so unwrap that to find the budget actually in force. None means no
    budget could be found, and the caller must not retry.
    """
    target = getattr(model, "bound", model)
    budget = getattr(target, "max_tokens", None)
    if budget is None:
        return None
    return min(budget * 2, MAX_RETRY_MAX_TOKENS)


UPSTREAM_RETRY_CODES = {408, 429, 500, 502, 503, 504}
UPSTREAM_RETRY_BACKOFF_S = (10, 30)


def _is_transient_upstream_error(exc: BaseException) -> bool:
    """A ValueError from langchain_openai wrapping an OpenRouter upstream failure.

    OpenRouter reports an upstream failure (timeout, rate limit, 5xx) as an
    `error` object inside an HTTP-200 body -- e.g. {'message': 'Provider timed
    out after 353032ms', 'code': 504, 'metadata': {...}}. langchain_openai turns
    that into a plain ValueError(response_dict["error"]), so the OpenAI SDK's own
    max_retries (defect #6, set on the ChatOpenAI client) never sees it, and
    neither did this project's code until now.

    Narrow on purpose: only a ValueError whose first arg is a dict with an
    integer `code` in the retryable set counts as transient. Anything else -- a
    malformed-request ValueError, a plain string, a code outside this set (e.g.
    400, a client error that will not fix itself) -- re-raises immediately,
    unchanged.
    """
    if not isinstance(exc, ValueError) or not exc.args:
        return False
    payload = exc.args[0]
    if not isinstance(payload, dict):
        return False
    code = payload.get("code")
    return isinstance(code, int) and code in UPSTREAM_RETRY_CODES


def _invoke_or_retry_upstream(invoke, *, trace, node: str, bab_ref: str, model_id: str,
                              prompt: str, attempt: int, hypothesis_hash: str | None = None,
                              manifest_id: str | None = None, sleep=time.sleep):
    """Call invoke() -> response, retrying a transient upstream error (see
    _is_transient_upstream_error) up to len(UPSTREAM_RETRY_BACKOFF_S) times, with
    a short backoff between attempts (10s, then 30s).

    One slow OpenRouter upstream call used to kill a whole run: the error never
    reached the OpenAI client's own max_retries, and run_inquiry cannot resume.
    This retries the exact same call, unchanged -- distinct from call_model's/
    call_with_tools's own empty-answer retry (Fix 2), which doubles the output
    budget because a thinking model ate its cap. The two must not be conflated:
    this one is about the upstream connection dying mid-call, not about what was
    asked.

    Every failed attempt lands in the trace, per the project's rule that every
    model call is traced -- same convention as Fix 2's retry. After the last
    attempt fails, the original error is re-raised unchanged. `sleep` is
    injectable so tests do not wait out the real backoff.

    Returns (response, attempt): the attempt number the caller's own trace.log
    for the returned response should be recorded under (it only advances when a
    transient retry actually happened).
    """
    current_attempt = attempt
    for backoff in (*UPSTREAM_RETRY_BACKOFF_S, None):
        started = time.monotonic()
        try:
            return invoke(), current_attempt
        except ValueError as exc:
            if not _is_transient_upstream_error(exc):
                raise
            trace.log(
                node=node, bab_ref=bab_ref, prompt=prompt, response="",
                model_id=model_id, tokens_in=0, tokens_out=0,
                latency_s=time.monotonic() - started, attempt=current_attempt,
                error=f"upstream_{exc.args[0]['code']}",
                hypothesis_hash=hypothesis_hash, manifest_id=manifest_id,
            )
            if backoff is None:
                raise
            sleep(backoff)
            current_attempt += 1
    raise AssertionError("unreachable")  # the loop above always returns or raises


def _length_error():
    """The same condition as _hit_the_cap, raised instead of returned.

    On the structured-output path the OpenAI SDK parses the completion itself and
    throws LengthFinishReasonError before langchain builds a response, so the check
    above never sees it -- a retryable truncation killed a whole arm instead
    (mazandarani, 2026-09-17, 24 min in). Imported lazily: openai is
    langchain_openai's dependency, not ours, and the test fakes never raise it.
    """
    try:
        from openai import LengthFinishReasonError
    except ImportError:  # older SDK: nothing raises it, so nothing to catch
        return ()
    return (LengthFinishReasonError,)


def _parse_json(text: str) -> dict:
    return json.loads(_FENCE.sub("", text.strip()))


def call_json(model, system: str, user: str, *, trace, node: str, bab_ref: str,
              model_id: str, hypothesis_hash: str | None = None,
              manifest_id: str | None = None, sleep=time.sleep) -> dict:
    """Traced call that must return JSON. One repair retry, then fail loudly.

    Defect #6: a silent fallback here would put unparsed prose in the notes file,
    which is exactly the free-text problem defect #8 exists to kill.

    A transient upstream error (see _invoke_or_retry_upstream) is retried there,
    inside each repair attempt, with the same call unchanged; `sleep` is
    injectable so tests do not wait out the real backoff.
    """
    # vLLM 0.21 supports OpenAI's JSON response mode. Bind it only here; prose
    # calls such as naive_reading and consolidation remain free-form.
    json_model = model if getattr(model, "model_name", "") == "humain-m3-preview" else model.bind(
        response_format={"type": "json_object"}
    )
    attempt_user = user
    last_error: Exception | None = None
    current_attempt = 1

    for _ in (1, 2):
        prompt = f"<<SYSTEM>>\n{system}\n<<USER>>\n{attempt_user}"
        started = time.monotonic()
        try:
            response, current_attempt = _invoke_or_retry_upstream(
                lambda au=attempt_user: json_model.invoke(
                    [SystemMessage(content=system), HumanMessage(content=au)]),
                trace=trace, node=node, bab_ref=bab_ref, model_id=model_id, prompt=prompt,
                attempt=current_attempt, hypothesis_hash=hypothesis_hash,
                manifest_id=manifest_id, sleep=sleep,
            )
        except _length_error() as exc:
            text, usage = "", {}
            response = None
            parsed, error_label, last_error = None, "length", exc
        else:
            text = response.content if isinstance(response.content, str) else str(response.content)
            usage = getattr(response, "usage_metadata", None) or {}

            try:
                parsed = _parse_json(text)
                error_label = None
            except (json.JSONDecodeError, ValueError) as exc:
                parsed, error_label, last_error = None, type(exc).__name__, exc
            if _hit_the_cap(response):
                parsed, error_label = None, "length"
                last_error = ValueError("stopped at max_tokens, so the JSON is truncated")

        trace.log(
            node=node, bab_ref=bab_ref,
            prompt=prompt,
            response=text, model_id=model_id,
            tokens_in=usage.get("input_tokens", 0), tokens_out=usage.get("output_tokens", 0),
            latency_s=time.monotonic() - started, attempt=current_attempt, error=error_label,
            hypothesis_hash=hypothesis_hash, manifest_id=manifest_id,
            served_by=served_by(response),
        )

        if parsed is not None:
            return parsed

        # A truncated reply is not a formatting mistake, and telling the model its
        # JSON was invalid only makes it write more. Ask for less instead.
        if error_label == "length":
            attempt_user = (
                f"{user}\n\n---\nYour previous reply ran past the output limit and was "
                "cut off. Answer the same question far more briefly: the JSON object "
                "only, shortest values that are still accurate, no markdown fences."
            )
        else:
            attempt_user = (
                f"{user}\n\n---\nYour previous reply was not valid JSON ({error_label}). "
                "Reply with the JSON object only, no prose and no markdown fences."
            )
        current_attempt += 1

    raise ValueError(
        f"{node} on {bab_ref} did not return valid JSON after one repair retry: {last_error}"
    )


# Qwen writes a tool call as TEXT when the server does not parse it into tool_calls:
#
#     <tool_call><function=lookup_word><parameter=word>بوائقه</parameter></function></tool_call>
#
# The graph saw no tool_calls field, ended the turn, and recorded that markup as the
# answer -- job compare-20260910-192047 scored a cell 0/0 on a question the same
# condition had answered 7/10 minutes earlier (08feea5). Recovering the call keeps the
# turn alive. Defined here rather than in local-agent-ui/agent.py so the inquiry graph
# and the comparison harness check against ONE definition.
_TEXT_CALL = re.compile(
    r"<tool_call>\s*<function=(?P<name>\w+)>(?P<args>.*?)</function>\s*</tool_call>", re.S)
_TEXT_ARG = re.compile(r"<parameter=(?P<key>\w+)>(?P<value>.*?)</parameter>", re.S)
TOOL_CALL_MARKUP = re.compile(r"<tool_call>|<function=|<parameter=")


def recover_text_tool_calls(message: dict) -> dict:
    """Turn a text-mode tool call back into the structured form the loop expects."""
    if message.get("tool_calls") or not message.get("content"):
        return message
    calls = []
    for index, match in enumerate(_TEXT_CALL.finditer(message["content"])):
        args = {m.group("key"): m.group("value").strip()
                for m in _TEXT_ARG.finditer(match.group("args"))}
        calls.append({"id": f"text-{index}", "type": "function",
                      "function": {"name": match.group("name"),
                                   "arguments": json.dumps(args, ensure_ascii=False)}})
    if not calls:
        return message
    residue = _TEXT_CALL.sub("", message["content"]).strip()
    return {**message, "content": residue or None, "tool_calls": calls}


def _as_openai_message(response) -> dict:
    """langchain's AIMessage -> the OpenAI dict shape the tool loop carries."""
    content = response.content if isinstance(response.content, str) else str(response.content)
    calls = [{"id": call.get("id") or f"call-{i}", "type": "function",
              "function": {"name": call["name"],
                           "arguments": json.dumps(call.get("args", {}), ensure_ascii=False)}}
             for i, call in enumerate(getattr(response, "tool_calls", None) or [])]
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return recover_text_tool_calls(message)


def call_with_tools(model, messages: list[dict], tool_specs: list[dict], *, trace,
                    node: str, bab_ref: str, model_id: str,
                    manifest_id: str | None = None, attempt: int = 1,
                    sleep=time.sleep) -> dict:
    """One traced tool-calling round. Returns the assistant message as an OpenAI dict.

    Same contract as call_model: nothing in this project talks to a model without
    leaving a line in the trace. A tool round is a model call.

    Fix 2: an empty, length-capped reply with no tool calls either is the same
    thinking-budget failure call_model retries -- see there. Retried once with
    double the budget; both attempts are traced.

    A transient upstream error (see _invoke_or_retry_upstream) is retried there,
    with the same call unchanged -- not to be confused with Fix 2 above. `sleep`
    is injectable so tests do not wait out the real backoff.
    """
    # No specs means OMIT `tools`, not send an empty array. bind_tools([]) serialises to
    # `"tools": []`, which SAIA rejects with a 400 ("must not be an empty array ... or
    # omit the field entirely"). That is the wrap-up call in tool_loop.run_tools, whose
    # whole point is to offer no tools -- so every stage that exhausted its tool rounds
    # died at the one call meant to rescue it. It killed the cayni arm three times.
    bound = model.bind_tools(tool_specs) if tool_specs else model
    lc_messages = _to_langchain(messages)
    prompt = json.dumps(messages, ensure_ascii=False)
    invoke_kwargs: dict = {}
    current_attempt = attempt
    retried = False

    while True:
        started = time.monotonic()
        response, current_attempt = _invoke_or_retry_upstream(
            lambda kw=invoke_kwargs: bound.invoke(lc_messages, **kw),
            trace=trace, node=node, bab_ref=bab_ref, model_id=model_id, prompt=prompt,
            attempt=current_attempt, manifest_id=manifest_id, sleep=sleep,
        )
        message = _as_openai_message(response)
        usage = getattr(response, "usage_metadata", None) or {}
        capped = _hit_the_cap(response)

        # A round with no tool call is a plain answer -- the trace's response field holds
        # the answer text itself, matching call_model, so a reviewer reading the trace
        # sees prose rather than a JSON blob. A round WITH a tool call has no single text
        # to show, so the whole message (including the call) is serialised instead.
        response_field = message["content"] if not message.get("tool_calls") else json.dumps(
            message, ensure_ascii=False)

        trace.log(
            node=node, bab_ref=bab_ref,
            prompt=prompt,
            response=response_field,
            model_id=model_id,
            tokens_in=usage.get("input_tokens", 0), tokens_out=usage.get("output_tokens", 0),
            latency_s=time.monotonic() - started, attempt=current_attempt,
            error="length" if capped else None,
            manifest_id=manifest_id,
            tool_calls=[c["function"]["name"] for c in message.get("tool_calls", [])],
            served_by=served_by(response),
        )

        empty = not (message.get("content") or "").strip() and not message.get("tool_calls")
        if not (empty and capped) or retried:
            return message
        budget = _next_budget(bound)
        if budget is None:
            return message
        invoke_kwargs = {"max_tokens": budget}
        current_attempt += 1
        retried = True


def _to_langchain(messages: list[dict]) -> list:
    """OpenAI dicts -> langchain messages, including tool results."""
    from langchain_core.messages import AIMessage, ToolMessage

    out = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            out.append(SystemMessage(content=m.get("content") or ""))
        elif role == "user":
            out.append(HumanMessage(content=m.get("content") or ""))
        elif role == "tool":
            out.append(ToolMessage(content=m.get("content") or "",
                                   tool_call_id=m.get("tool_call_id", "")))
        else:
            out.append(AIMessage(
                content=m.get("content") or "",
                tool_calls=[{"name": c["function"]["name"],
                             "args": json.loads(c["function"]["arguments"] or "{}"),
                             "id": c.get("id", "")}
                            for c in m.get("tool_calls", [])]))
    return out
