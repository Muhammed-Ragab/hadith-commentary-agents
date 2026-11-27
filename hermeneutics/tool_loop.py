"""The assistant->tools->assistant loop, bounded.

Lifted from local-agent-ui/agent.py's build_library_graph tools branch, rewired through
llm.call_with_tools so every round is traced. Knows nothing about graphs or stages: it
takes a system prompt, a user prompt and a corpus, and returns what the model said plus
what it read.
"""
import json
import os
from dataclasses import dataclass, field

from hermeneutics.llm import call_with_tools, empty_answer_hint, provider_of

# Five tools (search, verify, lexicon, rijal, isnad network) need more rounds than one
# did: a grounded answer may search, verify two quotations and check a narrator.
MAX_TOOL_ROUNDS = int(os.environ.get("INQUIRY_TOOL_ROUNDS", "8"))

WRAP_UP = ("You have used the research budget for this turn. Answer now from the sources "
           "you already retrieved, citing them. If they do not settle the question, say "
           "exactly that.")


@dataclass
class ToolResult:
    text: str
    sources: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    rounds: int = 0
    wrapped_up: bool = False


def _dispatch(corpus, name: str, args: dict):
    if name == "search_sources":
        return corpus.search(args["query"], args.get("source", "all"))
    if name == "verify_quote":
        return corpus.verify_quote(args["quote"], args.get("source", "all"))
    if name == "lookup_word":
        return corpus.lookup_word(args["word"], args.get("lexicon", "all"))
    if name == "lookup_narrator":
        return corpus.lookup_narrator(args["name"], args.get("school", "all"))
    if name == "lookup_isnad_network":
        return corpus.lookup_isnad_network(args["name"], args.get("corpus", "all"))
    raise ValueError(f"Unsupported tool: {name}")


def _empty_answer(node: str, bab_ref: str, rounds: int, wrapped_up: bool,
                  model_id: str, model) -> ValueError:
    stage = "the wrap-up call (no tools offered)" if wrapped_up else "a tool-calling round"
    return ValueError(
        f"{node} on {bab_ref} returned an empty answer from {model_id} after {rounds} "
        f"tool round(s), from {stage}. "
        f"{empty_answer_hint(model_id, provider=provider_of(model))}"
    )


def run_tools(model, system: str, user: str, corpus, *, trace, node: str, bab_ref: str,
              model_id: str, manifest_id: str | None = None,
              max_rounds: int = MAX_TOOL_ROUNDS) -> ToolResult:
    """Let the model work the tools, then answer. Always returns prose, never markup."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    result = ToolResult(text="")

    while True:
        message = call_with_tools(model, messages, corpus.tool_specs(), trace=trace,
                                  node=node, bab_ref=bab_ref, model_id=model_id,
                                  manifest_id=manifest_id)
        messages.append(message)

        if not message.get("tool_calls"):
            text = message.get("content") or ""
            if not text.strip():
                raise _empty_answer(node, bab_ref, result.rounds, wrapped_up=False,
                                    model_id=model_id, model=model)
            result.text = text
            return result

        if result.rounds >= max_rounds:
            messages.append({"role": "user", "content": WRAP_UP})
            # Correction 1: the wrap-up call must not be able to return another tool
            # call -- that would leave the loop with nothing to say (an ignored call
            # and an empty `content`). Pass no tool specs at all, mirroring agent.py's
            # wrap_up node, which calls client.chat(messages) with no `tools` argument.
            final = call_with_tools(model, messages, [], trace=trace,
                                    node=f"{node}_wrap_up", bab_ref=bab_ref,
                                    model_id=model_id, manifest_id=manifest_id)
            text = final.get("content") or ""
            if not text.strip():
                raise _empty_answer(node, bab_ref, result.rounds, wrapped_up=True,
                                    model_id=model_id, model=model)
            result.text = text
            result.wrapped_up = True
            return result

        result.rounds += 1
        for call in message["tool_calls"]:
            function = call.get("function", {})
            name = function.get("name", "")
            try:
                args = json.loads(function.get("arguments") or "{}")
                found = _dispatch(corpus, name, args)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                args, found = {}, {"error": str(exc)}
            result.calls.append({"tool": name, "args": args})
            hits = found if isinstance(found, list) else found.get("occurrences", [])
            result.sources.extend(item for item in hits if isinstance(item, dict)
                                  and "source" in item)
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": json.dumps(found, ensure_ascii=False)})
