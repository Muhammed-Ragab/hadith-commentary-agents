"""Inquiry runner. Spec §3.2, defect #10.

Usage:
  python scripts/run_inquiry.py -q "..." --corpus kafi --condition A
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langgraph.types import Command

from hermeneutics.conditions import CONDITIONS
from hermeneutics.inquiry_session import (  # noqa: F401 -- re-exported: tests import these from here
    CORPUS_FOR,
    DEFAULT_METHOD_RUN,
    RunParams,
    default_corpus,
    load_framework,
    load_passages_for,
    method_path,
    open_graph,
    prepare,
    write_session,
)
from hermeneutics.llm import get_chat_model
from hermeneutics.manifest import manifest_id  # noqa: F401 -- re-exported for tests
from hermeneutics.openrouter import DEFAULT_MODEL as OPENROUTER_DEFAULT_MODEL
from hermeneutics.openrouter import (
    DEFAULT_PROVIDER_ROUTE,
    OPENROUTER_BASE_URL,
    PRICE_SORT_ROUTE,
    openrouter_key,
)
from scripts.check_corpus import COMMENTATORS

ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one inquiry session.")
    p.add_argument("-q", "--question", required=True)
    # None, not a default: the commentator picks the corpus unless told otherwise, and
    # an explicit --corpus is then visible in the manifest as a choice.
    p.add_argument("--corpus", choices=("bukhari", "kafi", "muslim"), default=None)
    p.add_argument("--condition", choices=tuple(CONDITIONS), default="agent_method")
    p.add_argument("--commentator", choices=COMMENTATORS, default="nawawi")
    p.add_argument("--method-run", default=DEFAULT_METHOD_RUN,
                   help="study run that produced the method files")
    p.add_argument("--passages-file", default=None,
                   help="JSON: {question: [passage, ...]} chosen by the researcher")
    p.add_argument("--run-id", default=None)
    p.add_argument("--auto-approve", action="store_true",
                   help="run the three stage gates without a human. Recorded in the "
                        "intervention log as 'auto_approve', never as 'approve'.")
    # 2000 killed an arm outright and truncated calls in three others. qwen3 is a
    # thinking model: reasoning_content comes out of the SAME budget as the answer, so a
    # call that reasons hard returns tokens_out == max_tokens, error='length' and an
    # EMPTY content. run_tools raises on that when the call offered no tool calls to
    # fall back on, which is how the cayni arm died on 2026-09-22 after four minutes.
    #
    # ponytail: a ceiling, not a cure -- it makes the failure rarer, not impossible. If
    # it recurs, the fix is a retry on an empty length-stopped answer, not a bigger
    # number. Comparable runs must share this value; it lands in the manifest so a
    # session that did not can be told apart.
    p.add_argument("--max-tokens", type=int, default=4000,
                   help="output budget per call, shared with the model's reasoning")
    p.add_argument("--provider", choices=("saia", "openrouter"), default="saia",
                   help="backend to call. saia (default): the vLLM endpoint at "
                        "VLLM_BASE_URL. openrouter: OpenRouter, for a frontier "
                        "no_sources run or a Qwen run pinned to a specific upstream "
                        "serving stack.")
    p.add_argument("--model-id", default=None,
                   help="override the model id (default: VLLM_MODEL_ID for --provider "
                        f"saia, {OPENROUTER_DEFAULT_MODEL} for --provider openrouter)")
    p.add_argument("--provider-route", default=DEFAULT_PROVIDER_ROUTE,
                   help="OpenRouter upstream serving stack to pin, e.g. deepinfra. "
                        "Pass '' (empty) for a frontier model that should not be "
                        "pinned to any one stack. Pass 'price' (reserved) to sort "
                        "by price instead and take the cheapest upstream, with no "
                        "pin and no quantization filter -- meant for a frontier "
                        "model in condition no_sources. Ignored for --provider saia.")
    p.add_argument("--quantization", default="bf16",
                   help="quantization to pin alongside --provider-route. Ignored "
                        "for --provider saia, when --provider-route is empty, or "
                        "when --provider-route is 'price'.")
    return p


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    # '' means "no pin" (a frontier model, condition no_sources); argparse cannot
    # spell None on a command line, so the empty string stands in for it.
    provider_route = args.provider_route or None
    # quantization means nothing without a pinned route to filter against -- an
    # unpinned frontier run (no route, or the reserved "price" sort), or a SAIA
    # run, records None rather than a value that was never actually sent to any
    # endpoint.
    quantization = (args.quantization
                     if provider_route and provider_route != PRICE_SORT_ROUTE
                     else None)

    if args.provider == "openrouter":
        model_id = args.model_id or OPENROUTER_DEFAULT_MODEL
        base_url, api_key = OPENROUTER_BASE_URL, openrouter_key()
    else:
        model_id = args.model_id or os.environ["VLLM_MODEL_ID"]
        base_url, api_key = os.environ.get("VLLM_BASE_URL"), os.environ.get("VLLM_API_KEY")
        provider_route = None  # only OpenRouter runs pin an upstream stack
        quantization = None

    params = RunParams(
        question=args.question, commentator=args.commentator, corpus=args.corpus,
        condition=args.condition, model_id=model_id,
        max_tokens=args.max_tokens, method_run=args.method_run,
        passages_file=args.passages_file, auto_approve=args.auto_approve,
        run_id=args.run_id, provider=args.provider, provider_route=provider_route,
        quantization=quantization)
    model = get_chat_model(max_tokens=args.max_tokens, model_id=model_id, base_url=base_url,
                           api_key=api_key, provider=args.provider,
                           provider_route=provider_route, quantization=quantization)
    prepared = prepare(params, model=model, root=ROOT)

    with open_graph(prepared) as (graph, cfg):
        result = graph.invoke(prepared.initial_state, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            stage = payload.get("stage", "UNKNOWN")
            output = payload.get("output", str(payload))
            print(f"\n=== {stage.upper()} ===\n{output}\n")
            if args.auto_approve:
                print("[auto-approve: no human reviewed this stage]")
                decision = {"action": "auto_approve"}
            else:
                action = input("approve / edit [approve]: ").strip() or "approve"
                decision = {"action": action}
                if action == "edit":
                    decision["note"] = input("note: ").strip()
            result = graph.invoke(Command(resume=decision), cfg)

    out = write_session(prepared, result)
    print(f"\nSession saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
