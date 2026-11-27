"""Frontier chatbot baseline: one plain call to a model, no pipeline.

The existing `no_sources` condition in run_inquiry.py still runs the three-stage
inquiry graph (isnad / matn / synthesis) and hands the model computed isnad-network
metrics and a corpus label -- that is not a plain chatbot answer. This script is: one
system prompt (conditions.BASE), one user message (the question, nothing else), one
traced model call. What it quotes is then checked against the WHOLE corpus the same
way the Query-Agent gate checks it, so "verified" means the same thing on both sides.
Nothing is stripped -- the answer is stored exactly as the model wrote it.

Usage:
  python scripts/run_frontier_baseline.py --model-id openai/gpt-5.6-luna
  python scripts/run_frontier_baseline.py --model-id openai/gpt-5.6-luna \
      --only q1_ray,q5_taqiyya
"""
import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermeneutics import conditions
from hermeneutics.corpus_tools import CorpusTools
from hermeneutics.grounding import normalize_arabic
from hermeneutics.llm import call_model, get_chat_model
from hermeneutics.openrouter import OPENROUTER_BASE_URL, PRICE_SORT_ROUTE, openrouter_key
from hermeneutics.trace import TraceWriter
from scripts.run_question_set import LUNA_CELLS, QUESTIONS

ROOT = Path(__file__).resolve().parent.parent


def make_run_id(model_id: str, when: time.struct_time | None = None) -> str:
    """frontier_<model-slug>_<UTC stamp>. `when` is injectable so tests are deterministic."""
    slug = model_id.replace("/", "_").replace(":", "_")
    stamp = time.strftime("%Y%m%dT%H%M%S", when or time.gmtime())
    return f"frontier_{slug}_{stamp}"


def _quote_record(quote: str, question: str, corpus_tools: CorpusTools) -> dict:
    """Verify one quotation against the whole corpus, the same call the Query-Agent
    synthesis gate makes (conditions.verify_quotation), and log it the way that gate
    logs it -- minus foreign_commentary and in_session, which do not apply here: this
    condition is tied to no single commentator's book and never retrieved a session's
    worth of sources to begin with."""
    outcome = conditions.verify_quotation(quote, corpus_tools)
    quote_norm = normalize_arabic(quote)
    return {
        "quote": quote,
        "verified": bool(outcome["verified"]),
        "elided": outcome["elided"],
        "checked": outcome["checked"],
        "found_in": outcome.get("sources", []),
        "echoes_question": quote_norm in normalize_arabic(question),
        "ts": datetime.now(UTC).isoformat(),
    }


def run_question(model, qid: str, question: str, *, corpus_tools: CorpusTools,
                 trace: TraceWriter, model_id: str, provider_route: str | None,
                 max_tokens: int, out_dir: Path) -> dict:
    """One plain call, one grounding check, one JSON file. Returns the record written."""
    answer = call_model(model, conditions.BASE, question, trace=trace,
                        node="frontier_baseline", bab_ref=qid, model_id=model_id)

    quotes = conditions.quotations(answer)
    grounding_log = [_quote_record(q, question, corpus_tools) for q in quotes]
    verified = sum(1 for entry in grounding_log if entry["verified"])

    # call_model does not return token counts; the trace it just wrote does. Its last
    # line is the call that produced the answer just returned (retries, if any, wrote
    # their own lines earlier in this same call).
    last_line = trace.path.read_text(encoding="utf-8").strip().splitlines()[-1]
    usage = json.loads(last_line)

    record = {
        "qid": qid,
        "question": question,
        "model_id": model_id,
        "provider": "openrouter",
        "provider_route": provider_route,
        "max_tokens": max_tokens,
        "system": conditions.BASE,
        "answer": answer,
        "grounding_log": grounding_log,
        "quotations": len(quotes),
        "verified": verified,
        "tokens_in": usage.get("tokens_in", 0),
        "tokens_out": usage.get("tokens_out", 0),
        "ts": datetime.now(UTC).isoformat(),
    }
    (out_dir / f"{qid}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{qid}: {len(quotes)} quotations, {verified} verified")
    return record


def run(model, questions: dict, *, model_id: str, provider_route: str | None,
       max_tokens: int, root: Path, run_id: str | None = None) -> Path:
    """Run every question in `questions` (qid -> text) through the baseline. Returns
    the run directory."""
    run_id = run_id or make_run_id(model_id)
    out_dir = Path(root) / "outputs" / "frontier" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_tools = CorpusTools(root)

    with TraceWriter(run_dir=out_dir, phase="trace") as trace:
        index = {
            "run_id": run_id, "model_id": model_id, "provider": "openrouter",
            "provider_route": provider_route, "max_tokens": max_tokens,
            "system": conditions.BASE, "questions": {},
        }
        for qid, question in questions.items():
            record = run_question(
                model, qid, question, corpus_tools=corpus_tools, trace=trace,
                model_id=model_id, provider_route=provider_route,
                max_tokens=max_tokens, out_dir=out_dir)
            index["questions"][qid] = {
                "quotations": record["quotations"], "verified": record["verified"],
            }

    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-id", required=True,
                   help="OpenRouter model id, e.g. openai/gpt-5.6-luna")
    p.add_argument("--provider-route", default=PRICE_SORT_ROUTE,
                   help="OpenRouter routing. Default 'price': no pin, cheapest upstream "
                        "(hermeneutics.openrouter.PRICE_SORT_ROUTE). Pass '' for no pin "
                        "and no price sort either.")
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--only", default=None,
                   help="comma-separated question ids, e.g. q1_ray,q5_taqiyya. "
                        "Default: every id in run_question_set.LUNA_CELLS.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.only:
        qids = [q.strip() for q in args.only.split(",") if q.strip()]
        unknown = [q for q in qids if q not in QUESTIONS]
        if unknown:
            raise SystemExit(f"unknown question id(s): {', '.join(unknown)}")
    else:
        qids = LUNA_CELLS
    questions = {qid: QUESTIONS[qid] for qid in qids}

    provider_route = args.provider_route or None
    model = get_chat_model(max_tokens=args.max_tokens, model_id=args.model_id,
                           base_url=OPENROUTER_BASE_URL, api_key=openrouter_key(),
                           provider="openrouter", provider_route=provider_route,
                           quantization=None)
    out_dir = run(model, questions, model_id=args.model_id, provider_route=provider_route,
                 max_tokens=args.max_tokens, root=ROOT)
    print(f"\nRun saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
