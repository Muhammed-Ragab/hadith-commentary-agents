"""LLM pass over answer units the rule-based tagger (hermeneutics/answer_moves.py)
left as `other` (rule `no_rule_fired`) -- one call per ANSWER, not per unit.

Each call's prompt carries the fixed move definitions (prompt_template, hashed so the
manifest can tell which prompt version produced a given llm_moves.csv), the question,
and every unit of that answer, numbered, with FULL text -- units a rule already tagged
show that tag ("do not relabel"); only units tagged `other` are asked for. The model
must reply with strict JSON {unit_index: move}. A reply with an unknown label or a
missing/unexpected index gets exactly one retry (a tightened prompt naming the
problem); anything still wrong after that is left as `other` and logged.

    python scripts/tag_moves_llm.py --dry-run                      # sizing only, no calls
    python scripts/tag_moves_llm.py --execute --out-dir outputs/comparison/<stamp>

Every call goes through hermeneutics.llm.call_model (traced) and
hermeneutics.llm.get_chat_model (provider="openrouter"), the same wiring
scripts/run_frontier_baseline.py uses. temperature: get_chat_model hardcodes
temperature=0 for every backend ("spec §3.4 reproducibility contract") -- there is no
temperature parameter to pass, and this script does not edit hermeneutics/llm.py to add
one.

Cost is tracked from the ACTUAL traced tokens at --price-in/--price-out ($1/$5 per
million by default, Claude Haiku 4.5's OpenRouter list price at the time this was
written). --cost-cap (default $0.60) is a hard stop: once the running actual cost
plus a conservative estimate of the next call would exceed it, every remaining answer
is left rule-tagged (no LLM call) and logged as skipped.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hermeneutics.answer_moves import MOVES, Unit, llm_tag_prompt, tag_answer
from hermeneutics.llm import call_model, get_chat_model
from hermeneutics.openrouter import OPENROUTER_BASE_URL, openrouter_key
from hermeneutics.trace import TraceWriter

DEFAULT_LEXICONS = ROOT / "scripts" / "compare_lexicons.json"
DEFAULT_MODEL_ID = "anthropic/claude-haiku-4.5"
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def _load_lexicons(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _estimate_tokens(text: str) -> int:
    """Rough estimate: ~1 token per 4 characters, adequate for a cost projection only."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- prompt

def prompt_template() -> str:
    """The fixed system prompt: move definitions + the required JSON output format.
    Hashed (see PROMPT_HASH) so a manifest can pin exactly which wording produced a
    given llm_moves.csv -- changing this text changes the hash."""
    move_lines = "\n".join(f"  - {m}: {_MOVE_DEFINITIONS[m]}" for m in MOVES)
    return (
        "You are labelling the rhetorical MOVE of units (paragraphs, list items or "
        "quotations) inside answers to hadith-interpretation questions. The moves are:\n"
        f"{move_lines}\n\n"
        "You will be given the question and every unit of one answer, numbered. Units "
        "a deterministic rule already classified show that label and are marked "
        "'do not relabel' -- keep them exactly as shown, they are context only. Only "
        "units marked 'NEEDS LABEL' need your judgement; `other` is a legitimate answer "
        "for one of those if none of the other moves genuinely fits.\n\n"
        "Reply with STRICT JSON ONLY: a single object mapping each NEEDS-LABEL unit's "
        "index (as a string) to exactly one move name from the list above. No prose, "
        "no markdown code fences, no keys other than the NEEDS-LABEL indices, no "
        "missing indices. Example: {\"3\": \"claim\", \"7\": \"other\"}"
    )


_MOVE_DEFINITIONS = {
    "claim": "the answerer's own explicit assertion or conclusion (e.g. \"the correct "
             "view is...\", \"we hold that...\") -- NOT merely reporting someone else's view.",
    "report": "a reported view attributed to someone else (qala fulan, \"according to "
               "X\"), not quote-dominated and not the answerer's own assertion.",
    "evidence_quote": "a unit dominated by a delimited quotation from a source text.",
    "isnad_grading": "assessing a hadith's chain of transmission (sahih/hasan/da'if/...).",
    "lexical_gloss": "explaining what a word or phrase means.",
    "objection": "raising an objection or difficulty to be addressed.",
    "concession": "conceding a point before qualifying or countering it.",
    "reply": "answering an objection raised earlier in the same answer.",
    "limit": "stating that the sources available do not settle something, or that the "
             "answerer cannot establish a claim from them.",
    "proposal": "an interpretive proposal or overall conclusion, usually near the end.",
    "other": "none of the above genuinely fits.",
}

PROMPT_HASH = hashlib.sha256(prompt_template().encode("utf-8")).hexdigest()[:16]


def render_answer_prompt(question: str, units: list[Unit]) -> tuple[str, list[int]]:
    """The per-answer user prompt, and the list of unit indices it actually asks for."""
    lines = [f"QUESTION: {question}", "", "UNITS:"]
    target_indices = []
    for u in units:
        if u.move == "other":
            lines.append(f"[{u.index}] (other -- NEEDS LABEL): {u.text}")
            target_indices.append(u.index)
        else:
            lines.append(f"[{u.index}] (already tagged: {u.move}, do not relabel): {u.text}")
    lines.append("")
    lines.append(f"Label ONLY these unit indices, one JSON object, all of them: {target_indices}")
    return "\n".join(lines), target_indices


def parse_response(text: str, target_indices: list[int]) -> tuple[dict[int, str], list[str]]:
    """(valid map {index: move}, problems). Any problem means at least one retry is
    warranted; `valid` may still be partially usable even when `problems` is non-empty."""
    cleaned = _FENCE.sub("", (text or "").strip())
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {}, [f"json_decode_error: {e}"]
    if not isinstance(obj, dict):
        return {}, ["response is not a JSON object"]

    target_set = set(target_indices)
    valid: dict[int, str] = {}
    problems: list[str] = []
    for k, v in obj.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            problems.append(f"non-integer key: {k!r}")
            continue
        if idx not in target_set:
            problems.append(f"unexpected index: {idx}")
            continue
        if v not in MOVES:
            problems.append(f"unknown label for unit {idx}: {v!r}")
            continue
        valid[idx] = v
    missing = target_set - set(valid)
    if missing:
        problems.append(f"missing indices: {sorted(missing)}")
    return valid, problems


def merge_final_move(rule_move: str, llm_move: str | None) -> str:
    """final_move: the rule's label unless it was `other`, in which case the LLM's
    label if one was obtained, else `other` (the safe fallback)."""
    if rule_move != "other":
        return rule_move
    return llm_move if llm_move else "other"


# --------------------------------------------------------------------------- labelling (mockable)

def label_answer(call_fn, *, question: str, units: list[Unit], max_retries: int = 1) -> dict:
    """Label one answer's `other` units. `call_fn(system, user) -> str` is the only
    network-touching seam, injected so tests can mock it with no API access.

    Returns {"labels": {index: move}, "status": "no_other_units"|"ok"|"failed_validation",
    "attempts": int, "problems": [...]} -- `labels` may be partial when status is
    failed_validation; callers fall back to `other` for any index it does not cover.
    """
    target = [u.index for u in units if u.move == "other"]
    if not target:
        return {"labels": {}, "status": "no_other_units", "attempts": 0, "problems": []}

    system = prompt_template()
    user, target_indices = render_answer_prompt(question, units)
    problems_log: list[dict] = []
    valid: dict[int, str] = {}

    for attempt_num in range(1, max_retries + 2):
        text = call_fn(system, user)
        valid, problems = parse_response(text, target_indices)
        if not problems:
            return {"labels": valid, "status": "ok", "attempts": attempt_num, "problems": []}
        problems_log.append({"attempt": attempt_num, "problems": problems})
        user = (
            user + "\n\nYour previous reply was invalid: " + "; ".join(problems) +
            f"\nReply again with STRICT JSON ONLY, covering EXACTLY these indices: "
            f"{target_indices}"
        )

    return {"labels": valid, "status": "failed_validation",
            "attempts": max_retries + 1, "problems": problems_log}


# --------------------------------------------------------------------------- hand-check sampling

def stratified_sample(llm_labelled_rows: list[dict], *, seed: int, n_per_system: int = 15) -> list[dict]:
    """Up to `n_per_system` rows per system ("agent"/"luna", inferred from id prefix),
    drawn with a fixed seed from rows the LLM actually labelled (status == "ok" or a
    valid partial from failed_validation -- i.e. rows carrying a real llm_move, not a
    cost-cap skip or a total validation failure with nothing usable)."""
    by_system: dict[str, list[dict]] = {"agent": [], "luna": []}
    for row in llm_labelled_rows:
        if row["id"].startswith("agent_"):
            by_system["agent"].append(row)
        elif row["id"].startswith("luna_"):
            by_system["luna"].append(row)
        # any other id prefix is neither system and is dropped, not folded into luna

    rng = random.Random(seed)
    sample = []
    for system in ("agent", "luna"):
        pool = by_system[system]
        take = min(n_per_system, len(pool))
        sample.extend(rng.sample(pool, take))
    return sample


# --------------------------------------------------------------------------- execution

def _read_last_trace_usage(trace_path: Path) -> dict:
    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1]) if lines else {}


def run_llm_tagging(records: list[dict], lexicons: dict, *, model, model_id: str,
                    trace, price_in: float, price_out: float, cost_cap: float,
                    max_retries: int = 1) -> dict:
    """The whole batch: one label_answer call per record, cost-capped. Returns a dict
    with `rows` (per-unit llm_moves.csv rows), `tokens_in`, `tokens_out`, `cost`,
    `calls`, `retries`, `skipped_ids` (cost-cap), `failed_ids` (validation)."""
    rows: list[dict] = []
    tokens_in = tokens_out = calls = retries = 0
    cost = 0.0
    skipped_ids: list[str] = []
    failed_ids: list[str] = []
    aborted = False

    def call_fn(system: str, user: str) -> str:
        nonlocal tokens_in, tokens_out, calls, cost
        text = call_model(model, system, user, trace=trace, node="tag_moves_llm",
                          bab_ref=rec["id"], model_id=model_id)
        usage = _read_last_trace_usage(trace.path)
        tokens_in += usage.get("tokens_in", 0)
        tokens_out += usage.get("tokens_out", 0)
        cost += (usage.get("tokens_in", 0) / 1_000_000 * price_in +
                usage.get("tokens_out", 0) / 1_000_000 * price_out)
        calls += 1
        return text

    for rec in records:
        units = tag_answer(rec["answer"], lexicons)
        target_n = sum(1 for u in units if u.move == "other")

        if aborted or target_n == 0:
            if aborted and target_n:
                skipped_ids.append(rec["id"])
            for u in units:
                rows.append({"id": rec["id"], "unit_index": u.index, "rule_move": u.move,
                             "llm_move": "", "final_move": merge_final_move(u.move, None)})
            continue

        # Conservative pre-call guard: estimate this call's cost from the rendered
        # prompt before sending it, so the running total cannot blow well past the cap
        # between the check and the actual (usually cheaper) result.
        user_preview, _ = render_answer_prompt(rec["question"], units)
        est_in = _estimate_tokens(prompt_template() + user_preview)
        est_out = target_n * 15  # a short JSON label per unit
        est_cost = est_in / 1_000_000 * price_in + est_out / 1_000_000 * price_out
        if cost + est_cost > cost_cap:
            aborted = True
            skipped_ids.append(rec["id"])
            for u in units:
                rows.append({"id": rec["id"], "unit_index": u.index, "rule_move": u.move,
                             "llm_move": "", "final_move": merge_final_move(u.move, None)})
            continue

        result = label_answer(call_fn, question=rec["question"], units=units,
                              max_retries=max_retries)
        retries += max(0, result["attempts"] - 1)
        if result["status"] == "failed_validation":
            failed_ids.append(rec["id"])
        labels = result["labels"]
        for u in units:
            llm_move = labels.get(u.index, "") if u.move == "other" else ""
            rows.append({"id": rec["id"], "unit_index": u.index, "rule_move": u.move,
                         "llm_move": llm_move,
                         "final_move": merge_final_move(u.move, llm_move or None)})

    return {"rows": rows, "tokens_in": tokens_in, "tokens_out": tokens_out, "cost": cost,
            "calls": calls, "retries": retries, "skipped_ids": skipped_ids,
            "failed_ids": failed_ids, "aborted": aborted}


# --------------------------------------------------------------------------- CLI

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def collect_targets(answers: list[str], lexicons: dict, target_moves: set[str]) -> list[str]:
    """Prompts for every unit whose rule-based move is in `target_moves` (legacy
    per-unit sizing path, kept for --answers-file dry-run estimation of arbitrary text
    that hasn't gone through the loaders in scripts/compare_systems.py)."""
    prompts = []
    for answer in answers:
        for unit in tag_answer(answer, lexicons):
            if unit.move in target_moves:
                prompts.append(llm_tag_prompt(unit))
    return prompts


def _load_records(index_path: Path, luna_level1: Path, luna_level2: Path | None) -> list[dict]:
    from scripts.compare_systems import find_level2_luna_dir, load_agent_answers, load_luna_answers

    level2 = luna_level2 or find_level2_luna_dir()
    return load_agent_answers(index_path) + load_luna_answers(luna_level1, level2)


def main() -> int:
    from scripts.compare_systems import LEVEL1_LUNA_DEFAULT, QUESTION_SET_INDEX

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers-file", type=Path,
                     help="Legacy sizing-only mode: JSON file, list of answer strings "
                          "or {id: answer}, used only with --dry-run (no --out-dir).")
    ap.add_argument("--lexicons", type=Path, default=DEFAULT_LEXICONS)
    ap.add_argument("--index", type=Path, default=QUESTION_SET_INDEX)
    ap.add_argument("--luna-level1-dir", type=Path, default=LEVEL1_LUNA_DEFAULT)
    ap.add_argument("--luna-level2-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path,
                     help="An outputs/comparison/<stamp> dir to write llm_moves.csv, "
                          "the manifest, and the hand-check files into. Required for "
                          "--execute.")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--price-in", type=float, default=1.0, help="$/M input tokens (actual cost).")
    ap.add_argument("--price-out", type=float, default=5.0, help="$/M output tokens (actual cost).")
    ap.add_argument("--cost-cap", type=float, default=0.60, help="Hard stop, USD.")
    ap.add_argument("--est-output-tokens", type=int, default=8,
                     help="Legacy --answers-file dry-run: output tokens per unit.")
    ap.add_argument("--hand-check-n", type=int, default=15,
                     help="Rows per system in the hand-check sample.")
    ap.add_argument("--seed", type=int, default=20260928, help="Hand-check sampling seed.")
    ap.add_argument("--dry-run", action="store_true", default=True,
                     help="Default. Sizes the batch and exits without calling the API.")
    ap.add_argument("--execute", action="store_true",
                     help="Actually call the OpenRouter API. Requires --out-dir and "
                          "OPENROUTER_API_KEY.")
    args = ap.parse_args()

    lexicons = _load_lexicons(args.lexicons)

    if not args.execute:
        if args.answers_file:
            raw = json.loads(args.answers_file.read_text(encoding="utf-8"))
            answers = list(raw.values()) if isinstance(raw, dict) else list(raw)
            prompts = collect_targets(answers, lexicons, {"other"})
            tokens_in = sum(_estimate_tokens(p) for p in prompts)
            tokens_out = len(prompts) * args.est_output_tokens
            cost = tokens_in / 1_000_000 * args.price_in + tokens_out / 1_000_000 * args.price_out
            print(f"answers: {len(answers)}", flush=True)
            print(f"units targeted (other): {len(prompts)}", flush=True)
        else:
            records = _load_records(args.index, args.luna_level1_dir, args.luna_level2_dir)
            all_units = [u for rec in records for u in tag_answer(rec["answer"], lexicons)]
            other_units = [u for u in all_units if u.move == "other"]
            est_in = sum(
                _estimate_tokens(prompt_template() +
                                 render_answer_prompt(rec["question"],
                                                      tag_answer(rec["answer"], lexicons))[0])
                for rec in records)
            est_out = len(other_units) * 15
            tokens_in, tokens_out = est_in, est_out
            cost = tokens_in / 1_000_000 * args.price_in + tokens_out / 1_000_000 * args.price_out
            print(f"answers: {len(records)}", flush=True)
            print(f"units targeted (other): {len(other_units)} of {len(all_units)} total",
                  flush=True)
        print(f"estimated tokens in: {tokens_in}, out: {tokens_out}", flush=True)
        print(f"estimated cost: ${cost:.4f} at ${args.price_in}/M in, ${args.price_out}/M out",
              flush=True)
        print("Dry run only. No API call was made. Pass --execute --out-dir <dir> to run "
              "for real.", flush=True)
        return 0

    # --execute
    if not args.out_dir:
        raise SystemExit("--execute requires --out-dir <outputs/comparison/<stamp>>")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(args.index, args.luna_level1_dir, args.luna_level2_dir)
    all_units = [u for rec in records for u in tag_answer(rec["answer"], lexicons)]
    other_units = [u for u in all_units if u.move == "other"]
    est_in = sum(
        _estimate_tokens(prompt_template() +
                         render_answer_prompt(rec["question"], tag_answer(rec["answer"], lexicons))[0])
        for rec in records)
    est_out = len(other_units) * 15
    est_cost = est_in / 1_000_000 * args.price_in + est_out / 1_000_000 * args.price_out
    print(f"answers: {len(records)}, units targeted (other): {len(other_units)} of "
          f"{len(all_units)} total", flush=True)
    print(f"estimated tokens in: {est_in}, out: {est_out}, estimated cost: ${est_cost:.4f}",
          flush=True)
    if est_cost > args.cost_cap:
        raise SystemExit(f"estimated cost ${est_cost:.4f} already exceeds --cost-cap "
                         f"${args.cost_cap:.2f}; aborting before any call.")

    key = openrouter_key()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set (checked env and the User registry "
                         "fallback in hermeneutics.openrouter.openrouter_key).")

    model = get_chat_model(max_tokens=1024, model_id=args.model_id,
                           base_url=OPENROUTER_BASE_URL, api_key=key,
                           provider="openrouter", provider_route=None, quantization=None)

    trace_dir = args.out_dir / "llm_moves_trace"
    with TraceWriter(run_dir=trace_dir, phase="tag_moves_llm") as trace:
        result = run_llm_tagging(records, lexicons, model=model, model_id=args.model_id,
                                 trace=trace, price_in=args.price_in, price_out=args.price_out,
                                 cost_cap=args.cost_cap)

    _write_csv(args.out_dir / "llm_moves.csv", result["rows"],
              ["id", "unit_index", "rule_move", "llm_move", "final_move"])

    llm_labelled_rows = [r for r in result["rows"] if r["llm_move"]]
    sample = stratified_sample(llm_labelled_rows, seed=args.seed, n_per_system=args.hand_check_n)

    records_by_id = {r["id"]: r for r in records}
    check_rows, key_rows = [], []
    for i, row in enumerate(sample):
        rec = records_by_id[row["id"]]
        units = {u.index: u for u in tag_answer(rec["answer"], lexicons)}
        unit_text = units[row["unit_index"]].text if row["unit_index"] in units else ""
        sample_id = f"hc_{i:03d}"
        check_rows.append({"sample_id": sample_id, "answer_id": row["id"],
                           "unit_text": unit_text, "question": rec["question"],
                           "user_move": ""})
        key_rows.append({"sample_id": sample_id, "answer_id": row["id"],
                         "unit_index": row["unit_index"], "final_move": row["final_move"]})
    _write_csv(args.out_dir / "hand_check.csv", check_rows,
              ["sample_id", "answer_id", "unit_text", "question", "user_move"])
    _write_csv(args.out_dir / "hand_check_key.csv", key_rows,
              ["sample_id", "answer_id", "unit_index", "final_move"])

    manifest = {
        "model_id": args.model_id, "temperature": 0,
        "temperature_note": "hermeneutics.llm.get_chat_model hardcodes temperature=0 "
                            "for every backend; there is no parameter to pass, and "
                            "this script does not modify that shared module.",
        "prompt_hash": PROMPT_HASH,
        "date": datetime.now(UTC).isoformat(),
        "answers": len(records), "calls": result["calls"], "retries": result["retries"],
        "tokens_in": result["tokens_in"], "tokens_out": result["tokens_out"],
        "cost_usd": round(result["cost"], 4),
        "price_in_per_m": args.price_in, "price_out_per_m": args.price_out,
        "cost_cap_usd": args.cost_cap, "aborted_on_cost_cap": result["aborted"],
        "skipped_ids": result["skipped_ids"], "failed_validation_ids": result["failed_ids"],
        "hand_check_seed": args.seed, "hand_check_n_per_system": args.hand_check_n,
        "hand_check_rows": len(sample),
    }
    (args.out_dir / "llm_moves_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\ncalls: {result['calls']}, retries: {result['retries']}", flush=True)
    print(f"tokens in/out: {result['tokens_in']}/{result['tokens_out']}", flush=True)
    print(f"actual cost: ${result['cost']:.4f}", flush=True)
    if result["skipped_ids"]:
        print(f"skipped (cost cap): {len(result['skipped_ids'])} answers", flush=True)
    if result["failed_ids"]:
        print(f"failed validation after retry: {len(result['failed_ids'])} answers", flush=True)
    print(f"wrote {args.out_dir / 'llm_moves.csv'}", flush=True)
    print(f"wrote {args.out_dir / 'hand_check.csv'} ({len(sample)} rows) and hand_check_key.csv",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
