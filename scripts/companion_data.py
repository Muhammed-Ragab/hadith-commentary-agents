"""Load the final question-set batch for the companion PDF: 26 Query-Agent sessions
(`outputs/question_set/index.json`) and 15 Luna answers (two frontier runs).

Plain dicts, no HTML. Counts are raw grounding_log counts (as on the slides), never
re-verified: the PDF shows the verdicts the runs recorded.
"""
import json
import re
from pathlib import Path

from scripts.build_session_view import SCAR

LUNA_RUNS = ("frontier_openai_gpt-5.6-luna_20260924T161907",   # level 1, q1-q8
             "frontier_openai_gpt-5.6-luna_20260927T151354")   # level 2, q9-q11, 16k cap
COMMENTATORS = ("cayni", "nawawi", "mazandarani")
VARIANTS = ("ar", "en", "colloq")
KEY_RE = re.compile(r"^set_((q\d+)_[a-z]+(?:_(en|colloq))?)_(cayni|nawawi|mazandarani)$")
QID_RE = re.compile(r"^((q\d+)_[a-z]+(?:_(en|colloq))?)$")


def parse_key(key: str) -> tuple:
    m = KEY_RE.match(key)
    if not m:
        raise ValueError(f"unexpected question-set key: {key}")
    return m.group(1), m.group(2), m.group(3) or "ar", m.group(4)


def _counts(log: list) -> dict:
    return {"quotations": len(log),
            "verified": sum(1 for e in log if e.get("verified")),
            "echoes": sum(1 for e in log if e.get("echoes_question"))}


def _level(chapter: str) -> int:
    return 1 if int(chapter[1:]) <= 8 else 2


def load_agent(outputs: Path) -> list:
    index = json.loads((outputs / "question_set" / "index.json").read_text(encoding="utf-8"))
    answers = []
    for key, entry in index.items():
        qid, chapter, variant, commentator = parse_key(key)
        path = Path(entry["session"])
        if not path.exists():  # index stores absolute paths; fall back to outputs/sessions
            path = outputs / "sessions" / path.name
        s = json.loads(path.read_text(encoding="utf-8"))
        if s.get("corrections"):
            raise ValueError(f"{path.name} carries editorial corrections; refusing to print")
        log = s.get("grounding_log") or []
        text = s.get("synthesis_output") or ""
        stripped = sum(1 for e in log if not e.get("verified"))
        if text.count(SCAR) != stripped:
            raise ValueError(f"{path.name}: {text.count(SCAR)} scar(s) but {stripped} "
                             "unverified log entries; restoration would be a guess")
        manifest_path = outputs / "runs" / key / "manifest.json"
        manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.exists() else {})
        answers.append({
            "key": key, "system": "agent", "qid": qid, "chapter": chapter,
            "variant": variant, "commentator": commentator, "level": _level(chapter),
            "question": s.get("question", ""), "question_ar": s.get("question_ar", ""),
            "search_terms": s.get("search_terms") or [],
            "model_id": s.get("model_id"), "provider": s.get("provider"),
            "provider_route": s.get("provider_route"), "quantization": s.get("quantization"),
            "temperature": manifest.get("temperature", 0),
            "date": manifest.get("date") or path.stem.split("_")[-1][:8],
            "git_sha": manifest.get("git_sha", ""),
            "run_file": path.name, "text": text, "log": log,
            "calls": s.get("tool_calls_log") or [], **_counts(log)})
    return answers


def load_luna(outputs: Path) -> tuple:
    """Returns (answers, system_prompt)."""
    answers, system = [], None
    for run in LUNA_RUNS:
        d = outputs / "frontier" / run
        for path in sorted(d.glob("*.json")):
            if path.name == "index.json":
                continue
            r = json.loads(path.read_text(encoding="utf-8"))
            m = QID_RE.match(r["qid"])
            if not m:
                raise ValueError(f"unexpected Luna qid: {r['qid']}")
            system = system or r.get("system")
            log = r.get("grounding_log") or []
            answers.append({
                "key": f"luna_{r['qid']}", "system": "luna", "qid": r["qid"],
                "chapter": m.group(2), "variant": m.group(3) or "ar", "commentator": None,
                "level": _level(m.group(2)), "question": r.get("question", ""),
                "question_ar": "", "search_terms": [], "model_id": r.get("model_id"),
                "provider": r.get("provider"), "provider_route": r.get("provider_route"),
                "max_tokens": r.get("max_tokens"), "date": (r.get("ts") or "")[:10],
                "run_file": f"{run}/{path.name}", "text": r.get("answer", ""), "log": log,
                "calls": [], **_counts(log)})
    return answers, system


def load_batch(outputs: Path) -> list:
    """All answers in document order, each with a stable id A01..An."""
    luna, _ = load_luna(outputs)
    answers = load_agent(outputs) + luna

    def order(a):
        return (int(a["chapter"][1:]), 0 if a["system"] == "agent" else 1,
                COMMENTATORS.index(a["commentator"]) if a["commentator"] else 0,
                VARIANTS.index(a["variant"]))

    answers.sort(key=order)
    for i, a in enumerate(answers, 1):
        a["aid"] = f"A{i:02d}"
    return answers
