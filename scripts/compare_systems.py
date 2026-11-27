"""Quantitative + NLP-structural comparison of the Query-Agent (qwen) and the Luna
frontier baseline (openai/gpt-5.6-luna) on the conference question set.

Reads the agent's saved sessions (outputs/question_set/index.json -> session JSON) and
Luna's per-question JSON files (outputs/frontier/frontier_openai_gpt-5.6-luna_*/), and
writes a full comparison bundle to outputs/comparison/<UTC stamp>/: metrics.csv,
entities.csv, moves.csv, keyness.csv, similarity.csv, figures/, review/<qid>.md and a
README.md explaining every column. n=1 per cell throughout -- this is descriptive, not
inferential; no significance test is computed anywhere in this script.

All lexicons (hedges, boosters, grading terms, move markers, the level-2 gold variant
list, etc.) live in scripts/compare_lexicons.json, editable without touching this file.

    python scripts/compare_systems.py
    python scripts/compare_systems.py --luna-level1-dir ... --luna-level2-dir ...
"""
import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hermeneutics.answer_moves import (
    move_sequence,
    move_shares,
    normalized_edit_distance,
    split_units,
    tag_answer,
)
from hermeneutics.grounding import normalize_arabic
from scripts.check_corpus import COMMENTATORS, load_babs
from scripts.run_question_set import LUNA_CELLS, QUESTIONS

LEXICONS_PATH = ROOT / "scripts" / "compare_lexicons.json"
QUESTION_SET_INDEX = ROOT / "outputs" / "question_set" / "index.json"
FRONTIER_DIR = ROOT / "outputs" / "frontier"
LEVEL1_LUNA_DEFAULT = FRONTIER_DIR / "frontier_openai_gpt-5.6-luna_20260924T161907"
RUNS_DIR = ROOT / "outputs" / "runs"
CACHE_DIR = ROOT / "outputs" / "comparison" / "cache"

TRADITION = {"cayni": "sunni", "nawawi": "sunni", "mazandarani": "imami", "none": "none"}
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟۔])\s+|\n+")
# Main Arabic block plus the presentation-forms blocks, so a single character like the
# ﷺ ligature (U+FDFA, Arabic Presentation Forms-A) is still classified as "Arabic" for
# lexicon matching rather than falling through to the Latin/lowercase path.
_ARABIC_CHAR_RE = re.compile(r"[؀-ۿﭐ-﷿ﹰ-﻿]")

# One fixed colour per move type, reused across every figure. Colourblind-safe
# categorical palette (Okabe-Ito derived).
MOVE_COLORS = {
    "claim": "#0072B2", "report": "#332288", "evidence_quote": "#E69F00",
    "isnad_grading": "#009E73", "lexical_gloss": "#56B4E9", "objection": "#D55E00",
    "concession": "#CC79A7", "reply": "#F0E442", "limit": "#999999",
    "proposal": "#000000", "other": "#DDCC77",
}


# --------------------------------------------------------------------------- lexicons

def load_lexicons(path: Path = LEXICONS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- loading

def find_level2_luna_dir(frontier_dir: Path = FRONTIER_DIR) -> Path | None:
    """Newest frontier_openai_gpt-5.6-luna_* dir whose index.json has max_tokens==16000.

    The 20260927T141345 dir is truncated at 4000 tokens and must never be picked, even
    though its timestamp is newer than some valid runs -- max_tokens is the filter, the
    timestamp only breaks ties among directories that pass it.
    """
    candidates = []
    for d in sorted(frontier_dir.glob("frontier_openai_gpt-5.6-luna_*")):
        idx = d / "index.json"
        if not idx.exists():
            continue
        try:
            data = json.loads(idx.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("max_tokens") == 16000:
            candidates.append(d)
    return candidates[-1] if candidates else None


def _sum_jsonl_usage(run_dir: Path) -> tuple[int, int, float]:
    """Sum tokens_in, tokens_out, latency_s across every *.jsonl row in a cell's run
    dir. Per-call usage lives in outputs/runs/set_<qid>_<commentator>/inquiry_*.jsonl."""
    tin = tout = 0
    lat = 0.0
    for jf in run_dir.glob("*.jsonl"):
        for line in jf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tin += row.get("tokens_in") or 0
            tout += row.get("tokens_out") or 0
            lat += row.get("latency_s") or 0.0
    return tin, tout, lat


def load_agent_answers(index_path: Path = QUESTION_SET_INDEX) -> list[dict]:
    """One record per set_<qid>_<commentator> cell that has a saved session."""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    out = []
    for cell_id, entry in index.items():
        if entry.get("exit") != 0 or not entry.get("session"):
            continue
        m = re.match(r"^set_(.+)_(cayni|nawawi|mazandarani)$", cell_id)
        if not m:
            continue
        qid, commentator = m.group(1), m.group(2)
        session_path = Path(entry["session"])
        if not session_path.exists():
            print(f"WARNING: missing session file for {cell_id}: {session_path}",
                  flush=True)
            continue
        session = json.loads(session_path.read_text(encoding="utf-8"))
        run_dir = RUNS_DIR / cell_id
        tin, tout, lat = _sum_jsonl_usage(run_dir) if run_dir.exists() else (0, 0, 0.0)
        out.append({
            "id": f"agent_{qid}_{commentator}", "system": "agent", "qid": qid,
            "commentator": commentator, "question": QUESTIONS.get(qid, session.get("question", "")),
            "answer": session.get("synthesis_output", ""), "session": session,
            "session_path": str(session_path), "tokens_in": tin, "tokens_out": tout,
            "latency_s": round(lat, 3), "tool_calls": len(session.get("tool_calls_log", [])),
            "max_tokens": None,
        })
    return out


def load_luna_answers(level1_dir: Path, level2_dir: Path | None) -> list[dict]:
    out = []
    for qid in LUNA_CELLS:
        d = level2_dir if (level2_dir and qid in ("q9_shum", "q10_salat", "q10_salat_en",
                                                     "q11_umirtu")) else level1_dir
        if d is None:
            continue
        f = d / f"{qid}.json"
        if not f.exists():
            print(f"WARNING: missing Luna answer for {qid} in {d}", flush=True)
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        out.append({
            "id": f"luna_{qid}_none", "system": "luna", "qid": qid, "commentator": "none",
            "question": rec.get("question", QUESTIONS.get(qid, "")),
            "answer": rec.get("answer", ""), "session": rec, "session_path": str(f),
            "tokens_in": rec.get("tokens_in", 0), "tokens_out": rec.get("tokens_out", 0),
            "latency_s": None, "tool_calls": None, "max_tokens": rec.get("max_tokens"),
        })
    return out


# --------------------------------------------------------------------------- structure

def word_count(text: str) -> int:
    return len(text.split())


def structure_counts(answer: str) -> dict:
    units = split_units(answer)
    headings = sum(1 for line in answer.splitlines() if HEADING_RE.match(line))
    return {
        "words": word_count(answer),
        "paragraphs": sum(1 for u in units if u.kind == "paragraph"),
        "headings": headings,
        "list_items": sum(1 for u in units if u.kind == "list_item"),
        "blockquotes": sum(1 for u in units if u.kind == "blockquote"),
    }


# --------------------------------------------------------------------------- length/format control

_MD_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MD_BLOCKQUOTE_LINE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_MD_LIST_LINE = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_MD_EMPHASIS = re.compile(r"\*{1,3}|_{2,3}|`+")


def strip_markdown(text: str) -> str:
    """Plain-prose text with markdown syntax removed (headings, list markers,
    blockquote '>' prefixes, bold/italic/code markers) but the WORDS kept -- used for
    the length/format confound check, so embeddings there see running prose rather than
    the agent's heavier markdown structure (headings, bullets) that Luna rarely uses."""
    text = _MD_HEADING_LINE.sub("", text)
    text = _MD_BLOCKQUOTE_LINE.sub("", text)
    text = _MD_LIST_LINE.sub("", text)
    text = _MD_EMPHASIS.sub("", text)
    return text


def chunk_words(text: str, chunk_size: int = 150, max_chunks: int = 8) -> list[str]:
    """Fixed ~chunk_size-word chunks, at most `max_chunks` of them, so a long answer
    does not dominate the mean embedding just by having more chunks than a short one."""
    words = text.split()
    if not words:
        return [""]
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks[:max_chunks]


# --------------------------------------------------------------------------- lexical

def _count_terms(text: str, terms: list[str]) -> int:
    """Non-overlapping occurrence count of a lexicon's terms in `text`.

    Arabic terms are matched against normalize_arabic(text); Latin terms against the
    lowercased raw text. Each group is compiled into ONE alternation, longest term
    first, and matched with re.finditer -- so a longer variant (e.g. "al-Nawawi") is
    not also double-counted through a shorter one it happens to contain ("Nawawi").
    Summing per-term text.count() calls independently would inflate the total on any
    lexicon with both a name and a substring of that name in it.
    """
    ar_terms = sorted((t for t in terms if _ARABIC_CHAR_RE.search(t)),
                      key=len, reverse=True)
    lat_terms = sorted((t for t in terms if not _ARABIC_CHAR_RE.search(t)),
                       key=len, reverse=True)
    total = 0
    if ar_terms:
        norm = normalize_arabic(text)
        pattern = "|".join(re.escape(normalize_arabic(t)) for t in ar_terms if normalize_arabic(t))
        if pattern:
            total += len(re.findall(pattern, norm))
    if lat_terms:
        text_low = text.lower()
        pattern = "|".join(re.escape(t.lower()) for t in lat_terms if t)
        if pattern:
            total += len(re.findall(pattern, text_low))
    return total


def commentator_mentions(answer: str, lexicons: dict) -> dict:
    names = lexicons.get("commentator_names", {})
    out = {}
    total = 0
    for commentator, variants in names.items():
        n = _count_terms(answer, variants)
        out[f"mentions_{commentator}"] = n
        total += n
    out["mentions_total"] = total
    return out


def honorific_counts(answer: str, lexicons: dict) -> dict:
    hon = lexicons.get("honorifics", {})
    imami_terms = hon.get("imami_style", [])
    sunni_terms = hon.get("sunni_style", [])
    return {
        "honorific_imami_style": _count_terms(answer, imami_terms),
        "honorific_sunni_style": _count_terms(answer, sunni_terms),
    }


def stance_metrics(answer: str, lexicons: dict) -> dict:
    words = max(word_count(answer), 1)
    per1k = 1000 / words
    hedges = _count_terms(answer, lexicons.get("hedges", []))
    boosters = _count_terms(answer, lexicons.get("boosters", []))
    attribution = _count_terms(answer, lexicons.get("attribution_verbs", []))

    sentences = [s for s in SENTENCE_SPLIT_RE.split(answer) if s.strip()]
    quote_re = re.compile(r"[«“\"]")
    attributed = 0
    attrib_norm = [normalize_arabic(t).lower() if _ARABIC_CHAR_RE.search(t) else t.lower()
                   for t in lexicons.get("attribution_verbs", [])]
    for s in sentences:
        s_norm = normalize_arabic(s).lower()
        s_low = s.lower()
        has_attrib = any((t in s_norm) if _ARABIC_CHAR_RE.search(t) else (t in s_low)
                          for t in attrib_norm)
        if has_attrib or quote_re.search(s):
            attributed += 1
    share = attributed / len(sentences) if sentences else 0.0

    return {
        "hedges_per_1k": round(hedges * per1k, 3),
        "boosters_per_1k": round(boosters * per1k, 3),
        "attribution_verbs_per_1k": round(attribution * per1k, 3),
        "attributed_sentence_share": round(share, 3),
    }


def mattr(tokens: list[str], window: int = 100) -> float:
    """Moving-average type-token ratio over dediacritized tokens."""
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n
    ratios = []
    for i in range(n - window + 1):
        w = tokens[i:i + window]
        ratios.append(len(set(w)) / window)
    return sum(ratios) / len(ratios)


def lexicon_metrics(answer: str, lexicons: dict) -> dict:
    tokens = normalize_arabic(answer).split()
    words = max(word_count(answer), 1)
    domain = _count_terms(answer, lexicons.get("domain_terms", []))
    return {
        "mattr_100": round(mattr(tokens, 100), 4),
        "domain_term_density_per_1k": round(domain * 1000 / words, 3),
    }


def stated_limits_count(answer: str, lexicons: dict) -> int:
    return _count_terms(answer, lexicons.get("stated_limits", []))


def grading_terms_count(answer: str, lexicons: dict) -> int:
    return _count_terms(answer, lexicons.get("grading_terms", []))


# --------------------------------------------------------------------------- quotations/grounding

def grounding_metrics(record: dict) -> dict:
    session = record["session"]
    grounding_log = session.get("grounding_log", [])
    n_quotations = session.get("quotations")
    if n_quotations is None:
        n_quotations = len(grounding_log)
    verified = session.get("verified")
    if verified is None:
        verified = sum(1 for g in grounding_log if g.get("verified"))
    words = max(word_count(record["answer"]), 1)
    out = {
        "n_quotations": n_quotations,
        "verified": verified,
        "unverified": n_quotations - verified,
        "verified_rate": round(verified / n_quotations, 3) if n_quotations else None,
        "quotes_per_1k_words": round(n_quotations * 1000 / words, 3),
    }
    if record["system"] == "agent":
        out["verified_outside_scope"] = sum(
            1 for g in grounding_log if g.get("verified") and g.get("outside_scope"))
        out["foreign_commentary"] = sum(1 for g in grounding_log if g.get("foreign_commentary"))
        max_tok = None
        truncated = False
    else:
        out["verified_outside_scope"] = ""
        out["foreign_commentary"] = ""
        max_tok = record.get("max_tokens")
        truncated = bool(max_tok and record["tokens_out"] and record["tokens_out"] >= max_tok)
    out["truncated"] = truncated
    return out


# --------------------------------------------------------------------------- level-2 gold recall

def _core_fragments_match(fragment_groups: list, sentences_norm: list[str]) -> bool:
    """True when ANY one fragment group's fragments ALL occur within the SAME sentence.

    This is the "optional words" robustness the gold list needs: a gold item can be
    phrased as a full surface string (exact-ish, word order fixed) OR as one or more
    fragment groups, e.g. [["لا يقطع", "شي"]] -- both fragments must land in one
    sentence, but anything may sit between or around them ("لا يقطع صلاة المسلم شيء"
    matches this though the surface string "لا يقطع الصلاة شيء" does not, because the
    words are reordered and "المسلم" is inserted between them).
    """
    for group in fragment_groups:
        needles = [normalize_arabic(f) for f in group]
        if any(all(n in sent for n in needles) for sent in sentences_norm):
            return True
    return False


def level2_recall(record: dict, lexicons: dict) -> dict:
    """key-variant recall per gold item for level-2 questions. A = surface form found
    (normalized substring, anywhere in the answer) OR a gold item's `core_fragments`
    all co-occurring within one sentence (see _core_fragments_match). R (agent only) =
    the same test applied to `sources[].excerpt`. Tool-call *results* are not stored in
    this session schema (only tool/args/stage), so R is computed over `sources` only --
    documented in the README.

    A gold item may carry `"status": "candidate — user to confirm"` (currently only
    q9_shum's kafi_shum_gloss, added from RESULTS.md but not yet verified against the
    actual al-Kāfī wording by a human). Candidate items are still scored per-item (the
    A_<item>/R_<item> columns are always present) but excluded from the aggregate
    A_recall/R_recall denominator so an unconfirmed guess cannot move the headline
    number -- confirm the surface forms, then drop the status field to include it.
    """
    qid = record["qid"]
    gold = lexicons.get("level2_gold_variants", {}).get(qid)
    if not gold:
        return {}
    answer_norm = normalize_arabic(record["answer"])
    answer_sentences_norm = [normalize_arabic(s) for s in SENTENCE_SPLIT_RE.split(record["answer"])
                             if s.strip()]
    a_hits = r_hits = counted = 0
    per_item = {}
    excerpts_norm = ""
    excerpt_sentences_norm: list[str] = []
    if record["system"] == "agent":
        excerpts = [s.get("excerpt", "") for s in record["session"].get("sources", [])]
        excerpts_norm = normalize_arabic(" ".join(excerpts))
        excerpt_sentences_norm = [normalize_arabic(s)
                                  for e in excerpts for s in SENTENCE_SPLIT_RE.split(e) if s.strip()]
    for item in gold:
        surfaces = item.get("surfaces", [])
        fragment_groups = item.get("core_fragments", [])
        is_candidate = item.get("status", "").startswith("candidate")

        a_found = any(normalize_arabic(s) in answer_norm for s in surfaces)
        if not a_found and fragment_groups:
            a_found = _core_fragments_match(fragment_groups, answer_sentences_norm)
        per_item[f"A_{item['item']}"] = a_found
        if not is_candidate:
            a_hits += int(a_found)

        if record["system"] == "agent":
            r_found = any(normalize_arabic(s) in excerpts_norm for s in surfaces)
            if not r_found and fragment_groups:
                r_found = _core_fragments_match(fragment_groups, excerpt_sentences_norm)
            per_item[f"R_{item['item']}"] = r_found
            if not is_candidate:
                r_hits += int(r_found)
        if not is_candidate:
            counted += 1
    per_item["A_recall"] = round(a_hits / counted, 3) if counted else None
    if record["system"] == "agent":
        per_item["R_recall"] = round(r_hits / counted, 3) if counted else None
    else:
        per_item["R_recall"] = ""
    return per_item


# --------------------------------------------------------------------------- entities (NER + rijal fallback)

_NAME_CUE = re.compile(
    r"(?:أبو|أبي|ابن|بن)\s+[ء-ي]+(?:\s+[ء-ي]+){0,3}")


def _rijal_name_set(path: Path) -> set:
    data = json.loads(path.read_text(encoding="utf-8"))
    names = set()
    for entry in data.get("entries", []):
        label = entry.get("label", "")
        name = re.sub(r"^\s*\d+\s*-\s*", "", label).strip()
        if name:
            names.add(normalize_arabic(name))
    return names


def try_camel_ner() -> object | None:
    """The pretrained NER model, or None. Any failure (missing download, a broken cache
    path, an incompatible transformers version) falls back to the rijal-regex path in
    extract_entities -- there is no camel_tools exception hierarchy worth narrowing
    here, since the model loader can raise from several unrelated libraries."""
    try:
        from camel_tools.ner import NERecognizer
        return NERecognizer.pretrained()
    except Exception as e:  # noqa: BLE001
        print(f"NOTE: camel_tools NER unavailable ({e!r}); falling back to rijal-list "
              f"regex matching for entities.", flush=True)
        return None


def extract_entities(answer: str, ner) -> list[str]:
    """Person-name candidates. Uses camel_tools NER when available; otherwise falls
    back to a name-cue regex (أبو/أبي/ابن/بن + following tokens) over the raw answer,
    which is the fallback the task spec allows when the pretrained model can't load."""
    if ner is not None:
        try:
            tokens = answer.split()
            labels = ner.predict_sentence(tokens)
            names, cur = [], []
            for tok, lab in zip(tokens, labels):
                if lab.endswith("PERS"):
                    cur.append(tok)
                else:
                    if cur:
                        names.append(" ".join(cur))
                        cur = []
            if cur:
                names.append(" ".join(cur))
            return names
        except Exception as e:  # noqa: BLE001
            print(f"NOTE: camel_tools NER prediction failed ({e!r}); using rijal-regex "
                  f"fallback for this answer.", flush=True)
    return [m.group(0) for m in _NAME_CUE.finditer(answer)]


def entity_metrics(record: dict, ner, imami_names: set, sunni_names: set) -> tuple[dict, list[dict]]:
    ents = extract_entities(record["answer"], ner)
    seen = set()
    rows = []
    for e in ents:
        norm = normalize_arabic(e)
        if not norm:
            continue
        in_imami = any(norm in n or n in norm for n in imami_names) if len(norm) > 3 else False
        in_sunni = any(norm in n or n in norm for n in sunni_names) if len(norm) > 3 else False
        rows.append({"id": record["id"], "entity": e, "in_imami": in_imami, "in_sunni": in_sunni})
        seen.add(norm)
    n_unique = len(seen)
    n_in_imami = sum(1 for r in rows if r["in_imami"])
    n_in_sunni = sum(1 for r in rows if r["in_sunni"])
    metrics = {
        "n_person_entities": len(ents),
        "n_unique_persons": n_unique,
        "share_in_imami_rijal": round(n_in_imami / len(rows), 3) if rows else 0.0,
        "share_in_sunni_rijal": round(n_in_sunni / len(rows), 3) if rows else 0.0,
    }
    return metrics, rows


# --------------------------------------------------------------------------- text reuse (n-gram)

def _word_ngrams(tokens: list[str], n: int = 4) -> set:
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def build_ngram_cache(commentator: str, n: int = 4) -> set:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{commentator}_{n}grams.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    babs = load_babs(commentator)
    tokens = normalize_arabic(" ".join(b["text"] for b in babs)).split()
    grams = _word_ngrams(tokens, n)
    with open(cache_file, "wb") as f:
        pickle.dump(grams, f, protocol=pickle.HIGHEST_PROTOCOL)
    return grams


def reuse_share(answer: str, ngram_set: set, n: int = 4) -> float:
    tokens = normalize_arabic(answer).split()
    grams = _word_ngrams(tokens, n)
    if not grams:
        return 0.0
    return round(sum(1 for g in grams if g in ngram_set) / len(grams), 4)


# --------------------------------------------------------------------------- keyness (G^2)

def dunning_g2(a: int, b: int, total_a: int, total_b: int) -> float:
    """Log-likelihood (G^2) for one term given its counts in each group."""
    import math

    if a == 0 and b == 0:
        return 0.0
    e_a = total_a * (a + b) / (total_a + total_b) if (total_a + total_b) else 0
    e_b = total_b * (a + b) / (total_a + total_b) if (total_a + total_b) else 0
    g2 = 0.0
    if a > 0 and e_a > 0:
        g2 += 2 * a * math.log(a / e_a)
    if b > 0 and e_b > 0:
        g2 += 2 * b * math.log(b / e_b)
    return g2


def keyness_table(texts_a: list[str], texts_b: list[str], top_n: int = 50) -> list[dict]:
    """Log-likelihood (G^2) keywords contrasting two groups of Arabic answers."""

    def word_freq(texts):
        c = Counter()
        for t in texts:
            c.update(normalize_arabic(t).split())
        return c

    freq_a, freq_b = word_freq(texts_a), word_freq(texts_b)
    total_a, total_b = sum(freq_a.values()), sum(freq_b.values())
    vocab = set(freq_a) | set(freq_b)
    rows = []
    for term in vocab:
        a, b = freq_a.get(term, 0), freq_b.get(term, 0)
        g2 = dunning_g2(a, b, total_a, total_b)
        direction = "A" if (a / total_a if total_a else 0) >= (b / total_b if total_b else 0) else "B"
        rows.append({"term": term, "freq_a": a, "freq_b": b, "g2": round(g2, 3),
                      "direction": direction})
    rows.sort(key=lambda r: -r["g2"])
    top_a = [r for r in rows if r["direction"] == "A"][:top_n]
    top_b = [r for r in rows if r["direction"] == "B"][:top_n]
    return top_a + top_b


# --------------------------------------------------------------------------- LLM move overlay

def load_llm_moves(path: Path) -> dict[tuple[str, int], str]:
    """{(id, unit_index): final_move} from a scripts/tag_moves_llm.py llm_moves.csv."""
    import csv

    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["id"], int(row["unit_index"]))] = row["final_move"]
    return out


def apply_llm_moves(units: list, answer_id: str, llm_map: dict[tuple[str, int], str]) -> None:
    """Overrides each unit's .move in place with the LLM's final_move, for units the
    rule tagger left as `other` and llm_moves.csv has an entry for. Units the rule
    tagger already classified are untouched (final_move for those is just rule_move
    anyway -- see tag_moves_llm.merge_final_move). The rule field gets a `+llm`
    suffix so moves.csv/review sheets can tell an LLM-sourced label from a rule one."""
    for u in units:
        if u.move != "other":
            continue
        final_move = llm_map.get((answer_id, u.index))
        if final_move and final_move != u.move:
            u.move = final_move
            u.rule = f"{u.rule}+llm"


# --------------------------------------------------------------------------- main build

def build_metrics_rows(records: list[dict], lexicons: dict, ner, imami_names: set,
                       sunni_names: set, ngram_caches: dict,
                       llm_moves: dict[tuple[str, int], str] | None = None,
                       ) -> tuple[list[dict], list[dict]]:
    level2_qids = set(lexicons.get("level2_qids", []))
    rows, entity_rows = [], []
    for rec in records:
        answer = rec["answer"]
        qid = rec["qid"]
        level = "2" if qid in level2_qids else "1"
        lang = rec["session"].get("question_lang") if rec["system"] == "agent" else None
        if not lang:
            lang = "english" if qid.endswith("_en") else (
                "colloquial arabic" if "colloq" in qid else "arabic")

        row = {
            "id": rec["id"], "system": rec["system"], "qid": qid, "level": level,
            "lang": lang, "commentator": rec["commentator"],
            "tradition": TRADITION.get(rec["commentator"], "none"),
        }
        row.update(structure_counts(answer))
        row.update(grounding_metrics(rec))
        row.update(commentator_mentions(answer, lexicons))
        row["stated_limits"] = stated_limits_count(answer, lexicons)
        row["grading_terms"] = grading_terms_count(answer, lexicons)
        row.update(stance_metrics(answer, lexicons))
        row.update(honorific_counts(answer, lexicons))
        row.update(lexicon_metrics(answer, lexicons))
        row["tokens_in"] = rec["tokens_in"]
        row["tokens_out"] = rec["tokens_out"]
        row["tool_calls"] = rec["tool_calls"] if rec["tool_calls"] is not None else ""
        row["latency_s"] = rec["latency_s"] if rec["latency_s"] is not None else ""

        em, erows = entity_metrics(rec, ner, imami_names, sunni_names)
        row.update(em)
        entity_rows.extend(erows)

        for commentator in COMMENTATORS:
            row[f"reuse_{commentator}"] = reuse_share(answer, ngram_caches[commentator])

        units = tag_answer(answer, lexicons)
        if llm_moves:
            apply_llm_moves(units, rec["id"], llm_moves)
        row.update({f"move_share_{m}": round(v, 3) for m, v in move_shares(units).items()})
        row["move_sequence"] = move_sequence(units)

        if level == "2":
            row.update(level2_recall(rec, lexicons))
            row["hs_variants"] = ""
            row["hs_grading"] = ""
            row["hs_link"] = ""
            row["hs_proposal"] = ""
        row["_units"] = units
        row["_answer"] = answer
        rows.append(row)
    return rows, entity_rows


def write_csv(path: Path, rows: list[dict], skip_keys: tuple = ("_units", "_answer")):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in skip_keys and k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k not in skip_keys})


def write_moves_csv(path: Path, rows: list[dict]):
    out = []
    for r in rows:
        for u in r["_units"]:
            out.append({
                "id": r["id"], "unit_index": u.index,
                "unit_text": u.text[:200].replace("\n", " "),
                "move": u.move, "rule": u.rule,
            })
    write_csv(path, out, skip_keys=())


# --------------------------------------------------------------------------- similarity / embeddings

def _cosine_dist_matrix(mat):
    """Precomputed cosine-distance matrix, diagonal forced to exactly 0 and clipped at
    0 -- silhouette_score's precomputed-matrix check rejects a diagonal that floating-
    point noise left at +-1e-8 instead of exactly 0."""
    import numpy as np

    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    dist = 1 - (norm @ norm.T)
    np.fill_diagonal(dist, 0.0)
    return np.clip(dist, 0.0, None)


def _safe_silhouette(dist, labels) -> float | None:
    """silhouette_score, or None when the labels can't support one (fewer than 2
    groups, or every item its own singleton group)."""
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return None
    try:
        from sklearn.metrics import silhouette_score
        return round(float(silhouette_score(dist, labels, metric="precomputed")), 4)
    except ValueError:
        return None


def _length_tercile_labels(word_counts: list[int]) -> list[str]:
    """'short'/'medium'/'long' by tercile of this population's word counts."""
    ordered = sorted(word_counts)
    n = len(ordered)
    t1 = ordered[n // 3]
    t2 = ordered[(2 * n) // 3]

    def label(w):
        if w <= t1:
            return "short"
        if w <= t2:
            return "medium"
        return "long"

    return [label(w) for w in word_counts]


def compute_similarity(rows: list[dict], out_dir: Path, records_by_id: dict) -> None:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        (out_dir / "similarity_SKIPPED.txt").write_text(
            f"sentence-transformers unavailable: {e}\n", encoding="utf-8")
        return

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = {}
    for r in rows:
        paras = [u.text for u in r["_units"] if u.kind == "paragraph" and u.text.strip()]
        if not paras:
            paras = [r["_answer"][:2000]] if r["_answer"].strip() else [""]
        vecs = model.encode(paras, show_progress_bar=False)
        embeddings[r["id"]] = np.mean(vecs, axis=0)

    ids = [r["id"] for r in rows]
    mat = np.stack([embeddings[i] for i in ids])
    norms = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    cos = norms @ norms.T

    sim_rows = []
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if j <= i:
                continue
            sim_rows.append({"id_a": a, "id_b": b, "cosine": round(float(cos[i, j]), 4)})

    arabic_idx = [i for i, r in enumerate(rows) if r["lang"] and "english" not in r["lang"]]
    silhouette = {}  # "raw" silhouettes, kept for comparison against the confound checks
    pc1_corr = None
    if len(arabic_idx) >= 3:
        sub = mat[arabic_idx]
        dist = _cosine_dist_matrix(sub)
        for label_key in ("system", "tradition", "qid"):
            labels = [rows[i][label_key] for i in arabic_idx]
            silhouette[label_key] = _safe_silhouette(dist, labels)

        # 4a. Confound check: within the agent's OWN Arabic answers only, does
        # tradition (sunni cayni+nawawi vs imami mazandarani) separate at all? This
        # isolates tradition from the much bigger agent-vs-luna system gap.
        agent_arabic_idx = [i for i in arabic_idx if rows[i]["system"] == "agent"]
        if len(agent_arabic_idx) >= 3:
            agent_sub = mat[agent_arabic_idx]
            agent_dist = _cosine_dist_matrix(agent_sub)
            agent_labels = [rows[i]["tradition"] for i in agent_arabic_idx]
            silhouette["agent_only_tradition"] = _safe_silhouette(agent_dist, agent_labels)
        else:
            silhouette["agent_only_tradition"] = None

        # 4d. Spearman(word count, PC1) + silhouette by length tercile, on the same raw
        # embeddings/population as the raw silhouettes above, to show how much of the
        # raw embedding space's dominant axis is just answer length.
        from scipy.stats import spearmanr
        from sklearn.decomposition import PCA

        word_counts = [rows[i]["words"] for i in arabic_idx]
        if len(arabic_idx) >= 2:
            pc1 = PCA(n_components=1).fit_transform(sub)[:, 0]
            corr, pval = spearmanr(word_counts, pc1)
            pc1_corr = (round(float(corr), 4), round(float(pval), 4))
        tercile_labels = _length_tercile_labels(word_counts)
        silhouette["length_tercile"] = _safe_silhouette(dist, tercile_labels)
    else:
        for key in ("system", "tradition", "qid", "agent_only_tradition", "length_tercile"):
            silhouette[key] = None

    # 4b. Length/format confound check: strip markdown (headings/bullets/bold/blockquote
    # markers -- the agent uses far more of these than Luna) and represent each answer
    # as the mean of up to 8 fixed ~150-word chunks, so raw answer length also cannot
    # by itself drive the embedding the way one giant/short mean-pooled vector can.
    controlled_silhouette = {}
    if len(arabic_idx) >= 3:
        controlled_embeddings = {}
        for i in arabic_idx:
            r = rows[i]
            plain = strip_markdown(r["_answer"])
            chunks = chunk_words(plain, chunk_size=150, max_chunks=8)
            vecs = model.encode(chunks, show_progress_bar=False)
            controlled_embeddings[r["id"]] = np.mean(vecs, axis=0)
        controlled_mat = np.stack([controlled_embeddings[rows[i]["id"]] for i in arabic_idx])
        controlled_dist = _cosine_dist_matrix(controlled_mat)
        for label_key in ("system", "tradition", "qid"):
            labels = [rows[i][label_key] for i in arabic_idx]
            controlled_silhouette[label_key] = _safe_silhouette(controlled_dist, labels)
    else:
        for label_key in ("system", "tradition", "qid"):
            controlled_silhouette[label_key] = None

    # 4c. Level-2 Arabic answers only (Luna vs agent, comparable question difficulty
    # and, per RESULTS.md, comparable answer length at level 2): silhouette by system.
    level2_silhouette = None
    level2_idx = [i for i in arabic_idx if rows[i]["level"] == "2"]
    if len(level2_idx) >= 3:
        level2_mat = mat[level2_idx]
        level2_dist = _cosine_dist_matrix(level2_mat)
        level2_labels = [rows[i]["system"] for i in level2_idx]
        level2_silhouette = _safe_silhouette(level2_dist, level2_labels)

    evidence_closeness = {}
    for r in rows:
        if r["system"] != "agent":
            continue
        rec = records_by_id[r["id"]]
        sources = rec["session"].get("sources", [])
        excerpts = [s.get("excerpt", "") for s in sources if s.get("excerpt", "").strip()]
        if not excerpts:
            evidence_closeness[r["id"]] = None
            continue
        src_vecs = model.encode(excerpts[:200], show_progress_bar=False)
        src_norm = src_vecs / (np.linalg.norm(src_vecs, axis=1, keepdims=True) + 1e-12)
        paras = [u.text for u in r["_units"] if u.kind == "paragraph" and u.text.strip()]
        if not paras:
            evidence_closeness[r["id"]] = None
            continue
        para_vecs = model.encode(paras, show_progress_bar=False)
        para_norm = para_vecs / (np.linalg.norm(para_vecs, axis=1, keepdims=True) + 1e-12)
        sims = para_norm @ src_norm.T
        evidence_closeness[r["id"]] = round(float(np.mean(np.max(sims, axis=1))), 4)

    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    edit_rows = []
    for qid, group in by_q.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                d = normalized_edit_distance(group[i]["move_sequence"], group[j]["move_sequence"])
                edit_rows.append({"qid": qid, "id_a": group[i]["id"], "id_b": group[j]["id"],
                                   "normalized_edit_distance": round(d, 4)})

    with open(out_dir / "similarity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "a", "b", "value", "extra"])
        for row in sim_rows:
            w.writerow(["cosine", row["id_a"], row["id_b"], row["cosine"], ""])
        # raw: the original mean-of-paragraph-embeddings silhouettes, kept for
        # comparison against the confound checks below.
        for key in ("system", "tradition", "qid"):
            w.writerow([f"silhouette_raw_by_{key}", "", "", silhouette.get(key), ""])
        # 4a. within the agent's own Arabic answers: does tradition separate at all?
        w.writerow(["silhouette_agent_only_by_tradition", "", "", silhouette.get("agent_only_tradition"),
                    "agent Arabic answers only: sunni(cayni+nawawi) vs imami(mazandarani)"])
        # 4b. length/format-controlled: markdown stripped, <=8 fixed ~150-word chunks.
        for key in ("system", "tradition", "qid"):
            w.writerow([f"silhouette_controlled_by_{key}", "", "", controlled_silhouette.get(key),
                        "markdown-stripped, chunked to <=8x~150-word chunks, mean-pooled"])
        # 4c. level-2 Arabic answers only, Luna vs agent.
        w.writerow(["silhouette_level2_by_system", "", "", level2_silhouette,
                    "level-2 Arabic answers only (q9/q10/q10_en/q11)"])
        # 4d. how much of the raw embedding's PC1 is just answer length.
        if pc1_corr is not None:
            w.writerow(["spearman_wordcount_pc1", "", "", pc1_corr[0],
                        f"p={pc1_corr[1]}; raw embeddings, Arabic answers"])
        w.writerow(["silhouette_raw_by_length_tercile", "", "", silhouette.get("length_tercile"),
                    "raw embeddings, tercile of answer word count (short/medium/long)"])
        for rid, val in evidence_closeness.items():
            w.writerow(["evidence_closeness", rid, "", val, ""])
        for row in edit_rows:
            w.writerow(["move_edit_distance", row["id_a"], row["id_b"],
                        row["normalized_edit_distance"], row["qid"]])

    make_figures(rows, mat, ids, out_dir / "figures")


def make_figures(rows: list[dict], mat, ids: list[str], fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig_dir.mkdir(parents=True, exist_ok=True)
    by_id = {r["id"]: r for r in rows}

    # (a) move-sequence strips, grouped by question, Luna next to agent
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    q_order = sorted(by_q, key=lambda q: (q not in QUESTIONS, q))
    strip_rows = []
    for q in q_order:
        group = sorted(by_q[q], key=lambda r: (r["system"] != "luna", r["commentator"]))
        strip_rows.extend(group)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(strip_rows))))
    for row_i, r in enumerate(strip_rows):
        units = r["_units"]
        n = len(units) or 1
        x = 0.0
        for u in units:
            w = 1.0 / n
            ax.barh(row_i, w, left=x, color=MOVE_COLORS.get(u.move, "#333333"), height=0.8)
            x += w
    ax.set_yticks(range(len(strip_rows)))
    ax.set_yticklabels([f"{r['qid']} / {r['system']}:{r['commentator']}" for r in strip_rows],
                        fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of answer")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MOVE_COLORS.values()]
    ax.legend(handles, list(MOVE_COLORS.keys()), fontsize=6, loc="upper right",
              bbox_to_anchor=(1.25, 1))
    fig.tight_layout()
    fig.savefig(fig_dir / "move_sequences.png", dpi=150)
    fig.savefig(fig_dir / "move_sequences.svg")
    plt.close(fig)

    # (b) 2D PCA, colour = system, marker = tradition, label = qid
    if len(ids) >= 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(mat)
        system_colors = {"agent": "#0072B2", "luna": "#D55E00"}
        tradition_markers = {"sunni": "o", "imami": "s", "none": "^"}
        fig, ax = plt.subplots(figsize=(8, 8))
        for i, rid in enumerate(ids):
            r = by_id[rid]
            ax.scatter(coords[i, 0], coords[i, 1],
                       c=system_colors.get(r["system"], "#333333"),
                       marker=tradition_markers.get(r["tradition"], "x"), s=60,
                       edgecolors="black", linewidths=0.3)
            ax.annotate(r["qid"], (coords[i, 0], coords[i, 1]), fontsize=5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Answer embeddings: PCA (colour=system, marker=tradition)")
        fig.tight_layout()
        fig.savefig(fig_dir / "pca_answers.png", dpi=150)
        fig.savefig(fig_dir / "pca_answers.svg")
        plt.close(fig)


# --------------------------------------------------------------------------- review sheets

def build_review_sheets(rows: list[dict], records_by_id: dict, out_dir: Path):
    review_dir = out_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)

    for qid, group in by_q.items():
        group = sorted(group, key=lambda r: (r["system"] != "luna", r["commentator"]))
        lines = [f"# {qid}", "", f"**Question:** {QUESTIONS.get(qid, group[0]['id'])}", ""]
        for r in group:
            rec = records_by_id[r["id"]]
            who = "Luna" if r["system"] == "luna" else f"Agent / {r['commentator']}"
            lines.append(f"## {who}")
            lines.append("")
            grounding_log = rec["session"].get("grounding_log", [])
            verified_quotes = {normalize_arabic(g.get("quote", ""))
                               for g in grounding_log if g.get("verified")}
            unverified_quotes = {normalize_arabic(g.get("quote", ""))
                                 for g in grounding_log if not g.get("verified")}
            for u in r["_units"]:
                tag = f"[{u.move}]"
                text = u.text
                mark = ""
                nt = normalize_arabic(text)
                if any(v and v in nt for v in verified_quotes):
                    mark = " ✓"
                elif any(v and v in nt for v in unverified_quotes):
                    mark = " ✗"
                lines.append(f"**{tag}** {text}{mark}")
                lines.append("")
            lines.append("**Metrics**")
            lines.append("")
            lines.append("| metric | value |")
            lines.append("|---|---|")
            for key in ("words", "n_quotations", "verified", "verified_rate",
                       "mentions_total", "stated_limits", "grading_terms"):
                lines.append(f"| {key} | {r.get(key, '')} |")
            lines.append("")
            if r["level"] == "2":
                lines.append("**Hand score (0/1/2, fill in)**")
                lines.append("")
                lines.append("| hs_variants | hs_grading | hs_link | hs_proposal |")
                lines.append("|---|---|---|---|")
                lines.append("|  |  |  |  |")
                lines.append("")
        (review_dir / f"{qid}.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- README

def write_readme(out_dir: Path, *, level1_dir: Path, level2_dir: Path | None,
                 ner_available: bool, n_agent: int, n_luna: int,
                 missing: list[str], llm_moves_path: Path | None = None) -> None:
    lines = [
        "# Query-Agent vs Luna comparison",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Inputs",
        f"- Agent sessions index: `{QUESTION_SET_INDEX.relative_to(ROOT)}` ({n_agent} cells loaded)",
        f"- Luna level-1 dir: `{level1_dir}`",
        f"- Luna level-2 dir: `{level2_dir}`" if level2_dir else
        "- Luna level-2 dir: NOT FOUND -- level-2 Luna rows are missing from this run.",
        (f"- LLM move overrides: `{llm_moves_path}` (scripts/tag_moves_llm.py) -- units "
         "the rule tagger left as `other` are relabelled by its final_move; see "
         "move_share_<move>/move_sequence below and moves.csv (rule column gets a "
         "'+llm' suffix where this applied)." if llm_moves_path else
         "- LLM move overrides: none (rule-only tagging; see scripts/tag_moves_llm.py "
         "for the optional LLM pass over `other` units)."),
        f"- Luna records loaded: {n_luna}",
        f"- NER: {'camel_tools NERecognizer.pretrained() loaded' if ner_available else 'UNAVAILABLE -- fell back to rijal-list regex matching only (see extract_entities in compare_systems.py)'}",
        "",
        "## Caveats",
        "- n=1 per cell throughout. This is descriptive only -- no significance test is computed anywhere.",
        "- Luna answers one plain call per question (no commentator); the agent answers once per (question, commentator) cell. Counts are not directly comparable 1:1 across systems for commentator-scoped metrics.",
        "- Level-2 'R' (retrieved) recall is computed only against `sources[].excerpt`; the session schema does not store tool-call *results*, only tool/args/stage, so retrieval recall through tool_calls_log could not be computed.",
        "- 'unverified' for Luna means 'not found anywhere in the local corpus' -- not necessarily fabricated (the corpus is not exhaustive of everything a frontier model may correctly know).",
        "- A gold key-variant item marked `\"status\": \"candidate — user to confirm\"` in compare_lexicons.json (currently q9_shum's kafi_shum_gloss) is still scored per-item (A_<item>/R_<item> columns) but excluded from the A_recall/R_recall denominator until a human confirms its surface forms against the actual source text.",
        "- Gold items may also carry `core_fragments`: a list of fragment-groups, e.g. [[\"لا يقطع\", \"شي\"]]. An item matches if EITHER a full `surfaces` string is found, OR all fragments of any one group co-occur within a single sentence (SENTENCE_SPLIT_RE) of the answer (A) or of a source excerpt (R). This makes matching robust to word-order/insertion and to a lexicon surface list that only wrote one side's (Sunni) phrasing.",
        "",
        "## Files",
        "- `metrics.csv`: one row per answer. See column groups below.",
        "- `entities.csv`: (answer id, entity, in_imami, in_sunni) -- every extracted candidate name.",
        "- `moves.csv`: (id, unit_index, unit_text[:200], move, rule) -- one row per rhetorical unit.",
        "- `keyness.csv`: Dunning G^2 keywords, Arabic answers only, for Luna-vs-agent and Sunni(cayni+nawawi)-vs-Mazandarani.",
        "- `similarity.csv`: pairwise cosine similarity; silhouette scores raw AND under three confound checks (see below); evidence closeness (agent only); pairwise normalized move-sequence edit distance within each question.",
        "- `figures/move_sequences.{png,svg}`: one horizontal strip per answer, segments coloured by move (including the new `report` move), grouped by question.",
        "- `figures/pca_answers.{png,svg}`: 2D PCA of RAW answer embeddings, colour=system, marker=tradition, label=qid.",
        "- `review/<qid>.md`: side-by-side qualitative sheet, moves tagged inline, quotations marked (verified) / (unverified), metrics table, empty hand-score fields for level-2 questions.",
        "",
        "## similarity.csv: raw silhouettes vs the confound checks",
        "Rows keyed by `kind`:",
        "- `silhouette_raw_by_{system,tradition,qid}`: the original silhouette (cosine, mean-of-paragraph embeddings), Arabic answers only.",
        "- `silhouette_agent_only_by_tradition`: silhouette by tradition (sunni=cayni+nawawi vs imami=mazandarani) computed ONLY over the agent's own Arabic answers -- isolates a tradition effect from the much larger agent-vs-Luna system gap that dominates the raw-by-tradition number.",
        "- `silhouette_controlled_by_{system,tradition,qid}`: same three silhouettes, but each answer is first run through strip_markdown (headings/bullets/bold/blockquote markers removed) and chunked into at most 8 fixed ~150-word chunks before mean-pooling (hermeneutics-free length/format control -- the agent's answers use far more markdown structure and run longer than Luna's, and this isolates the system signal from that).",
        "- `silhouette_level2_by_system`: silhouette by system restricted to the 4 level-2 Arabic questions (Luna vs agent), where the two systems' answer lengths are closer.",
        "- `spearman_wordcount_pc1`: Spearman correlation between answer word count and PC1 of the RAW embeddings (Arabic answers), extra column carries the p-value -- shows how much of the raw embedding space's dominant axis is explained by length alone.",
        "- `silhouette_raw_by_length_tercile`: raw-embedding silhouette when answers are labelled purely by a word-count tercile (short/medium/long) instead of system/tradition/qid -- a direct measure of how separable length alone is in that same space.",
        "",
        "## metrics.csv column groups",
        "- id/system/qid/level/lang/commentator/tradition: row identity.",
        "- words/paragraphs/headings/list_items/blockquotes: structure counts (hermeneutics/answer_moves.split_units).",
        "- n_quotations/verified/unverified/verified_rate/quotes_per_1k_words: from the stored grounding_log (agent) or grounding_log/quotations/verified (Luna).",
        "- verified_outside_scope/foreign_commentary: agent only, from grounding_log flags.",
        "- mentions_<commentator>/mentions_total: commentator name mentions (Arabic+Latin variants, scripts/compare_lexicons.json:commentator_names).",
        "- stated_limits: count of the stated-limits lexicon.",
        "- A_<item>/R_<item>/A_recall/R_recall/hs_*: level-2 only, gold key-variant recall (surfaces OR core_fragments, see above) and empty hand-score fields. Candidate items (status field set) are excluded from A_recall/R_recall.",
        "- grading_terms: proxy count of isnad-grading vocabulary.",
        "- tokens_in/tokens_out/tool_calls/latency_s/truncated: cost. Agent tokens/latency summed from outputs/runs/set_<qid>_<commentator>/*.jsonl; Luna from its own record. truncated flags tokens_out>=max_tokens (Luna only; N/A for the agent, which has no single max_tokens ceiling).",
        "- hedges_per_1k/boosters_per_1k/attribution_verbs_per_1k/attributed_sentence_share: stance/attribution lexicons.",
        "- honorific_imami_style/honorific_sunni_style: honorific-form counts. imami_style includes عليه السلام/عليهم السلام/عليها السلام/صلوات الله عليه/صلى الله عليه وآله/صلى الله عليه وآله وسلم; sunni_style includes رضي الله عنه/عنها/عنهم/صلى الله عليه وسلم and the ﷺ ligature (U+FDFA). Both sides are matched via the same normalize_arabic-based, non-overlapping alternation (see _count_terms) so diacritics and hamza-form differences on either side of the comparison never cause a spurious miss; the longer imami phrase صلى الله عليه وآله وسلم does not also register as the shorter sunni phrase صلى الله عليه وسلم because the intervening وآله breaks that exact substring.",
        "- n_person_entities/n_unique_persons/share_in_imami_rijal/share_in_sunni_rijal: entity extraction (NER or rijal-regex fallback) cross-referenced against data/rijal/*.",
        "- mattr_100/domain_term_density_per_1k: lexical richness and hadith-science term density.",
        "- reuse_cayni/reuse_nawawi/reuse_mazandarani: share of the answer's word 4-grams occurring in each commentator's babs text (cached in outputs/comparison/cache/).",
        "- move_share_<move>/move_sequence: rhetorical-move shares (moves: claim, report, evidence_quote, isnad_grading, lexical_gloss, objection, concession, reply, limit, proposal, other) and the compact per-unit sequence string.",
        "",
        "## Move tagger change: claim vs report",
        "`claim` now fires ONLY on an explicit assertion marker (والصحيح أن، الراجح، الحق أن، نرى أن، \"the correct view\", \"we hold\"...). A unit that merely carries an attribution verb (قال، روى، ذكر...) or a report marker (ذهب الجمهور إلى، \"according to\"...), and is not already quote-dominated, is tagged `report` instead -- a reported view, not the answerer's own assertion. Being a list item is no longer sufficient on its own for `claim`; with no marker present a unit stays `other` (rule `no_rule_fired`).",
        "",
    ]
    if missing:
        lines.append("## Data problems noticed")
        lines.extend(f"- {m}" for m in missing)
        lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", type=Path, default=QUESTION_SET_INDEX)
    p.add_argument("--luna-level1-dir", type=Path, default=LEVEL1_LUNA_DEFAULT)
    p.add_argument("--luna-level2-dir", type=Path, default=None,
                   help="Default: newest outputs/frontier/frontier_openai_gpt-5.6-luna_* "
                        "dir with max_tokens==16000.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--skip-similarity", action="store_true",
                   help="Skip embeddings/silhouette/figures (faster iteration/tests).")
    p.add_argument("--llm-moves", type=Path, default=None,
                   help="A scripts/tag_moves_llm.py llm_moves.csv. When given, every "
                        "unit the rule tagger left as `other` is overridden with that "
                        "file's final_move before move shares, move_sequence, the "
                        "figures and the review sheets are built.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or (ROOT / "outputs" / "comparison" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    lexicons = load_lexicons()
    missing: list[str] = []

    agent_records = load_agent_answers(args.index)
    level2_dir = args.luna_level2_dir or find_level2_luna_dir()
    if level2_dir is None:
        missing.append("No 16000-token Luna level-2 dir found; level-2 Luna rows are absent.")
    luna_records = load_luna_answers(args.luna_level1_dir, level2_dir)

    records = agent_records + luna_records
    records_by_id = {r["id"]: r for r in records}
    print(f"Loaded {len(agent_records)} agent answers, {len(luna_records)} Luna answers",
          flush=True)

    ner = try_camel_ner()
    imami_names = _rijal_name_set(ROOT / "data" / "rijal" / "imami_rijal.json")
    sunni_names = _rijal_name_set(ROOT / "data" / "rijal" / "sunni_rijal.json")

    print("Building n-gram reuse caches (first run is slow)...", flush=True)
    ngram_caches = {c: build_ngram_cache(c) for c in COMMENTATORS}

    llm_moves = None
    if args.llm_moves:
        llm_moves = load_llm_moves(args.llm_moves)
        print(f"Loaded {len(llm_moves)} LLM-sourced move overrides from {args.llm_moves}",
              flush=True)

    rows, entity_rows = build_metrics_rows(records, lexicons, ner, imami_names,
                                           sunni_names, ngram_caches, llm_moves)

    write_csv(out_dir / "metrics.csv", rows)
    write_csv(out_dir / "entities.csv", entity_rows, skip_keys=())
    write_moves_csv(out_dir / "moves.csv", rows)

    arabic_rows = [r for r in rows if r["lang"] and "english" not in r["lang"]]
    luna_texts = [r["_answer"] for r in arabic_rows if r["system"] == "luna"]
    agent_texts = [r["_answer"] for r in arabic_rows if r["system"] == "agent"]
    sunni_texts = [r["_answer"] for r in arabic_rows if r["tradition"] == "sunni"]
    mazandarani_texts = [r["_answer"] for r in arabic_rows if r["tradition"] == "imami"]
    keyness_rows = []
    if luna_texts and agent_texts:
        keyness_rows += [{**row, "contrast": "luna_vs_agent"}
                         for row in keyness_table(luna_texts, agent_texts)]
    if sunni_texts and mazandarani_texts:
        keyness_rows += [{**row, "contrast": "sunni_vs_mazandarani"}
                         for row in keyness_table(sunni_texts, mazandarani_texts)]
    write_csv(out_dir / "keyness.csv", keyness_rows, skip_keys=())

    if not args.skip_similarity:
        try:
            compute_similarity(rows, out_dir, records_by_id)
        except Exception as e:  # noqa: BLE001
            # Embeddings/PCA/silhouette touch several optional, heavy libraries; any of
            # them can fail in ways worth recording rather than crashing the whole run.
            (out_dir / "similarity_ERROR.txt").write_text(str(e), encoding="utf-8")
            missing.append(f"similarity/figures step raised: {e!r}")

    build_review_sheets(rows, records_by_id, out_dir)

    write_readme(out_dir, level1_dir=args.luna_level1_dir, level2_dir=level2_dir,
                ner_available=ner is not None, n_agent=len(agent_records),
                n_luna=len(luna_records), missing=missing, llm_moves_path=args.llm_moves)

    print(f"\nWrote comparison bundle: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
