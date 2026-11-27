"""Inquiry graph. Spec §3.2.

Three stages, three interrupts, state checkpointed before each one so an abort never
loses work. Stage 1 receives COMPUTED network metrics from the §3.5 export - never a
raw JSON dump, and never recomputed here (defect #9).
"""
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from hermeneutics.conditions import CONDITIONS
from hermeneutics.corpus_tools import outside_scope
from hermeneutics.llm import call_json, call_model
from hermeneutics.state import InquiryState
from hermeneutics.tool_loop import run_tools

TOP_NODES = 25  # the whole node list is far too large for a prompt

# A short glossary of the classical vocabulary the commentators actually use, for
# terms an English or colloquial-Arabic question is likely to reach for. Kept short
# and put in one place: the canonicalise prompt cites it directly, so a new mapping
# belongs here, not scattered across stage prompts.
CANONICALISE_GLOSSARY = """\
taqiyya -> التقية
dissimulation -> التقية / الكتمان
free will / predestination / مسيّر / مخيّر -> الجبر والقدر / المشيئة / الاختيار
analogy / qiyas -> القياس / الرأي
mursal -> المرسل
munqati -> المنقطع
conflicting hadith -> اختلاف الحديث / مختلف الحديث
paraphrase transmission -> الرواية بالمعنى
supplication / dua -> الدعاء"""

CANONICALISE_SYSTEM = (
    "You canonicalise a hadith research question into classical Arabic before "
    "retrieval. The corpus and every commentary in it are classical Arabic; a "
    "question in English, colloquial Arabic or modern standard Arabic must be turned "
    "into the terms the commentators themselves use, or a literal-term search over "
    "the corpus finds nothing.\n\n"
    "Common English/colloquial forms and the classical terms they map to:\n"
    f"{CANONICALISE_GLOSSARY}\n\n"
    'Reply with a JSON object only: {"question_ar": "...", "terms": ["...", ...], '
    '"lang": "..."}.\n'
    "question_ar: a faithful classical-Arabic rendering of the question, adding no "
    "claim the asker did not make.\n"
    "terms: 3 to 8 Arabic search terms in the commentators' technical vocabulary.\n"
    "lang: the language of the ORIGINAL question (e.g. \"english\", \"colloquial "
    "arabic\", \"modern standard arabic\")."
)

# The matn stage reads the corpus under study, not every raw file on disk.
# corpus_tools.SOURCES key per corpus; "muslim_matn" because "muslim" is the cluster file.
MATN_SOURCE = {"bukhari": "bukhari", "kafi": "kafi", "muslim": "muslim_matn"}

# A real run can hand synthesis 100+ accumulated sources at ~1500 chars each -- far past
# a sane prompt, and get_chat_model(max_tokens=...) only bounds the OUTPUT. Cap what
# gets rendered, after dedup so the cap isn't spent on repeats of the same excerpt.
SYNTHESIS_SOURCE_CAP = 40

# Real inquiry sessions have argued, from the isnad stage's own centrality figures,
# that a ban is "one rule from the central source" -- a network property presented as
# doctrinal warrant. Appended to the isnad and synthesis stage prompts below (spec item
# 4). Kept local to this graph, not in hermeneutics.prompts.TEMPLATES: that module's
# prompt_hashes() backs the FROZEN study pipeline's published results (nodes.py,
# study_graph.py), and this sentence belongs to the Query-Agent only.
NETWORK_METRICS_NOTE = (
    "Network metrics describe the structure of transmission only. Never cite degree, "
    "betweenness, eigenvector centrality, or community membership as evidence for a "
    "doctrinal or legal position."
)


@dataclass
class InquiryDeps:
    model: Any
    trace: Any
    manifest_id: str
    metrics: dict
    framework: str          # method file text; empty unless the condition needs it
    commentator: str
    corpus: str             # the hadith corpus under study: bukhari | kafi | muslim
    corpus_tools: Any = None   # CorpusTools; None means this condition retrieves nothing
    condition: str = "agent_method"


def load_metrics(path: Path) -> dict:
    """Load the §3.5 export. Absent means defect #9 is still open."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run scripts/export_network_metrics.py "
            "(or notebook 01) before the inquiry graph. Spec §3.5."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported schema_version in {path}: expected 1, got {data.get('schema_version')!r}. "
            "Re-run scripts/export_network_metrics.py to regenerate."
        )
    return data


def _metrics_digest(metrics: dict) -> str:
    """Compact, prompt-sized rendering of the computed topology."""
    g = metrics["graph"]
    top = sorted(metrics["nodes"], key=lambda n: -n["betweenness"])[:TOP_NODES]
    lines = [
        f"nodes={g['n_nodes']} edges={g['n_edges']} modularity={g['modularity']} "
        f"components={g['n_components']} sizes={g['component_sizes'][:10]}",
        "",
        "Top narrators by betweenness (degree / betweenness / eigenvector / community):",
    ]
    lines += [
        f"- {n['id']}: {n['degree']} / {n['betweenness']} / {n['eigenvector']} / c{n['community']}"
        for n in top
    ]
    return "\n".join(lines)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage(name: str, system: str, build_user, out_key: str, prefetch=None,
           record_fetched: bool = True):
    """One stage node plus its interrupt.

    prefetch(state, deps) -> list[dict] of sources the stage receives BEFORE it may call
    a tool. The matn stage ran on a prompt that said "read the matn" and carried none; a
    stage that depends on a text gets that text whether or not the model thinks to
    search for it.

    record_fetched controls whether `fetched` is written into the "sources" channel
    alongside whatever the stage's own tool calls newly retrieved. True (the default) is
    right when `fetched` is new to the session, as the matn stage's corpus search is.
    Pass False for a stage whose prefetch REPLAYS sources an earlier stage already wrote
    (synthesis, which prefetches state["sources"] itself) -- otherwise every one of those
    entries gets written a second time into an operator.add channel and the final source
    count no longer matches what the run actually retrieved.
    """

    def node(state: dict, deps: InquiryDeps) -> dict:
        spec = CONDITIONS.get(deps.condition, {})
        retrieves = bool(spec.get("tools")) and deps.corpus_tools is not None
        fetched = prefetch(state, deps) if (prefetch and retrieves) else []
        user = build_user(state, deps, fetched)

        if not retrieves:
            # no_sources and passages must not retrieve. compare.py:92 already refuses
            # to record a no-retrieval condition that returned references.
            text = call_model(
                deps.model, system, user,
                trace=deps.trace, node=f"inquiry_{name}",
                bab_ref=f"{deps.corpus}/{name}",
                model_id=getattr(deps.model, "model_name", "fake"),
                manifest_id=deps.manifest_id,
            )
            return {out_key: text}

        result = run_tools(
            deps.model, system, user, deps.corpus_tools,
            trace=deps.trace, node=f"inquiry_{name}",
            bab_ref=f"{deps.corpus}/{name}",
            model_id=getattr(deps.model, "model_name", "fake"),
            manifest_id=deps.manifest_id,
        )
        return {
            out_key: result.text,
            "sources": [*fetched, *result.sources] if record_fetched else result.sources,
            "tool_calls_log": [{**call, "stage": name} for call in result.calls],
        }

    def review(state: dict, deps: InquiryDeps) -> dict:
        decision = interrupt({"stage": name, "output": state.get(out_key, "")})
        action = (decision or {}).get("action", "approve")
        # "auto_approve" is its own kind, not folded into "approve". The intervention log
        # is the only evidence the paper has that a human saw a stage, and a run nobody
        # watched must not be indistinguishable from one a scholar signed off.
        return {"intervention_log": [{"kind": action
                                      if action in ("approve", "edit", "auto_approve")
                                      else "approve",
                                      "kitab": name, "payload": dict(decision or {}),
                                      "ts": _now()}]}

    return node, review


def _validate_canonical(parsed) -> dict:
    """Raise on anything short of the contract canonicalise promises callers.

    A question_ar that is empty, or terms outside 3-8 items, is not a formatting
    mistake the way invalid JSON is -- but it is just as unusable, so it is treated
    the same way: retried once, then raised. Silently falling back to the raw
    question here would be exactly the bug this node exists to fix.
    """
    if not isinstance(parsed, dict):
        raise ValueError("canonicalise did not return a JSON object")  # noqa: TRY004 -- matches call_json's contract
    question_ar = parsed.get("question_ar")
    terms = parsed.get("terms")
    lang = parsed.get("lang")
    if not isinstance(question_ar, str) or not question_ar.strip():
        raise ValueError("canonicalise: question_ar is missing or empty")
    if (not isinstance(terms, list) or not (3 <= len(terms) <= 8)
            or not all(isinstance(t, str) and t.strip() for t in terms)):
        raise ValueError("canonicalise: terms must be a list of 3 to 8 non-empty strings")
    if not isinstance(lang, str) or not lang.strip():
        raise ValueError("canonicalise: lang is missing or empty")
    return {"question_ar": question_ar.strip(), "search_terms": [t.strip() for t in terms],
            "question_lang": lang.strip()}


def _canonicalise(state: dict, deps: InquiryDeps) -> dict:
    """Turn the asker's question into one classical-Arabic research question plus
    Arabic search terms, before any stage runs. Everything after this node is
    identical whatever language or register the question arrived in.

    Skipped entirely under no_sources (conditions.CONDITIONS[...]["canonicalise"]):
    the baseline must stay a model with no pipeline help, not even a rewritten
    question.
    """
    spec = CONDITIONS.get(deps.condition, {})
    if not spec.get("canonicalise", True):
        return {}

    question = state["question"]
    user = f"QUESTION: {question}"
    model_id = getattr(deps.model, "model_name", "fake")
    last_error: Exception | None = None

    for attempt in (1, 2):
        node = "inquiry_canonicalise" if attempt == 1 else "inquiry_canonicalise_retry"
        parsed = call_json(
            deps.model, CANONICALISE_SYSTEM, user, trace=deps.trace, node=node,
            bab_ref=f"{deps.corpus}/canonicalise", model_id=model_id,
            manifest_id=deps.manifest_id,
        )
        try:
            return _validate_canonical(parsed)
        except ValueError as exc:
            last_error = exc
            user = (
                f"QUESTION: {question}\n\n---\nYour previous reply did not satisfy the "
                f"contract ({exc}). Reply again with the JSON object only: "
                'question_ar (non-empty string), terms (3 to 8 non-empty strings), '
                'lang (non-empty string).'
            )

    raise ValueError(
        f"canonicalise on {deps.corpus} did not return valid question data after one "
        f"repair retry: {last_error}"
    )


def _prefetch_matn(state: dict, deps: InquiryDeps) -> list[dict]:
    """The hadith AND the commentator's own words on it, from the corpus and the
    commentary this commentator actually wrote.

    Searches with the CANONICAL Arabic question and terms, not the raw `question` --
    an English question against Arabic full text is zero retrieval every time (see
    the canonicalise node above), and colloquial Arabic misses the chapter headings'
    technical vocabulary almost as often.

    Two searches, not one: a matn stage that only ever retrieves the hadith text can
    answer every question without once reading what THIS commentator wrote about it,
    and two of four recorded mazandarani runs did exactly that -- his own commentary
    depended entirely on the model choosing to search for it. `kind` labels which
    result set is which so the prompt (and _render below) can tell them apart.
    """
    source = MATN_SOURCE.get(deps.corpus, "hadith")
    query = " ".join([state.get("question_ar", ""), *state.get("search_terms", [])]).strip()
    query = query or state["question"]
    hadith = deps.corpus_tools.search(query, source)
    commentary = deps.corpus_tools.search(query, "commentary")
    return ([{**item, "kind": "commentary"} for item in commentary] +
           [{**item, "kind": "hadith"} for item in hadith])


def _prefetch_synthesis(state: dict, deps: InquiryDeps) -> list[dict]:
    """What the isnad and matn stages already read.

    Not a fresh search: the evidence is in hand, and the stage that must quote it
    should see the same text the earlier stages reasoned over.
    """
    return list(state.get("sources", []))


def _dedup_sources(sources: list[dict]) -> list[dict]:
    """Drop repeats of the same excerpt, keeping the earliest occurrence.

    The same passage is often retrieved by both the isnad and matn stages; rendering it
    twice wastes SYNTHESIS_SOURCE_CAP slots on a repeat instead of new evidence.
    """
    seen = set()
    deduped = []
    for item in sources:
        key = (item.get("source"), item.get("reference"), item.get("excerpt"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _render(sources: list[dict]) -> str:
    # The Arabic author name leads when there is one: the prompt otherwise names a
    # commentator only by his Latin filename, and the model spells him from that.
    # `kind` (set by _prefetch_matn) labels commentary versus hadith so the prompt
    # shows which is which; absent elsewhere (synthesis's dedup'd mix, tool results),
    # where it renders as before.
    return "\n\n".join(
        f"[{item['kind'].upper() + ' — ' if item.get('kind') else ''}"
        f"{item['author'] + ' — ' if item.get('author') else ''}"
        f"{item.get('source', '?')} — {item.get('reference', '?')}]\n{item.get('excerpt', '')}"
        for item in sources) or "(nothing retrieved)"


def _question_block(state: dict) -> str:
    """Both the asker's own words and the canonical Arabic rendering, when there is
    one -- no_sources never canonicalises, so question_ar is absent there and this
    degrades to just QUESTION, unchanged from before Part 2."""
    original = f"QUESTION: {state['question']}"
    question_ar = state.get("question_ar")
    if not question_ar:
        return original
    return f"{original}\nCANONICAL ARABIC QUESTION (searched as): {question_ar}"


# Shared by every stage that retrieves: the corpus is classical Arabic, so a tool
# search in any other language or register finds nothing (the bug Part 2/3 fixes),
# and a quotation is only real evidence if the Arabic itself, not a paraphrase of
# it, sits inside the quotation marks.
_ARABIC_CORPUS_NOTE = (
    " The corpus is classical Arabic. Write every tool search in Arabic. Every "
    "quotation must be given verbatim in Arabic inside the quotation marks; put any "
    "translation or gloss outside them."
)


def _isnad_user(state: dict, deps: InquiryDeps, fetched: list[dict]) -> str:
    framework = f"\n\nINTERPRETIVE FRAMEWORK:\n{deps.framework}" if deps.framework else ""
    return (
        f"{_question_block(state)}\n\n"
        f"COMPUTED ISNĀD NETWORK METRICS ({deps.corpus}):\n{_metrics_digest(deps.metrics)}"
        f"{framework}\n\n"
        "What does the transmission structure show that bears on the question? "
        "Ground every claim in the metrics above; do not speculate about links not shown. "
        "Use lookup_narrator and lookup_isnad_network for any claim about a narrator. "
        f"{NETWORK_METRICS_NOTE}"
        f"{_ARABIC_CORPUS_NOTE}"
    )


def _matn_user(state: dict, deps: InquiryDeps, fetched: list[dict]) -> str:
    framework = f"\n\nINTERPRETIVE FRAMEWORK:\n{deps.framework}" if deps.framework else ""
    return (
        f"{_question_block(state)}\n\n"
        f"ISNĀD STAGE FINDINGS:\n{state.get('isnad_output', '')}\n\n"
        f"HADITH TEXT AND THE COMMENTATOR'S OWN COMMENTARY ON IT, FROM {deps.corpus.upper()}:\n"
        f"{_render(fetched)}"
        f"{framework}\n\n"
        "Read the matn above. What does it say, and how does it bear on the question? "
        "Search for more if these passages do not cover it, and check any word you "
        "gloss with lookup_word."
        f"{_ARABIC_CORPUS_NOTE}"
    )


def _in_session(quote: str, sources: list[dict]) -> bool:
    """Did this quotation come from what THIS session actually retrieved?"""
    from hermeneutics.grounding import normalize_arabic

    needle = normalize_arabic(quote)
    return any(needle in normalize_arabic(item.get("excerpt", "")) for item in sources)


def _strip_quote(text: str, quote: str) -> str:
    """Replace the occurrence of `quote` that conditions.quotations() found, using the
    ORIGINAL span in `text` rather than a naive text.replace(quote, ...).

    conditions.quotations() strips markdown from the matched text before returning it.
    Markdown wrapping the WHOLE quote (e.g. "**...**") leaves the inner run of
    characters contiguous, so a plain replace happens to still work; markdown INSIDE
    the quote (e.g. a stray `backtick`) does not, and text.replace(quote, ...) would
    silently no-op -- the log would say the quotation was censored while the text still
    shows it. Re-finding the match's raw span and clean-comparing it to `quote` avoids
    that.
    """
    from hermeneutics.conditions import _QUOTED, _clean

    for match in _QUOTED.finditer(text):
        raw = match.group(1) or match.group(2) or ""
        if _clean(raw) == quote:
            start, end = match.span(1) if match.group(1) is not None else match.span(2)
            return text[:start] + "[unverified quotation removed]" + text[end:]
    # Fallback: no matching span found (should not happen since `quote` came from
    # quotations() on this same text), but never leave a fabrication unmarked.
    return text.replace(quote, "[unverified quotation removed]")


def _foreign_commentary(found_in: list[str], commentator: str) -> bool:
    """True when a VERIFIED quotation was found only in a rival commentator's book.

    That is the mis-attribution signal: text credited to `commentator` but actually
    drawn from someone else's commentary. Quoting the lexica, the Qur'an, the rijal
    works or the hadith corpus is legitimate and commentators do it constantly, so only
    a source that starts with "commentary:" and is not `{commentator}_babs` counts.
    An empty `found_in` (the welded-fragments case, or nothing verified) is never
    flagged -- there is no source to accuse of being foreign.
    """
    if not found_in:
        return False
    own_stem = f"{commentator}_babs"
    return all(
        src.startswith("commentary:") and Path(src.split(":", 1)[1]).stem != own_stem
        for src in found_in
    )


def _synthesis_gate(state: dict, deps: InquiryDeps) -> dict:
    """Strip every quotation that does not occur in the corpus, and log the lot.

    Uses conditions.verify_quotation, which honours elisions ("كان رجال... فقالوا"): a
    scholar's synthesis elides, and a verifier that demands the whole string verbatim
    would censor honest quotations.

    Unlike the study graph's write gate, this rate VARIES. RESULTS.md records that
    real_grounding is 1.0 in every study run by construction -- the gate substitutes the
    span before verifying it, so an ungrounded move cannot be written and the rate
    cannot falsify anything. Here the model writes first and is checked after.

    Three things are RECORDED here but never used to strip, because the text is real
    and stripping it would be wrong -- the human reviewer is the arbiter, this gate's
    job is only to make the fact visible on the review sheet:
      - foreign_commentary: verified, but every source it occurs in is a RIVAL
        commentary -- a quotation attributed to `commentator` but actually drawn from
        someone else's book.
      - outside_scope: verified, but every source it occurs in lies outside this run's
        scope (corpus_tools.TRADITION_SCOPE) -- a superset of foreign_commentary, since
        it also catches a RIVAL HADITH COLLECTION quoted from memory (a cayni run
        producing al-Kafi text), which foreign_commentary deliberately ignores because
        quoting the hadith corpus itself is legitimate. Verification against it stays
        whole-corpus (corpus_tools.CorpusTools.verify_quote is never scope-filtered) so
        this flag can even be computed; only the flag, not the verdict, is scoped.
      - echoes_question: the quotation is just a fragment of the question itself,
        scare-quoted or topic-marked rather than cited as evidence. It still fails
        verification and gets stripped on that basis (it is not a real quotation), but
        the flag lets the reported rate be corrected for it without re-deriving which
        entries were question echoes from the raw log.
    """
    text = state.get("synthesis_output", "") or ""
    if deps.corpus_tools is None:
        return {}

    from hermeneutics.conditions import quotations, verify_quotation
    from hermeneutics.grounding import normalize_arabic

    sources = state.get("sources", [])
    # run_inquiry.py folds the researcher-selected passages into state["question"] for
    # the `passages` condition, so this would not always be the RAW question there. No
    # live bug: `passages` gets corpus_tools=None, so this gate returns {} above before
    # reaching here. If `passages` ever gains tools, a genuine quotation drawn from the
    # researcher's own passages could be misread as echoing the question.
    question_norm = normalize_arabic(state.get("question", "") or "")
    # question_ar is the model's OWN rendering of the question -- an echo of it must
    # be caught the same way an echo of the raw question is, or a quotation that
    # simply repeats the canonical Arabic question back would be misread as a real
    # citation from the corpus.
    question_ar_norm = normalize_arabic(state.get("question_ar", "") or "")
    log = []
    for quote in quotations(text):
        outcome = verify_quotation(quote, deps.corpus_tools)
        found_in = outcome.get("sources", [])
        verified = bool(outcome["verified"])
        log.append({"quote": quote, "verified": verified,
                    "elided": outcome["elided"], "checked": outcome["checked"],
                    "in_session": _in_session(quote, sources),
                    "found_in": found_in,
                    "foreign_commentary": verified and
                                          _foreign_commentary(found_in, deps.commentator),
                    "outside_scope": verified and outside_scope(
                        found_in, deps.commentator,
                        getattr(deps.corpus_tools, "poc_root", None)),
                    "echoes_question": normalize_arabic(quote) in question_norm
                                       or normalize_arabic(quote) in question_ar_norm,
                    "ts": _now()})
        if not outcome["verified"]:
            # Remove the quotation, leave a visible scar. A silent deletion would make
            # the synthesis read as though the claim had never been made.
            text = _strip_quote(text, quote)

    return {"synthesis_output": text, "grounding_log": log}


def _synthesis_user(state: dict, deps: InquiryDeps, fetched: list[dict]) -> str:
    framework = f"\n\nINTERPRETIVE FRAMEWORK:\n{deps.framework}" if deps.framework else ""
    sources = _dedup_sources(fetched)[:SYNTHESIS_SOURCE_CAP]
    lang = state.get("question_lang")
    answer_in = (f" Write the final answer in {lang}, the language of the original "
                if lang else " Answer in the language of the original "
                ) + "question, even though the corpus and your quotations are Arabic."
    return (
        f"{_question_block(state)}\n\n"
        f"ISNĀD FINDINGS:\n{state.get('isnad_output', '')}\n\n"
        f"MATN FINDINGS:\n{state.get('matn_output', '')}\n\n"
        f"RETRIEVED SOURCES:\n{_render(sources)}"
        f"{framework}\n\n"
        "Synthesise an answer. Every quotation must be checked with verify_quote before "
        "you present it, and cited by its reference. State clearly what the evidence "
        f"does not settle. {NETWORK_METRICS_NOTE}"
        f"{_ARABIC_CORPUS_NOTE}{answer_in}"
    )


def build_inquiry_graph(deps: InquiryDeps, checkpointer):
    isnad, isnad_review = _stage("isnad", "You analyse hadith transmission networks.",
                                 _isnad_user, "isnad_output")
    matn, matn_review = _stage("matn", "You analyse hadith content.",
                               _matn_user, "matn_output", prefetch=_prefetch_matn)
    synth, synth_review = _stage("synthesis", "You synthesise hadith evidence.",
                                 _synthesis_user, "synthesis_output",
                                 prefetch=_prefetch_synthesis, record_fetched=False)

    g = StateGraph(InquiryState)
    for name, fn in [
        ("canonicalise", _canonicalise),
        ("isnad_stage", isnad), ("isnad_review", isnad_review),
        ("matn_stage", matn), ("matn_review", matn_review),
        ("synthesis", synth), ("synthesis_gate", _synthesis_gate),
        ("synthesis_review", synth_review),
    ]:
        g.add_node(name, partial(fn, deps=deps))

    g.add_edge(START, "canonicalise")
    g.add_edge("canonicalise", "isnad_stage")
    g.add_edge("isnad_stage", "isnad_review")
    g.add_edge("isnad_review", "matn_stage")
    g.add_edge("matn_stage", "matn_review")
    g.add_edge("matn_review", "synthesis")
    g.add_edge("synthesis", "synthesis_gate")
    g.add_edge("synthesis_gate", "synthesis_review")
    g.add_edge("synthesis_review", END)

    return g.compile(checkpointer=checkpointer)
