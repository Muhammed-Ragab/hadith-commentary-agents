"""Study-graph nodes. Spec §3.1, §4.

Nodes take (state, deps) and return a partial state dict. The graph binds deps with
functools.partial at build time - no globals, so every node is directly testable.
"""
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from hermeneutics.chunking import chunk_text
from hermeneutics.grounding import normalize_arabic
from hermeneutics.llm import call_json, call_model
from hermeneutics.markers import sharh_only
from hermeneutics.prompts import render
from hermeneutics.schemas import CommentaryReading, DiffDraft, DiffRecord
from hermeneutics.state import hypothesis_hash

MIN_TEXT_CHARS = 100  # below this a bāb carries no commentary to study
EVIDENCE_SPAN_TOKENS = 256

# B5: measured on run 15789363, arabic_evidence was the whole cited span (the old
# defect: catalog[evidence_id] copied in wholesale) - 757-910 chars median, 1012 max,
# and even the SHORTEST span on record was 159 chars. A genuine verbatim quotation
# supporting one interpretive move is a phrase, not a paragraph; the ceiling is set
# well under that 159-char floor so a copied chunk cannot pass as a quote.
# A quotation is bounded two ways, because one bound cannot do it. The absolute cap
# must clear the text the model is asked to quote: commentary sentences run to a median
# of 99 (ʿAynī) to 188 (Nawawī, Māzandarānī) characters over the first 40 bābs of each
# arm, and a cap under that rejects legitimate quotations -- which under greedy decoding
# are rejected twice (§8.10), becoming declared failures that B4 caps at 10% of an arm.
# A mis-set constant could fail a whole arm.
#
# But a cap generous enough for a sentence would also admit a whole SHORT chunk, which
# is the original defect (evidence copied from the span, grounded by construction). So
# the share of the cited span is bounded too: a real excerpt is part of its span, never
# most of it.
MAX_QUOTE_CHARS = 400
MAX_QUOTE_FRACTION = 0.4


@dataclass
class Deps:
    """Everything the nodes need that is not graph state."""

    model: Any
    trace: Any
    manifest_id: str
    babs: list[dict]
    commentator: str
    naive: dict[str, str]          # bab_ref -> precomputed naive reading (batch phase)
    seg: dict[str, int]            # bab_ref -> matn_end_char (segmentation pass)
    token_len: Callable[[str], int]
    max_chunk_tokens: int = 6000
    # Scholar-review cadence: interrupt every Nth kitāb boundary. 1 = every kitāb
    # (the design default, and what Nawawi uses). ʿAynī has 87 kitābs after the
    # 2026-08-23 reparse, so it runs at 5 to keep total sittings near the original
    # budget. See handoff.md §10.6.
    review_group_size: int = 1
    pairing: dict[int, int] | None = None  # control only: local index -> local index
    notes_path: str = ""
    consolidate_model: Any = None
    kitab_summaries: dict[str, str] = field(default_factory=dict)
    # Control runs process a ~10% subset, so position in `babs` is not the global bāb
    # id. This maps local position -> global index so every bab_ref stays global and
    # score_control.py can still find each unit's commentary source.
    global_index: list[int] | None = None
    # B4: refs segment_one gave up on (e.g. cayni/2232, cayni/2872 -- an isnād the
    # 20,000-char window never reaches). run_segmentation.py logs these to
    # <name>_seg_failures.*.jsonl instead of writing a seg record, so they are also
    # absent from `seg` -- deliberately distinct from `seg` missing a ref entirely
    # (a real misconfiguration, which must still raise; see _segmentation_span).
    seg_failures: frozenset[str] = frozenset()
    # Ablation toggles for the two saturation-curve mechanisms (run 15789363). Both
    # default to the FIXED behaviour so a normal run gets the fix without asking;
    # scripts/run_ablation.sh flips them on one at a time to reproduce the defect on
    # purpose (condition A: both True: condition B: legacy_example_prompt False,
    # legacy_hypothesis_menu True; condition C, the default: both False).
    legacy_example_prompt: bool = False   # mechanism 1: prompts.read_commentary_legacy_example
    legacy_hypothesis_menu: bool = False  # mechanism 2: counted, frequency-ranked menu


def bab_ref(commentator: str, index: int) -> str:
    return f"{commentator}/{index}"


def _segmentation_span(deps: Deps, ref: str, text: str) -> tuple[int, int]:
    """The (start, end) of the matn inside this unit."""
    if ref not in deps.seg:
        raise ValueError(f"missing segmentation for {ref}")
    span = deps.seg[ref]
    start, end = span if isinstance(span, tuple) else (0, span)
    if not 0 <= start < end < len(text):
        raise ValueError(
            f"invalid segmentation for {ref}: matn is {start}-{end}, "
            f"text length={len(text)}"
        )
    return start, end


def select_next_bab(state: dict, deps: Deps) -> dict:
    """Advance the cursor to the next substantive bāb.

    Skips units under MIN_TEXT_CHARS. Defect #7: these are skipped here but counted
    and labelled by the parser regression suite, not silently dropped.

    Also skips units whose commentary source segment_one declared unsegmentable
    (B4). Skipping in run_segmentation alone is not enough: _segmentation_span
    below raises "missing segmentation" for any ref absent from deps.seg, so an
    omitted record would just move the crash from segment_one to here. Checking
    seg_failures specifically (not "ref not in deps.seg") keeps that guard live for
    a real misconfiguration -- an empty deps.seg with no declared failures still
    raises, as it always has.
    """
    cursor = state.get("cursor", 0)

    def to_global(local: int) -> int:
        if deps.global_index:
            if local >= len(deps.global_index):
                raise ValueError(
                    f"local index {local} out of range for global_index "
                    f"of length {len(deps.global_index)}"
                )
            return deps.global_index[local]
        return local

    def source_ref_at(idx: int) -> str:
        source_index = idx
        if state.get("control_mode") and deps.pairing:
            source_index = deps.pairing.get(idx, idx)
        return bab_ref(deps.commentator, to_global(source_index))

    while cursor < len(deps.babs) and (
        len(deps.babs[cursor]["text"].strip()) < MIN_TEXT_CHARS
        or source_ref_at(cursor) in deps.seg_failures
    ):
        cursor += 1

    if cursor >= len(deps.babs):
        return {
            "cursor": cursor,
            "current_bab": None,
            "current_bab_ref": "",
            "commentary_source_ref": "",
            "commentary_text": "",
        }

    bab = deps.babs[cursor]
    source_index = cursor
    if state.get("control_mode") and deps.pairing:
        source_index = deps.pairing.get(cursor, cursor)

    current_ref = bab_ref(deps.commentator, to_global(cursor))
    source_ref = bab_ref(deps.commentator, to_global(source_index))
    source_text = deps.babs[source_index]["text"]
    # ponytail: the sharh is what follows the matn. In an ʿAynī bāb that opens by
    # glossing Bukhārī's chapter heading, that opening gloss is commentary too and
    # is dropped here. Prepend source_text[:matn_start] if the heading gloss turns
    # out to carry moves the study should count.
    matn_start, commentary_start = _segmentation_span(deps, source_ref, source_text)

    if state.get("matn_only_mode"):
        # The no-sharh arm (2026-09-20, EXPLORATORY -- not pre-registered). Removing
        # the commentary outright cannot be the experiment: build_evidence_catalog
        # would produce no spans, no evidence_id would exist to cite, every move would
        # be rejected by _ungrounded_quotes, and the arm would score 0.0 by
        # construction -- measuring the code's write gate, not the model's priors.
        # Feeding the matn keeps the catalog, the citation contract and the grounding
        # check intact, so yield stays comparable with the study and control arms and
        # the question it answers is the real one: can the pipeline manufacture
        # hermeneutical "operations" from the hadith alone, with no sharh to read?
        commentary = source_text[matn_start:commentary_start]
    else:
        # sharh_only strips al-Kāfī's embedded `الأصل:` matn blocks; a no-op on
        # Nawawī and ʿAynī, which carry no such markers (§8.15's leak).
        commentary = sharh_only(source_text[commentary_start:])

    return {
        "cursor": cursor + 1,
        "current_bab": bab,
        "current_bab_ref": current_ref,
        "current_kitab": bab["kitab"],
        "commentary_source_ref": source_ref,
        "commentary_text": commentary,
    }


def naive_reading(state: dict, deps: Deps) -> dict:
    """Spec §4 call 1. Matn only - no commentary, no method hypothesis.

    Normally served from the batch phase (§6.2). The live fallback exists so a
    10-bāb smoke run works without a prior batch pass.
    """
    ref = state["current_bab_ref"]
    if ref in deps.naive:
        return {"naive_reading": deps.naive[ref]}

    bab_text = state["current_bab"]["text"]
    start, end = _segmentation_span(deps, ref, bab_text)
    matn = bab_text[start:end]
    text = call_model(
        deps.model,
        "You interpret hadith on their own terms.",
        render("naive_reading", matn=matn),
        trace=deps.trace, node="naive_reading", bab_ref=ref,
        model_id=getattr(deps.model, "model_name", "fake"),
        manifest_id=deps.manifest_id,
    )
    return {"naive_reading": text}


def _exact_chunks(text: str, max_tokens: int,
                  token_len: Callable[[str], int]) -> list[str]:
    """Token-bounded source slices whose text is copied exactly from the input."""
    words = list(re.finditer(r"\S+", text))
    chunks: list[str] = []
    start = end = None

    for word in words:
        if token_len(word.group()) > max_tokens:
            if start is not None:
                chunks.append(text[start:end])
                start = end = None
            pos = word.start()
            while pos < word.end():
                lo, hi, best = pos + 1, word.end(), pos + 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if token_len(text[pos:mid]) <= max_tokens:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                chunks.append(text[pos:best])
                pos = best
            continue

        if start is None:
            start, end = word.start(), word.end()
        elif token_len(text[start:word.end()]) <= max_tokens:
            end = word.end()
        else:
            chunks.append(text[start:end])
            start, end = word.start(), word.end()

    if start is not None:
        chunks.append(text[start:end])
    return chunks


def build_evidence_catalog(commentary: str, ref: str,
                           token_len: Callable[[str], int],
                           max_batch_tokens: int,
                           max_span_tokens: int = EVIDENCE_SPAN_TOKENS,
                           ) -> tuple[dict[str, str], list[str]]:
    """Build prompt batches of numbered spans while retaining exact source text."""
    spans = _exact_chunks(commentary, min(max_span_tokens, max_batch_tokens), token_len)
    grouped: list[list[str]] = []
    current: list[str] = []
    for span in spans:
        if current and token_len("\n\n".join(current + [span])) > max_batch_tokens:
            grouped.append(current)
            current = [span]
        else:
            current.append(span)
    if current:
        grouped.append(current)

    catalog: dict[str, str] = {}
    batches: list[str] = []
    for chunk_index, group in enumerate(grouped):
        rendered = []
        for span_index, span in enumerate(group):
            evidence_id = f"{ref}:c{chunk_index:03d}:s{span_index:03d}"
            catalog[evidence_id] = span
            rendered.append(f"[{evidence_id}]\n{span}")
        batches.append("\n\n".join(rendered))
    return catalog, batches



def _unknown_id_error(unknown: list[str], allowed) -> ValueError:
    """Name the format and one real id. "use the IDs printed above" was ignored twice.

    mazandarani/8 answered "1" and "7" on both attempts of job 15789363: greedy
    decoding repeats a format error unless the feedback shows the shape it wants.
    """
    example = next(iter(sorted(allowed)), "")
    hint = f' Evidence IDs look like "{example}"; copy one verbatim.' if example else ""
    return ValueError("unknown evidence_id: " + ", ".join(unknown) + "." + hint)


def _ungrounded_quotes(moves: list, catalog: dict) -> ValueError | None:
    """B5: verify each move's arabic_evidence, don't trust it.

    Rejects a quote that is too long to be a real excerpt (see MAX_QUOTE_CHARS) or
    that does not occur - after grounding.normalize_arabic folding, same as
    is_grounded itself - inside the span the move cites. Feeds the caller's existing
    corrective-retry loop instead of writing a record that grounds by construction.
    """
    problems = []
    for move in moves:
        quote = move.arabic_evidence
        chunk = catalog.get(move.evidence_id, "")
        too_long = len(quote) > MAX_QUOTE_CHARS
        # The share rule only means anything for a span long enough to select within.
        # Test fixtures and real short spans alike are legitimately quoted almost
        # whole, and rejecting that would be a rule about span size, not about quoting.
        too_much = (len(chunk) > MAX_QUOTE_CHARS
                    and len(quote) > MAX_QUOTE_FRACTION * len(chunk))
        if too_long or too_much:
            problems.append(
                f"{move.evidence_id}: arabic_evidence is {len(quote)} chars of a "
                f"{len(chunk)}-char span, over the {MAX_QUOTE_CHARS}-char limit or "
                f"{MAX_QUOTE_FRACTION:.0%} of the span. Quote the phrase that supports "
                "this move, not the whole span."
            )
            continue
        if normalize_arabic(quote) not in normalize_arabic(chunk):
            problems.append(
                f"{move.evidence_id}: arabic_evidence does not occur verbatim in that "
                "evidence span. Copy the excerpt exactly as printed there; do not "
                "paraphrase, alter, or invent it."
            )
    if not problems:
        return None
    return ValueError("quoted evidence failed grounding: " + " | ".join(problems))


def _read_commentary_batch(state: dict, deps: Deps, evidence_spans: str,
                           allowed_ids: set[str], h_hash: str,
                           ) -> tuple[CommentaryReading | None, str]:
    ref = state["current_bab_ref"]
    retry_feedback = ""
    last_error = ""
    # B (fix 1) toggle: legacy_example_prompt selects the pre-fix template, whose
    # copyable JSON example prints a live "lexical analysis" value instead of a
    # placeholder. Ablation only (Deps.legacy_example_prompt); real runs default False.
    template_name = ("read_commentary_legacy_example" if deps.legacy_example_prompt
                     else "read_commentary")
    for _ in range(2):
        try:
            raw = call_json(
                deps.model,
                "You study how a commentator interprets, not what the hadith says.",
                render(
                    template_name,
                    method_hypothesis=state.get("method_hypothesis", ""),
                    evidence_spans=evidence_spans,
                    retry_feedback=retry_feedback,
                ),
                trace=deps.trace, node="read_commentary", bab_ref=ref,
                model_id=getattr(deps.model, "model_name", "fake"),
                hypothesis_hash=h_hash, manifest_id=deps.manifest_id,
            )
            reading = CommentaryReading.model_validate(raw)
            unknown = sorted({o.evidence_id for o in reading.observations} - allowed_ids)
            if unknown:
                raise _unknown_id_error(unknown, allowed_ids)
            return reading, ""
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            retry_feedback = (
                "Previous response failed validation. Use the required JSON shape and "
                "only the evidence IDs printed above. Error: " + last_error
            )
    return None, last_error


def _observations_json(observations: list[dict]) -> str:
    # ponytail: empty values are dropped. `operation_qualifier` is usually "" at the map
    # step, and this JSON is measured against max_chunk_tokens to decide how many reduce
    # batches to run -- carrying `"operation_qualifier": ""` on every observation bought
    # an extra reduce call per bab and nothing else.
    lean = [{k: v for k, v in observation.items() if v != ""}
            for observation in observations]
    return json.dumps(
        {"status": "moves", "observations": lean},
        ensure_ascii=False,
    )


def _reduce_commentary_batch(state: dict, deps: Deps, observations: list[dict],
                             h_hash: str) -> tuple[CommentaryReading | None, str]:
    ref = state["current_bab_ref"]
    allowed_ids = {o["evidence_id"] for o in observations}
    max_observations = max(1, len(observations) // 2)
    retry_feedback = ""
    last_error = ""
    for _ in range(2):
        try:
            raw = call_json(
                deps.model,
                "You combine structured commentary observations without changing evidence.",
                render(
                    "reduce_commentary",
                    readings=_observations_json(observations),
                    max_observations=max_observations,
                    retry_feedback=retry_feedback,
                ),
                trace=deps.trace, node="reduce_commentary", bab_ref=ref,
                model_id=getattr(deps.model, "model_name", "fake"),
                hypothesis_hash=h_hash, manifest_id=deps.manifest_id,
            )
            reading = CommentaryReading.model_validate(raw)
            if reading.status != "moves":
                raise ValueError("reducer cannot replace observations with no_move")
            if len(reading.observations) > max_observations:
                raise ValueError(
                    f"reducer returned {len(reading.observations)} observations; "
                    f"maximum is {max_observations}"
                )
            unknown = sorted({o.evidence_id for o in reading.observations} - allowed_ids)
            if unknown:
                raise _unknown_id_error(unknown, allowed_ids)
            return reading, ""
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            retry_feedback = (
                "Previous response failed validation. Keep at least one observation and "
                "use only evidence IDs present in the input. Error: " + last_error
            )
    return None, last_error


def _reduce_commentary_observations(state: dict, deps: Deps,
                                    observations: list[dict], catalog: dict[str, str],
                                    h_hash: str,
                                    ) -> tuple[list[dict] | None, str, int]:
    def payload_tokens(items: list[dict]) -> int:
        selected = {item["evidence_id"] for item in items}
        return deps.token_len(_observations_json(items)) + sum(
            deps.token_len(catalog[evidence_id]) for evidence_id in selected
        )

    current = observations
    calls = 0
    while payload_tokens(current) > deps.max_chunk_tokens:
        previous_len = payload_tokens(current)
        groups: list[list[dict]] = []
        group: list[dict] = []
        for observation in current:
            if group and deps.token_len(
                _observations_json(group + [observation])
            ) > deps.max_chunk_tokens:
                groups.append(group)
                group = [observation]
            else:
                group.append(observation)
        if group:
            groups.append(group)

        reduced: list[dict] = []
        for batch in groups:
            reading, error = _reduce_commentary_batch(state, deps, batch, h_hash)
            calls += 1
            if reading is None:
                return None, error, calls
            reduced.extend(o.model_dump() for o in reading.observations)

        if payload_tokens(reduced) >= previous_len:
            return None, "commentary reduction did not shrink input", calls
        current = reduced
    return current, "", calls


def read_commentary(state: dict, deps: Deps) -> dict:
    """Spec §4 call 2. Structured map over immutable source evidence IDs."""
    ref = state["current_bab_ref"]
    commentary = state.get("commentary_text", "")
    if not commentary.strip():
        raise ValueError(f"no commentary text for {ref}")

    hypothesis = state.get("method_hypothesis", "")
    h_hash = hypothesis_hash(hypothesis)
    catalog, batches = build_evidence_catalog(
        commentary, ref, deps.token_len, deps.max_chunk_tokens
    )

    observations = []
    no_move_reasons = []
    for chunk_index, batch in enumerate(batches):
        prefix = f"{ref}:c{chunk_index:03d}:"
        allowed = {evidence_id for evidence_id in catalog if evidence_id.startswith(prefix)}
        reading, error = _read_commentary_batch(state, deps, batch, allowed, h_hash)
        if reading is None:
            return {
                "commentary_reading": None,
                "commentary_error": error,
                "evidence_catalog": catalog,
                "n_chunks": len(batches),
                "n_reduce_calls": 0,
            }
        observations.extend(o.model_dump() for o in reading.observations)
        if reading.status == "no_move":
            no_move_reasons.append(reading.no_move_reason)

    if observations:
        observations, error, reduce_calls = _reduce_commentary_observations(
            state, deps, observations, catalog, h_hash
        )
        if observations is None:
            return {
                "commentary_reading": None,
                "commentary_error": error,
                "evidence_catalog": catalog,
                "n_chunks": len(batches),
                "n_reduce_calls": reduce_calls,
            }
        combined = {"status": "moves", "observations": observations,
                    "no_move_reason": ""}
    else:
        reduce_calls = 0
        combined = {
            "status": "no_move",
            "observations": [],
            "no_move_reason": " ".join(no_move_reasons),
        }
    return {
        "commentary_reading": combined,
        "commentary_error": "",
        "evidence_catalog": catalog,
        "n_chunks": len(batches),
        "n_reduce_calls": reduce_calls,
    }


def _attempt_diff(state: dict, deps: Deps, ref: str, h_hash: str,
                  retry_feedback: str = "") -> dict:
    selected_ids = {
        observation.get("evidence_id", "")
        for observation in state.get("commentary_reading", {}).get("observations", [])
    }
    evidence_spans = "\n\n".join(
        f"[{evidence_id}]\n{text}"
        for evidence_id, text in state.get("evidence_catalog", {}).items()
        if evidence_id in selected_ids
    )
    raw = call_json(
        deps.model,
        "You make the difference between two readings explicit.",
        render("horizon_diff", naive=state["naive_reading"],
               commentary_reading=json.dumps(
                   state["commentary_reading"], ensure_ascii=False
               ),
               evidence_spans=evidence_spans,
               max_quote_chars=MAX_QUOTE_CHARS,
               retry_feedback=retry_feedback),
        trace=deps.trace, node="horizon_diff", bab_ref=ref,
        model_id=getattr(deps.model, "model_name", "fake"),
        hypothesis_hash=h_hash, manifest_id=deps.manifest_id,
    )
    return raw


def _write_failure(state: dict, deps: Deps, error: str, attempts: int) -> None:
    if not deps.notes_path:
        return
    ref = state["current_bab_ref"]
    notes_path = Path(deps.notes_path)
    stem = notes_path.stem.removesuffix("_notes")
    failures_path = notes_path.with_name(f"{stem}_failures.jsonl")
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failure = {
        "bab_ref": ref,
        "kitab": state.get("current_kitab", ""),
        "commentary_source_ref": state.get("commentary_source_ref", ref),
        "hypothesis_hash": hypothesis_hash(state.get("method_hypothesis", "")),
        "manifest_id": deps.manifest_id,
        "control": bool(state.get("control_mode", False)),
        "attempts": attempts,
        "error": error,
    }
    with open(failures_path, "a", encoding="utf-8", buffering=1) as fh:
        fh.write(json.dumps(failure, ensure_ascii=False) + "\n")


def horizon_diff(state: dict, deps: Deps) -> dict:
    """Materialize a schema-valid diff from immutable evidence IDs."""
    ref = state["current_bab_ref"]
    h_hash = hypothesis_hash(state.get("method_hypothesis", ""))
    last_error: Exception | None = None
    retry_feedback = ""
    catalog = state.get("evidence_catalog", {})

    if state.get("commentary_reading") is None:
        error = state.get("commentary_error", "commentary reading failed")
        _write_failure(state, deps, error, attempts=2)
        return {"diff_record": None}

    for _ in range(2):
        try:
            raw = _attempt_diff(state, deps, ref, h_hash, retry_feedback)
        except ValueError as exc:
            last_error = exc
            retry_feedback = "Previous response was not valid JSON. Error: " + str(exc)
            continue

        try:
            draft = DiffDraft.model_validate(raw)
            unknown = sorted({move.evidence_id for move in draft.moves} - set(catalog))
            if unknown:
                raise _unknown_id_error(unknown, catalog)

            quote_error = _ungrounded_quotes(draft.moves, catalog)
            if quote_error:
                raise quote_error

            moves = [
                {
                    **move.model_dump(),
                    "evidence_chunk": catalog[move.evidence_id],
                }
                for move in draft.moves
            ]
            record = DiffRecord(
                bab_ref=ref,
                kitab=state.get("current_kitab", ""),
                status=draft.status,
                gap=draft.gap,
                moves=moves,
                no_move_reason=draft.no_move_reason,
                revised_reading=draft.revised_reading,
                hypothesis_hash=h_hash,
                manifest_id=deps.manifest_id,
                control=bool(state.get("control_mode", False)),
            )
        except (ValidationError, ValueError) as exc:
            last_error = exc
            retry_feedback = (
                "Previous response failed validation. Return every field in the required "
                "shape and use only evidence IDs printed above. Error:\n" + str(exc)
            )
            continue
        return {"diff_record": record.model_dump()}

    _write_failure(state, deps, str(last_error), attempts=2)
    return {"diff_record": None}


HYPOTHESIS_TOP_N = 25  # operations kept in the running hypothesis


def accumulate(state: dict, deps: Deps) -> dict:
    """Append the diff record and refresh the running method hypothesis.

    Writes the JSONL line immediately (defect #6: the old pipeline logged at the end
    and lost everything on a crash).
    """
    record = state.get("diff_record")
    if not record:
        return {}

    if deps.notes_path:
        path = Path(deps.notes_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", buffering=1) as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "notes": [record],
        "method_hypothesis": _update_hypothesis(
            state.get("method_hypothesis", ""), record,
            legacy_menu=deps.legacy_hypothesis_menu,
        ),
    }


_HYPOTHESIS_HEADER = "Operations observed so far (accumulated across prior bābs):"
_HYPOTHESIS_LEGACY_HEADER = "Operations observed so far, most frequent first:"


def _update_hypothesis(current: str, record: dict, legacy_menu: bool = False) -> str:
    """The running method hypothesis fed into every read_commentary prompt.

    ponytail: a counter rendered as text, not an LLM summarisation call. It goes into
    every read_commentary prompt, so it must stay small and deterministic - an
    LLM-written hypothesis would add ~4,500 calls and make replay depend on it.

    That comment used to end "swap in a summarisation call only if the operation list
    proves too thin to steer." Saturation-curve run 15789363 found the opposite: a
    frequency-ranked, counted menu ("most frequent first", "(12x)") steers too HARD --
    mazandarani collapsed to 2 distinct operation-SETS across 24 bābs, vocabulary
    complete after bāb 1, because the model read a salience-ranked list of its own
    earlier labels before every new bāb. The accumulation itself stays (it is the
    hermeneutic circle, the intellectual core of the project); what had to go is the
    menu and its salience gradient -- the counts and the frequency ordering. The fixed
    form below is an alphabetical, uncounted list of distinct operations seen so far.

    `legacy_menu=True` restores the exact pre-fix counted/ranked render, parsed back the
    same way, for the ablation's condition A and B only (Deps.legacy_hypothesis_menu);
    every normal run defaults to the fixed form. The two renders are deliberately kept
    render/parse-symmetric on their own terms: `_update_hypothesis` re-parses its own
    previous output every call (see the "- name" / "- name (Nx)" loops below), and a
    mismatched round trip would silently reset accumulation on the next bāb.
    """
    if legacy_menu:
        counts: Counter[str] = Counter()
        for line in current.splitlines():
            if line.startswith("- ") and " (" in line:
                name, _, tail = line[2:].rpartition(" (")
                try:
                    counts[name] = int(tail.rstrip(")x"))
                except ValueError:
                    continue

        for move in record.get("moves", []):
            op = str(move.get("operation", "")).strip()[:60]  # cap to keep it bounded
            if op:
                counts[op] += 1

        top = counts.most_common(HYPOTHESIS_TOP_N)
        if not top:
            return current
        lines = "\n".join(f"- {name} ({n}x)" for name, n in top)
        return f"{_HYPOTHESIS_LEGACY_HEADER}\n{lines}"

    seen: dict[str, None] = {}
    for line in current.splitlines():
        if line.startswith("- "):
            name = line[2:].strip()
            if name:
                seen[name] = None

    for move in record.get("moves", []):
        op = str(move.get("operation", "")).strip()[:60]  # cap to keep it bounded
        if op:
            seen[op] = None

    if not seen:
        return current
    # Alphabetical, not insertion or frequency order - a stable, deterministic order
    # that carries no signal about which operation to reach for next. Capped at
    # HYPOTHESIS_TOP_N the same way the legacy menu was, so the prompt stays small;
    # this project has never come close to that many genuinely distinct operations.
    lines = "\n".join(f"- {name}" for name in sorted(seen)[:HYPOTHESIS_TOP_N])
    return f"{_HYPOTHESIS_HEADER}\n{lines}"


def provenance_ratio(records: list[dict]) -> dict:
    """Compiler's own reasoning vs relayed material. Spec §4 - a publishable finding."""
    counts = Counter()
    for rec in records:
        for move in rec.get("moves", []):
            counts[move.get("provenance", {}).get("author", "compiler_own")] += 1

    own, transmitted = counts.get("compiler_own", 0), counts.get("transmitted", 0)
    total = own + transmitted
    return {
        "compiler_own": own,
        "transmitted": transmitted,
        "compiler_own_pct": round(100.0 * own / total, 1) if total else 0.0,
    }


# A sentence is about provenance only if it names the axis. Percentages about anything
# else - operation frequencies, confidence - are not provenance claims.
_PROVENANCE_TOPIC = re.compile(
    r"transmitted|relay|predecessor|compiler'?s own|own voice|provenance|"
    r"quoted from|earlier authorit",
    re.I,
)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per ?cent)", re.I)
# (?<![.\d]) / (?!\.\d) keep this off decimal numbers: "81.1 / 18.9" was read as
# fraction 1/18 (5.6%) and false-rejected a method file, killing a whole cluster run.
_FRACTION = re.compile(r"(?<![.\d])(\d+)\s*(?:of|out of|/)\s*(\d+)(?!\.\d)\b")
# Rounding, and "about 7%" for 6.8%. Wider than rounding error, narrower than the 53
# points that separated §8.15's claim from its notes.
PROVENANCE_TOLERANCE_PCT = 3.0


def kitab_provenance(records: list[dict]) -> dict[str, dict]:
    """provenance_ratio per kitāb, keyed by the note's own `kitab` field.

    The global ratio is not the only figure a method file may legitimately state. §5 of
    the prompt asks what the ratio shows, and the model routinely answers with a
    breakdown -- "al-ʿItq: 100% own; al-ʿIlm: 100%; al-Salām: 95.8%". Those are counts
    the notes support; judging them against the global ratio is what produced three
    false UNVERIFIED stamps on correct files. Counting per kitāb makes the breakdown
    checkable instead of merely excusable.
    """
    by_kitab: dict[str, list[dict]] = {}
    for record in records:
        by_kitab.setdefault(record.get("kitab", ""), []).append(record)
    return {kitab: provenance_ratio(recs) for kitab, recs in by_kitab.items()}


def _shares(provenance: dict) -> tuple[float, float] | None:
    """(own%, transmitted%) for a counted block, or None when nothing was counted."""
    total = provenance.get("compiler_own", 0) + provenance.get("transmitted", 0)
    if not total:
        return None
    own_pct = provenance.get("compiler_own_pct", 0.0)
    return own_pct, round(100.0 - own_pct, 1)


def check_provenance_claim(method_file: str, provenance: dict,
                           per_kitab: dict[str, dict] | None = None) -> str:
    """Return an error when a stated provenance figure contradicts the counted one.

    §8.15: Nawawī's method file claimed the compiler "frequently references named
    predecessors" from 6 of 88 moves, and ʿAynī's made the same claim from 0 of 76.
    `provenance_ratio` had counted it correctly all along; `finalize` discarded the
    counts, so the global call answered §5 from prior. This is the layer whose output
    reaches the paper, and nothing checked it.

    Only *figures* are judged. Either share may be quoted - 93.2% own and 6.8%
    transmitted are one fact - so a sentence passes if it matches either.

    ponytail: qualitative prose ("frequently") is deliberately not parsed. Matching
    adverbs against the counts needs negation and contrast handling - "usually in his
    own voice rather than relaying transmitted material" reads as a violation to any
    vocabulary test - and a false rejection of a correct method file costs an arm.
    The computed block appended by `consolidate_global` is what the paper cites, so an
    unquantified adverb is a style problem rather than a false number. Tighten by
    asking for the figure as JSON, never by classifying prose.
    """
    shares = _shares(provenance)
    if shares is None:
        return ""
    own_pct, transmitted_pct = shares
    total = provenance["compiler_own"] + provenance["transmitted"]

    # A figure is a CONTRADICTION only if it matches nothing the notes counted. With
    # per_kitab supplied, a per-kitāb breakdown is judged against that kitāb's own
    # counts instead of the global ratio -- which is what the caller wants and what
    # three false UNVERIFIED stamps on correct files cost. Without it the old strict
    # behaviour is unchanged, so the inline consolidate_global path is untouched.
    allowed = {own_pct, transmitted_pct}
    allowed_totals = {total}
    for counts in (per_kitab or {}).values():
        kitab_shares = _shares(counts)
        if kitab_shares:
            allowed.update(kitab_shares)
            allowed_totals.add(counts["compiler_own"] + counts["transmitted"])

    for sentence in re.split(r"(?<=[.!?])\s+|\n", method_file):
        if not _PROVENANCE_TOPIC.search(sentence):
            continue

        stated = [float(m) for m in _PERCENT.findall(sentence)]
        # A fraction carries its own denominator, and that denominator says which
        # population it describes: "58 of 278" is the global claim, "0 of 5 moves in
        # al-Iman" is one kitāb. Judging the second against the global ratio was a
        # category error, not a tolerance problem. A denominator matching no counted
        # population is still judged -- an invented denominator is the defect itself.
        for n, d in _FRACTION.findall(sentence):
            if not int(d):
                continue
            if int(d) not in allowed_totals:
                # The denominator names a population nothing counted. "3 of 7 moves"
                # where no kitab has 7 moves and the corpus has 100 is not a
                # rounding disagreement -- the figure was invented whole, which is
                # the §8.15 defect in its purest form.
                return (
                    f"stated provenance figure {n} of {d} cites a population the "
                    f"notes never counted (global total {total}; per-kitab totals "
                    f"{sorted(allowed_totals - {total})}). "
                    f"Sentence: {sentence.strip()!r}"
                )
            stated.append(100.0 * int(n) / int(d))

        for value in stated:
            if all(abs(value - a) > PROVENANCE_TOLERANCE_PCT for a in allowed):
                return (
                    f"stated provenance figure {value:g} contradicts the notes: "
                    f"{provenance['transmitted']} of {total} moves are transmitted "
                    f"({transmitted_pct}%), {provenance['compiler_own']} are "
                    f"the compiler's own ({own_pct}%). Sentence: {sentence.strip()!r}"
                )
    return ""


def _source_key(name: str) -> str:
    """Match spellings of one authority: diacritics, case and punctuation folded away.

    The model transliterates inconsistently across notes, so exact-string dedup listed
    Qāḍī ʿIyāḍ three times in matched_qwen3_20260917's Nawawī file (`al-Qadi Iyad`,
    `al-Qāḍī Iyāḍ`, `al-Qāḍī ʿIyāḍ`), and al-Khaṭṭābī and Mālik twice each. A roster
    that repeats one man inflates the apparent breadth of the sources a §5 claim rests
    on -- and it is the first thing a hadith scholar notices.
    """
    folded = "".join(c for c in unicodedata.normalize("NFKD", name)
                     if not unicodedata.combining(c))
    folded = folded.translate(str.maketrans("", "", "ʿʾ'`ʼʻ"))
    return re.sub(r"[^a-z0-9؀-ۿ]+", " ", folded.lower()).strip()


def transmitted_sources(records: list[dict]) -> list[str]:
    """Every predecessor actually named in a `quoted_from`, deduplicated.

    The counts alone did not explain §8.15. ʿAynī's method file names al-Karmānī,
    al-Jawharī and al-Nawawī as authorities the commentator references, and its notes
    record `quoted_from` on **0 of 76** moves - the names are invented outright, not a
    mis-stated ratio. So the roster goes into the prompt beside the counts: a model
    told the recorded set is empty has nothing to relay.

    Spellings of one name are folded together by `_source_key` and reported in their
    fullest form. ponytail: only variants that differ by diacritics, case or
    punctuation merge. `al-Maziri`/`al-Muziri` differ by a letter and stay separate,
    because guessing that two near-miss strings are one scholar risks silently erasing
    a real second authority -- a worse error than listing one twice, and one nobody
    would catch.
    """
    variants: dict[str, list[str]] = {}
    for record in records:
        for move in record.get("moves", []):
            for name in move.get("provenance", {}).get("quoted_from", []):
                if not name or not name.strip():
                    continue
                key = _source_key(name)
                if key:
                    variants.setdefault(key, []).append(name.strip())

    return sorted(
        sorted(group, key=lambda n: (-sum(ord(c) > 127 for c in n), -len(n), n))[0]
        for group in variants.values()
    )


def provenance_line(provenance: dict, sources: list[str] | None = None) -> str:
    """The counted ratio as one line, for a prompt and for the method file."""
    total = provenance.get("compiler_own", 0) + provenance.get("transmitted", 0)
    if not total:
        return "no moves recorded"
    own_pct = provenance.get("compiler_own_pct", 0.0)
    line = (f"{provenance['compiler_own']} of {total} moves in the compiler's own voice "
            f"({own_pct}%); {provenance['transmitted']} of {total} transmitted from "
            f"named predecessors ({round(100.0 - own_pct, 1)}%)")
    if sources is not None:
        named = "; ".join(sources) if sources else "NONE - name no predecessors"
        line += f". Predecessors actually named in the notes: {named}"
    return line


def consolidate_kitab(kitab: str, records: list[dict], deps: Deps) -> dict:
    """Per-kitāb mini-consolidation. Spec §3.1, defect #3.

    Chunks the record list so a 197-bāb ʿAynī kitāb still fits the context.
    """
    model = deps.consolidate_model or deps.model
    blob = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    chunks = chunk_text(blob, deps.max_chunk_tokens, deps.token_len) or [""]
    provenance = provenance_ratio(records)

    partials = [
        call_model(
            model, "You summarise interpretive practice.",
            render("consolidate_kitab", kitab=kitab, notes=chunk,
                   provenance=provenance_line(provenance,
                                              transmitted_sources(records))),
            trace=deps.trace, node="consolidate_kitab", bab_ref=f"{deps.commentator}/{kitab}",
            model_id=getattr(model, "model_name", "fake"), manifest_id=deps.manifest_id,
        )
        for chunk in chunks
    ]

    return {
        "kitab": kitab,
        "summary": "\n\n".join(partials),
        "n_records": len(records),
        "provenance": provenance,
    }


def consolidate_global(summaries: dict[str, str], deps: Deps,
                       provenance: dict, sources: list[str] | None = None) -> str:
    """Global reduce over per-kitāb summaries only. Defect #3.

    The old pipeline fed every observation into one call. 63 summaries is a few
    thousand tokens; 4,522 records is not representable.

    `provenance` is counted over every move and is required, not optional: §5 of the
    prompt asks what the ratio shows, the summaries do not carry the counts, and a
    model asked for a number it cannot see answers from prior (§8.15). It is passed
    in, checked against what comes back, and appended to the file so the paper cites
    the count rather than the prose.
    """
    model = deps.consolidate_model or deps.model
    joined = "\n\n".join(f"## {k}\n{v}" for k, v in sorted(summaries.items()))
    counted = provenance_line(provenance, sources)
    retry_feedback = ""
    last_error = ""

    for _ in range(2):
        text = call_model(
            model, "You consolidate a commentator's method.",
            render("consolidate_global", commentator=deps.commentator, summaries=joined,
                   provenance=counted, retry_feedback=retry_feedback),
            trace=deps.trace, node="consolidate_global",
            bab_ref=f"{deps.commentator}/global",
            model_id=getattr(model, "model_name", "fake"), manifest_id=deps.manifest_id,
        )
        last_error = check_provenance_claim(text, provenance)
        if not last_error:
            return f"{text}\n\n## Provenance (counted, not written by the model)\n\n{counted}\n"
        retry_feedback = (
            "\nYour previous answer stated a provenance figure the notes contradict. "
            + last_error + " Use the counted figure below verbatim.\n"
        )

    # Three false positives in two days, each costing an arm's consolidation on a
    # CORRECT method file: a decimal ratio read as the fraction 1/18 (fixed, 1d0f512),
    # and twice a per-kitab breakdown judged against the global ratio ("83.3%, 40%,
    # 33.3% respectively"; "0 of 5 moves ... in al-Iman"). check_provenance_claim treats
    # every figure in a provenance sentence as the global claim -- a design flaw, not a
    # typo, and narrowing it means classifying prose, which that function's own
    # docstring warns against. So declare the disagreement and write the file, the way
    # B4 declares a bad bab: the counted block is appended verbatim and is what the
    # paper cites, so unresolved prose is a flag for review, not grounds to lose an arm.
    warning = (
        "## Unresolved provenance disagreement (declared, not silently accepted)\n\n"
        "The model twice stated a figure the counted provenance contradicts. The prose "
        "below is UNVERIFIED on this point; cite the counted block, not the sentence."
        "\n\n" + f"    {last_error}\n"
    )
    return f"{text}\n\n{warning}\n## Provenance (counted, not written by the model)\n\n{counted}\n"
