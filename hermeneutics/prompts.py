"""Prompt templates and their hashes. Spec §3.4 and §4.

Every prompt in the project lives in TEMPLATES. A prompt anywhere else is a prompt
whose hash never reaches the manifest, which breaks the reproducibility contract.

Literal JSON braces are doubled ({{ }}) so str.format does not crash on them - that
was defect #6.
"""
import hashlib

from hermeneutics.schemas import OPERATIONS

# docs/2026-09-16-operation-vocabulary.md: printed into every prompt that asks a model
# to name `operation`, so read_commentary and reduce_commentary teach the same closed
# set that schemas.Move enforces downstream. One line per label, "key - definition".
_OPERATION_LABELS = "\n".join(f"- {key}: {definition}" for key, definition in OPERATIONS.items())

_NAIVE = """You are a specialist in classical Islamic hermeneutics reading a hadith \
for the first time, with no commentary in front of you.

Give your best interpretation of the following matn: what it means, what it obliges \
or permits, and what you are uncertain about. Do not speculate about what a \
commentator would say. This is your own horizon, on its own terms.

Write in English, under 250 words.

MATN:
{matn}"""

# Job 15789363's saturation_curve measurement: every arm converged on the ONE operation
# string printed here as a live, answerable value ("lexical analysis") -- 27/21/34 uses
# across nawawi/cayni/mazandarani, the only label shared by all three arms, while two
# operations that appeared only in a prose "e.g." list stayed at 0/0/0. The model was
# copying the worked example, same defect class as the copied no_move_reason and the
# copied confidence=0.0. The value is now a placeholder shape (`<...>`), matching every
# other example in this file (no_move_reason, segmentation's confidence sentinel), and
# schemas.reject_placeholder_operation rejects it if a weak model copies it anyway.
#
# docs/2026-09-16-operation-vocabulary.md: that fix stopped the copying, but 135 free
# labels for 135 moves followed regardless -- the prompt still asked for the operation
# "in your own words". `operation` is now one label from a closed set (OPERATIONS,
# schemas.py), checked case-insensitively against it once it reaches Move; the free
# description goes in the new `operation_qualifier` field instead.
_READ_COMMENTARY = """You are studying how a classical hadith commentator reads and \
interprets a hadith.

The source spans below are numbered. Identify interpretive operations and cite only \
their evidence IDs. Never reproduce or rewrite the Arabic text.

Emit one JSON object only in either form:

{{"status":"moves","observations":[{{"operation":"<one label from the closed set \
below>","operation_qualifier":"<what is specific to this instance, in your own \
words>","evidence_id":"...","provenance":{{"author":"compiler_own",\
"quoted_from":[]}},"confidence":0.8}}]}}

{{"status":"no_move","observations":[],\
"no_move_reason":"<what you looked for and why it is absent>"}}

An operation reports what the commentator DOES, not merely what the hadith means. \
Pick exactly one label for "operation" from the closed set below; use "other" only \
when none of the rest fits. Put what is specific to this instance in \
"operation_qualifier", not in "operation".

""" + _OPERATION_LABELS + """

Use only evidence IDs printed below. For transmitted material, set the provenance \
author to "transmitted" and name at least one source in quoted_from, written as \
the commentary writes him (e.g. "al-Qadi Iyad"), never as an identifier such as \
"judge_iyad".

METHOD HYPOTHESIS SO FAR (may be empty on the first bāb):
{method_hypothesis}

SOURCE SPANS:
{evidence_spans}

RETRY FEEDBACK (empty on the first attempt):
{retry_feedback}"""

# Pre-fix text, kept only so the ablation (scripts/run_ablation.sh, conditions A and B)
# can reproduce the defect on purpose. Deps.legacy_example_prompt selects this template
# in hermeneutics.nodes._read_commentary_batch; every normal run uses _READ_COMMENTARY
# above. Do not add this template's name to prompts.REQUIRED-style "must exist" checks
# for new work -- it exists to be turned off.
_READ_COMMENTARY_LEGACY_EXAMPLE = """You are studying how a classical hadith commentator reads and \
interprets a hadith.

The source spans below are numbered. Identify interpretive operations and cite only \
their evidence IDs. Never reproduce or rewrite the Arabic text.

Emit one JSON object only in either form:

{{"status":"moves","observations":[{{"operation":"lexical analysis",\
"evidence_id":"...","provenance":{{"author":"compiler_own",\
"quoted_from":[]}},"confidence":0.8}}]}}

{{"status":"no_move","observations":[],\
"no_move_reason":"<what you looked for and why it is absent>"}}

An operation reports what the commentator DOES, not merely what the hadith means. \
Use only evidence IDs printed below. For transmitted material, set the provenance \
author to "transmitted" and name at least one source in quoted_from, written as \
the commentary writes him (e.g. "al-Qadi Iyad"), never as an identifier such as \
"judge_iyad".

METHOD HYPOTHESIS SO FAR (may be empty on the first bāb):
{method_hypothesis}

SOURCE SPANS:
{evidence_spans}

RETRY FEEDBACK (empty on the first attempt):
{retry_feedback}"""

# docs/2026-09-16-operation-vocabulary.md: every observation's "operation" already
# arrived as one of the closed labels below (_READ_COMMENTARY), so this step must not
# reintroduce free text -- copy the label through, never rename, merge, or summarise it.
_REDUCE_COMMENTARY = """You have several partial readings of one passage from a \
classical hadith commentary.

Select at most {max_observations} representative observations. Prioritise distinct \
interpretive operations, named authorities, and evidence spans. Never invent, alter, \
or replace an evidence ID. Every "operation" below is already one of the closed \
labels here; copy it through exactly as given, never rename, merge, or summarise it. \
Do not reproduce Arabic text.

""" + _OPERATION_LABELS + """

Emit one JSON object only:

{{"status":"moves","observations":[{{"operation":"...",\
"operation_qualifier":"...","evidence_id":"...","provenance":{{"author":"compiler_own",\
"quoted_from":[]}},"confidence":0.8}}]}}

PARTIAL READINGS:
{readings}

RETRY FEEDBACK (empty on the first attempt):
{retry_feedback}"""

_HORIZON_DIFF = """You are a specialist in classical Islamic hermeneutics.

You read a hadith on your own, then read a classical commentator's sharh on it. \
Your task is to make the difference between those two readings explicit.

Emit a JSON object only - no prose, no markdown fences - in one of these shapes:

{{
  "status": "moves",
  "gap": "how your naive reading differed from the commentator's",
  "moves": [
    {{
      "operation": "<one label from the closed set below - never your own wording>",
      "operation_qualifier": "what is specific to this instance, in your own words - a short verb phrase, never a paraphrase of what the hadith means",
      "evidence_id": "an evidence ID from the commentary reading below",
      "arabic_evidence": "a short excerpt, at most {max_quote_chars} characters, copied verbatim from inside that evidence span - the phrase that actually supports this move, not the whole span",
      "closes_gap_by": "what this move resolved",
      "provenance": {{"author": "compiler_own", "quoted_from": []}},
      "confidence": 0.0
    }}
  ],
  "revised_reading": "your reading after the commentator"
}}

or:

{{"status":"no_move","gap":"","moves":[],\
"no_move_reason":"<what you looked for and why it is absent>",\
"revised_reading":"..."}}

Rules:
- Select an "evidence_id" printed below, then copy an "arabic_evidence" excerpt of at \
most {max_quote_chars} characters straight out of that span, character for character. \
Quote only the phrase that supports this move, not the whole span. Never paraphrase, \
alter, translate, or invent the Arabic - every character of "arabic_evidence" must \
occur verbatim in the span you cite.
- "provenance.author" is "compiler_own" when the commentator reasons in his own voice, \
or "transmitted" when he relays a named predecessor. If "transmitted", list the names \
in "quoted_from" as the commentary writes them, never as identifiers; an empty \
list is invalid.
- "confidence" is between 0.0 and 1.0.
- Report interpretive OPERATIONS, not content paraphrases.
- "operation" must be exactly one label from the closed set below, copied character \
for character. Use "other" only when none of the rest fits. Everything specific to \
this instance goes in "operation_qualifier", never in "operation".

""" + _OPERATION_LABELS + """

YOUR NAIVE READING:
{naive}

STRUCTURED COMMENTARY READING:
{commentary_reading}

SOURCE EVIDENCE IDS AND TEXT:
{evidence_spans}

RETRY FEEDBACK (empty on the first attempt):
{retry_feedback}"""

_CONSOLIDATE_KITAB = """You have studied every bāb of one kitāb from a classical sharh \
work and recorded the interpretive moves in each.

Summarise this kitāb's evidence about the commentator's method: the operations used \
most, any sequencing pattern, and the adjudication criteria visible.

The compiler's own voice versus relayed material has already been counted for you and \
is given below. State that figure as given and say what it shows; do not estimate it \
from the records and do not substitute a figure of your own.

Write in English, under 400 words.

KITAB: {kitab}

PROVENANCE (counted, authoritative): {provenance}

RECORDS:
{notes}"""

_CONSOLIDATE_GLOBAL = """Below are per-kitāb summaries of a classical commentator's \
interpretive practice, covering the whole work.

Consolidate them into a structured account of {commentator}'s hermeneutical method as \
demonstrated in practice, not as theorised. Organise as:

1. Primary interpretive operations - most frequent, with description and example
2. Secondary operations - used in specific circumstances
3. Sequencing patterns - is there a consistent order of operations?
4. Adjudication criteria - how are competing interpretations settled?
5. Compiler's own reasoning vs transmitted material - what the given ratio shows
6. Limits observed - what the commentator does not do

Section 5 has been counted for you over every recorded move and is given below. Use \
that figure verbatim; any number you state there must match it. Do not estimate the \
ratio from the summaries, which do not contain the counts.

Write in English. Be precise and scholarly.
{retry_feedback}
PROVENANCE (counted over all moves, authoritative): {provenance}

KITAB SUMMARIES:
{summaries}"""

_SEGMENTATION = """Below is the raw text of one bāb from a printed hadith commentary. \
It contains one hadith matn (with its isnād) somewhere inside it, and the \
commentator's sharh around it.

The matn does NOT always come first. Many bābs open with the commentator glossing \
the chapter heading — text beginning أي, يعني or قوله — and only then quote the \
hadith, typically at an isnād formula such as حدثنا, حدثني or أخبرنا. Others do \
begin with the matn itself.

Report where the matn STARTS and where it ENDS, as character offsets into the text \
below.

matn_start_char is 0 when the bāb opens with the matn. Both offsets are less than \
the text length, and matn_start_char is less than matn_end_char.
Emit a JSON object only, for example: \
{{"matn_start_char": 0, "matn_end_char": 1234, "confidence": 0.9}}

If matn and sharh are interleaved rather than contiguous, bound the first matn \
passage and set confidence below 0.5.

TEXT:
{text}

RETRY FEEDBACK (empty on the first attempt):
{retry_feedback}"""

TEMPLATES: dict[str, str] = {
    "naive_reading": _NAIVE,
    "read_commentary": _READ_COMMENTARY,
    "read_commentary_legacy_example": _READ_COMMENTARY_LEGACY_EXAMPLE,
    "reduce_commentary": _REDUCE_COMMENTARY,
    "horizon_diff": _HORIZON_DIFF,
    "consolidate_kitab": _CONSOLIDATE_KITAB,
    "consolidate_global": _CONSOLIDATE_GLOBAL,
    "segmentation": _SEGMENTATION,
}


def prompt_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        for name, text in sorted(TEMPLATES.items())
    }


def render(name: str, **fields) -> str:
    """Render a template. Raises KeyError on an unknown template or a missing field."""
    return TEMPLATES[name].format(**fields)
