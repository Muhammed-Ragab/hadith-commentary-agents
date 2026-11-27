"""Strict output schemas. Spec §4, defect #8."""
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# The prompts print a no_move object as an example of the required JSON shape. Weak
# models copy its reason string verbatim instead of answering: 8 of the 9 no_move
# records written before this check said exactly "No identifiable interpretive
# operation." across three different models, including bab nawawi/0, which the same
# model had read as 5 real moves in job 15688358. A copied escape hatch is not a
# finding, so reject it and let the caller's existing retry ask again. Same defect as
# the copied confidence=0.0 in run_segmentation.reject_reason.
COPIED_NO_MOVE_REASONS = {
    "no identifiable interpretive operation.",
    "no identifiable interpretive operation",
    "...",
}


def _is_example_placeholder(value: str) -> bool:
    """Every prompt's fill-in-the-blank example is shaped `<like this>`."""
    stripped = value.strip()
    return stripped.startswith("<") and stripped.endswith(">")


def reject_no_move_reason(reason: str) -> str | None:
    """Why this no_move_reason is a copied placeholder, or None if it is an answer."""
    if not reason.strip():
        return "status='no_move' requires no_move_reason"
    stripped = reason.strip()
    if _is_example_placeholder(stripped):
        return "no_move_reason is the example placeholder, not an answer."
    if stripped.lower() in COPIED_NO_MOVE_REASONS:
        return (
            "no_move_reason repeats the example text. State what you actually looked "
            "for in this commentary and why no interpretive operation was present."
        )
    return None


# Saturation-curve run 15789363: `operation` used to be printed in _READ_COMMENTARY's
# copyable JSON example as a real, answerable value ("lexical analysis") rather than a
# placeholder shape. Every arm converged on exactly that string (27/21/34 uses, the
# only operation shared by all three arms) while two operations that appeared only in
# prose stayed at 0/0/0. The prompt now prints `<the interpretive operation, in your
# own words>` instead (hermeneutics/prompts.py); this is the schema half of that fix,
# same defect class and same precedent as reject_no_move_reason above -- prose alone
# ("in your own words") has failed three times in this project, so a copyable-shaped
# placeholder is rejected here too and left to the caller's existing retry.
def reject_placeholder_operation(operation: str) -> str | None:
    """Why this operation is the prompt's copied placeholder, or None if it is an answer."""
    if _is_example_placeholder(operation):
        return "operation is the example placeholder, not an answer."
    return None


# docs/2026-09-16-operation-vocabulary.md: the ablation measured sprawl 1.00 -- 135
# free-text operation labels for 135 moves, because the prompt asked for the operation
# "in your own words". This is the closed set that replaces it (§"The set - 17 labels
# plus an escape hatch"); definitions copied verbatim from that table. `tawil_imposition`
# drops the doc's hamza (taʾwil) so the key is a plain ASCII identifier -- the docs file
# writes it with the diacritic in prose, this is the code spelling.
OPERATIONS: dict[str, str] = {
    "variant_adjudication": (
        "chooses among manuscript or recitation variants; morphology, ellipsis and "
        "syntax in service of fixing a reading"
    ),
    "lexical_gloss": "defines a term by its root or by lexicographic testimony",
    "referent_specification": "fixes a vague or opaque expression to a concrete referent",
    "sense_bifurcation": "splits one term or payload into two or more distinct senses",
    "scope_restriction": "limits a general text by addressee, condition, or exclusion",
    "madhhab_adjudication": (
        "tarjīḥ among named positions; declaring one al-ṣaḥīḥ"
    ),
    "counterargument_refutation": (
        "declares an opposing argument invalid or its reliance false"
    ),
    "tawil_imposition": "mandates a non-literal reading to protect a doctrinal commitment",
    "speech_act_reclassification": "re-describes an utterance's illocutionary force",
    "ruling_extraction": "converts a narrative report into an explicit legal ruling",
    "scope_extension": (
        "generalizes beyond the immediate case to a wider class or question"
    ),
    "principle_extraction": (
        "derives an ethical or uṣūlī principle not present on the surface"
    ),
    "authority_citation": "grounds a claim in ijmāʿ or a named authority",
    "narration_harmonization": (
        "reconciles conflicting or variant narrations into one account"
    ),
    "intent_exoneration": "defends a companion's intent against an accusatory reading",
    "occasion_dating": (
        "limits or explains the text by when it was said - early Islam, before an "
        "obligation, naskh"
    ),
    "literal_affirmation": (
        "affirms a literal reading as doctrine against a metaphorical one"
    ),
    "other": "none of the above; the free description carries it",
}


def reject_unknown_operation(operation: str) -> str | None:
    """Why this operation isn't in the closed vocabulary, or None if it matches.

    Matches case- and whitespace-insensitively (docs/2026-09-16-operation-vocabulary.md
    doesn't mandate a casing convention and weak models won't reliably hit one), so the
    caller normalizes the label from OPERATIONS itself rather than trusting the input.
    """
    if operation.strip().lower() in OPERATIONS:
        return None
    examples = ", ".join(list(OPERATIONS)[:4]) + ", ... or 'other'"
    return (
        f"operation {operation!r} is not in the closed vocabulary. Pick exactly one "
        f"label, e.g. {examples}; put the specific description in operation_qualifier."
    )


# Job 15789363 lost one bāb in each of two arms to a malformed evidence_id, twice over,
# because greedy decoding repeats a format error verbatim. cayni/10 returned each id
# wrapped in the brackets the prompt prints around it ("[cayni/10:c000:s003]");
# mazandarani/8 returned ordinals ("1", "7"). The bracket case is unambiguous, so
# recover it here rather than burning a retry; the ordinal case is not recoverable and
# is left to the caller's retry, which now names the format and a real id.
_BRACKETED_ID = re.compile(r"^[\[\('\"]+(?P<id>[^\[\]\(\),'\"]+?)[\]\)'\"]+$")


def normalize_evidence_id(value):
    """Strip the brackets the prompt prints around an id. Reject a collapsed list."""
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(
                f"evidence_id must be one id, not a list of {len(value)}; "
                "emit a separate move for each span you cite"
            )
        value = value[0]
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if "," in stripped:
        raise ValueError(
            "evidence_id must be one id, not a comma-separated list; "
            "emit a separate move for each span you cite"
        )
    match = _BRACKETED_ID.match(stripped)
    return match.group("id").strip() if match else stripped


# A bare lower-case slug: two or more words joined by underscores, no capitals and no
# spaces. Real names carry spaces, capitals or Arabic; "judge_iyad" carries none.
_SLUG = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)+")


class Provenance(BaseModel):
    """Fixes validity threat V2: is this the compiler reasoning, or quoting a predecessor?"""

    author: Literal["compiler_own", "transmitted"]
    quoted_from: list[str] = Field(default_factory=list)

    @field_validator("quoted_from", mode="before")
    @classmethod
    def names_not_identifiers(cls, value):
        """A predecessor is named as the commentary names him, not slugged.

        Nawawī's 15789363 notes carry both `al-Qadi 'Iyad` and `judge_iyad` for one
        man, because the field had no shape guidance at all. Two labels for one
        predecessor inflate any distinct-source count, and the roster is now fed back
        into the consolidation prompt (B1), where a slug would be presented to the
        model as a scholar.
        """
        if not isinstance(value, list):
            return value
        cleaned = [s.strip() for s in value if isinstance(s, str) and s.strip()]
        for name in cleaned:
            if _SLUG.fullmatch(name):
                raise ValueError(
                    f"quoted_from {name!r} is an identifier, not a name. Write the "
                    "predecessor as the commentary writes him, in Arabic or "
                    "transliteration (e.g. 'al-Qadi Iyad')."
                )
        return cleaned

    @model_validator(mode="after")
    def transmitted_needs_a_source(self):
        if self.author == "transmitted" and not self.quoted_from:
            raise ValueError(
                "provenance.author='transmitted' requires at least one quoted_from name"
            )
        return self


# The model names the operation and then, on later babs, appends what it applied to:
# "reconciliation of apparent contradiction (ẓāhir vs. ijmāʿ)". Job cayni2 returned
# 12 moves over 2 babs that scored 12 distinct operations and a sprawl of 1.00 -- yet
# they were six operations each named twice, once bare and once qualified. Sprawl is
# the project's measure of an uncontrolled vocabulary (compare_methods, spec open item
# #2), and it cannot mean that while a parenthesis counts as a new category.
#
# The qualifier is not noise, though: it says WHICH contradiction, which is what a
# footnote cites. So it is separated here rather than stripped in the consumer, and
# both halves reach the notes file.
_OPERATION_QUALIFIER = re.compile(r"\s*[\(\[](?P<qualifier>[^()\[\]]*)[\)\]]\s*$")


def split_operation(value: str) -> tuple[str, str]:
    """Separate an operation label from its trailing scope note."""
    text = (value or "").strip()
    match = _OPERATION_QUALIFIER.search(text)
    if not match:
        return text, ""
    bare = text[: match.start()].strip()
    # A label that is nothing but a parenthetical is the whole answer, not a qualifier.
    return (bare, match.group("qualifier").strip()) if bare else (text, "")


class Move(BaseModel):
    operation: str = Field(min_length=1)
    operation_qualifier: str = ""
    arabic_evidence: str = Field(min_length=1)
    # B5: arabic_evidence is now the model's own short quotation (verified in
    # nodes.horizon_diff), not the whole cited span. The span stays here under its
    # own key so downstream context (e.g. the scholar-review sheet) isn't lost.
    evidence_chunk: str = ""
    evidence_id: str | None = None
    closes_gap_by: str = Field(min_length=1)
    provenance: Provenance
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("operation")
    @classmethod
    def _reject_placeholder_operation(cls, value):
        rejection = reject_placeholder_operation(value)
        if rejection:
            raise ValueError(rejection)
        return value

    @model_validator(mode="after")
    def _split_the_operation_qualifier(self):
        if not self.operation_qualifier:
            self.operation, self.operation_qualifier = split_operation(self.operation)
        # Closed vocabulary check runs after the split, so "lexical_gloss (root
        # meaning)" is judged on "lexical_gloss", not on the whole string.
        rejection = reject_unknown_operation(self.operation)
        if rejection:
            raise ValueError(rejection)
        self.operation = self.operation.strip().lower()
        return self


# ponytail: not split here. An observation is an intermediate that is serialised back
# into the reduce and diff prompts, so an extra field on it costs tokens on every
# chunk; compare_methods reads Move, and that is where the split has to hold.
class CommentaryObservation(BaseModel):
    operation: str = Field(min_length=1)
    # Not folded through split_operation or checked against OPERATIONS here (see the
    # ponytail note above CommentaryObservation): this is the map-step intermediate,
    # reduced and re-serialised before it ever reaches a Move. The closed-set label is
    # still asked for in the prompt (prompts._READ_COMMENTARY) so the model forms the
    # habit early, but only Move.operation is validated against it.
    operation_qualifier: str = ""
    evidence_id: str = Field(min_length=1)
    provenance: Provenance
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("operation")
    @classmethod
    def _reject_placeholder_operation(cls, value):
        rejection = reject_placeholder_operation(value)
        if rejection:
            raise ValueError(rejection)
        return value

    _clean_evidence_id = field_validator("evidence_id", mode="before")(
        lambda cls, v: normalize_evidence_id(v)
    )


class CommentaryReading(BaseModel):
    status: Literal["moves", "no_move"]
    observations: list[CommentaryObservation]
    no_move_reason: str = ""

    @model_validator(mode="after")
    def status_matches_observations(self):
        if self.status == "moves" and not self.observations:
            raise ValueError("status='moves' requires at least one observation")
        if self.status == "no_move":
            if self.observations:
                raise ValueError("status='no_move' requires no observations")
            rejection = reject_no_move_reason(self.no_move_reason)
            if rejection:
                raise ValueError(rejection)
        return self


class ReferencedMove(BaseModel):
    operation: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    # B5 (handoff.md §8.15 correction): the old code substituted the whole cited span
    # for arabic_evidence, so is_grounded was true by construction - 88/88, 76/76,
    # 84/84 on run 15789363. The model must now pick the quotation itself; nodes.py
    # verifies it actually occurs in catalog[evidence_id] and is short (see
    # nodes.MAX_QUOTE_CHARS) before it is trusted.
    arabic_evidence: str = Field(min_length=1)
    closes_gap_by: str = Field(min_length=1)
    provenance: Provenance
    confidence: float = Field(ge=0.0, le=1.0)

    _clean_evidence_id = field_validator("evidence_id", mode="before")(
        lambda cls, v: normalize_evidence_id(v)
    )


class DiffDraft(BaseModel):
    status: Literal["moves", "no_move"]
    gap: str
    moves: list[ReferencedMove]
    no_move_reason: str = ""
    revised_reading: str

    @model_validator(mode="after")
    def status_matches_moves(self):
        if self.status == "moves" and not self.moves:
            raise ValueError("status='moves' requires at least one move")
        if self.status == "no_move":
            if self.moves:
                raise ValueError("status='no_move' requires an empty moves list")
            rejection = reject_no_move_reason(self.no_move_reason)
            if rejection:
                raise ValueError(rejection)
        return self


class DiffRecord(BaseModel):
    """One unit's horizon diff. Written append-only to {name}_notes.jsonl."""

    bab_ref: str
    kitab: str
    status: Literal["moves", "no_move"] = "moves"
    gap: str
    moves: list[Move]
    no_move_reason: str = ""
    revised_reading: str
    hypothesis_hash: str  # spec §3.4 - makes the trajectory replayable
    manifest_id: str
    control: bool = False

    @model_validator(mode="after")
    def status_matches_moves(self):
        if self.status == "moves" and not self.moves:
            raise ValueError("status='moves' requires at least one move")
        if self.status == "no_move":
            if self.moves:
                raise ValueError("status='no_move' requires an empty moves list")
            rejection = reject_no_move_reason(self.no_move_reason)
            if rejection:
                raise ValueError(rejection)
        return self


class SegRecord(BaseModel):
    """Where the matn sits inside a unit. Spec §3.6.

    matn_start_char defaults to 0, which is both the common case and what every
    segmentation file written before ʿAynī forced the question already means.
    """

    bab_ref: str
    matn_start_char: int = Field(ge=0, default=0)
    matn_end_char: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)


class Intervention(BaseModel):
    """Citable human-in-the-loop data + agent re-read decisions. Spec §3.4."""

    kind: Literal["approve", "edit", "flag_reread", "agent_reread"]
    kitab: str
    payload: dict
    ts: str
