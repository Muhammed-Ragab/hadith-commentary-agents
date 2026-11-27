"""Graph state schemas. Spec §3.3.

Two schemas, not one: the study and inquiry graphs share no fields.

Reducers are mandatory on accumulating channels. LangGraph's default channel
overwrites on every write, so an append-only list without Annotated[..., operator.add]
silently keeps only the last write. test_state.py guards this.
"""
import hashlib
import operator
from typing import Annotated, Any, Literal, TypedDict


class StudyState(TypedDict, total=False):
    """Per-commentator study graph. Spec §3.1, §3.3."""

    commentator: str
    cursor: int                       # index into the bāb list
    current_bab: dict | None
    current_kitab: str
    naive_reading: str
    commentary_text: str
    diff_record: dict | None
    control_mode: bool
    matn_only_mode: bool
    method_hypothesis: str
    current_bab_ref: str
    commentary_source_ref: str
    commentary_reading: dict | None
    commentary_error: str
    evidence_catalog: dict[str, str]
    n_chunks: int
    notes_path: str
    run_manifest: dict
    re_read_cursor: int
    method_file: str

    # Append-only channels - reducers are not optional here.
    notes: Annotated[list[dict], operator.add]
    reviewed_kitabs: Annotated[list[str], operator.add]
    re_read_queue: Annotated[list[int], operator.add]
    intervention_log: Annotated[list[dict], operator.add]


class InquiryState(TypedDict, total=False):
    """Three-stage inquiry graph. Spec §3.2, §3.3."""

    condition: str                    # a key of hermeneutics.conditions.CONDITIONS
    question: str
    commentator: str
    # muslim belongs here: al-Nawawi's Minhaj commentates Muslim, so an inquiry under
    # his method file reads Muslim, not Bukhari.
    corpus: Literal["bukhari", "kafi", "muslim"]
    network_metrics: dict[str, Any]  # auditability: the topology the model saw; nodes read deps.metrics
    framework: str                    # the method file text, empty under Condition B
    isnad_output: str
    matn_output: str
    synthesis_output: str
    run_manifest: dict
    # Written by the canonicalise node, before isnad. `question` above stays the
    # asker's original words untouched; every stage after canonicalise searches and
    # is prompted with question_ar/search_terms instead, so retrieval works the same
    # way whatever language or register the question arrived in. Empty under
    # no_sources, which never canonicalises (see conditions.CONDITIONS).
    question_ar: str
    search_terms: list[str]
    question_lang: str

    # Append-only channels - reducers are not optional here. Each stage writes its own
    # sources and calls; without a reducer the matn stage would erase the isnad stage's
    # and the synthesis gate would verify against a fraction of what was actually read.
    intervention_log: Annotated[list[dict], operator.add]
    sources: Annotated[list[dict], operator.add]
    tool_calls_log: Annotated[list[dict], operator.add]
    grounding_log: Annotated[list[dict], operator.add]


def hypothesis_hash(hypothesis: str) -> str:
    """Hash of the method hypothesis in force at a call. Spec §3.4 replayability."""
    return hashlib.sha256(hypothesis.encode("utf-8")).hexdigest()[:16]


def initial_study_state(commentator: str, manifest: dict, notes_path: str,
                        control_mode: bool = False,
                        matn_only_mode: bool = False) -> StudyState:
    return {
        "commentator": commentator,
        "cursor": 0,
        "current_bab": None,
        "current_kitab": "",
        "current_bab_ref": "",
        "commentary_source_ref": "",
        "naive_reading": "",
        "commentary_text": "",
        "commentary_reading": None,
        "commentary_error": "",
        "evidence_catalog": {},
        "diff_record": None,
        "control_mode": control_mode,
        "matn_only_mode": matn_only_mode,
        "method_hypothesis": "",
        "notes_path": notes_path,
        "run_manifest": manifest,
        "notes": [],
        "reviewed_kitabs": [],
        "re_read_queue": [],
        "intervention_log": [],
        "re_read_cursor": 0,
        "method_file": "",
    }


def initial_inquiry_state(
    condition: str,
    question: str,
    commentator: str,
    corpus: str,
    manifest: dict,
    network_metrics: dict | None = None,
    framework: str = "",
) -> InquiryState:
    return {
        "condition": condition,
        "question": question,
        "commentator": commentator,
        "corpus": corpus,
        "run_manifest": manifest,
        "network_metrics": network_metrics or {},
        "framework": framework,
        "isnad_output": "",
        "matn_output": "",
        "synthesis_output": "",
        "intervention_log": [],
        "sources": [],
        "tool_calls_log": [],
        "grounding_log": [],
    }
