"""The reducer test is the point of this file. Everything else is scaffolding."""
import operator
from typing import Annotated, get_args, get_origin, get_type_hints

from langgraph.graph import END, START, StateGraph

from hermeneutics.state import InquiryState, StudyState, hypothesis_hash, initial_study_state


def _reducer_of(state_cls, field: str):
    """Return the reducer function attached to an Annotated field, or None."""
    hint = get_type_hints(state_cls, include_extras=True)[field]
    if get_origin(hint) is not Annotated:
        return None
    return get_args(hint)[1]


def test_append_only_study_fields_have_add_reducers():
    for field in ("notes", "re_read_queue", "intervention_log"):
        assert _reducer_of(StudyState, field) is operator.add, (
            f"StudyState.{field} has no operator.add reducer - LangGraph will "
            "overwrite it and the run will silently lose records (spec §3.3)"
        )


def test_scalar_study_fields_are_last_write_wins():
    for field in ("cursor", "method_hypothesis", "control_mode"):
        assert _reducer_of(StudyState, field) is None


def test_inquiry_intervention_log_has_a_reducer():
    assert _reducer_of(InquiryState, "intervention_log") is operator.add


def test_inquiry_state_has_every_spec_field():
    hints = get_type_hints(InquiryState, include_extras=True)
    for field in ("condition", "question", "commentator", "corpus", "network_metrics",
                  "framework", "isnad_output", "matn_output", "synthesis_output",
                  "intervention_log", "run_manifest"):
        assert field in hints, f"InquiryState missing {field} (spec §3.3)"


def test_two_nodes_appending_in_one_superstep_produce_two_records():
    """The actual failure mode: fan-out writes clobbering each other."""

    def node_a(state):
        return {"notes": [{"bab_ref": "nawawi/1"}]}

    def node_b(state):
        return {"notes": [{"bab_ref": "nawawi/2"}]}

    g = StateGraph(StudyState)
    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge("a", END)
    g.add_edge("b", END)

    result = g.compile().invoke({"notes": [], "re_read_queue": [], "intervention_log": []})
    refs = sorted(n["bab_ref"] for n in result["notes"])
    assert refs == ["nawawi/1", "nawawi/2"], (
        f"expected both records, got {result['notes']} - reducer is missing or wrong"
    )


def test_hypothesis_hash_is_stable_and_short():
    assert hypothesis_hash("abc") == hypothesis_hash("abc")
    assert hypothesis_hash("abc") != hypothesis_hash("abd")
    assert len(hypothesis_hash("abc")) == 16


def test_hypothesis_hash_of_empty_is_defined():
    assert hypothesis_hash("") == hypothesis_hash("")


def test_initial_study_state_has_empty_evidence_transport_fields():
    state = initial_study_state("nawawi", {}, "notes.jsonl")
    assert state["commentary_reading"] is None
    assert state["commentary_error"] == ""
    assert state["evidence_catalog"] == {}


def test_inquiry_accumulating_channels_have_reducers():
    """LangGraph's default channel overwrites. A stage's sources would erase the
    previous stage's, and the synthesis gate would check against the last stage only."""
    import operator
    import typing

    from hermeneutics.state import InquiryState

    hints = typing.get_type_hints(InquiryState, include_extras=True)
    for channel in ("sources", "tool_calls_log", "grounding_log", "intervention_log"):
        meta = getattr(hints[channel], "__metadata__", ())
        assert operator.add in meta, f"{channel} has no reducer and will overwrite"


def test_initial_inquiry_state_opens_every_channel():
    from hermeneutics.state import initial_inquiry_state

    state = initial_inquiry_state("agent_method", "q", "nawawi", "muslim", {"id": "m"})
    for channel in ("sources", "tool_calls_log", "grounding_log", "intervention_log"):
        assert state[channel] == []
