"""The condition table must be importable from the package.

run_inquiry.py is the abstract's two settings made runnable. It cannot reach them
through a sibling directory the cluster does not receive.
"""
import pytest


def test_all_four_conditions_are_importable_from_the_package():
    from hermeneutics.conditions import CONDITIONS

    assert set(CONDITIONS) == {"no_sources", "passages", "agent", "agent_method"}


def test_agent_method_is_the_abstracts_first_setting():
    from hermeneutics.conditions import CONDITIONS

    spec = CONDITIONS["agent_method"]
    assert spec["tools"] is True
    assert spec["needs"] == "method"


def test_passages_is_the_abstracts_second_setting():
    from hermeneutics.conditions import CONDITIONS

    spec = CONDITIONS["passages"]
    assert spec["tools"] is False
    assert spec["mode"] == "none"
    assert spec["needs"] == "passages"


def test_a_passages_run_without_passages_raises_rather_than_degrading():
    from hermeneutics.conditions import build_messages

    with pytest.raises(ValueError, match="passages"):
        build_messages("passages", "q", passages=None)
