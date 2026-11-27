import pytest

from hermeneutics.prompts import (TEMPLATES, _READ_COMMENTARY, prompt_hashes,
                                  render)
from hermeneutics.schemas import OPERATIONS

REQUIRED = {"naive_reading", "read_commentary", "horizon_diff",
            "consolidate_kitab", "consolidate_global", "segmentation"}


def test_every_required_template_exists():
    assert REQUIRED <= set(TEMPLATES)


def test_prompt_hashes_cover_every_template():
    assert set(prompt_hashes()) == set(TEMPLATES)


def test_hashes_are_stable_across_calls():
    assert prompt_hashes() == prompt_hashes()


def test_render_substitutes_named_fields():
    out = render("naive_reading", matn="THE MATN TEXT")
    assert "THE MATN TEXT" in out
    assert "{matn}" not in out


def test_render_rejects_an_unknown_template():
    with pytest.raises(KeyError):
        render("no_such_template")


def test_render_rejects_a_missing_field():
    with pytest.raises(KeyError):
        render("naive_reading")
    with pytest.raises(KeyError):
        render("horizon_diff")  # JSON template also raises on missing fields


def test_horizon_diff_template_demands_json_only():
    t = TEMPLATES["horizon_diff"]
    assert "JSON" in t
    assert "evidence_id" in t
    assert "compiler_own" in t and "transmitted" in t
    assert "no prose" in t or "no markdown" in t.lower()


def test_naive_reading_never_mentions_commentary_or_method():
    """The naive horizon is only naive if nothing leaks into it (validity threat V1)."""
    t = TEMPLATES["naive_reading"].lower()
    assert "{commentary}" not in t
    assert "{method_hypothesis}" not in t


def test_read_commentary_example_operation_is_a_placeholder_not_an_answer():
    """Mechanism 1 (saturation_curve, run 15789363): the copyable JSON example used to
    print a live, answerable operation value ('lexical analysis'), and every arm copied
    it verbatim (27/21/34 uses, the only label shared by all three arms). The fixed
    default template's example must not contain that or any other real-looking label."""
    t = TEMPLATES["read_commentary"]
    assert '"operation":"<' in t
    assert "lexical analysis" not in t


def test_read_commentary_legacy_example_preserves_the_pre_fix_text():
    """Ablation condition A/B (Deps.legacy_example_prompt) reproduces the defect on
    purpose, so this template must still print the answerable example."""
    t = TEMPLATES["read_commentary_legacy_example"]
    assert '"operation":"lexical analysis"' in t


def test_no_template_uses_bare_braces_that_would_crash_formatting():
    """Defect #6: literal JSON braces in a template crashed str.format."""
    for name in TEMPLATES:
        fields = {"matn", "commentary", "method_hypothesis", "naive", "commentary_reading",
                  "kitab", "notes", "summaries", "diff_json", "text", "commentator",
                  "readings", "retry_feedback", "evidence_spans", "max_observations",
                  "provenance", "max_quote_chars"}
        try:
            TEMPLATES[name].format(**{f: "X" for f in fields})
        except (IndexError, KeyError) as exc:
            pytest.fail(f"template {name!r} has unescaped braces: {exc}")


def test_no_prompt_names_a_real_operation_the_model_could_adopt():
    """The vocabulary-collapse fix is only complete if NO prompt supplies a usable label.

    `lexical analysis` was printed in read_commentary's copyable JSON example and used
    27/21/34 times across the three arms of 15789363, while `takhsis` and `harmonization`
    -- named only in horizon_diff's prose `e.g.` list -- were used ZERO times. So prose
    pulls far more weakly than a copyable value.

    But horizon_diff is the call that actually writes `Move.operation` into the notes,
    and it named `lexical analysis` too. Leaving it there would seed the very label under
    investigation in the ablation's fixed condition, so a flat curve could be blamed on
    the leftover seed rather than on the method. The templates carry no real label now.

    The live-template check is deliberately narrow: `read_commentary_legacy_example`
    exists precisely to reproduce the old prompt for ablation condition A/B and MUST
    keep its label.
    """
    live = {name: text for name, text in TEMPLATES.items()
            if not name.endswith("_legacy_example")}
    for label in ("lexical analysis", "takhsis"):
        offenders = sorted(name for name, text in live.items() if label in text)
        assert not offenders, f"{label!r} is still printed in {offenders}"

    # docs/2026-09-16-operation-vocabulary.md: read_commentary and reduce_commentary now
    # print the closed set ON PURPOSE -- that is the whole fix, and "harmonization" is a
    # substring of narration_harmonization. What must stay absent is a usable label in the
    # COPYABLE JSON example, which is the thing 15789363 showed the model actually copies.
    example = _READ_COMMENTARY.split("SOURCE SPANS:")[0].split("An operation reports")[0]
    for key in OPERATIONS:
        assert f'"operation":"{key}"' not in example, (
            f"{key!r} is a copyable example value in read_commentary")
    assert '"operation":"<' in example, "read_commentary lost its placeholder shape"
