"""B2 — al-Kāfī marks its own matn, so the Shia arm should not be guessing.

handoff.md §10.5 noted that al-Māzandarānī's edition separates matn from commentary
inline with `* الأصل:` / `* الشرح:` and that this was "not wired up". §8.15 then found
the cost of not wiring it: one sampled move's `arabic_evidence` ended with an isnād
fragment (`الأصل: 2 - محمد بن يحيى، عن محمد بن الحس`) inside what should be sharh-only
text, and the arm collapsed to 2 distinct operation-sets across 24 bābs.

Measured over the real corpus (466 units), the markers are messier than §10.5 implies:

    both markers present, loose form   350 / 466   (75.1%)
    ... and in matn-first order        340         (73.0%)
    starred form only                  325         (69.7%)

So this is a fast path, not a replacement: 73% of the arm gets an exact boundary at
zero model calls, and the rest still goes to the model. Nawawī and ʿAynī carry no such
markers at all, so every function here is a no-op for them.
"""
from hermeneutics.markers import matn_span, sharh_only

# The two shapes that occur. The starred form is the common one; `# الأصل:` without a
# star appears too, and requiring the star costs 25 real units.
STARRED = "عنوان الباب * الأصل: # 1 - محمد بن يحيى، عن أحمد * الشرح: قوله عليه السلام"
BARE = "عنوان الباب\n# الأصل: # 1 - محمد بن يحيى\n# الشرح: أي المراد به"


def test_the_starred_form_bounds_the_matn():
    start, end = matn_span(STARRED)
    assert "محمد بن يحيى" in STARRED[start:end]
    assert "قوله عليه السلام" not in STARRED[start:end]


def test_the_bare_form_bounds_the_matn_too():
    """Requiring the `*` costs 25 of 350 real units for nothing."""
    start, end = matn_span(BARE)
    assert "محمد بن يحيى" in BARE[start:end]


def test_a_text_with_no_markers_yields_no_span():
    """Nawawī and ʿAynī carry none; they must fall through to the model."""
    assert matn_span("النووي رحمه الله قال في شرحه") is None


def test_sharh_before_asl_yields_no_span():
    """10 real units open with a long gloss on the bāb title before any matn.

    Same shape as ʿAynī's heading gloss. Guessing a span here would reproduce §8.17's
    `cayni/0` failure on the other arm, so it falls through to the model instead.
    """
    text = "* الشرح: الجبر في اللغة الإكراه * الأصل: # 1 - علي بن محمد"
    assert matn_span(text) is None


def test_an_unterminated_matn_yields_no_span():
    assert matn_span("* الأصل: # 1 - محمد بن يحيى، عن أحمد بن محمد") is None


# ------------------------------------------------- the leak §8.15 found in the notes

def test_embedded_matn_blocks_are_removed_from_the_commentary():
    """The defect verbatim: a later `الأصل:` block inside sharh-only text."""
    text = ("* الشرح: قوله عليه السلام أي المراد "
            "* الأصل: # 2 - محمد بن يحيى، عن محمد بن الحسن "
            "* الشرح: وقوله هذا بيان للحكم")
    out = sharh_only(text)
    assert "محمد بن الحسن" not in out
    assert "قوله عليه السلام" in out and "بيان للحكم" in out


def test_commentary_before_the_first_marker_is_kept():
    """`select_next_bab` slices from the matn end, so the text opens mid-sharh."""
    text = "تتمة الشرح السابق * الأصل: # 2 - محمد بن يحيى * الشرح: ثم قال"
    out = sharh_only(text)
    assert "تتمة الشرح السابق" in out and "ثم قال" in out
    assert "محمد بن يحيى" not in out


def test_text_without_markers_passes_through_unchanged():
    """No-op for Nawawī and ʿAynī, which is how this stays one code path."""
    text = "قال النووي رحمه الله في شرح هذا الحديث"
    assert sharh_only(text) == text


def test_kept_segments_stay_verbatim_substrings_of_the_source():
    """Grounding checks evidence against the whole original bāb (validate_matched.py:92).

    Joining kept segments is only safe while each one is contiguous in the original —
    a quote spanning a join is two non-adjacent passages spliced, and failing it is
    correct.
    """
    text = "* الشرح: ألف باء * الأصل: # 2 - جيم * الشرح: دال هاء"
    for segment in sharh_only(text).split("\n"):
        assert segment in text
