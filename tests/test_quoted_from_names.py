"""B2 (folded in) — `quoted_from` is a name, not an identifier.

Nawawī's 15789363 notes record both `al-Qadi 'Iyad` and `judge_iyad` for one person:
the field is free text with no shape guidance, so the model sometimes emits a
snake_case slug instead of the name as the commentary writes it. Two labels for one
predecessor inflates any distinct-source count, and the roster now goes into the
consolidation prompt (B1), so a slug would be handed back to the model as though it
were a scholar.
"""
import pytest
from pydantic import ValidationError

from hermeneutics.nodes import transmitted_sources
from hermeneutics.schemas import Provenance


def test_a_real_name_is_accepted():
    p = Provenance(author="transmitted", quoted_from=["al-Qadi 'Iyad"])
    assert p.quoted_from == ["al-Qadi 'Iyad"]


def test_an_arabic_name_is_accepted():
    assert Provenance(author="transmitted", quoted_from=["القاضي عياض"]).quoted_from


def test_a_snake_case_slug_is_rejected():
    """The observed defect verbatim."""
    with pytest.raises(ValidationError, match="judge_iyad"):
        Provenance(author="transmitted", quoted_from=["judge_iyad"])


def test_surrounding_whitespace_is_stripped():
    p = Provenance(author="transmitted", quoted_from=["  Ibn Batal  "])
    assert p.quoted_from == ["Ibn Batal"]


def test_a_blank_entry_is_dropped_rather_than_kept():
    p = Provenance(author="transmitted", quoted_from=["Ibn Batal", "   "])
    assert p.quoted_from == ["Ibn Batal"]


def test_dropping_every_entry_still_fails_the_transmitted_rule():
    """An empty roster after cleaning must not pass as a named source."""
    with pytest.raises(ValidationError, match="quoted_from"):
        Provenance(author="transmitted", quoted_from=["  "])


def test_compiler_own_needs_no_names():
    assert Provenance(author="compiler_own").quoted_from == []


def test_a_multi_word_name_with_an_underscore_inside_a_title_is_not_a_slug():
    """Only a bare lower-case slug is refused; real names carry spaces or capitals."""
    assert Provenance(author="transmitted",
                      quoted_from=["Abu Hatam's Kitab Wusuf al-Iman"]).quoted_from


# ---------------------------------- one authority, several transliterations (2026-09-19)

def _rec(*names):
    return {"moves": [{"provenance": {"author": "transmitted", "quoted_from": list(names)}}]}


def test_transliterations_of_one_name_collapse_to_one_entry():
    """matched_qwen3_20260917's Nawawī roster listed Qāḍī ʿIyāḍ three ways."""
    out = transmitted_sources([_rec("al-Qadi Iyad", "al-Qāḍī Iyāḍ", "al-Qāḍī ʿIyāḍ")])
    assert out == ["al-Qāḍī ʿIyāḍ"], out


def test_the_fullest_spelling_is_the_one_reported():
    """Diacritics first, then length: the scholarly form, not whichever came first."""
    assert transmitted_sources([_rec("malik", "Mālik")]) == ["Mālik"]


def test_case_and_punctuation_alone_do_not_make_a_second_authority():
    assert transmitted_sources([_rec("Ibn Batal", "ibn batal")]) == ["Ibn Batal"]


def test_names_differing_by_a_letter_are_left_alone():
    """al-Maziri/al-Muziri differ by a letter, not a diacritic. Merging near-misses
    would silently erase a real second authority -- worse than listing one twice."""
    assert transmitted_sources([_rec("al-Maziri", "al-Muziri")]) == ["al-Maziri", "al-Muziri"]


def test_distinct_scholars_are_not_merged():
    out = transmitted_sources([_rec("Mālik", "Ahmad ibn Hanbal", "Ibn Khuzayma")])
    assert len(out) == 3


def test_an_arabic_script_name_survives_folding():
    assert transmitted_sources([_rec("القاضي عياض")]) == ["القاضي عياض"]


def test_names_are_collected_across_records_and_moves():
    out = transmitted_sources([_rec("Mālik"), _rec("al-Azhari")])
    assert out == ["Mālik", "al-Azhari"]
