"""B1 — the consolidation check. handoff.md §8.15, "highest-risk open item".

Nawawī's method file claimed the compiler "frequently references named predecessors"
from **6 of 88** moves; ʿAynī's made the same claim from **0 of 76**. `provenance_ratio`
had computed the real counts all along and `finalize` threw them away, so the global
call — whose prompt §5 asks for "what the ratio shows" — saw only prose and answered
from prior. That number reaches a slide.

Three properties are tested here, in the order they close the hole:
  1. the counts reach the prompts at all;
  2. a stated figure that contradicts the notes is caught;
  3. the method file carries the computed figure, so the paper cites code not prose.
"""
import pytest

from hermeneutics.nodes import (
    Deps,
    check_provenance_claim,
    consolidate_global,
    consolidate_kitab,
)


def _rec(kitab, op, author="compiler_own", sources=None):
    return {
        "bab_ref": f"nawawi/{op}", "kitab": kitab, "gap": "g",
        "moves": [{"operation": op, "arabic_evidence": "ع", "closes_gap_by": "c",
                   "provenance": {"author": author, "quoted_from": sources or []},
                   "confidence": 0.8}],
        "revised_reading": "r", "hypothesis_hash": "h", "manifest_id": "m1", "control": False,
    }


def _deps(fake_model, trace):
    return Deps(model=fake_model, trace=trace, manifest_id="m1", babs=[],
                commentator="nawawi", naive={}, seg={}, token_len=lambda s: len(s) // 4)


# The real shape of the defect: 6 transmitted of 88 moves, i.e. 6.8%.
NAWAWI_15789363 = {"compiler_own": 82, "transmitted": 6, "compiler_own_pct": 93.2}


# ---------------------------------------------------------------- the check itself

def test_a_stated_percentage_that_contradicts_the_notes_is_rejected():
    text = "5. The compiler relays transmitted material in roughly 60% of moves."
    assert check_provenance_claim(text, NAWAWI_15789363)


def test_a_stated_fraction_that_contradicts_the_notes_is_rejected():
    """§8.15's own shape: the claim written as a fraction rather than a percent."""
    text = "5. Named predecessors are relayed in 60 of 88 moves."
    assert check_provenance_claim(text, NAWAWI_15789363)


def test_the_correct_figure_passes():
    text = "5. Only 6.8% of moves relay transmitted material; the voice is his own."
    assert check_provenance_claim(text, NAWAWI_15789363) == ""


def test_rounding_is_tolerated():
    text = "5. Transmitted material accounts for about 7% of moves."
    assert check_provenance_claim(text, NAWAWI_15789363) == ""


def test_a_percentage_about_something_else_is_not_a_provenance_claim():
    """Operations are counted too. Only sentences about provenance are judged."""
    text = "1. Lexical analysis accounts for 40% of all interpretive operations."
    assert check_provenance_claim(text, NAWAWI_15789363) == ""


def test_the_compilers_own_share_is_read_as_its_own_figure():
    """93.2% own and 6.8% transmitted are the same fact; neither may be flagged."""
    text = "5. Some 93% of moves are made in the compiler's own voice."
    assert check_provenance_claim(text, NAWAWI_15789363) == ""


def test_prose_with_no_figure_at_all_is_not_judged():
    """Qualitative prose is not parsed — negation and contrast make that a guess.

    ponytail: the computed block appended to the method file is what the paper cites,
    so an unquantified adverb is a style problem, not a false number.
    """
    text = "5. The compiler frequently references named predecessors."
    assert check_provenance_claim(text, NAWAWI_15789363) == ""


def test_a_decimal_ratio_is_not_misread_as_a_fraction():
    """A decimal pair like "81.1 / 18.9" was read by `_FRACTION` as 1/18 (5.6%) and
    false-rejected a correct method file, killing a whole cluster run."""
    text = ("5. The 81.1 / 18.9 ratio thus reflects a method in which the commentator "
            "is primarily an independent analyst of transmitted material.")
    counts = {"compiler_own": 81, "transmitted": 19, "compiler_own_pct": 81.1}
    assert check_provenance_claim(text, counts) == ""


def test_a_real_fractional_lie_is_still_caught():
    text = "5. Only 6 of 88 moves are transmitted, the rest are the compiler's own."
    counts = {"compiler_own": 30, "transmitted": 7, "compiler_own_pct": 81.1}
    assert check_provenance_claim(text, counts)


def test_the_plain_fraction_form_still_parses():
    text = "5. Transmitted material appears in 7 of 37 moves."
    counts = {"compiler_own": 30, "transmitted": 7, "compiler_own_pct": 81.1}
    assert check_provenance_claim(text, counts) == ""


def test_an_empty_note_set_is_not_judged():
    assert check_provenance_claim("40% transmitted", {"compiler_own": 0,
                                                     "transmitted": 0,
                                                     "compiler_own_pct": 0.0}) == ""


# ---------------------------------------------------- the counts reach the prompts

def test_consolidate_kitab_puts_the_counts_in_the_prompt(fake_model, trace, tmp_path):
    records = [_rec("K1", "a"), _rec("K1", "b", "transmitted", ["al-Qadi Iyad"])]
    consolidate_kitab("K1", records, _deps(fake_model(["s"]), trace))
    prompt = (tmp_path / "test.jsonl").read_text(encoding="utf-8")
    assert "50.0" in prompt and "transmitted" in prompt


def test_consolidate_global_puts_the_counts_in_the_prompt(fake_model, trace, tmp_path):
    consolidate_global({"K1": "s1"}, _deps(fake_model(["m"]), trace), NAWAWI_15789363)
    prompt = (tmp_path / "test.jsonl").read_text(encoding="utf-8")
    assert "82" in prompt and "6" in prompt


# --------------------------------------------------- the artifact carries the number

def test_the_method_file_carries_the_computed_figure(fake_model, trace):
    out = consolidate_global({"K1": "s1"}, _deps(fake_model(["THE METHOD"]), trace),
                             NAWAWI_15789363)
    assert "THE METHOD" in out
    assert "6.8%" in out or "6 of 88" in out


def test_a_contradicting_method_file_is_retried_once(fake_model, trace):
    """§8.10's idiom: feed the contradiction back, then accept the corrected answer."""
    model = fake_model(["Transmitted material appears in 60% of moves.",
                        "Transmitted material appears in 6.8% of moves."])
    out = consolidate_global({"K1": "s1"}, _deps(model, trace), NAWAWI_15789363)
    assert "6.8%" in out


def test_a_method_file_that_contradicts_twice_is_declared_not_raised(fake_model, trace):
    """A red arm beats a fabricated finding (§8.14) -- but raising here destroyed three
    arms in two days on CORRECT files (§8.19), because check_provenance_claim reads a
    per-kitab breakdown as a global claim. The disagreement is now declared in the file
    and the counted block still appended, so nothing false is published and the arm
    survives for a human to judge."""
    model = fake_model(["60% of moves are transmitted.", "60% of moves are transmitted."])
    out = consolidate_global({"K1": "s1"}, _deps(model, trace), NAWAWI_15789363)
    assert "Unresolved provenance disagreement" in out
    assert "UNVERIFIED" in out
    # The unverified prose survives, but so does the counted block the paper cites.
    assert "60% of moves are transmitted." in out
    assert "6.8%" in out or "6 of 88" in out
    # The declaration is markdown a scholar reads, not an escaped string: an
    # accidental '\\n\\n' literal reached matched_qwen3_20260917's Nawawi file.
    assert "\\n" not in out


# ------------------------------------- the named predecessors, which were invented

def test_recorded_predecessor_names_reach_the_prompt(fake_model, trace, tmp_path):
    records = [_rec("K1", "a"),
               _rec("K1", "b", "transmitted", ["al-Qadi Iyad", "Ibn Batal"])]
    consolidate_kitab("K1", records, _deps(fake_model(["s"]), trace))
    prompt = (tmp_path / "test.jsonl").read_text(encoding="utf-8")
    assert "al-Qadi Iyad" in prompt and "Ibn Batal" in prompt


def test_an_empty_roster_is_stated_as_an_instruction_not_an_omission(fake_model, trace,
                                                                     tmp_path):
    """ʿAynī's real failure: 0 of 76 quoted_from, and the file named three authorities.

    Silence in the prompt is what let the model supply names from prior. Saying NONE
    out loud is the fix; an omitted field is not.
    """
    consolidate_kitab("K1", [_rec("K1", "a")], _deps(fake_model(["s"]), trace))
    prompt = (tmp_path / "test.jsonl").read_text(encoding="utf-8")
    assert "NONE" in prompt


def test_transmitted_sources_deduplicates_and_sorts():
    from hermeneutics.nodes import transmitted_sources
    records = [_rec("K1", "a", "transmitted", ["al-Khattabi", ""]),
               _rec("K1", "b", "transmitted", ["al-Khattabi", "Ibn Batal"])]
    assert transmitted_sources(records) == ["Ibn Batal", "al-Khattabi"]
