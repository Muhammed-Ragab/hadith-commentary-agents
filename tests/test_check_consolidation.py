"""The consolidation check: a method file judged against the notes it summarises."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hermeneutics.nodes import check_provenance_claim, kitab_provenance
from scripts.check_consolidation import check_arm, unrecorded_names


def _move(author, quoted_from=None):
    return {"arabic_evidence": "قال", "provenance": {"author": author,
                                                    "quoted_from": quoted_from or []}}


def _notes(spec):
    """spec: {kitab: (n_own, n_transmitted)} -> one note per kitab."""
    return [{"bab_ref": f"x/{i}", "kitab": kitab, "status": "moves",
             "moves": [_move("compiler_own")] * own
                      + [_move("transmitted", ["al-Khattabi"])] * trans}
            for i, (kitab, (own, trans)) in enumerate(spec.items())]


GLOBAL_ONLY = {"compiler_own": 79, "transmitted": 21, "compiler_own_pct": 79.0}


def test_a_figure_matching_nothing_counted_is_still_a_contradiction():
    """The narrowing must not relax the check it exists to sharpen."""
    per_kitab = kitab_provenance(_notes({"A": (5, 0), "B": (3, 2)}))
    claim = "The compiler relays from named predecessors in 55% of moves."
    assert check_provenance_claim(claim, GLOBAL_ONLY, per_kitab)


def test_the_global_figure_passes():
    claim = "Transmitted material accounts for 21% of moves."
    assert check_provenance_claim(claim, GLOBAL_ONLY, {}) == ""


def test_a_per_kitab_breakdown_is_judged_against_that_kitab_not_the_global_ratio():
    """Three false UNVERIFIED stamps on CORRECT files came from this. A: 100% own is
    true of A and says nothing about the global 79%.
    """
    per_kitab = kitab_provenance(_notes({"A": (5, 0), "B": (3, 2)}))
    claim = "Own voice by kitab: A: 100%; B: 60%."
    assert check_provenance_claim(claim, GLOBAL_ONLY, per_kitab) == ""
    # Without the per-kitab counts the same sentence is rejected -- the old behaviour,
    # kept so the inline consolidate_global path is provably unchanged.
    assert check_provenance_claim(claim, GLOBAL_ONLY)


def test_a_fraction_is_judged_only_against_a_denominator_that_was_counted():
    """"0 of 5 moves in al-Iman" is one kitab's count, not a global claim. Its
    denominator says so, which is a fact about the notes rather than about the prose.
    """
    per_kitab = kitab_provenance(_notes({"al-Iman": (5, 0), "B": (74, 21)}))
    assert check_provenance_claim(
        "0 of 5 moves are transmitted from predecessors in al-Iman.",
        GLOBAL_ONLY, per_kitab) == ""
    # A denominator matching no counted population is judged, not excused.
    assert check_provenance_claim(
        "3 of 7 moves are transmitted from predecessors.", GLOBAL_ONLY, per_kitab)


def test_unrecorded_names_ignores_the_commentator_and_partial_matches():
    recorded = ["Abu Nuʿaym al-Isbahani", "al-Khattabi"]
    prose = "al-Nawawi cites al-Isbahani and al-Khattabi, and also al-Bukhari."
    stray = unrecorded_names(prose, recorded, "nawawi")
    assert "al-Bukhari" in stray
    assert not any("Isbahani" in s or "Khattabi" in s or "Nawawi" in s for s in stray)


def _write_arm(tmp_path, name, method_text, notes):
    arm = tmp_path / name
    arm.mkdir(parents=True)
    (arm / f"{name}_method.md").write_text(method_text, encoding="utf-8")
    (arm / f"{name}_notes.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in notes),
        encoding="utf-8")
    return arm


def test_check_arm_passes_a_file_whose_figures_match(tmp_path):
    notes = _notes({"A": (79, 21)})
    text = ("Transmitted material is 21% of moves.\n\n"
            "## Provenance (counted, not written by the model)\n\nx\n")
    errors, _ = check_arm(_write_arm(tmp_path, "nawawi", text, notes), "nawawi")
    assert errors == []


def test_check_arm_fails_a_file_that_invents_a_figure(tmp_path):
    notes = _notes({"A": (79, 21)})
    text = ("The compiler relays from named predecessors in 55% of moves.\n\n"
            "## Provenance (counted, not written by the model)\n\nx\n")
    errors, _ = check_arm(_write_arm(tmp_path, "nawawi", text, notes), "nawawi")
    assert any("contradicts the notes" in e for e in errors)


def test_check_arm_fails_a_file_with_no_counted_block(tmp_path):
    """The counted block is what the paper cites; its absence is a defect even when
    every stated figure happens to be right."""
    notes = _notes({"A": (79, 21)})
    errors, _ = check_arm(
        _write_arm(tmp_path, "nawawi", "Transmitted material is 21% of moves.", notes),
        "nawawi")
    assert any("no counted provenance block" in e for e in errors)


def test_cli_help_runs_from_the_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/check_consolidation.py", "--help"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
