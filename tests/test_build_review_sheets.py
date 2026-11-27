"""Review sheets built from a finished run's notes, with no model calls."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_review_sheets import roster_section, sheet_for


def _rec(kitab, confidences, quoted_from=None):
    return {"bab_ref": "n/0", "kitab": kitab, "status": "moves",
            "moves": [{"operation": "gloss", "confidence": c,
                       "arabic_evidence": "قال",
                       "provenance": {"author": "transmitted" if quoted_from
                                      else "compiler_own",
                                      "quoted_from": quoted_from or []}}
                      for c in confidences]}


def test_moves_are_selected_relatively_not_by_the_dead_threshold():
    """LOW_CONFIDENCE is 0.5 and this model never reports below 0.70, so an absolute
    threshold selects nothing and every sheet comes out empty. The one sitting there
    is cannot be spent on blank pages.
    """
    sheet = sheet_for("K", [_rec("K", [0.95, 0.70, 0.88])])
    assert [m["confidence"] for m in sheet["low_confidence_moves"]] == [0.70, 0.88, 0.95]
    assert sheet["confidence_range"] == (0.70, 0.95)


def test_the_sheet_is_capped_so_one_kitab_stays_reviewable():
    from hermeneutics.study_graph import MAX_REVIEW_MOVES

    sheet = sheet_for("K", [_rec("K", [0.9] * (MAX_REVIEW_MOVES + 5))])
    assert len(sheet["low_confidence_moves"]) == MAX_REVIEW_MOVES
    assert sheet["n_moves"] == MAX_REVIEW_MOVES + 5


def test_a_kitab_with_no_moves_does_not_crash():
    sheet = sheet_for("K", [{"bab_ref": "n/1", "kitab": "K", "status": "no_move",
                             "moves": []}])
    assert sheet["low_confidence_moves"] == []
    assert sheet["confidence_range"] is None


def test_roster_lists_recorded_names_and_flags_unrecorded_ones():
    records = [_rec("K", [0.9], quoted_from=["al-Khattabi"])]
    section = roster_section(records, "nawawi", "The compiler follows al-Bukhari here.")
    assert "al-Khattabi" in section
    assert "al-Bukhari" in section
    assert "not a person" in section


def test_cli_help_runs_from_the_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/build_review_sheets.py", "--help"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
