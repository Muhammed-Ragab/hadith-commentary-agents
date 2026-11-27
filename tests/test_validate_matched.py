import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _write_arm(root, name, evidence):
    arm = root / name
    arm.mkdir(parents=True)
    (arm / f"{name}_method.md").write_text("method", encoding="utf-8")
    record = {
        "bab_ref": f"{name}/0",
        "moves": [{"arabic_evidence": evidence}],
    }
    (arm / f"{name}_notes.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def test_validate_arm_accepts_complete_grounded_arabic_output(tmp_path):
    from scripts.validate_matched import validate_arm

    _write_arm(tmp_path, "nawawi", "قال النووي")
    errors = validate_arm(tmp_path, "nawawi", 1, {"nawawi/0": "وقد قال النووي رحمه الله"})
    assert errors == []


def test_validate_arm_rejects_non_arabic_and_ungrounded_evidence(tmp_path):
    from scripts.validate_matched import validate_arm

    _write_arm(tmp_path, "cayni", "lexical analysis")
    errors = validate_arm(tmp_path, "cayni", 1, {"cayni/0": "قال العيني"})
    assert any("Arabic" in error for error in errors)
    assert any("not grounded" in error for error in errors)


def test_validate_arm_accepts_an_explicit_no_move_record(tmp_path):
    from scripts.validate_matched import validate_arm

    arm = tmp_path / "nawawi"
    arm.mkdir()
    (arm / "nawawi_method.md").write_text("method", encoding="utf-8")
    record = {
        "bab_ref": "nawawi/0",
        "status": "no_move",
        "moves": [],
        "no_move_reason": "Nawawi restates the matn and adds no gloss, isnad note, or lexical remark.",
    }
    (arm / "nawawi_notes.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    assert validate_arm(tmp_path, "nawawi", 1, {"nawawi/0": "source"}) == []


def test_a_bab_with_a_note_is_covered_even_if_it_also_logged_a_failure(tmp_path):
    """B4 widened _write_failure from segmentation-only to EVERY failure, so a
    move-level grounding rejection on a bāb that still produced a good note now writes
    a failure record too (cayni/22 in the matched study). That is a move-rejection, not
    a shortfall: the bāb is covered. Erroring on it reported a complete arm as broken,
    and counting it toward MAX_DECLARED_FAILURE_RATE pushed healthy arms at the ceiling.
    """
    from scripts.validate_matched import validate_arm

    _write_arm(tmp_path, "nawawi", "قال النووي")
    arm = tmp_path / "nawawi"
    (arm / "nawawi_failures.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/0", "error": "unknown evidence_id"}) + "\n",
        encoding="utf-8",
    )

    assert validate_arm(tmp_path, "nawawi", 1, {"nawawi/0": "قال النووي"}) == []


def test_a_bab_with_no_note_and_no_failure_is_still_a_silent_gap(tmp_path):
    """The completeness guard the test above must not have relaxed away."""
    from scripts.validate_matched import validate_arm

    _write_arm(tmp_path, "nawawi", "قال النووي")
    errors = validate_arm(tmp_path, "nawawi", 2,
                          {"nawawi/0": "قال النووي", "nawawi/1": "قال النووي"})
    assert any("neither a note nor a" in error for error in errors)


def test_cli_help_runs_from_the_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/validate_matched.py", "--help"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def _write_segmentation(root, name, offset, confidence):
    seg = root / "segmentation"
    seg.mkdir(exist_ok=True)
    (seg / f"{name}_seg.000-of-001.jsonl").write_text(
        json.dumps({"bab_ref": f"{name}/0", "matn_end_char": offset,
                    "confidence": confidence}) + "\n", encoding="utf-8")


def test_validate_arm_rejects_a_structurally_valid_but_empty_boundary(tmp_path):
    """Qwen job 15775708 passed `validated 1 babs` on matn_end_char=1, confidence=0."""
    from scripts.validate_matched import validate_arm

    _write_arm(tmp_path, "nawawi", "قال النووي")
    _write_segmentation(tmp_path, "nawawi", 1, 0.0)
    errors = validate_arm(tmp_path, "nawawi", 1, {"nawawi/0": "# وقد قال النووي رحمه الله"})
    assert any("not credible" in error for error in errors)


def test_validate_arm_accepts_a_credible_boundary(tmp_path):
    from scripts.validate_matched import validate_arm

    _write_arm(tmp_path, "nawawi", "قال النووي")
    _write_segmentation(tmp_path, "nawawi", 12, 0.9)
    assert validate_arm(tmp_path, "nawawi", 1, {"nawawi/0": "وقد قال النووي رحمه الله"}) == []


def test_validate_arm_is_silent_when_no_segmentation_was_written(tmp_path):
    from scripts.validate_matched import validate_arm

    _write_arm(tmp_path, "nawawi", "قال النووي")
    assert validate_arm(tmp_path, "nawawi", 1, {"nawawi/0": "وقد قال النووي رحمه الله"}) == []


def _sources(name, n, text="ا" * 500):
    return {f"{name}/{i}": text for i in range(n)}


def test_validate_arm_accepts_a_declared_segmentation_failure(tmp_path, capsys):
    """B4: one unsegmentable bāb (cayni/2232, cayni/2872 in the real corpus) must
    complete the arm, not abort it -- and must not validate silently at less than
    the requested count."""
    from scripts.validate_matched import validate_arm

    arm = tmp_path / "nawawi"
    arm.mkdir(parents=True)
    (arm / "nawawi_method.md").write_text("method", encoding="utf-8")
    (arm / "nawawi_notes.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/0", "moves": [{"arabic_evidence": "قال النووي"}]}) + "\n",
        encoding="utf-8",
    )
    seg_dir = tmp_path / "segmentation"
    seg_dir.mkdir()
    (seg_dir / "nawawi_seg_failures.000-of-001.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/1", "attempts": 2,
                    "error": "segmentation returned no credible boundary"}) + "\n",
        encoding="utf-8",
    )

    sources = {"nawawi/0": "وقد قال النووي رحمه الله", "nawawi/1": "ا" * 500}
    errors = validate_arm(tmp_path, "nawawi", 2, sources)
    assert errors == []
    assert "nawawi: 1/2 notes, 1 declared failure(s)" in capsys.readouterr().out


def test_validate_arm_rejects_a_missing_note_with_no_declared_failure(tmp_path):
    """The other half of B4: a gap with nothing recorded for it must still fail --
    otherwise the declared-shortfall rule is just the old guard removed."""
    from scripts.validate_matched import validate_arm

    arm = tmp_path / "nawawi"
    arm.mkdir(parents=True)
    (arm / "nawawi_method.md").write_text("method", encoding="utf-8")
    (arm / "nawawi_notes.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/0", "moves": [{"arabic_evidence": "قال النووي"}]}) + "\n",
        encoding="utf-8",
    )
    # nawawi/1 has no note and no failure record -- an undeclared, silent gap.

    errors = validate_arm(tmp_path, "nawawi", 2, _sources("nawawi", 2))
    assert any("neither a note nor a declared failure" in e and "nawawi/1" in e for e in errors)


def test_validate_arm_rejects_a_note_outside_the_requested_set(tmp_path):
    from scripts.validate_matched import validate_arm

    _write_arm(tmp_path, "nawawi", "قال النووي")  # writes a note for nawawi/0
    errors = validate_arm(tmp_path, "nawawi", 1, {"nawawi/9": "ا" * 500})
    assert any("outside the requested set" in e for e in errors)


def _arm_with(root, notes, failures):
    """An arm holding `notes` grounded notes and `failures` declared failures."""
    arm = root / "nawawi"
    arm.mkdir(parents=True)
    (root / "segmentation").mkdir(exist_ok=True)
    (arm / "nawawi_method.md").write_text("method", encoding="utf-8")
    with open(arm / "nawawi_notes.jsonl", "w", encoding="utf-8") as fh:
        for i in range(notes):
            fh.write(json.dumps({"bab_ref": f"nawawi/{i}", "status": "moves",
                                 "moves": [{"arabic_evidence": "نص"}]},
                                ensure_ascii=False) + "\n")
    if failures:
        with open(arm / "nawawi_failures.jsonl", "w", encoding="utf-8") as fh:
            for i in range(notes, notes + failures):
                fh.write(json.dumps({"bab_ref": f"nawawi/{i}", "error": "gave up"}) + "\n")
    return root


def test_an_arm_that_declared_away_most_of_its_babs_still_fails(tmp_path):
    """A declared shortfall is a relaxation, not an exemption.

    B4 made a bāb that cannot be segmented non-fatal. Without a ceiling that also
    passes an arm which produced nothing at all, as long as it logged a failure for
    every bāb -- the completeness guard would be gone above the 'declared' bar rather
    than relaxed. §8.14's rule stands: a red arm beats a fabricated finding.
    """
    from scripts.validate_matched import validate_arm
    root = _arm_with(tmp_path, notes=2, failures=8)
    errors = validate_arm(root, "nawawi", 10, {f"nawawi/{i}": "نص" for i in range(10)})
    assert any("declared failure" in e and "exceeds" in e for e in errors)


def test_a_shortfall_inside_the_ceiling_still_passes(tmp_path):
    from scripts.validate_matched import validate_arm
    root = _arm_with(tmp_path, notes=9, failures=1)
    errors = validate_arm(root, "nawawi", 10, {f"nawawi/{i}": "نص" for i in range(10)})
    assert not any("exceeds" in e for e in errors)
