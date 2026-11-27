from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_google_genai_pipeline_is_gone():
    """Defect #5: pyproject no longer ships google-generativeai, so this code
    cannot run. Dead code that looks runnable is worse than no code."""
    assert not (ROOT / "agents").exists()
    assert not (ROOT / "scripts" / "run_study.py").exists()


def test_no_source_file_imports_the_retired_sdks():
    banned = ("google.generativeai", "import anthropic")
    paths = list((ROOT / "hermeneutics").glob("*.py")) + list(
            (ROOT / "scripts").glob("*.py"))
    assert paths, "no Python source found to scan — did hermeneutics/ or scripts/ move?"
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name} still imports {token}"


def test_no_source_file_slices_text_to_a_fixed_length():
    """Defect #1 was text[:3000]. Guard against it creeping back."""
    import re

    pattern = re.compile(r"\[:\s*\d{4,}\s*\]")
    paths = list((ROOT / "hermeneutics").glob("*.py"))
    assert paths, "hermeneutics/ is empty or missing — guard cannot run"
    for path in paths:
        assert not pattern.search(path.read_text(encoding="utf-8")), (
            f"{path.name} has a fixed-length slice - use hermeneutics.chunking"
        )
