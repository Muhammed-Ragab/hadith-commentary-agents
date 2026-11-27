"""The reading copy: a banner, and the frozen body left alone."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.make_reading_copy import reading_copy

BODY = '''# Method file

The compiler "never adjudicates by personal preference" (al-Ilm).
'''


def test_the_body_is_not_modified():
    """The method file is a run output the paper cites. The banner is prepended; not
    one byte of the model's text is rewritten, because rewriting a frozen output
    silently breaks the reproducibility contract.
    """
    out = reading_copy(BODY)
    assert out.endswith(BODY)


def test_the_banner_warns_that_english_quotes_are_not_sharh_text():
    out = reading_copy(BODY)
    assert "Nothing in English quotation marks is a quotation" in out


def test_arabic_spans_are_counted_and_flagged_as_possibly_genuine():
    out = reading_copy(BODY + 'The reading is "اتفقت نسخ" there.\n')
    assert "Arabic-script spans (1 of them here)" in out
    assert "no Arabic-script quotations" not in out


def test_a_file_with_no_arabic_quotes_says_so():
    assert "no Arabic-script quotations" in reading_copy(BODY)


def test_the_unverified_note_appears_only_when_the_stamp_does():
    assert "confirmed false" not in reading_copy(BODY)
    stamped = BODY + "\n## Unresolved provenance disagreement\n"
    assert "confirmed false" in reading_copy(stamped)


def test_cli_help_runs_from_the_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/make_reading_copy.py", "--help"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
