"""Proof that build_isnad_sample.py reproduces the committed samples. Spec §3.5.

The central claim of this task: the Muslim isnad sample must be drawn by the exact
procedure that produced bukhari_sample.json and kafi_sample.json (notebook 01 cell 2,
seed 42), or its network metrics are not comparable to theirs. If the reproduction
test below fails, that is a real finding about the committed samples -- do not adjust
the test to make it pass, and do not generate the Muslim sample from an unproven
procedure.
"""
import json
from pathlib import Path

import pytest

from scripts.build_isnad_sample import DATA_DIR, RAW_PATHS, build_sample

BUKHARI_SAMPLE = DATA_DIR / "bukhari_sample.json"
KAFI_SAMPLE = DATA_DIR / "kafi_sample.json"

_raw_missing = not all(p.exists() for p in RAW_PATHS.values())
_committed_missing = not (BUKHARI_SAMPLE.exists() and KAFI_SAMPLE.exists())

pytestmark = pytest.mark.skipif(
    _raw_missing or _committed_missing,
    reason="raw corpora or committed sample files not present on disk",
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------- reproduction of committed samples

def test_bukhari_sample_reproduces_committed_file_byte_for_byte_in_content():
    rebuilt = build_sample("bukhari")
    committed = _load(BUKHARI_SAMPLE)
    assert rebuilt == committed


def test_kafi_sample_reproduces_committed_file_byte_for_byte_in_content():
    rebuilt = build_sample("kafi")
    committed = _load(KAFI_SAMPLE)
    assert rebuilt == committed


# ----------------------------------------------------------------- muslim sample shape

def test_muslim_sample_yields_500_records_with_isnad_and_matn():
    sample = build_sample("muslim")
    assert len(sample) == 500
    for h in sample:
        assert h.get("isnad")
        assert h.get("matn")


def test_same_seed_twice_gives_the_same_sample():
    a = build_sample("muslim", seed=42)
    b = build_sample("muslim", seed=42)
    assert a == b


def test_different_seed_gives_a_different_sample():
    a = build_sample("muslim", seed=42)
    b = build_sample("muslim", seed=7)
    assert a != b


# ------------------------------------------------------------------- overwrite guard

def test_overwrite_guard_refuses_without_force(tmp_path, monkeypatch):
    import scripts.build_isnad_sample as bis

    fake_data_dir = tmp_path
    fake_sample = fake_data_dir / "muslim_sample.json"
    fake_sample.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(bis, "DATA_DIR", fake_data_dir)
    monkeypatch.setattr(bis, "RAW_PATHS", {**bis.RAW_PATHS, "muslim": bis.RAW_PATHS["muslim"]})

    with pytest.raises(SystemExit):
        bis.main(["--corpus", "muslim"])

    # Untouched: still the placeholder, not a freshly built sample.
    assert fake_sample.read_text(encoding="utf-8") == "[]"
