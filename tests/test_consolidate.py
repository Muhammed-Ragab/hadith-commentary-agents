import json

from hermeneutics.nodes import Deps, consolidate_global, consolidate_kitab, provenance_ratio


def _rec(kitab, op, author="compiler_own", sources=None):
    return {
        "bab_ref": f"nawawi/{op}", "kitab": kitab, "gap": "g",
        "moves": [{"operation": op, "arabic_evidence": "ع", "closes_gap_by": "c",
                   "provenance": {"author": author, "quoted_from": sources or []},
                   "confidence": 0.8}],
        "revised_reading": "r", "hypothesis_hash": "h", "manifest_id": "m1", "control": False,
    }


# Counted over the notes, not written by the model. B1 / handoff.md §8.15.
_PROV = {"compiler_own": 1, "transmitted": 1, "compiler_own_pct": 50.0}


def _deps(fake_model, trace):
    return Deps(model=fake_model, trace=trace, manifest_id="m1", babs=[],
                commentator="nawawi", naive={}, seg={}, token_len=lambda s: len(s) // 4)


def test_provenance_ratio_counts_both_categories():
    records = [_rec("K1", "a"), _rec("K1", "b", "transmitted", ["al-Qadi Iyad"])]
    ratio = provenance_ratio(records)
    assert ratio == {"compiler_own": 1, "transmitted": 1, "compiler_own_pct": 50.0}


def test_provenance_ratio_on_empty_input():
    assert provenance_ratio([]) == {"compiler_own": 0, "transmitted": 0,
                                    "compiler_own_pct": 0.0}


def test_consolidate_kitab_summarises_one_kitab(fake_model, trace):
    out = consolidate_kitab("K1", [_rec("K1", "takhsis")], _deps(fake_model(["summary"]), trace))
    assert out["summary"] == "summary"
    assert out["n_records"] == 1


def test_consolidate_kitab_reports_provenance(fake_model, trace):
    records = [_rec("K1", "a"), _rec("K1", "b", "transmitted", ["X"])]
    out = consolidate_kitab("K1", records, _deps(fake_model(["s"]), trace))
    assert out["provenance"]["transmitted"] == 1


def test_consolidate_global_reduces_summaries_not_raw_records(fake_model, trace, tmp_path):
    """Defect #3: the global call must never see all 4,522 records."""
    summaries = {f"K{i}": "x" * 100 for i in range(63)}
    consolidate_global(summaries, _deps(fake_model(["method file"]), trace), _PROV)
    traced = (tmp_path / "test.jsonl").read_text(encoding="utf-8")
    assert "hypothesis_hash" in traced
    assert len(json.loads(traced.strip().split("\n")[0])["prompt"]) < 100_000


def test_consolidate_global_returns_the_method_text(fake_model, trace):
    out = consolidate_global({"K1": "s1"}, _deps(fake_model(["THE METHOD"]), trace), _PROV)
    # The counted provenance block is appended by code; see test_provenance_claim.py.
    assert out.startswith("THE METHOD")


def test_kitab_summaries_chunk_when_records_are_many(fake_model, trace):
    """A 197-bāb ʿAynī kitāb must not blow the context either."""
    records = [_rec("K1", f"op{i}") for i in range(200)]
    out = consolidate_kitab("K1", records, _deps(fake_model(["s1", "s2", "s3"]), trace))
    assert out["n_records"] == 200
    assert out["summary"]
