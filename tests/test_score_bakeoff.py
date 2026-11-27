import json

from scripts.score_bakeoff import projected_hours, score_candidate


def test_projected_hours_uses_sequential_calls_only():
    """Batch calls parallelise; only the sequential loop sets wall clock (spec §6.1)."""
    assert projected_hours(2.5, 14783) == 10.3
    assert projected_hours(5.0, 14783) == 20.5


def _write(path, records):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records),
                    encoding="utf-8")


def test_score_candidate_reports_grounding_and_latency(tmp_path, monkeypatch):
    import scripts.score_bakeoff as sb

    monkeypatch.setattr(sb, "ROOT", tmp_path)
    cand = tmp_path / "outputs" / "bakeoff" / "some_model"
    cand.mkdir(parents=True)
    _write(cand / "nawawi_notes.jsonl", [{
        "bab_ref": "nawawi/0",
        "moves": [{"arabic_evidence": "قال النووي"}, {"arabic_evidence": "غير موجود"}],
    }])

    traces = tmp_path / "outputs" / "runs" / "some_model"
    traces.mkdir(parents=True)
    _write(traces / "study_nawawi.jsonl", [
        {"latency_s": 2.0, "error": None},
        {"latency_s": 4.0, "error": "JSONDecodeError"},
    ])

    row = score_candidate(cand, "nawawi", {"nawawi/0": "وقد قال النووي رحمه الله"})
    assert row["grounding_rate"] == 0.5, "one of two moves is anchored in the source"
    assert row["s_per_call_median"] == 3.0
    assert row["json_repair_rate"] == 0.5
