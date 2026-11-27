import json

from hermeneutics.trace import TraceWriter


def test_writes_one_json_line_per_call(tmp_path):
    tw = TraceWriter(run_dir=tmp_path, phase="study")
    tw.log(node="naive_reading", bab_ref="nawawi/1", prompt="p1", response="r1",
           model_id="M", tokens_in=10, tokens_out=5, latency_s=1.2)
    tw.log(node="horizon_diff", bab_ref="nawawi/1", prompt="p2", response="r2",
           model_id="M", tokens_in=20, tokens_out=8, latency_s=2.0)

    lines = (tmp_path / "study.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["node"] == "naive_reading"


def test_stores_full_payloads_not_truncated(tmp_path):
    """Spec §6.3: sampled or metadata-only tracing cannot be the research record."""
    long_prompt = "ا" * 50_000
    tw = TraceWriter(run_dir=tmp_path, phase="study")
    tw.log(node="read_commentary", bab_ref="cayni/9", prompt=long_prompt, response="ok",
           model_id="M", tokens_in=1, tokens_out=1, latency_s=0.1)

    rec = json.loads((tmp_path / "study.jsonl").read_text(encoding="utf-8"))
    assert rec["prompt"] == long_prompt
    assert rec["response"] == "ok"


def test_flushes_incrementally_not_at_end(tmp_path):
    """Defect #6: a crash must not lose the run."""
    tw = TraceWriter(run_dir=tmp_path, phase="study")
    tw.log(node="n", bab_ref="x/0", prompt="p", response="r",
           model_id="M", tokens_in=1, tokens_out=1, latency_s=0.1)
    # No close() call - the line must already be on disk.
    assert (tmp_path / "study.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_records_error_and_attempt_metadata(tmp_path):
    tw = TraceWriter(run_dir=tmp_path, phase="study")
    tw.log(node="horizon_diff", bab_ref="nawawi/2", prompt="p", response="",
           model_id="M", tokens_in=1, tokens_out=0, latency_s=0.1,
           attempt=2, error="JSONDecodeError", hypothesis_hash="h9")
    rec = json.loads((tmp_path / "study.jsonl").read_text(encoding="utf-8"))
    assert rec["attempt"] == 2
    assert rec["error"] == "JSONDecodeError"
    assert rec["hypothesis_hash"] == "h9"


def test_every_record_carries_a_timestamp(tmp_path):
    tw = TraceWriter(run_dir=tmp_path, phase="control")
    tw.log(node="n", bab_ref="x/0", prompt="p", response="r",
           model_id="M", tokens_in=1, tokens_out=1, latency_s=0.1)
    rec = json.loads((tmp_path / "control.jsonl").read_text(encoding="utf-8"))
    assert rec["ts"].startswith("20")
