import json
import sys

import pytest

import scripts.run_segmentation as run_segmentation
from scripts.run_segmentation import build_parser, segment_one, shard


def test_shard_splits_evenly_and_covers_everything():
    items = list(range(100))
    shards = [shard(items, i, 4) for i in range(4)]
    assert sum(len(s) for s in shards) == 100
    assert sorted(x for s in shards for x in s) == items


def test_single_shard_is_the_whole_list():
    assert shard([1, 2, 3], 0, 1) == [1, 2, 3]


def test_segment_one_returns_a_valid_record(fake_model, trace):
    model = fake_model([json.dumps({"matn_end_char": 120, "confidence": 0.9})])
    rec = segment_one({"kitab": "K", "bab": "b", "text": "ا" * 500},
                      "nawawi/0", model, trace, "m1")
    assert rec["matn_end_char"] == 120
    assert rec["bab_ref"] == "nawawi/0"


def test_segment_one_retries_an_offset_past_the_text_end(fake_model, trace):
    model = fake_model([
        json.dumps({"matn_end_char": 99999, "confidence": 0.9}),
        json.dumps({"matn_end_char": 80, "confidence": 0.9}),
    ])
    rec = segment_one({"kitab": "K", "bab": "b", "text": "ا" * 100},
                      "nawawi/0", model, trace, "m1")
    assert rec["matn_end_char"] == 80


def test_segment_one_retries_a_zero_offset(fake_model, trace):
    model = fake_model([
        json.dumps({"matn_end_char": 0, "confidence": 0.0}),
        json.dumps({"matn_end_char": 40, "confidence": 0.9}),
    ])
    rec = segment_one({"kitab": "K", "bab": "b", "text": "ا" * 100},
                      "nawawi/0", model, trace, "m1")
    assert rec["matn_end_char"] == 40


def test_segment_one_rejects_a_negative_offset(fake_model, trace):
    model = fake_model([json.dumps({"matn_end_char": -5, "confidence": 0.9})])
    with pytest.raises(Exception):
        segment_one({"kitab": "K", "bab": "b", "text": "ا" * 100},
                    "nawawi/0", model, trace, "m1")


def test_segment_one_respects_a_small_model_input_budget(fake_model, trace):
    model = fake_model([json.dumps({"matn_end_char": 10, "confidence": 0.9})])
    segment_one(
        {"kitab": "K", "bab": "b", "text": "A" * 100},
        "nawawi/0", model, trace, "m1", max_chars=20,
    )

    row = json.loads(trace.path.read_text(encoding="utf-8").splitlines()[0])
    assert "A" * 20 in row["prompt"]
    assert "A" * 21 not in row["prompt"]


def test_parser_accepts_shard_arguments():
    args = build_parser().parse_args(
        ["nawawi", "--shard", "3", "--n-shards", "8", "--output-dir", "seg"]
    )
    assert args.shard == 3
    assert args.n_shards == 8
    assert args.output_dir == "seg"


def test_parser_defaults_to_a_single_shard():
    args = build_parser().parse_args(["cayni"])
    assert args.shard == 0
    assert args.n_shards == 1


def test_limit_is_applied_before_sharding():
    items = list(enumerate(["a", "b", "c", "d"]))
    assert run_segmentation.select_units(items, limit=2, index=0, n_shards=1) == items[:2]


def test_main_disables_thinking_for_segmentation(monkeypatch, tmp_path):
    requested = {}

    def build_model(**kwargs):
        requested.update(kwargs)
        return object()

    monkeypatch.setattr(run_segmentation, "get_chat_model", build_model)
    monkeypatch.setattr(run_segmentation, "load_babs", lambda _name: [])
    monkeypatch.setattr(run_segmentation, "ROOT", tmp_path)
    monkeypatch.setenv("VLLM_MODEL_ID", "Qwen/Qwen3.8-27B")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_segmentation.py", "nawawi", "--limit", "0", "--output-dir", str(tmp_path)],
    )

    assert run_segmentation.main() == 0
    assert requested == {"max_tokens": 200, "enable_thinking": False}


def test_segment_one_rejects_the_character_one_non_answer(fake_model, trace):
    """Qwen job 15775708 returned this for Nawawi and Cayni; it passed 0 < n < len."""
    model = fake_model([json.dumps({"matn_end_char": 1, "confidence": 0.0})] * 2)
    with pytest.raises(ValueError):
        segment_one({"kitab": "K", "bab": "b", "text": "ا" * 20_000},
                    "nawawi/0", model, trace, "m1")


def test_segment_one_retries_a_tiny_prefix_then_accepts(fake_model, trace):
    model = fake_model([
        json.dumps({"matn_end_char": 1, "confidence": 0.0}),
        json.dumps({"matn_end_char": 1468, "confidence": 0.9}),
    ])
    rec = segment_one({"kitab": "K", "bab": "b", "text": "ا" * 20_000},
                      "nawawi/0", model, trace, "m1")
    assert rec["matn_end_char"] == 1468


def test_segment_one_rejects_zero_confidence_on_a_plausible_offset(fake_model, trace):
    model = fake_model([json.dumps({"matn_end_char": 1468, "confidence": 0.0})] * 2)
    with pytest.raises(ValueError):
        segment_one({"kitab": "K", "bab": "b", "text": "ا" * 20_000},
                    "nawawi/0", model, trace, "m1")


def test_segment_one_keeps_the_documented_interleaved_low_confidence(fake_model, trace):
    """The prompt asks for confidence below 0.5 when matn and sharh interleave."""
    model = fake_model([json.dumps({"matn_end_char": 900, "confidence": 0.3})])
    rec = segment_one({"kitab": "K", "bab": "b", "text": "ا" * 20_000},
                      "nawawi/0", model, trace, "m1")
    assert rec["confidence"] == 0.3


def test_a_short_nawawi_lemma_boundary_is_accepted(fake_model, trace):
    """Nawawi quotes a lemma then comments; 79 chars is a real boundary, not a stub."""
    text = "# 46 " + "ق" * 800
    model = fake_model([json.dumps({"matn_end_char": 79, "confidence": 0.9})])
    rec = segment_one({"kitab": "K", "bab": "b", "text": text},
                      "nawawi/22", model, trace, "m1")
    assert rec["matn_end_char"] == 79


def test_a_degenerate_two_character_bab_does_not_abort_the_arm(fake_model, trace):
    """16 corpus babs are under 100 chars; the floor scales so they stay segmentable."""
    model = fake_model([json.dumps({"matn_end_char": 1, "confidence": 0.9})])
    rec = segment_one({"kitab": "K", "bab": "b", "text": "اب"},
                      "nawawi/0", model, trace, "m1")
    assert rec["matn_end_char"] == 1


def test_segment_one_accepts_a_matn_that_starts_after_a_heading_gloss(fake_model, trace):
    """The ʿAynī case: 93% of its 3,959 babs gloss Bukhari's chapter heading first,
    then quote the hadith. Asked only where the matn *ended*, the model answered 1 --
    correct, and fatal to the arm (jobs 15849569, saia2, saia3)."""
    model = fake_model([json.dumps(
        {"matn_start_char": 480, "matn_end_char": 1310, "confidence": 0.9})])
    rec = segment_one({"kitab": "K", "bab": "b", "text": "ا" * 20_000},
                      "cayni/0", model, trace, "m1")
    assert (rec["matn_start_char"], rec["matn_end_char"]) == (480, 1310)


def test_segment_one_defaults_the_start_to_the_top_of_the_bab(fake_model, trace):
    """Nawawi and Mazandarani mostly do open with the matn, and so did every
    segmentation file written before the span existed."""
    model = fake_model([json.dumps({"matn_end_char": 1468, "confidence": 0.9})])
    rec = segment_one({"kitab": "K", "bab": "b", "text": "ا" * 20_000},
                      "nawawi/0", model, trace, "m1")
    assert rec["matn_start_char"] == 0


def test_segment_one_rejects_a_span_that_holds_no_matn(fake_model, trace):
    """A start past the end is not a bab that opens with a gloss, it is a non-answer."""
    model = fake_model([json.dumps(
        {"matn_start_char": 900, "matn_end_char": 400, "confidence": 0.9})] * 2)
    with pytest.raises(ValueError):
        segment_one({"kitab": "K", "bab": "b", "text": "ا" * 20_000},
                    "cayni/0", model, trace, "m1")


def test_segment_one_rejects_an_empty_span_inside_a_long_bab(fake_model, trace):
    model = fake_model([json.dumps(
        {"matn_start_char": 480, "matn_end_char": 481, "confidence": 0.9})] * 2)
    with pytest.raises(ValueError):
        segment_one({"kitab": "K", "bab": "b", "text": "ا" * 20_000},
                    "cayni/0", model, trace, "m1")


def test_anchor_is_the_top_of_a_bab_with_no_marker():
    from scripts.run_segmentation import anchor_offset
    assert anchor_offset("ا" * 500) == 0


def test_anchor_is_the_openiti_hadith_marker_when_present():
    """94.1% of ʿAynī's babs carry '### $'; Nawawi and Mazandarani carry none."""
    from scripts.run_segmentation import anchor_offset
    text = "أي هذا باب " * 50 + "### $ 1 - # حدثنا عبيد الله" + "ا" * 400
    assert anchor_offset(text) == text.index("###")


def test_anchor_falls_back_to_the_isnad_only_past_the_window():
    """cayni/0: 57,117 chars, first isnād at 34,815, so a 20,000-char window from
    the top holds no matn and the model answered 0-0 (job cayni1)."""
    from scripts.run_segmentation import anchor_offset
    text = "ا" * 30_000 + "حدثنا فلان" + "ب" * 1_000
    assert anchor_offset(text, max_chars=20_000) == 30_000


def test_a_short_bab_is_never_anchored_on_an_isnad_in_its_sharh():
    """Inside the window nothing can be hidden, and Nawawi mentions chains mid-sharh."""
    from scripts.run_segmentation import anchor_offset
    text = "مطن" * 100 + "حدثنا فلان" + "ب" * 100
    assert anchor_offset(text, max_chars=20_000) == 0


def test_main_logs_an_unsegmentable_bab_instead_of_aborting_the_arm(monkeypatch, tmp_path):
    """B4: one bad bāb (cayni/2232, cayni/2872 in the real corpus) must not kill the
    whole arm. segment_one still raises -- that's unit-tested above -- but main()'s
    loop must catch it, log a failure record, and keep writing the babs after it."""
    babs = [
        {"kitab": "K", "bab": "b0", "text": "ا" * 500},
        {"kitab": "K", "bab": "b1", "text": "ب" * 500},
        {"kitab": "K", "bab": "b2", "text": "ج" * 500},
    ]
    # bab 1 never answers with a credible span; 0 and 2 do on the first try.
    responses = iter([
        json.dumps({"matn_end_char": 50, "confidence": 0.9}),
        json.dumps({"matn_end_char": 0, "confidence": 0.0}),
        json.dumps({"matn_end_char": 0, "confidence": 0.0}),
        json.dumps({"matn_end_char": 50, "confidence": 0.9}),
    ])
    monkeypatch.setattr(run_segmentation, "get_chat_model", lambda **kw: object())
    monkeypatch.setattr(run_segmentation, "load_babs", lambda _name: babs)
    monkeypatch.setattr(run_segmentation, "ROOT", tmp_path)
    monkeypatch.setattr(
        run_segmentation, "call_json",
        lambda *a, **kw: json.loads(next(responses)),
    )
    monkeypatch.setenv("VLLM_MODEL_ID", "Qwen/Qwen3.8-27B")
    monkeypatch.setattr(
        sys, "argv",
        ["run_segmentation.py", "nawawi", "--output-dir", str(tmp_path)],
    )

    assert run_segmentation.main() == 0

    out_path = tmp_path / "nawawi_seg.000-of-001.jsonl"
    refs = [json.loads(line)["bab_ref"] for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert refs == ["nawawi/0", "nawawi/2"]  # nawawi/1 is absent, not a fabricated record

    failures_path = tmp_path / "nawawi_seg_failures.000-of-001.jsonl"
    failures = [json.loads(line) for line in failures_path.read_text(encoding="utf-8").splitlines()]
    assert [f["bab_ref"] for f in failures] == ["nawawi/1"]
    assert failures[0]["attempts"] == 2
    assert "no credible boundary" in failures[0]["error"]


def test_segment_one_reports_offsets_absolute_to_the_whole_bab(fake_model, trace):
    """The model answers in window coordinates; the study graph slices bab['text']."""
    text = "أي هذا باب " * 50 + "### $ حدثنا" + "ا" * 2_000
    anchor = text.index("###")
    model = fake_model([json.dumps(
        {"matn_start_char": 0, "matn_end_char": 300, "confidence": 0.9})])
    rec = segment_one({"kitab": "K", "bab": "b", "text": text},
                      "cayni/1", model, trace, "m1")
    assert rec["matn_start_char"] == anchor
    assert rec["matn_end_char"] == anchor + 300
