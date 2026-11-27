import json

import pytest

import scripts.run_study_graph as run_study_graph
from scripts.run_study_graph import (
    build_parser,
    load_naive,
    render_review_sheet,
)


def test_parser_requires_a_commentator():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_accepts_the_documented_flags():
    args = build_parser().parse_args(
        ["nawawi", "--limit", "10", "--run-id", "r1", "--chunk-tokens", "2000",
         "--segmentation-dir", "seg", "--naive-dir", "naive/r1", "--live-naive"]
    )
    assert args.commentator == "nawawi"
    assert args.limit == 10
    assert args.chunk_tokens == 2000
    assert args.segmentation_dir == "seg"
    assert args.naive_dir == "naive/r1"
    assert args.live_naive is True


def test_parser_ablation_toggles_default_off():
    """Deps.legacy_example_prompt / legacy_hypothesis_menu default to the fixed
    behaviour so a normal run gets the fix without asking (ablation conditions A/B
    are opt-in)."""
    args = build_parser().parse_args(["nawawi"])
    assert args.legacy_example_prompt is False
    assert args.legacy_hypothesis_menu is False


def test_parser_accepts_the_ablation_toggles():
    args = build_parser().parse_args(
        ["nawawi", "--legacy-example-prompt", "--legacy-hypothesis-menu"]
    )
    assert args.legacy_example_prompt is True
    assert args.legacy_hypothesis_menu is True


def test_parser_naive_dir_defaults_to_none():
    """None signals main() to fall back to outputs/naive/<run-id> -- B3."""
    assert build_parser().parse_args(["nawawi"]).naive_dir is None


def test_parser_rejects_an_unknown_commentator():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["majlisi"])


def test_parser_uses_the_model_specific_chunk_budget_from_the_environment(monkeypatch):
    monkeypatch.setenv("STUDY_CHUNK_TOKENS", "2500")
    args = build_parser().parse_args(["nawawi"])
    assert args.chunk_tokens == 2500


def test_model_output_budgets_can_fit_a_4k_context(monkeypatch):
    monkeypatch.setenv("STUDY_MAX_OUTPUT_TOKENS", "800")
    monkeypatch.setenv("STUDY_CONSOLIDATE_MAX_TOKENS", "1000")
    assert run_study_graph.model_output_budgets() == (800, 1000)


def test_load_naive_reads_the_batch_jsonl(tmp_path):
    p = tmp_path / "nawawi_naive.jsonl"
    p.write_text(
        json.dumps({"bab_ref": "nawawi/0", "naive_reading": "r0"}) + "\n"
        + json.dumps({"bab_ref": "nawawi/1", "naive_reading": "r1"}) + "\n",
        encoding="utf-8",
    )
    assert load_naive(p) == {"nawawi/0": "r0", "nawawi/1": "r1"}


def test_load_naive_returns_empty_when_the_file_is_absent(tmp_path):
    assert load_naive(tmp_path / "missing.jsonl") == {}


def test_load_segmentation_merges_shards_by_bab_ref(tmp_path):
    (tmp_path / "nawawi_seg.000-of-002.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/0", "matn_end_char": 120}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "nawawi_seg.001-of-002.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/1", "matn_end_char": 90}) + "\n",
        encoding="utf-8",
    )
    assert run_study_graph.load_segmentation(tmp_path, "nawawi") == {
        "nawawi/0": (0, 120),
        "nawawi/1": (0, 90),
    }


def test_load_segmentation_ignores_the_failures_shards(tmp_path):
    """B4: nawawi_seg_failures.*.jsonl lives in the same directory and has no
    matn_end_char -- load_segmentation's glob must not sweep it in."""
    (tmp_path / "nawawi_seg.000-of-001.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/0", "matn_end_char": 50}) + "\n", encoding="utf-8"
    )
    (tmp_path / "nawawi_seg_failures.000-of-001.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/1", "attempts": 2, "error": "no boundary"}) + "\n",
        encoding="utf-8",
    )
    assert run_study_graph.load_segmentation(tmp_path, "nawawi") == {"nawawi/0": (0, 50)}


def test_load_segmentation_reads_a_matn_that_does_not_start_the_bab(tmp_path):
    """93% of ʿAynī's babs gloss the chapter heading before quoting the hadith."""
    (tmp_path / "cayni_seg.000-of-001.jsonl").write_text(
        json.dumps({"bab_ref": "cayni/0", "matn_start_char": 480,
                    "matn_end_char": 1310}) + "\n",
        encoding="utf-8",
    )
    assert run_study_graph.load_segmentation(tmp_path, "cayni") == {"cayni/0": (480, 1310)}


def test_load_segmentation_failures_merges_shards(tmp_path):
    (tmp_path / "nawawi_seg_failures.000-of-002.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/2232", "attempts": 2, "error": "no boundary"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "nawawi_seg_failures.001-of-002.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/2872", "attempts": 2, "error": "no boundary"}) + "\n",
        encoding="utf-8",
    )
    assert run_study_graph.load_segmentation_failures(tmp_path, "nawawi") == frozenset(
        {"nawawi/2232", "nawawi/2872"}
    )


def test_load_segmentation_failures_is_empty_when_no_failures_were_logged(tmp_path):
    assert run_study_graph.load_segmentation_failures(tmp_path, "nawawi") == frozenset()


def test_load_segmentation_failures_does_not_pick_up_ordinary_seg_shards(tmp_path):
    """The failures glob must not collide with the boundary-record glob used by
    load_segmentation() and validate_matched.py."""
    (tmp_path / "nawawi_seg.000-of-001.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/0", "matn_end_char": 50}) + "\n", encoding="utf-8"
    )
    assert run_study_graph.load_segmentation_failures(tmp_path, "nawawi") == frozenset()


def test_review_sheet_renders_the_required_sections():
    sheet = {
        "kitab": "كتاب الإيمان", "n_records": 17, "summary": "the summary",
        "provenance": {"compiler_own": 10, "transmitted": 7, "compiler_own_pct": 58.8},
        "low_confidence_moves": [
            {"bab_ref": "nawawi/3", "operation": "takhsis",
             "confidence": 0.3, "evidence": "ع"}
        ],
        "reread_cap": 2,
    }
    out = render_review_sheet(sheet)
    assert "كتاب الإيمان" in out
    assert "58.8" in out
    assert "nawawi/3" in out
    assert "2" in out


def test_a_missing_batch_naive_file_raises_instead_of_going_live(tmp_path):
    """B3: silence here is a silent change of experimental condition.

    `load_naive` returns {} for an absent file, and {} means the study graph computes
    the naive reading inline with the method hypothesis already in play -- a different
    condition, chosen by accident. `--live-naive` is how you ask for that on purpose.
    """
    from scripts.run_study_graph import require_naive
    with pytest.raises(FileNotFoundError, match="--live-naive"):
        require_naive(tmp_path / "nawawi_naive.jsonl")


def test_a_present_batch_naive_file_is_returned(tmp_path):
    p = tmp_path / "nawawi_naive.jsonl"
    p.write_text('{"bab_ref":"nawawi/0","naive_reading":"r0"}\n', encoding="utf-8")
    from scripts.run_study_graph import require_naive
    assert require_naive(p) == {"nawawi/0": "r0"}


class _Saver:
    def __init__(self, checkpoint):
        self._checkpoint = checkpoint

    def get(self, cfg):
        return self._checkpoint


def test_a_thread_with_no_history_starts_from_the_initial_state():
    assert run_study_graph.has_checkpoint(_Saver(None), {}) is False


def test_a_thread_that_already_has_a_checkpoint_resumes():
    """A shared RUN_TAG restarted the ʿAynī arm from bāb 1 and appended duplicates
    (2026-09-18) because main() passed a fresh initial state on an existing thread.
    graph.invoke(None, cfg) is what resumes; this predicate is what chooses it."""
    assert run_study_graph.has_checkpoint(_Saver({"v": 1, "id": "abc"}), {}) is True
