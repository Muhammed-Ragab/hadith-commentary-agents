"""Skip integration checks only when their unpublished fixtures are absent."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_BABS = [f"data/babs/{name}_babs.json" for name in ("nawawi", "cayni", "mazandarani")]
_CORPUS_TESTS = {
    "test_chunking.py": "test_count_chunks_matches_spec_fanout_for_cayni",
    "test_corpus_guard.py": "test_corpus_counts_match_spec test_totals test_sunni_arm_totals_still_hold test_no_arm_leaks_front_matter_or_a_whole_undivided_kitab test_the_shia_arm_starts_on_a_real_bab",
    "test_corpus_scope.py": "test_every_tradition_scope_file_exists_on_disk test_cayni_scoped_search_all_never_returns_kafi_or_mazandarani test_mazandarani_scoped_search_never_returns_cayni_or_nawawi test_an_unscoped_corpus_tools_still_searches_everything test_verify_quote_is_not_scope_filtered test_resolve_scope_lists_the_concrete_files_for_the_manifest test_outside_scope_false_for_a_quotation_found_in_the_runs_own_files test_outside_scope_false_for_shared_lexica_or_quran",
    "test_corpus_tools_move.py": "test_the_named_corpus_files_exist test_a_hadith_is_cited_by_its_number_not_by_its_filename test_verify_quote_accepts_a_quotation_free_of_corpus_line_wraps test_verify_quote_finds_a_bab_heading test_a_heading_record_cannot_be_matched_across_its_junction_with_the_body test_a_commentary_hit_carries_the_commentators_arabic_name",
    "test_inquiry_graph.py": "test_the_gate_flags_outside_scope_for_a_verified_quotation_found_only_in_kafi test_the_gate_does_not_flag_outside_scope_for_the_runs_own_commentary",
    "test_review_cadence.py": "test_configured_cadence_lands_near_the_pre_reparse_budget test_grouping_does_not_shrink_the_reread_budget",
}
_OTHER_TESTS = {
    "test_export_metrics.py": ("data/bukhari_sample.json", "test_build_and_export_writes_a_real_isnad_graph_from_the_sample test_build_and_export_is_deterministic_across_two_runs test_a_known_narrator_appears_with_real_metrics test_an_invented_name_never_appears_in_the_graph"),
    "test_parse_muslim.py": ("data/openiti/muslim_raw.json", "test_cluster_refs_are_unique_across_the_corpus test_transmission_refs_are_unique_across_the_corpus"),
    "test_quran_corpus_quality.py": ("data/quran/quran_roots.json", "*"),
    "test_humain_comparison_page.py": ("outputs/comparison/20260927T153436Z/review", "*"),
    "test_inquiry_session.py": ("outputs/networks/muslim_metrics.json", "test_a_default_run_params_manifest_records_no_provider_route test_the_provider_and_route_reach_the_manifest_and_session test_the_chosen_model_reaches_the_manifest_and_its_id test_corpus_follows_the_commentator_unless_given test_session_json_carries_model_and_resumes test_resume_after_a_failure_restarts_the_failed_stage_not_the_run"),
    "test_ui_runs.py": ("outputs/networks/muslim_metrics.json", "test_a_gated_run_waits_at_each_stage_and_records_the_note test_auto_approve_runs_straight_through test_one_run_at_a_time test_a_failed_run_resumes_from_its_checkpoint test_resume_refuses_a_run_that_did_not_stop test_a_run_left_active_by_a_dead_server_comes_back_interrupted test_cancel_at_a_gate_writes_no_session"),
    "test_serve_ui.py": ("outputs/networks/muslim_metrics.json", "test_index_is_served test_models_are_cached_until_refreshed test_models_warning_is_passed_through test_a_full_gated_run_over_http test_second_start_is_refused_while_a_run_is_active test_bad_input_is_a_400 test_decision_on_a_run_that_is_not_waiting_is_a_409 test_unknown_run_and_path_tricks_are_404 test_the_key_never_reaches_the_page"),
    "test_run_inquiry.py": ("outputs/method_files_matched/matched_qwen3_20260917", "test_the_method_path_points_at_a_file_that_exists"),
    "test_run_inquiry_e2e.py": ("outputs/networks/muslim_metrics.json", "*"),
}


def pytest_collection_modifyitems(items):
    for item in items:
        filename = item.path.name
        name = item.originalname or item.name
        if name in _CORPUS_TESTS.get(filename, "").split():
            item.add_marker(pytest.mark.skipif(
                not all((ROOT / path).exists() for path in _BABS),
                reason="Excluded commentary corpus is unavailable",
            ))
        if filename in _OTHER_TESTS:
            path, names = _OTHER_TESTS[filename]
            if names == "*" or name in names.split():
                item.add_marker(pytest.mark.skipif(
                    not (ROOT / path).exists(), reason=f"Excluded fixture unavailable: {path}",
                ))


"""Shared fixtures. No test in this suite may call a live model."""
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


@pytest.fixture
def fake_model():
    """Factory: fake_model(["reply1", "reply2"]) yields them in order."""

    def _make(responses: list[str]):
        return GenericFakeChatModel(messages=iter(responses))

    return _make


@pytest.fixture
def failing_model():
    """Factory: failing_model(["reply1"]) answers those in order, then raises the way a
    SAIA 400 does. Exhausting a plain fake raises StopIteration, which is not what a
    real failure looks like and can be swallowed by generator machinery."""

    def _make(responses: list[str]):
        def _replies():
            yield from responses
            raise ConnectionError("SAIA returned 400: model does not support tools")

        return GenericFakeChatModel(messages=_replies())

    return _make


@pytest.fixture
def trace(tmp_path):
    from hermeneutics.trace import TraceWriter

    return TraceWriter(run_dir=tmp_path, phase="test")
