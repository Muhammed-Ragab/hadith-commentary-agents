import json

import pytest

from scripts.run_inquiry import (build_parser, default_corpus, load_framework,
                                 load_passages_for, method_path)


def test_all_four_conditions_are_reachable_from_the_cli():
    for condition in ("no_sources", "passages", "agent", "agent_method"):
        args = build_parser().parse_args(["-q", "x", "--condition", condition])
        assert args.condition == condition


def test_the_old_ab_flags_are_gone():
    """--condition B loaded the empty string and was reported as the abstract's second
    setting. It was neither setting: it was 'tools, no method file'."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-q", "x", "--condition", "B"])


def test_muslim_is_reachable_as_a_corpus():
    assert build_parser().parse_args(["-q", "x", "--corpus", "muslim"]).corpus == "muslim"


def test_parser_rejects_an_unknown_corpus():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-q", "x", "--corpus", "muwatta"])


def test_corpus_defaults_to_the_one_the_commentator_commentates():
    """al-Nawawi's Minhaj is a commentary on MUSLIM. Running his method file against
    Bukhari pairs a method with a corpus it was never derived from."""
    assert default_corpus("nawawi") == "muslim"
    assert default_corpus("cayni") == "bukhari"
    assert default_corpus("mazandarani") == "kafi"


def test_an_explicit_corpus_still_wins():
    args = build_parser().parse_args(
        ["-q", "x", "--commentator", "nawawi", "--corpus", "bukhari"])
    assert args.corpus == "bukhari"


def test_corpus_is_none_when_not_given_so_the_default_can_apply():
    assert build_parser().parse_args(["-q", "x"]).corpus is None


def test_only_agent_method_loads_the_method_file(tmp_path):
    mf = tmp_path / "nawawi_method.md"
    mf.write_text("THE METHOD", encoding="utf-8")

    assert load_framework("agent_method", mf) == "THE METHOD"
    assert load_framework("agent", mf) == ""
    assert load_framework("no_sources", mf) == ""
    assert load_framework("passages", mf) == ""


def test_agent_method_with_no_method_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_study_graph"):
        load_framework("agent_method", tmp_path / "missing.md")


def test_the_method_path_points_at_a_file_that_exists():
    """outputs/method_files/{c}_method.md has never existed. The real files live under
    the study run that produced them."""
    for commentator in ("nawawi", "cayni", "mazandarani"):
        assert method_path(commentator, "matched_qwen3_20260917").is_file()


def test_the_method_path_uses_the_model_copy_not_the_reading_copy():
    """*_method.reading.md is a human reading copy (ed072d4), not the model's input."""
    assert method_path("nawawi", "matched_qwen3_20260917").name == "nawawi_method.md"


def test_the_method_run_tag_is_a_flag_not_a_constant():
    args = build_parser().parse_args(["-q", "x", "--method-run", "some_other_run"])
    assert args.method_run == "some_other_run"


def test_passages_condition_needs_a_passages_file():
    with pytest.raises(ValueError, match="passages"):
        load_passages_for("passages", None, "ما معنى النية")


def test_passages_are_looked_up_by_question(tmp_path):
    pf = tmp_path / "passages.json"
    pf.write_text(json.dumps({"ما معنى النية": ["فقرة أولى", "فقرة ثانية"]},
                             ensure_ascii=False), encoding="utf-8")

    assert load_passages_for("passages", pf, "ما معنى النية") == ["فقرة أولى", "فقرة ثانية"]


def test_conditions_that_need_no_passages_get_none():
    assert load_passages_for("agent", None, "q") is None


def test_question_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
