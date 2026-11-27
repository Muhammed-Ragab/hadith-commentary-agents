import pytest

from scripts.run_control import (build_control_deps_args, build_parser,
                                 resolve_fraction)


def test_parser_requires_a_commentator():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_defaults_to_the_documented_seed():
    args = build_parser().parse_args(["nawawi"])
    assert args.seed == 20260821
    # None, not 0.10: --sample changes what the default should be, so the parser
    # records "unset" and resolve_fraction decides.
    assert args.fraction is None


def test_a_whole_corpus_control_keeps_the_documented_tenth():
    assert resolve_fraction(None, None) == 0.10


def test_a_sampled_control_takes_the_whole_sample():
    """0.10 of 5 bābs per kitāb is 2, so a sampled control left at the corpus default
    would be a third the size of the study arm the 0.80 rule divides it by."""
    assert resolve_fraction(None, "14:5:20260914") == 1.0


def test_an_explicit_fraction_always_wins():
    assert resolve_fraction(0.25, "14:5:20260914") == 0.25
    assert resolve_fraction(0.25, None) == 0.25


def test_the_control_arm_takes_the_study_sample_spec():
    """The study walks a stratified sample; a control drawn from the whole corpus
    compares different kitābs, and 10% of ʿAynī is 396 bābs (~66 h)."""
    args = build_parser().parse_args(["nawawi", "--sample", "14:5:20260914"])
    assert args.sample == "14:5:20260914"


def test_the_control_arm_takes_a_segmentation_dir():
    """_select_bab splits matn from sharh using the PAIRED bāb's boundary, so seg={}
    ran the control on a different footing from the arm it is compared against."""
    args = build_parser().parse_args(["nawawi", "--segmentation-dir", "seg"])
    assert args.segmentation_dir == "seg"


def test_sample_and_segmentation_dir_default_to_none():
    args = build_parser().parse_args(["nawawi"])
    assert args.sample is None
    assert args.segmentation_dir is None
    assert args.notes_dir is None
    assert args.real_method is None


def test_the_isolation_check_takes_its_own_path():
    """Not --notes-dir: compare_methods falls back to glob('*_notes.jsonl'), and
    nawawi_control_notes.jsonl matches it AND sorts before nawawi_notes.jsonl, so a
    control written beside the study arm would be read as the study arm."""
    args = build_parser().parse_args(
        ["nawawi", "--notes-dir", "outputs/control/nawawi",
         "--real-method", "outputs/study/nawawi/nawawi_method.md"])
    assert args.notes_dir == "outputs/control/nawawi"
    assert args.real_method == "outputs/study/nawawi/nawawi_method.md"


def test_control_uses_a_distinct_thread_id_and_checkpointer():
    """Spec §4.1: separate thread id and checkpointer namespace, or the arms leak."""
    real = build_control_deps_args("nawawi", control=False)
    control = build_control_deps_args("nawawi", control=True)

    assert real["thread_id"] != control["thread_id"]
    assert real["db_name"] != control["db_name"]
    assert real["notes_name"] != control["notes_name"]
    assert real["trace_phase"] != control["trace_phase"]


def test_control_notes_go_to_the_control_file():
    assert build_control_deps_args("cayni", control=True)["notes_name"] == (
        "cayni_control_notes.jsonl"
    )


def test_the_control_arm_takes_a_naive_dir():
    """B3: the control arm read outputs/naive/<name>_naive.jsonl, which nothing writes
    any more. Its own run_id is computed from a manifest carrying control_pairing, so
    it can never equal the naive pre-pass's -- the directory has to be passed in."""
    args = build_parser().parse_args(["nawawi", "--naive-dir", "outputs/naive/r1"])
    assert args.naive_dir == "outputs/naive/r1"


def test_naive_dir_defaults_to_none_so_the_run_id_is_used():
    assert build_parser().parse_args(["nawawi"]).naive_dir is None
