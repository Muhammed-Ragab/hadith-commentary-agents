import json

from scripts.compare_methods import collect_arms, count_failures, distinct_operation_sets, main, profile


def test_profile_reports_no_move_separately_from_technical_output():
    records = [
        {
            "status": "moves",
            "moves": [{"operation": "takhsis", "arabic_evidence": "دليل"}],
        },
        {
            "status": "no_move",
            "moves": [],
            "no_move_reason": "No identifiable operation.",
        },
    ]

    operations, verified, total, no_moves = profile(records)
    assert operations["takhsis"] == 1
    assert (verified, total, no_moves) == (1, 1, 1)


def test_count_failures_sums_graph_and_segmentation_failures(tmp_path):
    """B4: horizon_diff failures live under <arm>/<name>_failures.jsonl; segment_one
    failures live under segmentation/ instead, since that pass runs before the arm
    directory exists. Both are declared shortfall and must both count."""
    arm = tmp_path / "nawawi"
    arm.mkdir()
    (arm / "nawawi_failures.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/5", "error": "unknown evidence_id"}) + "\n",
        encoding="utf-8",
    )
    seg = tmp_path / "segmentation"
    seg.mkdir()
    (seg / "nawawi_seg_failures.000-of-001.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/1", "error": "no credible boundary"}) + "\n",
        encoding="utf-8",
    )
    assert count_failures(tmp_path, "nawawi") == 2


def test_count_failures_is_zero_for_a_complete_arm(tmp_path):
    (tmp_path / "nawawi").mkdir()
    (tmp_path / "segmentation").mkdir()
    assert count_failures(tmp_path, "nawawi") == 0


def test_main_flags_an_arm_short_of_a_full_run(tmp_path, capsys):
    arm = tmp_path / "nawawi"
    arm.mkdir()
    (arm / "nawawi_notes.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/0", "status": "moves",
                    "moves": [{"operation": "takhsis", "arabic_evidence": "دليل"}]}) + "\n",
        encoding="utf-8",
    )
    (arm / "nawawi_failures.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/1", "error": "no credible boundary"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "segmentation").mkdir()

    import sys
    old_argv = sys.argv
    sys.argv = ["compare_methods.py", str(tmp_path)]
    try:
        assert main() == 0
    finally:
        sys.argv = old_argv

    out = capsys.readouterr().out
    assert "SHORT of a full run" in out
    assert "nawawi" in out


# --------------------------------------------------------------- S3: rescoring

def test_a_qualifier_written_before_the_schema_split_is_still_folded():
    """S3. `4767842` split the trailing scope note into `operation_qualifier`, but only
    for records written AFTER it. Every note file on disk from 15789363 and earlier
    still carries the decorated string, and `profile` read `operation` raw -- so those
    runs were scored the old way and their numbers are not comparable with new ones.
    Fold at read time so one scorer gives one answer for both shapes.
    """
    records = [{"status": "moves", "moves": [
        {"operation": "attribution of transmitted interpretation", "arabic_evidence": "ن"},
        {"operation": "attribution of transmitted interpretation (Islam/Iman identity)",
         "arabic_evidence": "ن"},
        {"operation": "attribution of transmitted interpretation (kamāl vs. dharūriyyāt)",
         "arabic_evidence": "ن"},
    ]}]
    operations, _, total, _ = profile(records)
    assert len(operations) == 1, "one operation named three ways is still one operation"
    assert operations["attribution of transmitted interpretation"] == 3
    assert total == 3


def test_an_already_split_record_is_unchanged_by_the_fold():
    records = [{"status": "moves", "moves": [
        {"operation": "takhsis", "operation_qualifier": "zahir vs ijma",
         "arabic_evidence": "ن"},
    ]}]
    operations, _, _, _ = profile(records)
    assert operations["takhsis"] == 1


# --------------------------------------------------------------- S2: saturation

def test_saturation_curve_counts_distinct_operations_as_babs_accumulate():
    """S2: does the induced vocabulary stop growing? A flat tail is the evidence that
    the sample was large enough; a climbing one is a finding, not a reason to run more.
    """
    from scripts.compare_methods import saturation_curve

    records = [
        {"status": "moves", "moves": [{"operation": "a"}]},
        {"status": "moves", "moves": [{"operation": "a"}]},
        {"status": "moves", "moves": [{"operation": "b"}, {"operation": "a"}]},
    ]
    assert saturation_curve(records) == [1, 1, 2]


def test_saturation_curve_counts_a_no_move_bab_as_a_read_bab():
    """A bāb that yielded nothing is still a bāb read; dropping it would shorten the
    x-axis and make the vocabulary look like it saturated sooner than it did."""
    from scripts.compare_methods import saturation_curve

    records = [
        {"status": "moves", "moves": [{"operation": "a"}]},
        {"status": "no_move", "moves": []},
    ]
    assert saturation_curve(records) == [1, 1]


def test_saturation_curve_folds_qualifiers_like_the_profile_does():
    from scripts.compare_methods import saturation_curve

    records = [
        {"status": "moves", "moves": [{"operation": "takhsis"}]},
        {"status": "moves", "moves": [{"operation": "takhsis (zahir vs ijma)"}]},
    ]
    assert saturation_curve(records) == [1, 1]


# ----------------------------------------------------- operation-set count (this task)

def test_distinct_operation_sets_counts_combinations_not_labels():
    """The measured defect: mazandarani had few distinct LABELS (3) but almost every
    bāb recombined the same one or two of them -- 2 distinct operation-SETS across 24
    bābs. `profile`'s `distinct` column alone does not show that collapse."""
    records = [
        {"status": "moves", "moves": [{"operation": "a"}, {"operation": "b"}]},
        {"status": "moves", "moves": [{"operation": "b"}, {"operation": "a"}]},  # same set
        {"status": "moves", "moves": [{"operation": "a"}]},  # a different set
        {"status": "no_move", "moves": []},  # no set contributed
    ]
    n_sets, n_babs = distinct_operation_sets(records)
    assert (n_sets, n_babs) == (2, 3)


def test_distinct_operation_sets_folds_qualifiers():
    records = [
        {"status": "moves", "moves": [{"operation": "takhsis"}]},
        {"status": "moves", "moves": [{"operation": "takhsis (zahir vs ijma)"}]},
    ]
    assert distinct_operation_sets(records) == (1, 2)


# --------------------------------------------------- ablation (multi-root) reporting

def test_collect_arms_reads_the_ablation_layout_across_multiple_roots(tmp_path):
    """scripts/run_ablation.sh writes one commentator's notes under a per-condition
    --notes-dir, so there is no arm-named subdirectory (the bake-off shape). Passing
    multiple condition roots must label each arm by its root's own directory name."""
    for condition in ("A", "B", "C"):
        d = tmp_path / condition
        d.mkdir()
        (d / "nawawi_notes.jsonl").write_text(
            json.dumps({"bab_ref": "nawawi/0", "status": "moves",
                        "moves": [{"operation": condition, "arabic_evidence": "ع"}]}) + "\n",
            encoding="utf-8",
        )
    located = collect_arms([tmp_path / c for c in ("A", "B", "C")])
    assert set(located) == {"A", "B", "C"}
    _, stem, records = located["B"]
    assert stem == "nawawi"
    assert records[0]["moves"][0]["operation"] == "B"


def test_main_prints_the_ablation_conditions_side_by_side(tmp_path, capsys):
    for condition in ("A", "B", "C"):
        d = tmp_path / condition
        d.mkdir()
        (d / "nawawi_notes.jsonl").write_text(
            json.dumps({"bab_ref": "nawawi/0", "status": "moves",
                        "moves": [{"operation": condition, "arabic_evidence": "ع"}]}) + "\n",
            encoding="utf-8",
        )

    import sys
    old_argv = sys.argv
    sys.argv = ["compare_methods.py", str(tmp_path / "A"), str(tmp_path / "B"),
                str(tmp_path / "C")]
    try:
        assert main() == 0
    finally:
        sys.argv = old_argv

    out = capsys.readouterr().out
    assert "A" in out and "B" in out and "C" in out
