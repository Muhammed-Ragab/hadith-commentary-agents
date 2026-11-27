import json

from scripts.run_naive_batch import build_parser, merge_shards, naive_one, resolve_run_id


def test_naive_one_returns_a_record(fake_model, trace):
    rec = naive_one({"kitab": "K", "bab": "b", "text": "ا" * 300},
                    "nawawi/0", fake_model(["my naive reading"]), trace, "m1")
    assert rec == {"bab_ref": "nawawi/0", "naive_reading": "my naive reading"}


def test_naive_one_prompt_contains_no_method_hypothesis(fake_model, trace, tmp_path):
    """This is why the batch phase exists at all."""
    naive_one({"kitab": "K", "bab": "b", "text": "MATN"}, "nawawi/0",
              fake_model(["r"]), trace, "m1")
    rec = json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8"))
    assert "MATN" in rec["prompt"]
    assert "Operations observed so far" not in rec["prompt"]


def test_merge_shards_concatenates_in_bab_order(tmp_path):
    (tmp_path / "nawawi_naive.001-of-008.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/10", "naive_reading": "b"}) + "\n", encoding="utf-8")
    (tmp_path / "nawawi_naive.000-of-008.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/2", "naive_reading": "a"}) + "\n", encoding="utf-8")

    out = merge_shards(tmp_path, "nawawi")
    refs = [json.loads(line)["bab_ref"]
            for line in out.read_text(encoding="utf-8").strip().split("\n")]
    assert refs == ["nawawi/2", "nawawi/10"]


def test_merge_shards_deduplicates_a_rerun_shard(tmp_path):
    (tmp_path / "nawawi_naive.000-of-001.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/1", "naive_reading": "first"}) + "\n"
        + json.dumps({"bab_ref": "nawawi/1", "naive_reading": "second"}) + "\n",
        encoding="utf-8")
    out = merge_shards(tmp_path, "nawawi")
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["naive_reading"] == "second"  # last write wins


def test_parser_accepts_merge_mode():
    assert build_parser().parse_args(["nawawi", "--merge"]).merge is True


def test_limit_applies_before_sharding():
    """The naive pre-pass must cover exactly the bābs the study run will read."""
    from scripts.run_segmentation import select_units

    items = list(enumerate(range(100)))
    assert len(select_units(items, 25, 0, 1)) == 25
    # limit first, then split: two shards of a 25-bāb limit cover 25, not 50
    a = select_units(items, 25, 0, 2)
    b = select_units(items, 25, 1, 2)
    assert len(a) + len(b) == 25
    assert [i for i, _ in a + b] == list(range(25))
    assert len(select_units(items, None, 0, 1)) == 100


def test_naive_parser_accepts_limit():
    from scripts.run_naive_batch import build_parser

    assert build_parser().parse_args(["nawawi", "--limit", "25"]).limit == 25
    assert build_parser().parse_args(["nawawi"]).limit is None


def test_parser_accepts_run_id_on_merge():
    args = build_parser().parse_args(["nawawi", "--merge", "--run-id", "run_a"])
    assert args.merge is True
    assert args.run_id == "run_a"


def test_resolve_run_id_uses_explicit_flag():
    assert resolve_run_id("nawawi", "run_a") == "run_a"


def test_resolve_run_id_defaults_to_manifest_id(monkeypatch):
    """Absent --run-id, output still lands under a per-run directory, not a shared one."""
    from hermeneutics.manifest import build_manifest, manifest_id

    monkeypatch.setenv("VLLM_MODEL_ID", "some-model")
    expected = manifest_id(build_manifest(commentator="nawawi", model_id="some-model"))
    assert resolve_run_id("nawawi", None) == expected


def test_two_runs_do_not_collide_on_shard_files_or_merge(tmp_path):
    """The actual defect: concurrent runs of the same commentator must not see
    each other's shards, and --merge of one run must not pick up the other's."""
    run_a_dir = tmp_path / "run_a"
    run_b_dir = tmp_path / "run_b"
    run_a_dir.mkdir()
    run_b_dir.mkdir()

    # Same commentator, same shard index, different runs -- the exact collision
    # that six concurrent shards per arm across two models would otherwise hit.
    (run_a_dir / "nawawi_naive.000-of-001.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/0", "naive_reading": "from run a"}) + "\n",
        encoding="utf-8")
    (run_b_dir / "nawawi_naive.000-of-001.jsonl").write_text(
        json.dumps({"bab_ref": "nawawi/0", "naive_reading": "from run b"}) + "\n",
        encoding="utf-8")

    out_a = merge_shards(run_a_dir, "nawawi")
    rec_a = json.loads(out_a.read_text(encoding="utf-8").strip())
    assert rec_a["naive_reading"] == "from run a"

    out_b = merge_shards(run_b_dir, "nawawi")
    rec_b = json.loads(out_b.read_text(encoding="utf-8").strip())
    assert rec_b["naive_reading"] == "from run b"

    # merging run A must not have written anything into run B's directory
    assert not (run_b_dir / "nawawi_naive.jsonl").read_text(encoding="utf-8").count("from run a")
