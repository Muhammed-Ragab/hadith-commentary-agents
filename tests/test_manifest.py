from hermeneutics.manifest import build_manifest, file_sha256, manifest_id


def test_file_sha256_is_stable(tmp_path):
    f = tmp_path / "x.json"
    f.write_text("hello", encoding="utf-8")
    assert file_sha256(f) == file_sha256(f)
    assert len(file_sha256(f)) == 64


def test_file_sha256_changes_with_content(tmp_path):
    f = tmp_path / "x.json"
    f.write_text("hello", encoding="utf-8")
    first = file_sha256(f)
    f.write_text("goodbye", encoding="utf-8")
    assert file_sha256(f) != first


def test_manifest_has_every_required_field():
    m = build_manifest(commentator="nawawi", model_id="Qwen/Qwen2.5-32B-Instruct")
    for key in ("git_sha", "model_id", "prompt_hashes", "corpus_hashes",
                "parser_version", "temperature", "date", "commentator"):
        assert key in m, f"manifest missing {key} (spec §3.4)"
    assert m["temperature"] == 0


def test_manifest_id_is_deterministic_for_same_inputs():
    m1 = build_manifest(commentator="nawawi", model_id="X", date="2026-08-22")
    m2 = build_manifest(commentator="nawawi", model_id="X", date="2026-08-22")
    assert manifest_id(m1) == manifest_id(m2)


def test_manifest_id_changes_when_model_changes():
    m1 = build_manifest(commentator="nawawi", model_id="X", date="2026-08-22")
    m2 = build_manifest(commentator="nawawi", model_id="Y", date="2026-08-22")
    assert manifest_id(m1) != manifest_id(m2)
