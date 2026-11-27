import json

import pytest

import hermeneutics.nodes as nodes
from hermeneutics.nodes import Deps, naive_reading, read_commentary

BAB = {"kitab": "K1", "bab": "b0", "text": "ا" * 400}


def _reading(evidence_id: str, operation: str = "lexical analysis") -> str:
    return json.dumps({
        "status": "moves",
        "observations": [{
            "operation": operation,
            "evidence_id": evidence_id,
            "provenance": {"author": "compiler_own", "quoted_from": []},
            "confidence": 0.8,
        }],
    })


def _deps(fake_model, trace, **kw):
    base = dict(model=fake_model, trace=trace, manifest_id="m1", babs=[BAB],
                commentator="nawawi", naive={}, seg={}, token_len=lambda s: len(s) // 4,
                max_chunk_tokens=6000, pairing=None)
    base.update(kw)
    return Deps(**base)


def test_naive_reading_uses_the_precomputed_batch_result(fake_model, trace):
    deps = _deps(fake_model([]), trace, naive={"nawawi/0": "precomputed reading"})
    out = naive_reading({"current_bab_ref": "nawawi/0", "current_bab": BAB}, deps)
    assert out["naive_reading"] == "precomputed reading"


def test_naive_reading_falls_back_to_a_live_call_when_missing(fake_model, trace):
    deps = _deps(fake_model(["live reading"]), trace, naive={}, seg={"nawawi/0": 100})
    out = naive_reading({"current_bab_ref": "nawawi/0", "current_bab": BAB}, deps)
    assert out["naive_reading"] == "live reading"


def test_naive_reading_refuses_to_send_commentary_when_segmentation_is_missing(
        fake_model, trace):
    deps = _deps(fake_model(["must not be called"]), trace, naive={}, seg={})
    with pytest.raises(ValueError, match="missing segmentation"):
        naive_reading({"current_bab_ref": "nawawi/0", "current_bab": BAB}, deps)


def test_naive_reading_never_sees_the_method_hypothesis(fake_model, trace, tmp_path):
    """Validity: the naive horizon is only naive if nothing leaks in."""
    deps = _deps(fake_model(["r"]), trace, naive={}, seg={"nawawi/0": 100})
    naive_reading({"current_bab_ref": "nawawi/0", "current_bab": BAB,
                   "method_hypothesis": "SECRET METHOD"}, deps)
    line = (tmp_path / "test.jsonl").read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(line)
    assert "SECRET METHOD" not in rec["prompt"]


def test_read_commentary_single_chunk_returns_one_reading(fake_model, trace):
    deps = _deps(fake_model([_reading("nawawi/0:c000:s000")]), trace)
    out = read_commentary({"current_bab_ref": "nawawi/0", "commentary_text": "short text",
                           "method_hypothesis": ""}, deps)
    assert out["commentary_reading"]["status"] == "moves"
    assert out["n_chunks"] == 1


def test_read_commentary_maps_over_chunks_without_losing_observations(fake_model, trace):
    long_text = "alpha beta gamma delta"
    def token_len(text):
        return 1 if text.lstrip().startswith("{") else len(text.split())

    deps = _deps(fake_model([
        json.dumps({
            "status": "no_move",
            "observations": [],
            "no_move_reason": "No move in this span.",
        }),
        _reading("cayni/0:c001:s000", "part B"),
    ]), trace, max_chunk_tokens=3, token_len=token_len)
    out = read_commentary({"current_bab_ref": "cayni/0", "commentary_text": long_text,
                           "method_hypothesis": ""}, deps)
    operations = [o["operation"] for o in out["commentary_reading"]["observations"]]
    assert operations == ["part B"]
    assert out["n_chunks"] == 2


def test_read_commentary_reduces_map_outputs_to_a_bounded_diff_input(fake_model, trace):
    commentary = " ".join("word" for _ in range(900))
    model = fake_model([
        _reading("mazandarani/1:c000:s000", "operation A"),
        _reading("mazandarani/1:c001:s000", "operation B"),
        _reading("mazandarani/1:c002:s000", "operation C"),
        _reading("mazandarani/1:c000:s000", "operation AB"),
    ])
    def token_len(text):
        stripped = text.lstrip()
        return len(text) if stripped.startswith("{") or stripped.startswith("[{") \
            else len(text.split())

    deps = _deps(model, trace, max_chunk_tokens=500, token_len=token_len)
    out = read_commentary({"current_bab_ref": "mazandarani/1",
                           "commentary_text": commentary,
                           "method_hypothesis": ""}, deps)
    assert len(out["commentary_reading"]["observations"]) == 1
    assert out["n_chunks"] == 3
    assert out["n_reduce_calls"] == 1
    selected_tokens = sum(
        deps.token_len(out["evidence_catalog"][observation["evidence_id"]])
        for observation in out["commentary_reading"]["observations"]
    )
    reading_tokens = deps.token_len(json.dumps(out["commentary_reading"]))
    assert reading_tokens + selected_tokens <= deps.max_chunk_tokens


def test_reduce_commentary_retries_a_nonshrinking_result(fake_model, trace):
    observations = [
        {
            "operation": "lexical analysis",
            "evidence_id": "mazandarani/0:c000:s000",
            "provenance": {"author": "compiler_own", "quoted_from": []},
            "confidence": 0.8,
        },
        {
            "operation": "grammatical analysis",
            "evidence_id": "mazandarani/0:c000:s001",
            "provenance": {"author": "compiler_own", "quoted_from": []},
            "confidence": 0.8,
        },
    ]
    model = fake_model([
        json.dumps({"status": "moves", "observations": observations}),
        _reading("mazandarani/0:c000:s000"),
    ])
    deps = _deps(model, trace)

    reading, error = nodes._reduce_commentary_batch(
        {"current_bab_ref": "mazandarani/0"}, deps, observations, "h1"
    )

    assert error == ""
    assert len(reading.observations) == 1


def test_read_commentary_passes_the_method_hypothesis(fake_model, trace, tmp_path):
    deps = _deps(fake_model([_reading("nawawi/0:c000:s000")]), trace)
    read_commentary({"current_bab_ref": "nawawi/0", "commentary_text": "text",
                     "method_hypothesis": "PRIOR METHOD"}, deps)
    line = (tmp_path / "test.jsonl").read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(line)
    assert "PRIOR METHOD" in rec["prompt"]


def test_read_commentary_on_empty_text_raises(fake_model, trace):
    deps = _deps(fake_model([]), trace)
    with pytest.raises(ValueError, match="no commentary text"):
        read_commentary({"current_bab_ref": "nawawi/0", "commentary_text": "  ",
                         "method_hypothesis": ""}, deps)


def test_evidence_catalog_preserves_exact_source_spans():
    source = "alpha   beta\ngamma delta"
    catalog, batches = nodes.build_evidence_catalog(
        source,
        "nawawi/7",
        token_len=lambda text: len(text.split()),
        max_batch_tokens=4,
        max_span_tokens=2,
    )

    assert list(catalog.values()) == ["alpha   beta", "gamma delta"]
    assert list(catalog) == ["nawawi/7:c000:s000", "nawawi/7:c000:s001"]
    assert all(span in source for span in catalog.values())
    assert len(batches) == 1


def test_read_commentary_returns_structured_observations_with_evidence_catalog(
        fake_model, trace):
    response = json.dumps({
        "status": "moves",
        "observations": [{
            "operation": "lexical analysis",
            "evidence_id": "nawawi/0:c000:s000",
            "provenance": {"author": "compiler_own", "quoted_from": []},
            "confidence": 0.8,
        }],
    })
    deps = _deps(fake_model([response]), trace, token_len=lambda text: len(text.split()))

    out = read_commentary({
        "current_bab_ref": "nawawi/0",
        "commentary_text": "alpha beta",
        "method_hypothesis": "",
    }, deps)

    assert out["commentary_reading"]["status"] == "moves"
    assert out["commentary_reading"]["observations"][0]["evidence_id"] == (
        "nawawi/0:c000:s000"
    )
    assert out["evidence_catalog"]["nawawi/0:c000:s000"] == "alpha beta"


def test_read_commentary_retries_an_unknown_evidence_id(fake_model, trace):
    bad = json.dumps({
        "status": "moves",
        "observations": [{
            "operation": "lexical analysis",
            "evidence_id": "invented",
            "provenance": {"author": "compiler_own", "quoted_from": []},
            "confidence": 0.8,
        }],
    })
    good = json.dumps({
        "status": "moves",
        "observations": [{
            "operation": "lexical analysis",
            "evidence_id": "nawawi/0:c000:s000",
            "provenance": {"author": "compiler_own", "quoted_from": []},
            "confidence": 0.8,
        }],
    })
    deps = _deps(fake_model([bad, good]), trace, token_len=lambda text: len(text.split()))

    out = read_commentary({
        "current_bab_ref": "nawawi/0",
        "commentary_text": "alpha beta",
        "method_hypothesis": "",
    }, deps)

    assert out["commentary_reading"]["observations"][0]["evidence_id"].endswith("s000")
    rows = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    prompts = [row["prompt"] for row in rows if row["node"] == "read_commentary"]
    assert prompts[0] != prompts[1]
    assert "unknown evidence_id" in prompts[1].lower()


def test_read_commentary_keeps_technical_failure_distinct_from_no_move(fake_model, trace):
    bad = json.dumps({
        "status": "moves",
        "observations": [{
            "operation": "lexical analysis",
            "evidence_id": "invented",
            "provenance": {"author": "compiler_own", "quoted_from": []},
            "confidence": 0.8,
        }],
    })
    deps = _deps(fake_model([bad, bad]), trace, token_len=lambda text: len(text.split()))

    out = read_commentary({
        "current_bab_ref": "nawawi/0",
        "commentary_text": "alpha beta",
        "method_hypothesis": "",
    }, deps)

    assert out["commentary_reading"] is None
    assert "unknown evidence_id" in out["commentary_error"].lower()


def test_legacy_example_prompt_toggle_changes_the_rendered_prompt(fake_model, trace):
    """Deps.legacy_example_prompt (mechanism 1, ablation condition A/B) must actually
    switch which template is sent -- otherwise the toggle is a no-op."""
    deps = _deps(fake_model([_reading("nawawi/0:c000:s000")]), trace,
                token_len=lambda text: len(text.split()), legacy_example_prompt=True)
    read_commentary({"current_bab_ref": "nawawi/0", "commentary_text": "alpha beta",
                     "method_hypothesis": ""}, deps)
    rows = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    prompt = next(row["prompt"] for row in rows if row["node"] == "read_commentary")
    assert '"operation":"lexical analysis"' in prompt


def test_default_prompt_neutralizes_the_example_operation(fake_model, trace):
    deps = _deps(fake_model([_reading("nawawi/0:c000:s000")]), trace,
                token_len=lambda text: len(text.split()))
    read_commentary({"current_bab_ref": "nawawi/0", "commentary_text": "alpha beta",
                     "method_hypothesis": ""}, deps)
    rows = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    prompt = next(row["prompt"] for row in rows if row["node"] == "read_commentary")
    assert "lexical analysis" not in prompt
    assert '"operation":"<' in prompt


def test_read_commentary_preserves_explicit_no_move(fake_model, trace):
    response = json.dumps({
        "status": "no_move",
        "observations": [],
        "no_move_reason": "No identifiable operation.",
    })
    deps = _deps(fake_model([response]), trace, token_len=lambda text: len(text.split()))

    out = read_commentary({
        "current_bab_ref": "nawawi/0",
        "commentary_text": "alpha beta",
        "method_hypothesis": "",
    }, deps)

    assert out["commentary_reading"]["status"] == "no_move"
