"""scripts/tag_moves_llm.py with the LLM mocked throughout -- no network, no API key
needed. Covers JSON parsing/validation, the one-retry policy, final_move merging, and
stratified hand-check sampling."""
from hermeneutics.answer_moves import Unit


def _unit(index, move="other", rule="no_rule_fired", text="نص عادي هنا"):
    return Unit(index=index, text=text, kind="paragraph", move=move, rule=rule)


def test_parse_response_accepts_well_formed_json():
    from scripts.tag_moves_llm import parse_response

    valid, problems = parse_response('{"3": "claim", "7": "other"}', [3, 7])

    assert valid == {3: "claim", 7: "other"}
    assert problems == []


def test_parse_response_strips_markdown_code_fences():
    from scripts.tag_moves_llm import parse_response

    text = "```json\n{\"0\": \"report\"}\n```"
    valid, problems = parse_response(text, [0])

    assert valid == {0: "report"}
    assert problems == []


def test_parse_response_rejects_unknown_move_label():
    from scripts.tag_moves_llm import parse_response

    valid, problems = parse_response('{"0": "not_a_real_move"}', [0])

    assert valid == {}
    assert any("unknown label" in p for p in problems)


def test_parse_response_rejects_unexpected_index():
    from scripts.tag_moves_llm import parse_response

    valid, problems = parse_response('{"0": "claim", "99": "claim"}', [0])

    assert valid == {0: "claim"}
    assert any("unexpected index: 99" in p for p in problems)


def test_parse_response_flags_missing_indices():
    from scripts.tag_moves_llm import parse_response

    valid, problems = parse_response('{"3": "claim"}', [3, 7])

    assert valid == {3: "claim"}
    assert any("missing indices" in p and "7" in p for p in problems)


def test_parse_response_malformed_json_reports_decode_error():
    from scripts.tag_moves_llm import parse_response

    valid, problems = parse_response("not json at all", [0])

    assert valid == {}
    assert problems and "json_decode_error" in problems[0]


def test_parse_response_non_object_json_is_rejected():
    from scripts.tag_moves_llm import parse_response

    valid, problems = parse_response("[1, 2, 3]", [0])

    assert valid == {}
    assert problems == ["response is not a JSON object"]


def test_merge_final_move_keeps_rule_label_when_not_other():
    from scripts.tag_moves_llm import merge_final_move

    assert merge_final_move("limit", None) == "limit"
    assert merge_final_move("evidence_quote", "claim") == "evidence_quote"  # rule wins


def test_merge_final_move_uses_llm_label_when_rule_was_other():
    from scripts.tag_moves_llm import merge_final_move

    assert merge_final_move("other", "claim") == "claim"
    assert merge_final_move("other", "other") == "other"


def test_merge_final_move_falls_back_to_other_with_no_llm_label():
    from scripts.tag_moves_llm import merge_final_move

    assert merge_final_move("other", None) == "other"
    assert merge_final_move("other", "") == "other"


def test_label_answer_no_other_units_makes_no_call():
    from scripts.tag_moves_llm import label_answer

    calls = []
    units = [_unit(0, move="limit", rule="stated_limits_lexicon")]
    result = label_answer(lambda s, u: calls.append(1) or "{}", question="q?", units=units)

    assert result == {"labels": {}, "status": "no_other_units", "attempts": 0, "problems": []}
    assert calls == []


def test_label_answer_succeeds_on_first_try():
    from scripts.tag_moves_llm import label_answer

    units = [_unit(0), _unit(1, move="limit", rule="stated_limits_lexicon")]
    result = label_answer(lambda s, u: '{"0": "claim"}', question="q?", units=units)

    assert result["status"] == "ok"
    assert result["attempts"] == 1
    assert result["labels"] == {0: "claim"}
    assert result["problems"] == []


def test_label_answer_retries_once_then_succeeds():
    from scripts.tag_moves_llm import label_answer

    units = [_unit(0), _unit(1)]
    responses = iter(['{"0": "claim"}', '{"0": "claim", "1": "report"}'])
    calls = []

    def fake_call(system, user):
        calls.append(user)
        return next(responses)

    result = label_answer(fake_call, question="q?", units=units, max_retries=1)

    assert result["status"] == "ok"
    assert result["attempts"] == 2
    assert result["labels"] == {0: "claim", 1: "report"}
    assert len(calls) == 2
    # the retry prompt must name the problem so the model can actually fix it
    assert "missing indices" in calls[1]


def test_label_answer_exhausts_retries_and_reports_failed_validation():
    from scripts.tag_moves_llm import label_answer

    units = [_unit(0), _unit(1)]
    calls = []

    def always_bad(system, user):
        calls.append(user)
        return '{"0": "claim"}'  # unit 1 always missing

    result = label_answer(always_bad, question="q?", units=units, max_retries=1)

    assert result["status"] == "failed_validation"
    assert result["attempts"] == 2
    assert len(calls) == 2
    # partial result is preserved so the caller can still use what validated
    assert result["labels"] == {0: "claim"}


def test_run_llm_tagging_merges_final_move_and_tracks_tokens(monkeypatch):
    import scripts.tag_moves_llm as tml

    monkeypatch.setattr(tml, "call_model", lambda *a, **k: '{"0": "claim"}')
    monkeypatch.setattr(tml, "_read_last_trace_usage",
                        lambda path: {"tokens_in": 100, "tokens_out": 10})

    lexicons = {"stated_limits": [], "move_markers": {}, "attribution_verbs": []}
    records = [{"id": "agent_x", "question": "سؤال؟",
               "answer": "نص عادي بلا أي علامة هنا على الإطلاق تماما"}]

    class _FakeTrace:
        path = None

    result = tml.run_llm_tagging(records, lexicons, model=object(), model_id="m",
                                 trace=_FakeTrace(), price_in=1.0, price_out=5.0,
                                 cost_cap=0.60)

    assert result["calls"] == 1
    assert result["tokens_in"] == 100
    assert result["tokens_out"] == 10
    assert result["cost"] == 100 / 1_000_000 * 1.0 + 10 / 1_000_000 * 5.0
    final_moves = {r["unit_index"]: r["final_move"] for r in result["rows"]}
    assert final_moves[0] == "claim"


def test_run_llm_tagging_respects_cost_cap(monkeypatch):
    """A cap set below even one call's estimated cost must skip every answer and make
    zero calls -- the pre-call guard, not just a post-hoc report."""
    import scripts.tag_moves_llm as tml

    calls = []
    monkeypatch.setattr(tml, "call_model", lambda *a, **k: calls.append(1) or '{"0": "claim"}')

    lexicons = {"stated_limits": [], "move_markers": {}, "attribution_verbs": []}
    records = [{"id": "agent_x", "question": "سؤال؟",
               "answer": "نص طويل جدا بلا أي علامة خاصة " * 50}]

    class _FakeTrace:
        path = None

    result = tml.run_llm_tagging(records, lexicons, model=object(), model_id="m",
                                 trace=_FakeTrace(), price_in=1.0, price_out=5.0,
                                 cost_cap=0.0)

    assert calls == []
    assert result["aborted"] is True
    assert result["skipped_ids"] == ["agent_x"]
    assert all(r["final_move"] == "other" for r in result["rows"])


def test_stratified_sample_draws_up_to_n_per_system_and_is_seed_stable():
    from scripts.tag_moves_llm import stratified_sample

    rows = ([{"id": f"agent_q{i}"} for i in range(20)] +
            [{"id": f"luna_q{i}"} for i in range(10)])

    sample1 = stratified_sample(rows, seed=42, n_per_system=15)
    sample2 = stratified_sample(rows, seed=42, n_per_system=15)

    agent_ids = [r["id"] for r in sample1 if r["id"].startswith("agent_")]
    luna_ids = [r["id"] for r in sample1 if r["id"].startswith("luna_")]

    assert len(agent_ids) == 15
    assert len(luna_ids) == 10  # fewer than n_per_system available -> takes all of them
    assert sample1 == sample2  # same seed, same draw


def test_stratified_sample_only_draws_from_given_pool_not_all_rows():
    """Rows not belonging to "agent"/"luna" ids (or already excluded upstream, e.g. a
    cost-cap skip with no llm_move) must never appear in the sample."""
    from scripts.tag_moves_llm import stratified_sample

    rows = [{"id": "agent_a"}, {"id": "luna_b"}, {"id": "weird_c"}]
    sample = stratified_sample(rows, seed=1, n_per_system=15)

    assert all(r["id"] in ("agent_a", "luna_b") for r in sample)


def test_prompt_template_contains_every_move_and_hash_is_stable():
    from hermeneutics.answer_moves import MOVES
    from scripts.tag_moves_llm import PROMPT_HASH, prompt_template

    template = prompt_template()

    assert all(m in template for m in MOVES)
    assert PROMPT_HASH == __import__("hashlib").sha256(
        template.encode("utf-8")).hexdigest()[:16]


def test_render_answer_prompt_targets_only_other_units():
    from scripts.tag_moves_llm import render_answer_prompt

    units = [_unit(0, move="other"), _unit(1, move="limit", rule="stated_limits_lexicon"),
             _unit(2, move="other")]
    user, target = render_answer_prompt("سؤال؟", units)

    assert target == [0, 2]
    assert "NEEDS LABEL" in user
    assert "do not relabel" in user
