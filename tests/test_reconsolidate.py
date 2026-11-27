import json

from scripts.reconsolidate import build_parser, kitab_summaries_from_trace


def _trace(tmp_path, records):
    p = tmp_path / "study_nawawi.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                 encoding="utf-8")
    return p


def test_summaries_are_keyed_by_kitab(tmp_path):
    p = _trace(tmp_path, [
        {"node": "consolidate_kitab", "bab_ref": "nawawi/كتاب الإيمان", "response": "s1"},
        {"node": "consolidate_kitab", "bab_ref": "nawawi/كتاب السلام", "response": "s2"},
    ])
    assert kitab_summaries_from_trace(p, "nawawi") == {
        "كتاب الإيمان": "s1", "كتاب السلام": "s2",
    }


def test_other_nodes_are_ignored(tmp_path):
    p = _trace(tmp_path, [
        {"node": "read_commentary", "bab_ref": "nawawi/3", "response": "not a summary"},
        {"node": "consolidate_global", "bab_ref": "nawawi/global", "response": "neither"},
        {"node": "consolidate_kitab", "bab_ref": "nawawi/k", "response": "s"},
    ])
    assert kitab_summaries_from_trace(p, "nawawi") == {"k": "s"}


def test_a_kitab_summarised_in_chunks_is_rejoined_in_order(tmp_path):
    """consolidate_kitab chunks notes that do not fit one call and joins the partials
    with a blank line; recovery has to reproduce that, not keep only the last."""
    p = _trace(tmp_path, [
        {"node": "consolidate_kitab", "bab_ref": "nawawi/k", "response": "part one"},
        {"node": "consolidate_kitab", "bab_ref": "nawawi/k", "response": "part two"},
    ])
    assert kitab_summaries_from_trace(p, "nawawi") == {"k": "part one\n\npart two"}


def test_a_failed_attempt_is_dropped_so_a_retry_does_not_duplicate(tmp_path):
    p = _trace(tmp_path, [
        {"node": "consolidate_kitab", "bab_ref": "nawawi/k", "response": "",
         "attempt": 1, "error": "length"},
        {"node": "consolidate_kitab", "bab_ref": "nawawi/k", "response": "the real one",
         "attempt": 2, "error": None},
    ])
    assert kitab_summaries_from_trace(p, "nawawi") == {"k": "the real one"}


def test_an_empty_response_is_not_a_summary(tmp_path):
    """A thinking model that spends its budget before answering returns a successful
    call with empty content (§8.17); joining that in would blank part of a kitāb."""
    p = _trace(tmp_path, [
        {"node": "consolidate_kitab", "bab_ref": "nawawi/k", "response": "   "},
        {"node": "consolidate_kitab", "bab_ref": "nawawi/k", "response": "real"},
    ])
    assert kitab_summaries_from_trace(p, "nawawi") == {"k": "real"}


def test_parser_defaults_to_a_live_run(tmp_path):
    args = build_parser().parse_args(["nawawi", "--trace", "t.jsonl", "--notes", "n.jsonl"])
    assert args.dry_run is False
    assert args.out is None
