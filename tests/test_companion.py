"""Companion PDF: data loading, verbatim rendering, index numbers.

The PDF's claim is that every answer is printed exactly as the run stored it, with the
gate's verdicts as recorded. These tests pin that claim; the rest is layout.
"""
import json
import re
import sys
from html import unescape
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_session_view import SCAR
from scripts import companion_data as cd


def _entry(quote, verified, **flags):
    e = {"quote": quote, "verified": verified, "elided": False, "checked": 1,
         "in_session": verified, "found_in": ["commentary:cayni_babs.json"] if verified else [],
         "foreign_commentary": False, "outside_scope": False, "echoes_question": False,
         "ts": "2026-09-27T14:00:00+00:00"}
    e.update(flags)
    return e


@pytest.fixture
def outputs(tmp_path):
    out = tmp_path / "outputs"
    (out / "sessions").mkdir(parents=True)
    index = {}
    sessions = {
        "set_q1_ray_cayni": ("bukhari_agent_method_20260924T174004", "cayni",
                             "ما حكم الرأي؟", f"## الجواب\nقال «فهو محمود» ثم «{SCAR}» انتهى.",
                             [_entry("فهو محمود", True), _entry("كلام مخترع", False)]),
        "set_q1_ray_en_mazandarani": ("kafi_agent_method_20260924T181015", "mazandarani",
                                      "What is ra'y?", "Answer «إن دين الله لا يصاب بالقياس».",
                                      [_entry("إن دين الله لا يصاب بالقياس", True,
                                              echoes_question=True)]),
        "set_q10_salat_cayni": ("bukhari_agent_method_20260927T141350", "cayni",
                                "حديث يقطع الصلاة", "نص الجواب", []),
    }
    for key, (stem, comm, q, syn, log) in sessions.items():
        path = out / "sessions" / f"{stem}.json"
        path.write_text(json.dumps({
            "question": q, "question_ar": "سؤال قانوني", "search_terms": ["رأي", "قياس"],
            "question_lang": "x", "commentator": comm, "model_id": "qwen/qwen3.8-27b",
            "provider": "openrouter", "provider_route": "deepinfra", "quantization": "bf16",
            "synthesis_output": syn, "grounding_log": log,
            "tool_calls_log": [{"tool": "search_sources", "args": {"query": "رأي", "source": "all"},
                                "stage": "isnad"},
                               {"tool": "verify_quote", "args": {"quote": "فهو محمود"},
                                "stage": "synthesis"}],
        }, ensure_ascii=False), encoding="utf-8")
        index[key] = {"question": q, "exit": "0", "session": str(path)}
        (out / "runs" / key).mkdir(parents=True)
        (out / "runs" / key / "manifest.json").write_text(json.dumps(
            {"temperature": 0, "date": "2026-09-2" + stem[-8], "git_sha": "abc"}), encoding="utf-8")
    (out / "question_set").mkdir()
    (out / "question_set" / "index.json").write_text(json.dumps(index, ensure_ascii=False),
                                                    encoding="utf-8")
    for run, qids in (("frontier_openai_gpt-5.6-luna_20260924T161907", ["q1_ray", "q1_ray_en"]),
                      ("frontier_openai_gpt-5.6-luna_20260927T151354", ["q10_salat"])):
        d = out / "frontier" / run
        d.mkdir(parents=True)
        (d / "index.json").write_text(json.dumps({"system": "SYS", "max_tokens": 4000}),
                                      encoding="utf-8")
        (d / "trace.jsonl").write_text("", encoding="utf-8")
        for qid in qids:
            (d / f"{qid}.json").write_text(json.dumps({
                "qid": qid, "question": f"Q {qid}", "model_id": "openai/gpt-5.6-luna",
                "provider": "openrouter", "provider_route": "price", "max_tokens": 4000,
                "system": "SYS", "answer": "Luna says «شبهتمونا بالحمر» and «غير موجود».",
                "grounding_log": [_entry("شبهتمونا بالحمر", True), _entry("غير موجود", False)],
                "quotations": 2, "verified": 1, "tokens_in": 1, "tokens_out": 2,
                "ts": "2026-09-27T15:15:26+00:00"}, ensure_ascii=False), encoding="utf-8")
    return out


def test_parse_key():
    assert cd.parse_key("set_q10_salat_en_mazandarani") == ("q10_salat_en", "q10", "en", "mazandarani")
    assert cd.parse_key("set_q5_taqiyya_colloq_mazandarani") == ("q5_taqiyya_colloq", "q5", "colloq", "mazandarani")
    assert cd.parse_key("set_q1_ray_cayni") == ("q1_ray", "q1", "ar", "cayni")


def test_load_counts_and_order(outputs):
    answers = cd.load_batch(outputs)
    assert len(answers) == 6
    assert [a["aid"] for a in answers] == ["A01", "A02", "A03", "A04", "A05", "A06"]
    # chapter order, then agent before Luna, then commentator, then variant
    assert [(a["chapter"], a["system"], a["commentator"], a["variant"]) for a in answers] == [
        ("q1", "agent", "cayni", "ar"), ("q1", "agent", "mazandarani", "en"),
        ("q1", "luna", None, "ar"), ("q1", "luna", None, "en"),
        ("q10", "agent", "cayni", "ar"), ("q10", "luna", None, "ar")]


def test_counts_are_raw_log_counts(outputs):
    a = cd.load_batch(outputs)[0]
    assert (a["verified"], a["quotations"], a["echoes"]) == (1, 2, 0)
    assert cd.load_batch(outputs)[1]["echoes"] == 1


def test_agent_fields(outputs):
    a = cd.load_batch(outputs)[0]
    assert a["text"].startswith("## الجواب")
    assert a["temperature"] == 0 and a["date"] == "2026-09-24"
    assert a["run_file"] == "bukhari_agent_method_20260924T174004.json"
    assert a["level"] == 1 and len(a["calls"]) == 2


def test_luna_fields(outputs):
    luna = [a for a in cd.load_batch(outputs) if a["system"] == "luna"]
    assert luna[0]["text"].startswith("Luna says")
    assert luna[0]["calls"] == [] and luna[0]["date"] == "2026-09-27"
    assert luna[2]["level"] == 2


def test_corrections_fail_the_build(outputs):
    p = next((outputs / "sessions").glob("*174004.json"))
    s = json.loads(p.read_text(encoding="utf-8"))
    s["corrections"] = [{"from": "a", "to": "b", "count": 1, "field": "synthesis_output"}]
    p.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="corrections"):
        cd.load_batch(outputs)


def test_unpaired_scars_fail_the_build(outputs):
    p = next((outputs / "sessions").glob("*174004.json"))
    s = json.loads(p.read_text(encoding="utf-8"))
    s["grounding_log"] = [_entry("فهو محمود", True)]  # scar with no unverified entry
    p.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="scar"):
        cd.load_batch(outputs)


from scripts import build_companion as bc


def plain_words(html_text: str) -> list:
    """Visible answer words, annotations removed. Tags are deleted, not replaced by a
    space: inline marks sit inside words' punctuation («<span>…</span>»), and md() puts a
    newline between block elements, so word boundaries survive."""
    t = re.sub(r'<(sup|span) class="(ann|why|why-ok)"[^>]*>.*?</\1>', "", html_text, flags=re.S)
    t = re.sub(r"<[^>]+>", "", t)
    return unescape(t).split()


def stored_words(text: str, log: list) -> list:
    """What the run stored, scars replaced by the quotations the gate removed,
    markdown syntax dropped the way md() drops it."""
    removed = iter([e["quote"] for e in log if not e.get("verified")])
    parts = text.split(SCAR)
    text = parts[0] + "".join(next(removed) + p for p in parts[1:])
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("|"):  # pipe table: separator rows and the | characters are layout
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue
            s = " ".join(cells)
            s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
            lines.append(re.sub(r"`(.+?)`", r"\1", s))
            continue
        s = re.sub(r"^#+\s*", "", s)
        s = re.sub(r"^>\s?", "", s)
        s = re.sub(r"^[-*]\s+", "", s)
        # Paired markers only, exactly as md()'s inline(): a stored answer cut off
        # mid-bold ends in a lone "**" that md() correctly leaves in the text.
        s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
        lines.append(re.sub(r"`(.+?)`", r"\1", s))
    return " ".join(lines).split()


def test_agent_body_is_verbatim(outputs):
    for a in cd.load_batch(outputs):
        body = bc.answer_body(a)
        assert plain_words(body) == stored_words(a["text"], a["log"] if a["system"] == "agent" else [])


def test_verified_quote_marked_and_stripped_quote_struck(outputs):
    body = bc.answer_body(cd.load_batch(outputs)[0])
    assert '<span class="q ok" data-n="1">فهو محمود</span>' in body
    assert "<s>كلام مخترع</s>" in body and SCAR not in body


def test_luna_marks_both_verdicts_and_strikes_nothing(outputs):
    luna = [a for a in cd.load_batch(outputs) if a["system"] == "luna"][0]
    body = bc.answer_body(luna)
    assert 'class="q ok"' in body and 'class="q no"' in body and "<s>" not in body


def test_quotation_list_has_every_entry(outputs):
    a = cd.load_batch(outputs)[0]
    table = bc.quotation_list(a)
    assert table.count("<tr class=\"qrow") == len(a["log"])
    assert "removed by the gate" in table


def test_trail_is_unclipped_and_staged(outputs):
    a = cd.load_batch(outputs)[0]
    a["calls"].append({"tool": "verify_quote", "args": {"quote": "س" * 300}, "stage": "matn"})
    trail = bc.compact_trail(a["calls"])
    assert "س" * 300 in trail
    assert trail.index("Isnād") < trail.index("Matn") < trail.index("Synthesis")


def test_every_answer_once_and_front_matter(outputs):
    answers = cd.load_batch(outputs)
    page = bc.build_html(answers, "SYS")
    for a in answers:
        assert page.count(f' id="{a["aid"]}"') == 1
    assert bc.DISCLAIMER in page
    assert page.index(bc.DISCLAIMER) < page.index('class="answer"')
    assert page.index("Index of all answers") > page.rindex('class="answer"')


def test_index_numbers_match_data(outputs):
    answers = cd.load_batch(outputs)
    table = bc.index_table(answers, {})
    for a in answers:
        assert f'{a["verified"]} of {a["quotations"]}' in table
        assert f'<td class="num">{len(a["text"].split())}</td>' in table


def test_header_states_status_and_canonical_question(outputs):
    a = cd.load_batch(outputs)[0]
    h = bc.header(a)
    assert "not reviewed by a ḥadīth specialist" in h
    assert "سؤال قانوني" in h and "temperature 0" in h and a["run_file"] in h


def test_weak_run_note_present():
    assert "weak run" in bc.EDITOR_NOTES["set_q10_salat_en_mazandarani"]


def test_answer_pages_takes_the_second_occurrence(tmp_path):
    import fitz
    from scripts.print_companion import answer_pages
    doc = fitz.open()
    for text in ("[A01] [A02]", "intro", "[A01] answer", "[A02] answer"):
        doc.new_page().insert_text((72, 72), text)
    path = tmp_path / "x.pdf"
    doc.save(path)
    assert answer_pages(path, ["A01", "A02"]) == {"A01": 3, "A02": 4}


def test_answer_pages_refuses_ambiguity(tmp_path):
    import fitz
    from scripts.print_companion import answer_pages
    doc = fitz.open()
    for text in ("[A01]", "[A01]", "[A01]"):
        doc.new_page().insert_text((72, 72), text)
    path = tmp_path / "x.pdf"
    doc.save(path)
    with pytest.raises(RuntimeError):
        answer_pages(path, ["A01"])


def test_pipe_table_becomes_html_table_without_separator_row():
    html_text = bc.md_with_tables("قبل\n| a | ب |\n|---|---|\n| 1 | 2 |\nبعد")
    assert html_text.count("<tr>") == 2 and "---" not in html_text
    assert "<td" in html_text and "|" not in html_text
    assert plain_words(html_text) == ["قبل", "a", "ب", "1", "2", "بعد"]


def test_truncated_run_note_isolates_its_arabic(outputs):
    a = cd.load_batch(outputs)[0]
    a = dict(a, key="set_q11_umirtu_cayni")
    section = bc.answer_section(a)
    assert '<bdi class="ar">«حتى</bdi>' in section and "4,000 tokens" in section
