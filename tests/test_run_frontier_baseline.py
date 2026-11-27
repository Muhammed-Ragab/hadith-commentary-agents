"""The frontier baseline is one plain call, checked against the whole corpus. Offline:
a fake model stands in for OpenRouter, and a tiny fixture corpus stands in for the real
one (same shape CorpusTools already expects -- data/*_raw.json)."""
import json

import pytest

from hermeneutics import conditions
from scripts.run_frontier_baseline import make_run_id, run


@pytest.fixture
def corpus_root(tmp_path):
    """A one-hadith fixture corpus: enough for verify_quote to find one real quotation
    and correctly fail a fabricated one."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fixture_raw.json").write_text(
        json.dumps([{"hadith_number": "1", "matn": "إنما الأعمال بالنيات"}],
                   ensure_ascii=False),
        encoding="utf-8")
    return tmp_path


ANSWER = ("جاء في الحديث «إنما الأعمال بالنيات» وهذا يدل على أهمية النية، "
         "أما «هذا الكلام غير موجود إطلاقا في أي مصدر» فلا أصل له.")


def test_make_run_id_uses_model_slug_and_utc_stamp():
    import time

    run_id = make_run_id("openai/gpt-5.6-luna", when=time.gmtime(0))
    assert run_id == "frontier_openai_gpt-5.6-luna_19700101T000000"


def test_run_records_two_quotations_one_verified_answer_unmodified_system_is_base(
        fake_model, corpus_root):
    model = fake_model([ANSWER])

    out_dir = run(model, {"q1_ray": "ما حكم الرأي والقياس؟"}, model_id="openai/gpt-5.6-luna",
                 provider_route="price", max_tokens=4000, root=corpus_root,
                 run_id="frontier_test")

    record = json.loads((out_dir / "q1_ray.json").read_text(encoding="utf-8"))

    assert record["quotations"] == 2
    assert record["verified"] == 1
    assert record["answer"] == ANSWER, "nothing may be stripped from the answer"
    assert record["system"] == conditions.BASE
    assert record["qid"] == "q1_ray"
    assert record["model_id"] == "openai/gpt-5.6-luna"
    assert record["provider"] == "openrouter"
    assert record["provider_route"] == "price"
    assert record["max_tokens"] == 4000
    assert len(record["grounding_log"]) == 2
    verified_entries = [g for g in record["grounding_log"] if g["verified"]]
    assert verified_entries[0]["quote"] == "إنما الأعمال بالنيات"
    assert verified_entries[0]["found_in"], "a verified quote must record where it was found"


def test_run_writes_an_index_and_a_trace(fake_model, corpus_root):
    model = fake_model([ANSWER])

    out_dir = run(model, {"q1_ray": "ما حكم الرأي والقياس؟"}, model_id="openai/gpt-5.6-luna",
                 provider_route="price", max_tokens=4000, root=corpus_root,
                 run_id="frontier_test2")

    assert (out_dir / "trace.jsonl").exists()
    index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    assert index["questions"]["q1_ray"] == {"quotations": 2, "verified": 1}
    assert index["run_id"] == "frontier_test2"


def test_run_prints_one_line_per_question(fake_model, corpus_root, capsys):
    model = fake_model([ANSWER])

    run(model, {"q1_ray": "ما حكم الرأي والقياس؟"}, model_id="openai/gpt-5.6-luna",
       provider_route="price", max_tokens=4000, root=corpus_root, run_id="frontier_test3")

    assert "q1_ray: 2 quotations, 1 verified" in capsys.readouterr().out
