"""scripts/compare_systems.py: small synthetic fixtures only -- no big corpus files, no
model downloads. NER and embeddings are mocked/skipped (see the module docstrings on
try_camel_ner and compute_similarity for why they are optional at runtime)."""
import json
from pathlib import Path

LEXICONS = json.loads(
    (Path(__file__).resolve().parent.parent / "scripts" / "compare_lexicons.json")
    .read_text(encoding="utf-8"))


def test_structure_counts_counts_each_block_kind():
    from scripts.compare_systems import structure_counts

    answer = "# عنوان\nفقرة أولى هنا بعدة كلمات.\n\n- بند أول\n- بند ثان\n\n> نص مقتبس هنا"
    counts = structure_counts(answer)

    assert counts["headings"] == 1
    assert counts["paragraphs"] == 1
    assert counts["list_items"] == 2
    assert counts["blockquotes"] == 1
    assert counts["words"] > 0


def test_commentator_mentions_counts_variants_without_double_counting_substrings():
    """A longer variant like "al-Nawawi" must not also be counted through a shorter
    variant it contains ("Nawawi") -- see _count_terms' non-overlapping alternation."""
    from scripts.compare_systems import commentator_mentions

    text = "قال العيني كذا وذكر al-Nawawi كذلك"
    mentions = commentator_mentions(text, LEXICONS)

    assert mentions["mentions_cayni"] == 1
    assert mentions["mentions_nawawi"] == 1
    assert mentions["mentions_mazandarani"] == 0
    assert mentions["mentions_total"] == 2


def test_stated_limits_count_matches_lexicon_phrase():
    from scripts.compare_systems import stated_limits_count

    assert stated_limits_count("لم أجد هذا الحديث في المصادر المتوفرة", LEXICONS) == 1
    assert stated_limits_count("هذا الحديث صحيح ومشهور", LEXICONS) == 0


def test_key_variant_matching_ignores_diacritics():
    """Level-2 gold-variant recall matches on normalize_arabic, so a diacritized answer
    still hits an undiacritized gold surface form."""
    from hermeneutics.grounding import normalize_arabic

    surface = "شبهتمونا بالحمر والكلاب"
    diacritized_answer = "قالت عائشة: شَبَّهْتُمُونَا بِالْحُمُرِ وَالْكِلَابِ فذكرت ذلك"

    assert normalize_arabic(surface) in normalize_arabic(diacritized_answer)


def test_level2_recall_agent_computes_a_and_r():
    from scripts.compare_systems import level2_recall

    record = {
        "system": "agent", "qid": "q10_salat",
        "answer": "ورد أن لا يقطع الصلاة شيء، وهذا الحديث منسوخ عند بعضهم.",
        "session": {"sources": [{"excerpt": "شبهتمونا بالحمر والكلاب كما ورد"}]},
    }
    result = level2_recall(record, LEXICONS)

    assert result["A_la_yaqtau_shay"] is True
    assert result["A_naskh"] is True
    assert result["A_aisha_shabbahtumuna"] is False
    assert result["R_aisha_shabbahtumuna"] is True
    assert result["R_la_yaqtau_shay"] is False
    assert 0.0 <= result["A_recall"] <= 1.0


def test_honorific_counts_matches_diacritized_imami_and_sunni_forms():
    """Regression test: needle and haystack must be normalized identically. A fully
    diacritized «عليه السلام» and a diacritized «صلى الله عليه وسلم» must still be
    counted after normalize_arabic strips the diacritics from both sides."""
    from scripts.compare_systems import honorific_counts

    text = ("عن أبي عبد الله (عَلَيْهِ السَّلَامُ) قال، وروى ذلك عمن صَلَّى اللَّهُ "
            "عَلَيْهِ وَسَلَّمَ")
    counts = honorific_counts(text, LEXICONS)

    assert counts["honorific_imami_style"] == 1
    assert counts["honorific_sunni_style"] == 1


def test_honorific_counts_ligature_is_sunni_style():
    from scripts.compare_systems import honorific_counts

    counts = honorific_counts("قال النبي ﷺ كذا وكذا", LEXICONS)

    assert counts["honorific_sunni_style"] == 1
    assert counts["honorific_imami_style"] == 0


def test_honorific_counts_full_imami_phrase_not_also_counted_as_sunni():
    """«صلى الله عليه وآله وسلم» must count once as imami_style and must NOT also match
    the sunni_style «صلى الله عليه وسلم» substring (the extra «وآله» breaks it)."""
    from scripts.compare_systems import honorific_counts

    counts = honorific_counts("صلى الله عليه وآله وسلم", LEXICONS)

    assert counts["honorific_imami_style"] == 1
    assert counts["honorific_sunni_style"] == 0


def test_level2_recall_core_fragments_matches_reordered_imami_wording():
    """The real Māzandarānī q10_salat answer says «لا يقطع صلاةَ المسلم شئٌ» (word order
    swapped, "المسلم" inserted, alternate hamza spelling of "shay'"), which the plain
    surfaces list does not catch. The core_fragments group [["لا يقطع", "شي"]] must
    catch it because both fragments occur in the same sentence."""
    from scripts.compare_systems import level2_recall

    record = {
        "system": "agent", "qid": "q10_salat",
        "answer": "سألته عن ذلك، فقال: لا يقطع صلاةَ المسلمِ شئٌ ولكن ادرء ما استطعت.",
        "session": {"sources": []},
    }
    result = level2_recall(record, LEXICONS)

    assert result["A_la_yaqtau_shay"] is True


def test_level2_recall_candidate_item_excluded_from_aggregate_recall():
    """An item marked status=candidate is still scored per-item but must not move
    A_recall/R_recall, so an unconfirmed guess can't inflate the headline number.
    Uses its own gold list: the real one's candidates change as the user confirms them."""
    from scripts.compare_systems import level2_recall

    lexicons = {**LEXICONS, "level2_gold_variants": {"q9_shum": [
        {"item": "conditional_particle", "surfaces": ["إن كان الشؤم"]},
        {"item": "hikaya_can_ghayrihi", "surfaces": ["حكاية عن غيره"]},
        {"item": "aisha_rejection", "surfaces": ["كان أهل الجاهلية يقولون"]},
        {"item": "unconfirmed", "status": "candidate", "surfaces": ["غلاء مهرها"]},
    ]}}
    record_without_candidate_hit = {
        "system": "luna", "qid": "q9_shum",
        "answer": "إن كان الشؤم في شيء ما ذكرت العرب ذلك حكاية عن غيره لا اعتقادا",
        "session": {},
    }
    result = level2_recall(record_without_candidate_hit, lexicons)
    assert "A_unconfirmed" in result
    assert result["A_unconfirmed"] is False
    # 2 of 3 non-candidate items hit; the candidate is not in the denominator.
    assert result["A_recall"] == round(2 / 3, 3)


def test_level2_recall_returns_empty_for_level1_question():
    from scripts.compare_systems import level2_recall

    record = {"system": "luna", "qid": "q1_ray", "answer": "نص عادي", "session": {}}

    assert level2_recall(record, LEXICONS) == {}


def test_mattr_is_one_for_all_distinct_tokens_under_window():
    from scripts.compare_systems import mattr

    assert mattr(["a", "b", "c"], window=100) == 1.0


def test_mattr_is_lower_for_repeated_tokens():
    from scripts.compare_systems import mattr

    repeated = mattr(["a", "a", "a", "a"], window=100)
    varied = mattr(["a", "b", "c", "d"], window=100)

    assert repeated < varied
    assert repeated == 0.25


def test_mattr_windows_moving_average_beyond_window_size():
    from scripts.compare_systems import mattr

    tokens = ["a", "b"] * 60  # 120 tokens, alternating -> every 100-window has TTR 2/100
    assert abs(mattr(tokens, window=100) - 0.02) < 1e-9


def test_dunning_g2_is_zero_when_counts_are_equal_and_symmetric():
    from scripts.compare_systems import dunning_g2

    assert dunning_g2(0, 0, 100, 100) == 0.0
    assert dunning_g2(50, 50, 100, 100) == 0.0


def test_dunning_g2_is_positive_when_a_term_is_skewed_to_one_group():
    from scripts.compare_systems import dunning_g2

    skewed = dunning_g2(90, 10, 1000, 1000)
    even = dunning_g2(50, 50, 1000, 1000)

    assert skewed > even
    assert skewed > 0


def test_keyness_table_ranks_a_group_exclusive_word_high():
    from scripts.compare_systems import keyness_table

    texts_a = ["كلمة فريدة تتكرر هنا كثيرا"] * 5
    texts_b = ["نص آخر بلا تلك الكلمة إطلاقا"] * 5

    rows = keyness_table(texts_a, texts_b, top_n=5)
    # keyness_table normalizes with normalize_arabic, which folds ة -> ه.
    terms = {r["term"]: r for r in rows}

    assert "فريده" in terms
    assert terms["فريده"]["direction"] == "A"
    assert terms["فريده"]["g2"] > 0


def test_reuse_share_full_overlap_and_no_overlap():
    from scripts.compare_systems import _word_ngrams, reuse_share

    corpus_text = "هذا نص طويل يتكرر فيه الكلام كثيرا جدا ويشرح المسائل"
    ngram_set = _word_ngrams(corpus_text.split(), 4)

    reused_answer = "هذا نص طويل يتكرر فيه الكلام كثيرا جدا"
    fresh_answer = "كلام غريب تماما لا علاقة له بأي شيء آخر هنا اليوم"

    assert reuse_share(reused_answer, ngram_set, 4) > 0.5
    assert reuse_share(fresh_answer, ngram_set, 4) == 0.0


def test_truncated_flag_set_when_tokens_out_reaches_max_tokens():
    from scripts.compare_systems import grounding_metrics

    record = {
        "system": "luna", "qid": "q1_ray", "answer": "نص قصير هنا",
        "max_tokens": 4000, "tokens_in": 100, "tokens_out": 4000,
        "session": {"quotations": 0, "verified": 0, "grounding_log": []},
    }
    result = grounding_metrics(record)

    assert result["truncated"] is True


def test_truncated_flag_false_when_under_the_ceiling():
    from scripts.compare_systems import grounding_metrics

    record = {
        "system": "luna", "qid": "q1_ray", "answer": "نص قصير هنا",
        "max_tokens": 4000, "tokens_in": 100, "tokens_out": 1200,
        "session": {"quotations": 0, "verified": 0, "grounding_log": []},
    }
    result = grounding_metrics(record)

    assert result["truncated"] is False


def test_grounding_metrics_agent_reports_outside_scope_and_foreign_commentary():
    from scripts.compare_systems import grounding_metrics

    record = {
        "system": "agent", "qid": "q1_ray", "answer": "نص",
        "session": {
            "grounding_log": [
                {"verified": True, "outside_scope": True, "foreign_commentary": False},
                {"verified": True, "outside_scope": False, "foreign_commentary": True},
                {"verified": False, "outside_scope": False, "foreign_commentary": False},
            ],
        },
    }
    result = grounding_metrics(record)

    assert result["n_quotations"] == 3
    assert result["verified"] == 2
    assert result["unverified"] == 1
    assert result["verified_outside_scope"] == 1
    assert result["foreign_commentary"] == 1


def test_find_level2_luna_dir_prefers_16000_token_dir_over_truncated_ones(tmp_path):
    from scripts.compare_systems import find_level2_luna_dir

    truncated = tmp_path / "frontier_openai_gpt-5.6-luna_20260927T141345"
    truncated.mkdir()
    (truncated / "index.json").write_text(json.dumps({"max_tokens": 4000}), encoding="utf-8")

    good = tmp_path / "frontier_openai_gpt-5.6-luna_20260927T151354"
    good.mkdir()
    (good / "index.json").write_text(json.dumps({"max_tokens": 16000}), encoding="utf-8")

    older_good = tmp_path / "frontier_openai_gpt-5.6-luna_20260924T161907"
    older_good.mkdir()
    (older_good / "index.json").write_text(json.dumps({"max_tokens": 16000}), encoding="utf-8")

    result = find_level2_luna_dir(tmp_path)

    assert result == good  # newest dir that actually has max_tokens == 16000


def test_find_level2_luna_dir_returns_none_when_nothing_qualifies(tmp_path):
    from scripts.compare_systems import find_level2_luna_dir

    only_truncated = tmp_path / "frontier_openai_gpt-5.6-luna_20260927T141345"
    only_truncated.mkdir()
    (only_truncated / "index.json").write_text(json.dumps({"max_tokens": 4000}), encoding="utf-8")

    assert find_level2_luna_dir(tmp_path) is None


def test_write_csv_covers_every_row_key_across_heterogeneous_rows(tmp_path):
    from scripts.compare_systems import write_csv

    rows = [{"id": "a", "x": 1}, {"id": "b", "x": 2, "y": 3}]
    out = tmp_path / "out.csv"
    write_csv(out, rows)

    text = out.read_text(encoding="utf-8")
    assert "id" in text and "x" in text and "y" in text
    assert "a" in text and "b" in text


def test_strip_markdown_removes_headings_bullets_blockquotes_and_emphasis():
    from scripts.compare_systems import strip_markdown

    text = "## عنوان\n- بند أول\n> اقتباس هنا\n**نص عريض** و_مائل_"
    stripped = strip_markdown(text)

    assert "#" not in stripped
    assert not stripped.splitlines()[1].startswith("-")
    assert ">" not in stripped
    assert "*" not in stripped
    assert "عنوان" in stripped and "بند أول" in stripped and "نص عريض" in stripped


def test_chunk_words_caps_at_max_chunks_and_keeps_all_words_within_that_cap():
    from scripts.compare_systems import chunk_words

    text = " ".join(f"كلمة{i}" for i in range(2000))  # far more than 8*150 words
    chunks = chunk_words(text, chunk_size=150, max_chunks=8)

    assert len(chunks) == 8
    assert all(len(c.split()) <= 150 for c in chunks)


def test_chunk_words_short_text_is_a_single_chunk():
    from scripts.compare_systems import chunk_words

    chunks = chunk_words("نص قصير جدا", chunk_size=150, max_chunks=8)

    assert len(chunks) == 1
    assert chunks[0] == "نص قصير جدا"


def test_length_tercile_labels_splits_into_three_bands():
    from scripts.compare_systems import _length_tercile_labels

    counts = [10, 20, 30, 100, 200, 300, 1000, 2000, 3000]
    labels = _length_tercile_labels(counts)

    assert labels[:3] == ["short", "short", "short"]
    assert labels[7:] == ["long", "long"]
    assert set(labels) == {"short", "medium", "long"}


def test_cosine_dist_matrix_has_exact_zero_diagonal():
    import numpy as np

    from scripts.compare_systems import _cosine_dist_matrix

    rng = np.random.default_rng(0)
    mat = rng.normal(size=(6, 12))
    dist = _cosine_dist_matrix(mat)

    assert np.all(np.diag(dist) == 0.0)
    assert np.all(dist >= 0.0)


def test_safe_silhouette_returns_none_for_all_singleton_groups():
    import numpy as np

    from scripts.compare_systems import _safe_silhouette

    dist = np.zeros((3, 3))
    assert _safe_silhouette(dist, ["a", "b", "c"]) is None


def test_safe_silhouette_returns_none_for_a_single_group():
    import numpy as np

    from scripts.compare_systems import _safe_silhouette

    dist = np.zeros((3, 3))
    assert _safe_silhouette(dist, ["a", "a", "a"]) is None


def test_load_llm_moves_reads_id_unit_index_pairs(tmp_path):
    from scripts.compare_systems import load_llm_moves

    csv_path = tmp_path / "llm_moves.csv"
    csv_path.write_text(
        "id,unit_index,rule_move,llm_move,final_move\n"
        "agent_x,0,other,claim,claim\n"
        "agent_x,1,limit,,limit\n",
        encoding="utf-8")

    result = load_llm_moves(csv_path)

    assert result == {("agent_x", 0): "claim", ("agent_x", 1): "limit"}


def test_apply_llm_moves_overrides_only_other_units():
    from hermeneutics.answer_moves import Unit
    from scripts.compare_systems import apply_llm_moves

    units = [
        Unit(index=0, text="x", kind="paragraph", move="other", rule="no_rule_fired"),
        Unit(index=1, text="y", kind="paragraph", move="limit", rule="stated_limits_lexicon"),
    ]
    llm_map = {("agent_x", 0): "claim", ("agent_x", 1): "report"}  # index 1 must be ignored

    apply_llm_moves(units, "agent_x", llm_map)

    assert units[0].move == "claim"
    assert units[0].rule == "no_rule_fired+llm"
    assert units[1].move == "limit"  # untouched: rule already classified it
    assert units[1].rule == "stated_limits_lexicon"


def test_apply_llm_moves_leaves_unit_unchanged_when_no_override_present():
    from hermeneutics.answer_moves import Unit
    from scripts.compare_systems import apply_llm_moves

    units = [Unit(index=5, text="x", kind="paragraph", move="other", rule="no_rule_fired")]
    apply_llm_moves(units, "agent_x", {})

    assert units[0].move == "other"
    assert units[0].rule == "no_rule_fired"


def test_entity_metrics_falls_back_to_rijal_regex_when_ner_is_none():
    """No camel_tools model is loaded in this test (ner=None) -- extract_entities must
    still produce candidates via the أبو/ابن/بن name-cue regex, matched against small
    synthetic rijal sets rather than the real multi-megabyte files."""
    from scripts.compare_systems import entity_metrics

    record = {
        "id": "test_1", "system": "agent",
        "answer": "روى أبو هريرة هذا الحديث، وذكره ابن حجر في شرحه.",
    }
    imami_names = {"ابو رافع"}
    sunni_names = {"ابو هريره", "ابن حجر"}

    metrics, rows = entity_metrics(record, None, imami_names, sunni_names)

    assert metrics["n_person_entities"] >= 2
    assert any(r["in_sunni"] for r in rows)
    assert all(isinstance(r["in_imami"], bool) for r in rows)
