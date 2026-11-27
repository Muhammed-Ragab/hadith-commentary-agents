import pytest
from scripts.parse_openiti import (
    clean_text,
    parse_babs,
    parse_hadith_file,
    parse_lexicon_file,
    parse_rijal_file,
)

SAMPLE = """######OpenITI#

#META#Header#End#

# | 1 ( كتاب الإيمان )
# | 1 ( باب بيان الإيمان والإسلام )
~~هذا نص الحديث الأول PageV01P010
~~يواصل النص ms0001
~~وهنا يبدأ شرح النووي على الحديث المذكور
# | 1 ( باب الدليل على صحة الإسلام )
~~حديث آخر PageV01P015
~~وشرحه هنا
"""

def test_parse_babs_count():
    babs = parse_babs(SAMPLE)
    assert len(babs) == 2

def test_parse_babs_fields():
    babs = parse_babs(SAMPLE)
    assert babs[0]["kitab"] == "كتاب الإيمان"
    assert "باب بيان الإيمان" in babs[0]["bab"]
    assert "PageV01P010" not in babs[0]["text"]
    assert "ms0001" not in babs[0]["text"]

def test_clean_text_strips_markers():
    raw = "نص ~~الحديث PageV01P010 ms0001 هنا"
    assert clean_text(raw) == "نص الحديث هنا"

def test_empty_bab_skipped():
    babs = parse_babs(SAMPLE)
    for b in babs:
        assert b["text"].strip() != ""


# --- defect #7 regressions (handoff §10.1, measured 2026-08-23) ---

WRAPPED = """# | 1 ( كتاب الإيمان )
# | 1 ( باب بيان أن الإسناد من الدين وأن الرواية لا تكون
الا عن الثقات )
~~شرح النووي على هذا الباب
# | 1 ( فصل فى جملة المسائل )
~~تتمة الشرح داخل الباب نفسه
# | 1 ( باب الدليل على صحة الإسلام )
~~شرح آخر
"""


def test_wrapped_marker_title_starts_a_new_bab():
    """257 Nawawi markers wrap onto the next line; a same-line ')' regex merged them."""
    babs = parse_babs(WRAPPED)
    assert len(babs) == 2
    assert "الا عن الثقات" in babs[0]["bab"]


def test_fasl_is_absorbed_into_the_enclosing_bab():
    """فصل is sub-bāb: its body belongs to the bāb, and the marker is not text."""
    babs = parse_babs(WRAPPED)
    assert "تتمة الشرح" in babs[0]["text"]
    assert "فصل" not in babs[0]["text"]


CAYNI = """### | (كتاب الوحي )
### || 1 - (باب كتاب النبي صلى الله عليه وسلم إلى كسرى )
~~شرح العيني هنا
### || 2 - (باب بدء الوحي
وكيف كان )
~~شرح آخر
"""


def test_cayni_level_two_marker_is_a_bab_even_when_it_starts_with_kitab():
    """Level, not wording, sets the tier in a multi-level file."""
    babs = parse_babs(CAYNI)
    assert len(babs) == 2
    assert all(b["kitab"] == "كتاب الوحي" for b in babs)
    assert babs[0]["bab"].startswith("باب كتاب النبي")


def test_cayni_ordinal_prefix_is_stripped():
    babs = parse_babs(CAYNI)
    assert not babs[1]["bab"].startswith("2")
    assert "وكيف كان" in babs[1]["bab"]


BARE = """# | 1 ( كتاب الإيمان )
# | 1 ( باب بيان الإيمان )
~~شرح النووي
# | كتاب العتق
# قال أهل اللغة العتق الحرية يقال منه عتق يعتق عتقا
~~بفتحها أيضا حكاه صاحب المحكم
# | 1 ( باب بيان أن الولاء لمن أعتق )
~~شرح الباب
"""

CAYNI_BARE = """### | (كتاب الوحي )
### || 42 - (باب قول النبي الدين النصيحة لله ولرسوله ولأئمة
~~المسلمين
# وعامتهم وقوله تعالى {إذا نصحوا لله ورسوله} )
# الكلام فيه على وجوه
### || 48 - باب من ترك بعض الإختيار مخافة أن يقصر فهم بعض الناس عنه
~~منه
# أي: هذا باب في بيان من ترك ... الخ. وكلمة: من، موصولة
"""


def test_bare_marker_title_does_not_swallow_the_commentary():
    """17 of Nawawi's markers and 126 of ʿAynī's open no paren. Scanning them for a
    closing ')' that never comes ate the body: 3 of Nawawi's 46 kitāb labels carried
    ~300 chars of gloss. Kitāb identity is what the sampler's strata are built on."""
    kitabs = {b["kitab"] for b in parse_babs(BARE)}
    assert "كتاب العتق" in kitabs
    assert not any("قال أهل اللغة" in k for k in kitabs)


def test_bare_marker_still_wraps_onto_a_continuation_line():
    """A bare title does continue on '~~' -- ʿAynī 48 ends mid-phrase."""
    babs = parse_babs(CAYNI_BARE)
    assert any(b["bab"].endswith("منه") for b in babs), [b["bab"] for b in babs]
    assert not any("موصولة" in b["bab"] for b in babs)


def test_parenthesised_title_still_wraps_across_a_hash_line():
    """ʿAynī 42's genuine title crosses a '#' line before its ')'. Breaking on every '#'
    fixed the bare case by truncating this one -- the paren is what tells them apart."""
    babs = parse_babs(CAYNI_BARE)
    assert any("وعامتهم" in b["bab"] for b in babs), [b["bab"] for b in babs]
    assert not any("الكلام فيه على وجوه" in b["bab"] for b in babs)


# --- plain-header format (Māzandarānī, handoff §10.5) ---

PLAIN = """######OpenITI#

# مقدمة المؤلف وبعد فيقول المفتقر
# كتاب العلم باب فرض العلم * الأصل: نص الحديث الأول
# * الشرح: شرح المازندراني هنا
# باب صفة العلم وفضله * الأصل: حديث ثان
# كتاب الروضة * الشرح: كتاب بلا أبواب
"""


def test_plain_format_is_detected_without_pipe_markers():
    babs = parse_babs(PLAIN)
    assert babs, "a file with no '#|' markers must fall back to plain-header parsing"
    assert any(b["kitab"] == "كتاب العلم" for b in babs)


def test_plain_kitab_header_running_into_its_first_bab():
    """'# كتاب العلم باب فرض العلم ...' opens a kitāb and its first bāb at once."""
    babs = parse_babs(PLAIN)
    first = next(b for b in babs if b["kitab"] == "كتاب العلم")
    assert first["bab"] == "باب فرض العلم"
    assert "نص الحديث الأول" in first["text"]


def test_plain_kitab_without_babs_becomes_one_unit():
    """al-ʿAql wa-l-Jahl and al-Rawḍa are undivided in al-Kāfī; dropping them loses text."""
    babs = parse_babs(PLAIN)
    rawda = [b for b in babs if b["kitab"] == "كتاب الروضة"]
    assert len(rawda) == 1
    assert rawda[0]["bab"] == "كتاب الروضة"
    assert "بلا أبواب" in rawda[0]["text"]


def test_plain_preamble_is_kept_not_dropped():
    babs = parse_babs(PLAIN)
    assert any("\u0648\u0628\u0639\u062f \u0641\u064a\u0642\u0648\u0644" in b["text"] for b in babs)


def test_wave_parsers_keep_requested_shapes(tmp_path):
    hadith = tmp_path / "muslim.mARkdown"
    hadith.write_text(
        "### | 1 - book\n### || 1 - bab\n### $ 7 -\n### $\n"
        "~~chain \u0642\u0627\u0644: matn\n",
        encoding="utf-8",
    )
    parsed = parse_hadith_file(hadith, "Muslim")
    assert set(parsed) == {"book_name", "chapters"}
    item = parsed["chapters"][0]["sub_chapters"][0]["hadiths"][0]
    assert item == {"hadith_number": "7", "isnad": "chain", "matn": "\u0642\u0627\u0644: matn"}

    lexicon = tmp_path / "lexicon.mARkdown"
    lexicon.write_text(
        "### | \u062d\u0631\u0641 \u0627\u0644\u0623\u0644\u0641\n### | (\u0623\u0628\u062f)\n~~definition\n"
        "### | (\u0623\u0628\u0644)\n~~next\n",
        encoding="utf-8",
    )
    assert [e["entry"] for e in parse_lexicon_file(lexicon, "nihaya")] == ["\u0623\u0628\u062f", "\u0623\u0628\u0644"]

    rijal = tmp_path / "rijal.mARkdown"
    rijal.write_text("### | section\n### $ 1 - narrator\n~~\u062b\u0642\u0629\n", encoding="utf-8")
    entry = parse_rijal_file(rijal, "source")[0]
    assert entry["source"] == "source"
    assert entry["text"] == "\u062b\u0642\u0629"
