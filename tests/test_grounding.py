from hermeneutics.grounding import grounding_rate, is_grounded, normalize_arabic


def test_strips_diacritics():
    assert normalize_arabic("مُحَمَّدٌ") == normalize_arabic("محمد")


def test_unifies_alif_variants():
    assert normalize_arabic("أحمد") == normalize_arabic("احمد")
    assert normalize_arabic("إسلام") == normalize_arabic("اسلام")
    assert normalize_arabic("آمن") == normalize_arabic("امن")


def test_unifies_ta_marbuta_and_ya():
    assert normalize_arabic("صلاة") == normalize_arabic("صلاه")
    assert normalize_arabic("على") == normalize_arabic("علي")


def test_collapses_whitespace():
    assert normalize_arabic("كلمة    أخرى") == normalize_arabic("كلمة أخرى")


def test_strips_tatweel():
    assert normalize_arabic("محـــمد") == normalize_arabic("محمد")


def test_does_not_erase_arabic_indic_digits():
    assert normalize_arabic("حديث ١٢٣") == "حديث ١٢٣"


def test_is_grounded_finds_a_verbatim_quote():
    source = "قال النووي وهذا مخصوص بالمسافر في السفر الطويل"
    assert is_grounded("وهذا مخصوص بالمسافر", source) is True


def test_is_grounded_survives_diacritic_differences():
    assert is_grounded("مُخصوص بالمسافر", "وهذا مخصوص بالمسافر هنا") is True


def test_is_grounded_rejects_a_fabricated_quote():
    assert is_grounded("لم يقل هذا أبدا", "قال النووي وهذا مخصوص") is False


def test_empty_evidence_is_not_grounded():
    assert is_grounded("", "any source text") is False


def test_grounding_rate_counts_moves_not_records():
    records = [{
        "bab_ref": "nawawi/0",
        "moves": [
            {"arabic_evidence": "مخصوص"},      # grounded
            {"arabic_evidence": "غير موجود"},  # not grounded
        ],
    }]
    sources = {"nawawi/0": "وهذا مخصوص بالمسافر"}
    assert grounding_rate(records, sources) == 0.5


def test_grounding_rate_on_no_moves_is_zero():
    assert grounding_rate([], {}) == 0.0


def test_grounding_rate_skips_records_with_no_source():
    records = [{"bab_ref": "nawawi/9", "moves": [{"arabic_evidence": "مخصوص بالمسافر"}]}]
    assert grounding_rate(records, {}) == 0.0


def test_a_line_wrap_artifact_does_not_break_a_true_quotation():
    """bukhari_raw.json carries "ما ~~نوى" 33,160 times and kafi_raw.json 68,067.
    Left in, a model quoting the hadith CORRECTLY fails verification while one
    reproducing the artifact passes -- the gate censors honest quotations."""
    from hermeneutics.grounding import is_grounded

    source = "إنما الأعمال بالنيات وإنما لكل امرئ ما ~~نوى فمن كانت هجرته"

    assert is_grounded("لكل امرئ ما نوى", source)
    assert is_grounded("لكل امرئ ما ~~نوى", source)


def test_punctuation_does_not_break_a_true_quotation():
    """The OpenITI editions are largely unpunctuated and the model punctuates what it
    quotes. This is al-Nawawi's text as the edition carries it, against the quotation a
    recorded session was stripped for: identical but for two commas. 7 of 26 strips in
    the six sessions were this, so the gate was censoring honest quotations -- the same
    direction of failure as the ~~ artifact above."""
    source = ("لم يصح هذا الحديث عن النبي صلى الله عليه وسلم الا من رواية عمر بن الخطاب "
              "ولا عن عمر الا من رواية علقمة بن وقاص")

    assert is_grounded("عمر بن الخطاب، ولا عن عمر", source)
    assert is_grounded("عمر بن الخطاب ولا عن عمر", source)


def test_punctuation_folds_to_a_space_so_it_cannot_glue_two_words():
    """The edition writes "القصد/الوجه" with no space around the slash. Folding the
    slash away rather than to a space would produce "القصدالوجه", a token in neither
    text, and could match across a boundary that exists in the source."""
    assert normalize_arabic("القصد/الوجه") == normalize_arabic("القصد الوجه")
    assert not is_grounded("القصدالوجه", "القصد/الوجه")


def test_punctuation_folding_does_not_ground_a_fabrication():
    """Loosening the match must not turn a contraction into a quotation. A recorded
    session wrote "عظم موقعه" where al-Nawawi has "عظم موقع هذا الحديث"; that is a
    paraphrase presented as verbatim, which is exactly what the gate exists to catch."""
    source = "اجمع المسلمون على عظم موقع هذا الحديث وكثرة فوائده"

    assert not is_grounded("اجمع المسلمون على عظم موقعه", source)
