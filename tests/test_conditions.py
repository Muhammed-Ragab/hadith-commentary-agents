"""verify_quotation's provenance: it already asks a corpus WHERE a fragment occurs and
was throwing that away. These tests exercise the "sources" field added additively --
see conditions.py:verify_quotation -- and, specifically, that it is computed as an
INTERSECTION across an elided quotation's fragments, not a union.
"""


class _Corpus:
    """quote -> occurrences, exactly as CorpusTools.verify_quote returns them."""

    def __init__(self, occurrences_by_quote: dict):
        self._occurrences = occurrences_by_quote

    def verify_quote(self, quote, source="all"):
        occurrences = self._occurrences.get(quote, [])
        return {"verified": bool(occurrences), "occurrences": occurrences}


def test_an_elided_quotation_whose_fragments_share_one_source_reports_it():
    from hermeneutics.conditions import verify_quotation

    corpus = _Corpus({
        "كان رجال من أصحاب رسول الله": [
            {"source": "commentary:nawawi_babs.json", "reference": "بدء الوحي"}],
        "فجاءت عشائرهم فقالوا": [
            {"source": "commentary:nawawi_babs.json", "reference": "بدء الوحي"}],
    })

    result = verify_quotation(
        "كان رجال من أصحاب رسول الله... فجاءت عشائرهم فقالوا", corpus)

    assert result["verified"] is True
    assert result["sources"] == ["commentary:nawawi_babs.json"]


def test_an_elided_quotation_welded_from_different_sources_reports_no_source():
    """The proof of the intersection semantics: each fragment is real on its own, but
    they never occur TOGETHER in any one work -- welded text is not a quotation from
    any one source, even though it is, fragment by fragment, real. `verified` is
    UNCHANGED by this -- it still only asks whether every fragment occurs somewhere."""
    from hermeneutics.conditions import verify_quotation

    corpus = _Corpus({
        "كان رجال من أصحاب رسول الله": [
            {"source": "commentary:nawawi_babs.json", "reference": "بدء الوحي"}],
        "فجاءت عشائرهم فقالوا": [
            {"source": "commentary:cayni_babs.json", "reference": "كتاب الإيمان"}],
    })

    result = verify_quotation(
        "كان رجال من أصحاب رسول الله... فجاءت عشائرهم فقالوا", corpus)

    assert result["verified"] is True
    assert result["sources"] == []


def test_sources_is_empty_when_no_fragment_is_checkable():
    from hermeneutics.conditions import verify_quotation

    result = verify_quotation("قال... ثم... وقد", _Corpus({}))

    assert result == {"verified": False, "elided": True, "checked": 0, "sources": []}


def test_a_non_elided_verified_quotation_reports_its_single_sources_list():
    from hermeneutics.conditions import verify_quotation

    corpus = _Corpus({
        "نص كامل غير مقتطع يظهر في مصدر واحد فقط هنا": [
            {"source": "lexica:lisan_entries.json", "reference": "بوق"}],
    })

    result = verify_quotation("نص كامل غير مقتطع يظهر في مصدر واحد فقط هنا", corpus)

    assert result["verified"] is True
    assert result["sources"] == ["lexica:lisan_entries.json"]


def test_an_unverified_quotation_reports_no_sources():
    from hermeneutics.conditions import verify_quotation

    result = verify_quotation("نص لا وجود له في أي مصدر محلي إطلاقا هنا", _Corpus({}))

    assert result["verified"] is False
    assert result["sources"] == []
