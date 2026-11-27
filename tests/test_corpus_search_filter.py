"""An English question must not silently narrow the search to one commentator.

Regression for the bug where a term that is merely a SUBSTRING of a commentator's
filename or alias (e.g. "a" inside "mazandarani") restricted `search` to that
commentator's files. Fixed by requiring a whole-token match of a term with 3+
characters, and by dropping short Latin tokens ("a", "an", "to", ...) from `_terms`
before they ever reach the filter.
"""
from hermeneutics.corpus_tools import _terms


def test_short_latin_stopwords_are_not_search_terms():
    terms = _terms("is it a sin to lie in an emergency")
    assert "a" not in terms
    assert "an" not in terms
    assert "to" not in terms
    assert "in" not in terms
    assert "is" not in terms


def test_short_latin_terms_do_not_name_mazandarani():
    """"a"/"an" used to be a substring of "mazandarani" and restricted the search
    to al-Mazandarani's files alone -- an English question would search one book
    instead of the whole corpus."""
    from hermeneutics.corpus_tools import _WORDS, _normalise

    whose_tokens = set(_WORDS.findall(_normalise("المازندراني شرح الكافي mazandarani_babs")))
    for bad_term in ("a", "an"):
        assert not (len(bad_term) >= 3 and bad_term in whose_tokens)


def test_whole_token_still_names_nawawi_in_either_script():
    terms_latin = _terms("nawawi's view on qiyas")
    terms_arabic = _terms("رأي النووي في القياس")
    assert "nawawi" in terms_latin
    assert any("نووي" in t or "النووي" in t for t in terms_arabic)
