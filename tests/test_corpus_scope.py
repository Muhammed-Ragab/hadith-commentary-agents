"""Part 5: retrieval scoping. Each commentator's run reads its own tradition's files
(TRADITION_SCOPE), not "all" or a rival corpus, and a verified quotation found only
outside that scope is flagged (not stripped) so a human reviewer sees it.

Uses the real poc data, the way test_corpus_tools_move.py already does -- it is local,
read-only and requires no network, so it is offline in the sense the test suite cares
about even though it is not a synthetic fixture.
"""
from pathlib import Path

from hermeneutics.corpus_tools import (
    TRADITION_SCOPE,
    CorpusTools,
    outside_scope,
    resolve_scope,
    scope_for,
)

ROOT = Path(__file__).resolve().parents[1]


def test_every_tradition_scope_file_exists_on_disk():
    """A renamed metrics or rijal file must fail loudly here, not as a silently empty
    scope that looks like a working one."""
    for commentator, patterns in TRADITION_SCOPE.items():
        for pattern in patterns:
            assert list(ROOT.glob(pattern)), f"{commentator}: {pattern} matched no file"


def test_cayni_scoped_search_all_never_returns_kafi_or_mazandarani():
    corpus = CorpusTools(ROOT, scope=scope_for("cayni"))
    hits = corpus.search("الرأي والقياس", "all")
    assert hits, "expected some hits from the cayni-scoped corpus"
    assert not any("kafi" in h["source"] for h in hits)
    assert not any("mazandarani" in h["source"] for h in hits)


def test_cayni_scoped_search_named_kafi_source_returns_nothing():
    """The model can still name a source outside its scope (source="kafi"); the scope
    intersects with it rather than raising, so the tool loop keeps working -- and the
    result is empty, not a rival tradition's text."""
    corpus = CorpusTools(ROOT, scope=scope_for("cayni"))
    assert corpus.search("الرأي والقياس", "kafi") == []


def test_mazandarani_scoped_search_never_returns_cayni_or_nawawi():
    corpus = CorpusTools(ROOT, scope=scope_for("mazandarani"))
    hits = corpus.search("العلم", "all")
    assert hits, "expected some hits from the mazandarani-scoped corpus"
    assert not any("cayni" in h["source"] for h in hits)
    assert not any("nawawi" in h["source"] for h in hits)


def test_an_unscoped_corpus_tools_still_searches_everything():
    """scope=None (the default) is the pre-existing whole-corpus behaviour -- every
    other CorpusTools test in this suite relies on it staying unrestricted."""
    corpus = CorpusTools(ROOT)
    hits = corpus.search("العيني عمدة القاري", "commentary")
    assert any("cayni" in h["source"] for h in hits)


def test_lookup_narrator_stays_within_the_scoped_rijal_work():
    """cayni's scope carries sunni_rijal.json only; asking for the imami school
    explicitly must not reach imami_rijal.json."""
    corpus = CorpusTools(ROOT, scope=scope_for("cayni"))
    out = corpus.lookup_narrator("محمد بن اسماعيل", school="imami")
    assert out["matches"] == []
    assert out["total_matches"] == 0


def test_lookup_isnad_network_stays_within_the_scoped_metrics_file():
    corpus = CorpusTools(ROOT, scope=scope_for("mazandarani"))
    out = corpus.lookup_isnad_network("محمد", corpus="bukhari")
    assert out["matches"] == []
    assert out["total_matches"] == 0


def test_verify_quote_is_not_scope_filtered():
    """Verification stays whole-corpus (spec item 3) even for a scoped CorpusTools --
    otherwise outside_scope below could never be computed, since a quotation outside
    scope would just come back unverified instead of verified-but-foreign."""
    kafi_text = "ورأيت العظيم من المال ينفق في سخط الله عز وجل"
    unscoped = CorpusTools(ROOT).search(kafi_text, "kafi")
    assert unscoped, "fixture assumption: this text is really in kafi_raw.json"

    cayni_scoped = CorpusTools(ROOT, scope=scope_for("cayni"))
    assert cayni_scoped.search(kafi_text, "kafi") == []  # search stays scoped
    result = cayni_scoped.verify_quote(kafi_text)         # verify does not
    assert result["verified"] is True
    assert any("kafi_raw.json" in o["source"] for o in result["occurrences"])


def test_resolve_scope_lists_the_concrete_files_for_the_manifest():
    files = resolve_scope(ROOT, "nawawi")
    assert "data/babs/nawawi_babs.json" in files
    assert "data/openiti/muslim_raw.json" in files
    assert "outputs/networks/muslim_metrics.json" in files
    assert not any("cayni" in f or "kafi" in f for f in files)


def test_outside_scope_true_for_a_quotation_found_only_in_a_rival_hadith_file():
    """foreign_commentary deliberately ignores the hadith corpus (quoting it is
    legitimate); outside_scope is the superset that still catches a cayni run
    producing genuine al-Kafi text it should never have been scoped to search."""
    assert outside_scope(["hadith:kafi_raw.json"], "cayni", ROOT) is True


def test_outside_scope_false_for_a_quotation_found_in_the_runs_own_files():
    assert outside_scope(["commentary:cayni_babs.json"], "cayni", ROOT) is False
    assert outside_scope(["hadith:bukhari_raw.json"], "cayni", ROOT) is False


def test_outside_scope_false_for_shared_lexica_or_quran():
    assert outside_scope(["lexica:lisan_entries.json"], "cayni", ROOT) is False


def test_outside_scope_is_false_without_a_recognised_commentator_or_root():
    """A test double with no filesystem backing, or an old session with no commentator
    on record, must read as 'cannot tell', never as a false accusation."""
    assert outside_scope(["hadith:kafi_raw.json"], "cayni", None) is False
    assert outside_scope(["hadith:kafi_raw.json"], "", ROOT) is False
    assert outside_scope([], "cayni", ROOT) is False
