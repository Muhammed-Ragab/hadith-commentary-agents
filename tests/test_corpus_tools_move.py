"""The corpus layer must be importable from the package, not only from a sibling dir.

poc/ is what gets rsynced to KISSKI and run. A tool layer reachable only through
local-agent-ui/ is a tool layer the cluster does not have.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_corpus_tools_is_importable_from_the_package():
    from hermeneutics.corpus_tools import SOURCES, CorpusTools

    assert "hadith" in SOURCES
    assert callable(CorpusTools)


def test_tool_specs_still_lists_all_five_tools():
    from hermeneutics.corpus_tools import CorpusTools

    names = [spec["function"]["name"] for spec in CorpusTools.tool_specs()]
    assert names == ["search_sources", "verify_quote", "lookup_word",
                     "lookup_narrator", "lookup_isnad_network"]


def test_corpus_tools_resolves_paths_against_the_poc_root():
    from hermeneutics.corpus_tools import CorpusTools

    assert CorpusTools(ROOT).poc_root == ROOT.resolve()


def test_each_corpus_is_addressable_on_its_own():
    """The matn stage reads ONE corpus. A glob over data/*_raw.json is two."""
    from hermeneutics.corpus_tools import SOURCES

    assert SOURCES["bukhari"] == "data/bukhari_raw.json"
    assert SOURCES["kafi"] == "data/kafi_raw.json"
    assert SOURCES["muslim_matn"] == "data/openiti/muslim_raw.json"


def test_the_per_corpus_keys_reach_the_tool_schema():
    from hermeneutics.corpus_tools import CorpusTools

    spec = CorpusTools.tool_specs()[0]["function"]
    allowed = spec["parameters"]["properties"]["source"]["enum"]
    assert {"bukhari", "kafi", "muslim_matn"} <= set(allowed)


def test_the_named_corpus_files_exist():
    """A source key pointing at nothing fails silently as an empty search."""
    from hermeneutics.corpus_tools import SOURCES

    for key in ("bukhari", "kafi", "muslim_matn"):
        assert list(ROOT.glob(SOURCES[key])), f"{key} -> {SOURCES[key]} matched no file"


def test_a_hadith_is_cited_by_its_number_not_by_its_filename():
    """_reference looked for hadith_num; the raw corpora spell it hadith_number, so
    every hadith citation fell through to the bare filename stem. An answer citing
    "bukhari_raw" is the uncheckable answer this project argues against."""
    from hermeneutics.corpus_tools import CorpusTools

    hits = CorpusTools(ROOT).search("النية", "bukhari")

    assert hits, "expected hits for النية in Bukhari"
    assert hits[0]["reference"] != "bukhari_raw"
    assert hits[0]["reference"].strip()


def test_verify_quote_accepts_a_quotation_free_of_corpus_line_wraps():
    """The gate calls verify_quote. A scholar quotes the hadith as it reads, not as
    the OpenITI line-wrapper mangled it."""
    from hermeneutics.corpus_tools import CorpusTools

    corpus = CorpusTools(ROOT)
    clean = "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى"

    assert corpus.verify_quote(clean)["verified"], "a correct quotation must verify"


def test_verify_quote_finds_a_bab_heading():
    """_documents searched _record_text, which is the body only, while _chunks indexes
    the reference -- so the agent could FIND a heading through search_sources, quote it,
    and be told its correct quotation does not occur in the corpus. Two of the 26 strips
    in the recorded sessions were this heading."""
    from hermeneutics.corpus_tools import CorpusTools

    heading = "باب ما جاء أن الأعمال بالنية والحسبة ولكل امرىء ما نوى"

    result = CorpusTools(ROOT).verify_quote(heading)

    assert result["verified"], "a bab heading quoted verbatim must verify"
    assert result["occurrences"][0]["reference"].strip()


def test_a_heading_record_cannot_be_matched_across_its_junction_with_the_body():
    """The heading is a SEPARATE record, not concatenated onto the text. Concatenated,
    a needle spanning the junction would verify against a string that exists nowhere in
    the edition -- the fabrication this gate is built to catch."""
    from hermeneutics.corpus_tools import _documents

    docs = _documents(str(ROOT / "data" / "babs" / "cayni_babs.json"))
    by_ref = {}
    for ref, text in docs:
        by_ref.setdefault(ref, []).append(text)
    ref, texts = next((r, t) for r, t in by_ref.items() if len(t) > 1)

    assert len(texts) == 2, "each record contributes its body and its heading, separately"
    assert not any(texts[0] in t and texts[1] in t for t in texts), \
        "the two must not have been concatenated into one searchable string"


def test_a_commentary_hit_carries_the_commentators_arabic_name():
    """Without it the model sees only "mazandarani_babs.json" and spells the name itself --
    it wrote "مظاندراني" through all three stages of a recorded session."""
    from hermeneutics.corpus_tools import CorpusTools

    hits = CorpusTools(ROOT).search("المازندراني العلم", "commentary")
    maz = [h for h in hits if "mazandarani" in h["source"]]

    assert maz, "expected a hit from the mazandarani commentary"
    assert "المازندراني" in maz[0]["author"]
    assert maz[0]["source"].endswith("mazandarani_babs.json"), "source must stay unchanged"
