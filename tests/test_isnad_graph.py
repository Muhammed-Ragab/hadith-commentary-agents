import re

from hermeneutics.isnad_graph import build_isnad_graph, extract_narrators


def test_extract_narrators_splits_on_transmission_markers():
    # Fallback path (no camel-tools NER model installed in this environment).
    # "عنه" swallows the marker "عن" as a substring match, so the honorific-strip
    # regex (which expects the whole "عنه" token) never fires on the remainder --
    # a quirk inherited unchanged from the notebook's fallback splitter.
    narrators = extract_narrators("حدثنا سليمان بن حرب حدثنا حماد بن زيد عن ثابت عن أنس رضي الله عنه")
    assert narrators == ["سليمان بن حرب", "حماد بن زيد", "ثابت", "أنس"]  # was "أنس رضي الله": a wrap-truncated honorific, not a name


def test_extract_narrators_drops_blocklisted_relational_terms():
    # "أبيه" (his father) is a relational term, not a name -- it must not appear.
    narrators = extract_narrators("عن أحمد بن أبي عبد الله عن أبيه عن أبي البختري")
    assert "أبيه" not in narrators
    assert "أحمد بن أبي عبد الله" in narrators


def test_extract_narrators_empty_isnad_yields_no_narrators():
    assert extract_narrators("") == []
    assert extract_narrators(None) == []


def test_build_isnad_graph_links_consecutive_narrators():
    hadiths = [{"isnad": "حدثنا سليمان بن حرب حدثنا حماد بن زيد عن ثابت", "matn": "x"}]
    graph = build_isnad_graph(hadiths, "bukhari")
    assert graph.has_edge("سليمان بن حرب", "حماد بن زيد")
    assert graph.has_edge("حماد بن زيد", "ثابت")
    assert not graph.has_edge("سليمان بن حرب", "ثابت")


def test_build_isnad_graph_weights_edges_by_shared_hadith_count():
    hadiths = [
        {"isnad": "حدثنا سليمان بن حرب حدثنا حماد بن زيد"},
        {"isnad": "حدثنا سليمان بن حرب حدثنا حماد بن زيد"},
    ]
    graph = build_isnad_graph(hadiths, "bukhari")
    assert graph["سليمان بن حرب"]["حماد بن زيد"]["weight"] == 2


def test_build_isnad_graph_skips_hadiths_with_fewer_than_two_narrators():
    hadiths = [{"isnad": "قال أنس"}]  # single narrator after markers are stripped
    graph = build_isnad_graph(hadiths, "bukhari")
    assert graph.number_of_nodes() == 0


def test_build_isnad_graph_is_directed():
    hadiths = [{"isnad": "حدثنا سليمان بن حرب عن حماد بن زيد"}]
    graph = build_isnad_graph(hadiths, "bukhari")
    assert graph.is_directed()
    assert not graph.has_edge("حماد بن زيد", "سليمان بن حرب")


# --------------------------------------------- source markup is not part of a name

def test_wrap_markers_do_not_become_part_of_a_narrator():
    """The corpus wraps lines with `~~` and carries `msNNNN` manuscript sigla. Left in,
    they split one narrator across several nodes: the first build of
    bukhari_metrics.json held `أنس` AND `~~أنس رضي الله` as separate nodes, 38 nodes
    carrying latin/digits and 125 carrying honorifics, so every centrality was computed
    over a fragmented graph. These are markup, not linguistic judgements -- stripping
    them is not the NER path's job.
    """
    names = extract_narrators("حدثنا ms0272 مالك بن إسماعيل عن أنس رضي الله ~~")
    assert not any("~~" in n or "ms0" in n for n in names)
    assert any(n == "أنس" for n in names)


def test_a_truncated_honorific_is_still_stripped():
    """`رضي الله عنه` is the pattern, but a wrap can cut it to `رضي الله`."""
    assert extract_narrators("عن أبي هريرة رضي الله") == ["أبي هريرة"]


def test_trailing_punctuation_does_not_smuggle_a_blocklisted_term_through():
    """`أبيه` ("his father") is blocklisted as a relational term, but `أبيه،` is not the
    same string, so the comma walked it straight past the check -- and it came out as
    the HIGHEST-betweenness node in the Kāfī graph. "His father" is not a narrator, and
    that number would have gone on a slide."""
    assert "أبيه" not in extract_narrators("عن أبيه، عن جده")
    assert "أبيه،" not in extract_narrators("عن أبيه، عن جده")


def test_punctuation_does_not_fork_one_narrator_into_two_nodes():
    a = extract_narrators("عن محمد بن يحيى، عن أحمد")
    b = extract_narrators("عن محمد بن يحيى عن أحمد")
    assert a == b


def test_openiti_page_and_quote_markers_are_not_part_of_a_name():
    """al-Kāfī carries `PageV03P188 #` page markers and Bukhārī `@QE@` quote markers,
    and both rode inside narrator names: `PageV03P188 # ابن محبوب` was a separate node
    from `ابن محبوب`, so one transmitter's centrality was split across two."""
    names = extract_narrators("عن PageV03P188 # ابن محبوب عن 13 - علي بن إبراهيم @QE@")
    assert "ابن محبوب" in names
    assert not any(re.search(r"[A-Za-z@#]|\d", n) for n in names)


def test_first_person_transmission_verbs_split_the_chain_too():
    """The notebook's marker list had حدثنا/أخبرنا but not their first-person forms,
    though Bukhārī's 500-isnad sample carries حدثني 140 times and أخبرني 50. Each
    unsplit occurrence welds two narrators into one node."""
    assert extract_narrators("حدثنا مالك حدثني نافع أخبرني سالم") == [
        "مالك", "نافع", "سالم"]
