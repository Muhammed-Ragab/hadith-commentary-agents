from scripts.parse_muslim import split_chains, split_transmissions


REC1 = (
    "مقدمة قصيرة، حدثنا أحمد، عن ربعي بن حراش، قال: «اللفظ الأول الطويل» "
    "ح وحدثنا بكر، عن خالد، قال: «ثان» "
    "وأخبرنا عمر، عن زيد، يقول: «اللفظ الثالث»"
)
ABBREV = "حدثنا أحمد، عن زيد، بهذا الإسناد مثله"


def test_split_chains_returns_three_routes_with_their_starts():
    routes = split_chains(REC1)
    assert len(routes) == 3
    assert routes[0].startswith("مقدمة قصيرة، حدثنا أحمد")
    assert routes[1].startswith("ح وحدثنا بكر")
    assert routes[2].startswith("وأخبرنا عمر")


def test_tahwil_marker_stays_with_its_route():
    routes = split_chains(REC1)
    assert "ح وحدثنا بكر" in routes[1]
    assert len(routes) == 3


def test_split_transmissions_extracts_quoted_wordings():
    transmissions = split_transmissions(REC1, "muslim/1")
    assert [transmission.matn for transmission in transmissions] == [
        "اللفظ الأول الطويل",
        "ثان",
        "اللفظ الثالث",
    ]


def test_isnad_is_not_truncated_at_the_first_qala():
    transmissions = split_transmissions(REC1, "muslim/1")
    assert "ربعي بن حراش" in transmissions[0].isnad


def test_matn_does_not_include_narration_verbs():
    transmissions = split_transmissions(REC1, "muslim/1")
    assert all("حدثنا" not in transmission.matn for transmission in transmissions)
    assert all("أخبرنا" not in transmission.matn for transmission in transmissions)


def test_abbreviated_parallel_references_its_sibling():
    transmissions = split_transmissions(ABBREV, "muslim/1")
    assert len(transmissions) == 1
    assert transmissions[0].references_sibling is True
    assert transmissions[0].matn == ""


def test_short_route_without_wording_is_retained_as_sibling_evidence():
    transmissions = split_transmissions("حدثنا زيد", "muslim/1")
    assert len(transmissions) == 1
    assert transmissions[0].references_sibling is True
    assert transmissions[0].matn == ""


def test_abbreviated_parallel_after_qala_is_not_treated_as_matn():
    transmissions = split_transmissions("حدثنا أحمد، قال: بهذا الإسناد مثله", "muslim/1")
    assert len(transmissions) == 1
    assert transmissions[0].references_sibling is True
    assert transmissions[0].matn == ""


def test_route_ending_after_narration_formula_is_sibling_evidence():
    transmissions = split_transmissions("حدثنا أحمد، قال:", "muslim/1")
    assert len(transmissions) == 1
    assert transmissions[0].references_sibling is True
    assert transmissions[0].matn == ""


def test_transmission_refs_are_cluster_ref_with_route_index():
    transmissions = split_transmissions(REC1, "muslim/1")
    assert [transmission.ref for transmission in transmissions] == [
        "muslim/1#1",
        "muslim/1#2",
        "muslim/1#3",
    ]


# Muslim's isnads routinely end in an inflected narration verb: قالا when two
# transmitters are named, قالوا for three or more, قالت for a woman. A bare قال
# pattern matches only the prefix and strands the suffix as the matn ("ا:").
DUAL = "حدثني محمد بن عبد الله، وزهير بن حرب، قالا: حدثنا عبد الرزاق عن معمر"
PLURAL = "حدثنا أبو بكر، وعمرو الناقد، وزهير، قالوا: حدثنا سفيان عن الزهري"
FEM = "حدثنا يحيى، عن عائشة، قالت: «كان رسول الله يصلي من الليل»"


def test_dual_qala_is_not_split_mid_word():
    transmissions = split_transmissions(DUAL, "muslim/1")
    assert transmissions[0].matn != "ا:"
    assert "ا:" not in transmissions[0].matn


def test_plural_qalu_is_not_split_mid_word():
    transmissions = split_transmissions(PLURAL, "muslim/1")
    assert transmissions[0].matn != "وا:"
    assert not transmissions[0].matn.startswith("وا")


def test_feminine_qalat_is_not_split_mid_word():
    transmissions = split_transmissions(FEM, "muslim/1")
    assert transmissions[0].matn == "كان رسول الله يصلي من الليل"


def test_inflected_verb_stays_out_of_the_matn():
    for text in (DUAL, PLURAL, FEM):
        for transmission in split_transmissions(text, "muslim/1"):
            assert not transmission.matn.strip().startswith(("ا", "وا", "ت"))


def test_cluster_refs_are_unique_across_the_corpus():
    """hadith_number restarts in every kitab, so it cannot be a ref on its own:
    6299 hadiths carry only 522 distinct numbers, and bare "muslim/2" would name
    119 different clusters. A Claim.source_ref has to resolve to exactly one."""
    from scripts.parse_muslim import parse

    clusters = parse()
    refs = [cluster.cluster_ref for cluster in clusters]
    assert len(refs) == len(set(refs))


def test_transmission_refs_are_unique_across_the_corpus():
    from scripts.parse_muslim import parse

    refs = [t.ref for cluster in parse() for t in cluster.transmissions]
    assert len(refs) == len(set(refs))
