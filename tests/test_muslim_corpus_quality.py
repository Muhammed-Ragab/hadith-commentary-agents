import json
import re
from pathlib import Path

import pytest


CORPUS = Path(__file__).parents[1] / "data" / "openiti" / "muslim_clusters.json"
CHAIN_START = re.compile(r"^\s*و?(?:حدثنا|حدثني|أخبرنا|أخبرني)\b")


if not CORPUS.exists():
    pytest.skip("generated Muslim cluster corpus is unavailable", allow_module_level=True)


def test_muslim_clusters_preserve_transmission_structure():
    clusters = json.loads(CORPUS.read_text(encoding="utf-8"))
    transmissions = [
        transmission
        for cluster in clusters
        for transmission in cluster["transmissions"]
    ]
    nonempty_matns = [transmission["matn"] for transmission in transmissions if transmission["matn"].strip()]
    chain_started_matns = [matn for matn in nonempty_matns if CHAIN_START.match(matn)]
    sibling_refs = sum(transmission["references_sibling"] for transmission in transmissions)
    refs = [transmission["ref"] for transmission in transmissions]

    assert 6_000 <= len(clusters) <= 6_500
    assert all(transmission["isnad"].strip() for transmission in transmissions)
    assert len(chain_started_matns) / len(nonempty_matns) < 0.02
    assert sum(len(cluster["transmissions"]) > 1 for cluster in clusters) / len(clusters) > 0.5
    assert sibling_refs > 1_000
    assert len(refs) == len(set(refs))


# A narration verb agreeing with its transmitters (قالا / قالوا / قالت) split mid-word for
# 3068 matns before the boundary fix, leaving the inflection stranded as the whole text
# ("ا:", "وا:"). The count guards above all passed while that was true, so the corpus needs
# a guard on what a matn actually contains.
STRANDED_INFLECTION = re.compile(r"^(?:ا|وا|ت|تا|ان)\s*[:،]")


def test_matns_are_not_stranded_verb_inflections():
    clusters = json.loads(CORPUS.read_text(encoding="utf-8"))
    matns = [
        transmission["matn"].strip()
        for cluster in clusters
        for transmission in cluster["transmissions"]
        if transmission["matn"].strip()
    ]
    stranded = [matn for matn in matns if STRANDED_INFLECTION.match(matn)]
    very_short = [matn for matn in matns if len(matn) < 15]

    assert not stranded
    # Some genuinely short matns exist ("نعم"), so this is a regression ceiling, not a
    # claim that the remaining 21% are all clean.
    assert len(very_short) / len(matns) < 0.25


def test_cluster_refs_resolve_to_exactly_one_cluster():
    """Claim.source_ref is only checkable if the ref it carries is unique."""
    clusters = json.loads(CORPUS.read_text(encoding="utf-8"))
    refs = [cluster["cluster_ref"] for cluster in clusters]

    assert len(refs) == len(set(refs))


def test_most_clusters_carry_the_canonical_muslim_number():
    clusters = json.loads(CORPUS.read_text(encoding="utf-8"))
    numbered = [cluster for cluster in clusters if cluster["muslim_number"]]

    assert len(numbered) / len(clusters) > 0.70
    assert max(int(cluster["muslim_number"]) for cluster in numbered) <= 3_033
