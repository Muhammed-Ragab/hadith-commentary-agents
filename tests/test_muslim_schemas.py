"""Muslim's records are clusters of transmissions, not single hadiths."""
import pytest
from pydantic import ValidationError

from hermeneutics.muslim_schemas import Cluster, Transmission


def test_transmission_requires_an_isnad():
    with pytest.raises(ValidationError):
        Transmission(ref="muslim/1#1", isnad="", matn="نص")


def test_transmission_may_have_no_matn_when_it_references_a_sibling():
    t = Transmission(ref="muslim/1#2", isnad="حدثنا زهير", matn="",
                     references_sibling=True)
    assert t.references_sibling is True


def test_transmission_without_matn_or_reference_is_invalid():
    with pytest.raises(ValidationError):
        Transmission(ref="muslim/1#3", isnad="حدثنا زهير", matn="")


def test_cluster_refs_are_unique():
    with pytest.raises(ValidationError, match="transmission refs must be unique"):
        Cluster(cluster_ref="muslim/1", kitab="الإيمان", bab="", transmissions=[
            Transmission(ref="muslim/1#1", isnad="أ", matn="م"),
            Transmission(ref="muslim/1#1", isnad="ب", matn="ن"),
        ])


def test_cluster_needs_at_least_one_transmission():
    with pytest.raises(ValidationError):
        Cluster(cluster_ref="muslim/1", kitab="الإيمان", bab="", transmissions=[])


def test_cluster_requires_bab():
    with pytest.raises(ValidationError):
        Cluster(cluster_ref="muslim/1", kitab="الإيمان", transmissions=[
            Transmission(ref="muslim/1#1", isnad="أ", matn="م"),
        ])
