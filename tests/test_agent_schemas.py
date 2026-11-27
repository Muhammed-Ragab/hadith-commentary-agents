import pytest
from pydantic import ValidationError

from hermeneutics.agent_schemas import Claim, Diagnosis, Finding, Verdict


def test_claim_defaults_to_unchecked():
    claim = Claim(text="A claim", arabic_evidence="دليل", source_ref="source-1")

    assert claim.status == "unchecked"


def test_finding_rejects_empty_skill_id():
    with pytest.raises(ValidationError):
        Finding(skill_id="")


def test_verdict_allows_unsupported_tawaqquf():
    verdict = Verdict(kind="tawaqquf", reading="Withhold judgment")

    assert verdict.supported_by == []


def test_verdict_rejects_unsupported_tarjih():
    with pytest.raises(ValidationError):
        Verdict(kind="tarjih", reading="Prefer one reading")


def test_diagnosis_rejects_empty_skill_plan():
    with pytest.raises(ValidationError):
        Diagnosis(skill_plan=[])
