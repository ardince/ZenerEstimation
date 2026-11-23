"""Synthetic structural tests; no models or battery datasets are loaded."""
from dataclasses import FrozenInstanceError
import json

import pytest

from zenerestimation.assessment.eligibility import (
    CONTRACT_VERSION, AssessmentDeclaration, AssessmentTrack, ExposureStatus,
    TrustedVerificationEvidence, VerificationFinding, VerificationStatus,
    validate_evidence_binding,
)


def declaration(**overrides):
    values = dict(contract_version=CONTRACT_VERSION, assessment_id="case-001",
                  assessment_track=AssessmentTrack.RETROSPECTIVE,
                  dataset_id="battery-410", development_cutoff="2023-12-01",
                  forecast_dates=("2024-03-01", "2024-06-01"),
                  evidence_refs={"forecast": "artifact-1"})
    values.update(overrides)
    return AssessmentDeclaration(**values)


def finding(status=VerificationStatus.VERIFIED, **overrides):
    values = dict(status=status, evidence_ref="evidence-1", verifier_id="verifier-v1",
                  reason_codes=("CHECK_FAILED",) if status is VerificationStatus.FAILED else ())
    values.update(overrides)
    return VerificationFinding(**values)


def evidence(d=None, **overrides):
    d = d or declaration()
    values = dict(assessment_id=d.assessment_id, dataset_id=d.dataset_id,
                  declaration_digest=d.digest, technical_integrity=finding(),
                  protocol_compliance=finding(), target_exposure=ExposureStatus.EXPOSED,
                  exposure_evidence=finding(), forecast_freeze=finding(),
                  target_availability=finding(), chronology=finding(),
                  frozen_artifact_integrity=finding())
    values.update(overrides)
    return TrustedVerificationEvidence(**values)


def test_a01_retrospective_declaration():
    assert declaration().assessment_track is AssessmentTrack.RETROSPECTIVE


def test_a02_prospective_declaration():
    assert declaration(assessment_track=AssessmentTrack.PROSPECTIVE).assessment_track.value == "prospective"


def test_a03_unsupported_version():
    with pytest.raises(ValueError, match="version"):
        declaration(contract_version="v0")


@pytest.mark.parametrize("field", ["assessment_id", "dataset_id"])
def test_a04_empty_id(field):
    with pytest.raises(ValueError):
        declaration(**{field: ""})


def test_a05_invalid_cutoff():
    with pytest.raises(ValueError):
        declaration(development_cutoff="2023-02-30")


@pytest.mark.parametrize("dates", [
    ("2024-03-01", "2024-03-01"),
    ("2024-06-01", "2024-03-01"),
])
def test_a06_duplicate_or_unordered(dates):
    with pytest.raises(ValueError):
        declaration(forecast_dates=dates)


def test_a07_nonquarterly_calendar():
    with pytest.raises(ValueError, match="three-month"):
        declaration(forecast_dates=("2024-03-01", "2024-07-01"))


def test_a08_verified_missing_reference():
    with pytest.raises(ValueError, match="requires"):
        finding(evidence_ref=None)


def test_a09_verified_missing_verifier():
    with pytest.raises(ValueError, match="requires"):
        finding(verifier_id=None)


def test_a10_failed_missing_reason():
    with pytest.raises(ValueError, match="reason"):
        finding(VerificationStatus.FAILED, reason_codes=())


def test_a11_assessment_mismatch():
    d = declaration()
    with pytest.raises(ValueError, match="assessment_id mismatch"):
        validate_evidence_binding(d, evidence(d, assessment_id="other"))


def test_a12_dataset_mismatch():
    d = declaration()
    with pytest.raises(ValueError, match="dataset_id mismatch"):
        validate_evidence_binding(d, evidence(d, dataset_id="other"))


def test_a13_unverified_nonexposure_cannot_be_accepted():
    with pytest.raises(ValueError, match="exposure"):
        evidence(target_exposure=ExposureStatus.NOT_EXPOSED,
                 exposure_evidence=finding(VerificationStatus.UNKNOWN))


def test_a14_invalid_status():
    with pytest.raises(TypeError):
        finding(status="verified")


def test_a15_frozen_record_and_nested_refs():
    d = declaration()
    with pytest.raises(FrozenInstanceError):
        d.dataset_id = "changed"
    with pytest.raises(TypeError):
        d.evidence_refs["forecast"] = "changed"


def test_a16_serialized_verified_json_is_not_authentication():
    # JSON contains claims, but no public JSON -> trusted certificate pathway exists.
    record = json.loads(json.dumps(evidence().to_dict()))
    assert record["technical_integrity"]["status"] == "verified"
    assert not hasattr(evidence(), "prospective_independence_certified")


def test_a17_deterministic_digest_and_serialization():
    a = declaration(evidence_refs={"z": "ref-z", "a": "ref-a"})
    b = declaration(evidence_refs={"a": "ref-a", "z": "ref-z"})
    assert a.digest == b.digest
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)


def test_a18_changed_declaration_breaks_binding():
    original = declaration()
    trusted = evidence(original)
    changed = declaration(evidence_refs={"forecast": "artifact-2"})
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_evidence_binding(changed, trusted)


def test_valid_binding():
    d = declaration()
    validate_evidence_binding(d, evidence(d))


def test_unknown_exposure_with_unknown_finding():
    record = evidence(target_exposure=ExposureStatus.UNKNOWN,
                      exposure_evidence=finding(VerificationStatus.UNKNOWN))
    assert record.target_exposure is ExposureStatus.UNKNOWN


def test_no_implicit_forecast_origin_constraint():
    # Cutoff is a training boundary, not necessarily the forecast origin.
    assert declaration(forecast_dates=("2024-06-01", "2024-09-01")).forecast_dates[0] == "2024-06-01"
