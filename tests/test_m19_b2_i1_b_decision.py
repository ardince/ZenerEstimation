"""Synthetic policy tests: not an authentication or scientific certification test."""
from dataclasses import FrozenInstanceError, replace
import json
import pytest

from zenerestimation.assessment import (
    CONTRACT_VERSION, AssessmentDeclaration, AssessmentTrack, AssessmentValidity,
    ExposureStatus, ScientificClassification, TrustedVerificationEvidence,
    VerificationFinding, VerificationStatus, evaluate_scientific_eligibility,
)
from zenerestimation.assessment.decision import ScientificEligibilityDecision


def finding(status=VerificationStatus.VERIFIED):
    return VerificationFinding(status=status, evidence_ref="evidence-1", verifier_id="verifier-1",
                               reason_codes=("CHECK_FAILED",) if status is VerificationStatus.FAILED else ())


def setup(track=AssessmentTrack.PROSPECTIVE, exposure=ExposureStatus.NOT_EXPOSED, **changes):
    d = AssessmentDeclaration(CONTRACT_VERSION, "assessment-1", track, "battery-1", "2025-03-01", ("2025-06-01",), {})
    e = TrustedVerificationEvidence(
        assessment_id=d.assessment_id, dataset_id=d.dataset_id, declaration_digest=d.digest,
        technical_integrity=finding(), protocol_compliance=finding(), target_exposure=exposure,
        exposure_evidence=finding(VerificationStatus.UNKNOWN) if exposure is ExposureStatus.UNKNOWN else finding(),
        forecast_freeze=finding(), target_availability=finding(), chronology=finding(),
        frozen_artifact_integrity=finding(),
    )
    return d, replace(e, **changes)


def test_complete_prospective_is_eligible_but_not_certified():
    result = evaluate_scientific_eligibility(*setup())
    assert result.assessment_validity is AssessmentValidity.VALID
    assert result.scientific_classification is ScientificClassification.PROSPECTIVE_INDEPENDENT
    assert result.prospective_independence_eligible
    assert not result.prospective_independence_certified


def test_retrospective_exposed():
    r = evaluate_scientific_eligibility(*setup(AssessmentTrack.RETROSPECTIVE, ExposureStatus.EXPOSED))
    assert r.scientific_classification is ScientificClassification.CONTROLLED_RETROSPECTIVE
    assert "TARGET_EXPOSURE_CONFIRMED" in r.reason_codes


def test_retrospective_not_exposed_still_retrospective():
    r = evaluate_scientific_eligibility(*setup(AssessmentTrack.RETROSPECTIVE))
    assert r.scientific_classification is ScientificClassification.CONTROLLED_RETROSPECTIVE


def test_unknown_exposure_unverified():
    r = evaluate_scientific_eligibility(*setup(exposure=ExposureStatus.UNKNOWN))
    assert r.scientific_classification is ScientificClassification.INDEPENDENCE_UNVERIFIED
    assert "TARGET_EXPOSURE_UNKNOWN" in r.reason_codes


@pytest.mark.parametrize("field", ["technical_integrity", "protocol_compliance", "frozen_artifact_integrity"])
def test_core_failed_invalid(field):
    r = evaluate_scientific_eligibility(*setup(**{field: finding(VerificationStatus.FAILED)}))
    assert r.assessment_validity is AssessmentValidity.INVALID
    assert r.scientific_classification is None


@pytest.mark.parametrize("field", ["technical_integrity", "protocol_compliance", "frozen_artifact_integrity"])
def test_core_unknown_incomplete(field):
    r = evaluate_scientific_eligibility(*setup(**{field: finding(VerificationStatus.UNKNOWN)}))
    assert r.assessment_validity is AssessmentValidity.INCOMPLETE


@pytest.mark.parametrize("field", ["forecast_freeze", "target_availability", "chronology"])
def test_prospective_unknown_unverified(field):
    r = evaluate_scientific_eligibility(*setup(**{field: finding(VerificationStatus.UNKNOWN)}))
    assert r.scientific_classification is ScientificClassification.INDEPENDENCE_UNVERIFIED
    assert not r.prospective_independence_eligible


@pytest.mark.parametrize("field", ["forecast_freeze", "target_availability", "chronology"])
def test_prospective_failed_invalid(field):
    r = evaluate_scientific_eligibility(*setup(**{field: finding(VerificationStatus.FAILED)}))
    assert r.assessment_validity is AssessmentValidity.INVALID


def test_exposed_prospective_claim_invalid():
    r = evaluate_scientific_eligibility(*setup(exposure=ExposureStatus.EXPOSED))
    assert r.assessment_validity is AssessmentValidity.INVALID
    assert "ASSESSMENT_TRACK_MISMATCH" in r.reason_codes


def test_invalid_precedes_exposure_classification():
    r = evaluate_scientific_eligibility(*setup(exposure=ExposureStatus.EXPOSED, technical_integrity=finding(VerificationStatus.FAILED)))
    assert r.assessment_validity is AssessmentValidity.INVALID


def test_binding_mismatch_rejected():
    d, e = setup()
    with pytest.raises(ValueError, match="digest mismatch"):
        evaluate_scientific_eligibility(replace(d, development_cutoff="2025-02-01"), e)


def test_deterministic_output_and_serializable():
    d, e = setup(chronology=finding(VerificationStatus.UNKNOWN))
    a = evaluate_scientific_eligibility(d, e).to_dict()
    b = evaluate_scientific_eligibility(d, e).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_evidence_downgrade_never_upgrades():
    d, e = setup()
    assert evaluate_scientific_eligibility(d, e).prospective_independence_eligible
    for field in ("technical_integrity", "protocol_compliance", "frozen_artifact_integrity",
                  "forecast_freeze", "target_availability", "chronology"):
        for status in (VerificationStatus.UNKNOWN, VerificationStatus.FAILED):
            assert not evaluate_scientific_eligibility(d, replace(e, **{field: finding(status)})).prospective_independence_eligible
    assert not evaluate_scientific_eligibility(*setup(exposure=ExposureStatus.UNKNOWN)).prospective_independence_eligible


def test_immutable_decision_and_refs():
    r = evaluate_scientific_eligibility(*setup())
    with pytest.raises(FrozenInstanceError):
        r.assessment_id = "other"
    with pytest.raises(TypeError):
        r.evidence_refs["chronology"] = "changed"


def test_pure_decision_cannot_claim_certification():
    r = evaluate_scientific_eligibility(*setup())
    with pytest.raises(ValueError, match="cannot issue certification"):
        replace(r, prospective_independence_certified=True)


def test_no_unverified_certification_by_serialized_input():
    r = evaluate_scientific_eligibility(*setup())
    payload = r.to_dict()
    assert payload["prospective_independence_certified"] is False
    assert payload["prospective_independence_eligible"] is True
