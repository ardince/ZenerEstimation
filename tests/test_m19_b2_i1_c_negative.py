"""M19-B2-I1-C adversarial policy tests, synthetic evidence only.

These tests do NOT authenticate chronology, execution, or verifier identity.
"""
from dataclasses import FrozenInstanceError, replace
import json
import pytest

from zenerestimation.assessment import (
    CONTRACT_VERSION, AssessmentDeclaration, AssessmentTrack, AssessmentValidity,
    ExposureStatus, ScientificClassification, TrustedVerificationEvidence,
    VerificationFinding, VerificationStatus, evaluate_scientific_eligibility,
)
from zenerestimation.assessment.decision import ScientificEligibilityDecision


def finding(status=VerificationStatus.VERIFIED, tag="base"):
    return VerificationFinding(
        status=status, evidence_ref=f"evidence-{tag}", verifier_id=f"verifier-{tag}",
        reason_codes=("NEGATIVE_CHECK_FAILED",) if status is VerificationStatus.FAILED else (),
    )


def pair(track=AssessmentTrack.PROSPECTIVE, exposure=ExposureStatus.NOT_EXPOSED, **changes):
    d = AssessmentDeclaration(CONTRACT_VERSION, "assessment-a", track, "battery-a", "2025-03-01", ("2025-06-01", "2025-09-01"), {})
    e = TrustedVerificationEvidence(
        assessment_id=d.assessment_id, dataset_id=d.dataset_id, declaration_digest=d.digest,
        technical_integrity=finding(tag="tech"), protocol_compliance=finding(tag="protocol"),
        target_exposure=exposure,
        exposure_evidence=finding(VerificationStatus.UNKNOWN, "exposure") if exposure is ExposureStatus.UNKNOWN else finding(tag="exposure"),
        forecast_freeze=finding(tag="freeze"), target_availability=finding(tag="availability"),
        chronology=finding(tag="chronology"), frozen_artifact_integrity=finding(tag="artifact"),
    )
    return d, replace(e, **changes)


@pytest.mark.parametrize("field", ["technical_integrity", "protocol_compliance", "frozen_artifact_integrity"])
def test_c01_core_failure_dominates_complete_prospective(field):
    d, e = pair(**{field: finding(VerificationStatus.FAILED, field)})
    r = evaluate_scientific_eligibility(d, e)
    assert r.assessment_validity is AssessmentValidity.INVALID
    assert r.scientific_classification is None
    assert not r.prospective_independence_eligible
    assert not r.prospective_independence_certified


@pytest.mark.parametrize("field", ["technical_integrity", "protocol_compliance", "frozen_artifact_integrity"])
def test_c02_unknown_core_never_certifies_even_if_other_evidence_verified(field):
    r = evaluate_scientific_eligibility(*pair(**{field: finding(VerificationStatus.UNKNOWN, field)}))
    assert r.assessment_validity is AssessmentValidity.INCOMPLETE
    assert r.scientific_classification is None
    assert not r.prospective_independence_eligible


def test_c03_failed_core_dominates_unknown_core():
    r = evaluate_scientific_eligibility(*pair(technical_integrity=finding(VerificationStatus.FAILED), protocol_compliance=finding(VerificationStatus.UNKNOWN)))
    assert r.assessment_validity is AssessmentValidity.INVALID


def test_c04_prospective_exposure_conflict_rejected_despite_good_chronology():
    r = evaluate_scientific_eligibility(*pair(exposure=ExposureStatus.EXPOSED))
    assert r.assessment_validity is AssessmentValidity.INVALID
    assert "ASSESSMENT_TRACK_MISMATCH" in r.reason_codes
    assert not r.prospective_independence_certified


def test_c05_retrospective_unknown_exposure_not_upgraded_to_controlled():
    r = evaluate_scientific_eligibility(*pair(AssessmentTrack.RETROSPECTIVE, ExposureStatus.UNKNOWN))
    assert r.assessment_validity is AssessmentValidity.VALID
    assert r.scientific_classification is ScientificClassification.INDEPENDENCE_UNVERIFIED
    assert "TARGET_EXPOSURE_UNKNOWN" in r.reason_codes


def test_c06_retrospective_known_exposure_is_controlled_not_independent():
    r = evaluate_scientific_eligibility(*pair(AssessmentTrack.RETROSPECTIVE, ExposureStatus.EXPOSED))
    assert r.scientific_classification is ScientificClassification.CONTROLLED_RETROSPECTIVE
    assert not r.prospective_independence_eligible


@pytest.mark.parametrize("field", ["forecast_freeze", "target_availability", "chronology"])
def test_c07_unknown_prospective_component_blocks_eligibility(field):
    r = evaluate_scientific_eligibility(*pair(**{field: finding(VerificationStatus.UNKNOWN, field)}))
    assert r.scientific_classification is ScientificClassification.INDEPENDENCE_UNVERIFIED
    assert not r.prospective_independence_eligible


@pytest.mark.parametrize("field", ["forecast_freeze", "target_availability", "chronology"])
def test_c08_failed_prospective_component_invalidates(field):
    r = evaluate_scientific_eligibility(*pair(**{field: finding(VerificationStatus.FAILED, field)}))
    assert r.assessment_validity is AssessmentValidity.INVALID
    assert not r.prospective_independence_eligible


def test_c09_unknown_exposure_plus_failed_chronology_is_invalid():
    r = evaluate_scientific_eligibility(*pair(exposure=ExposureStatus.UNKNOWN, chronology=finding(VerificationStatus.FAILED)))
    assert r.assessment_validity is AssessmentValidity.INVALID


def test_c10_declared_nonexposure_without_verified_finding_rejected():
    with pytest.raises(ValueError, match="known exposure status requires verified"):
        pair(exposure_evidence=finding(VerificationStatus.UNKNOWN, "exposure"))


def test_c11_verified_exposure_finding_with_unknown_conclusion_rejected():
    with pytest.raises(ValueError, match="verified exposure finding"):
        pair(exposure=ExposureStatus.UNKNOWN, exposure_evidence=finding(tag="exposure"))


@pytest.mark.parametrize("change", [
    {"assessment_id": "other-assessment"},
    {"dataset_id": "other-battery"},
    {"development_cutoff": "2025-02-01"},
    {"forecast_dates": ("2025-09-01", "2025-12-01")},
])
def test_c12_changed_declaration_rejected_without_new_evidence(change):
    d, e = pair()
    with pytest.raises(ValueError, match="mismatch"):
        evaluate_scientific_eligibility(replace(d, **change), e)


def test_c13_changed_digest_to_wrong_but_well_formed_value_rejected():
    d, e = pair()
    with pytest.raises(ValueError, match="digest mismatch"):
        evaluate_scientific_eligibility(d, replace(e, declaration_digest="sha256:" + "0" * 64))


def test_c14_frozen_artifact_failure_blocks_eligibility():
    r = evaluate_scientific_eligibility(*pair(frozen_artifact_integrity=finding(VerificationStatus.FAILED, "mutated")))
    assert r.assessment_validity is AssessmentValidity.INVALID


def test_c15_downgrade_any_required_finding_never_upgrades():
    d, e = pair()
    assert evaluate_scientific_eligibility(d, e).prospective_independence_eligible
    for field in ("technical_integrity", "protocol_compliance", "frozen_artifact_integrity", "forecast_freeze", "target_availability", "chronology"):
        for status in (VerificationStatus.UNKNOWN, VerificationStatus.FAILED):
            r = evaluate_scientific_eligibility(d, replace(e, **{field: finding(status, field)}))
            assert not r.prospective_independence_eligible
            assert not r.prospective_independence_certified


def test_c16_deterministic_reasons_for_simultaneous_failures():
    d, e = pair(technical_integrity=finding(VerificationStatus.FAILED), protocol_compliance=finding(VerificationStatus.FAILED))
    r1 = evaluate_scientific_eligibility(d, e).to_dict()
    r2 = evaluate_scientific_eligibility(d, e).to_dict()
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    assert r1["reason_codes"] == sorted(set(r1["reason_codes"]))


def test_c17_pure_policy_never_certifies_even_with_all_verified():
    r = evaluate_scientific_eligibility(*pair())
    assert r.prospective_independence_eligible
    assert r.prospective_independence_certified is False
    assert r.to_dict()["prospective_independence_certified"] is False


def test_c18_direct_certificate_forgery_rejected():
    r = evaluate_scientific_eligibility(*pair())
    with pytest.raises(ValueError, match="cannot issue certification"):
        replace(r, prospective_independence_certified=True)


def test_c19_no_execution_or_chronology_authentication_claim():
    # A self-reported VERIFIED chronology is only a policy input. No signature,
    # trusted timestamp, measurement-availability audit or external execution proof
    # is established by this test or by the pure decision function.
    r = evaluate_scientific_eligibility(*pair())
    assert r.prospective_independence_eligible
    assert not r.prospective_independence_certified


def test_c20_decision_is_immutable():
    r = evaluate_scientific_eligibility(*pair())
    with pytest.raises(FrozenInstanceError):
        r.assessment_id = "forged"
    with pytest.raises(TypeError):
        r.evidence_refs["chronology"] = "forged"
