"""Focused D4-B2 contract tests; monkeypatch isolates the new adapter boundary.

Run adjacent D3/D2 real-path tests separately for upstream verification.
"""
import copy
import json

import pytest

from zenerestimation.assessment import consolidated_assessment as module
from zenerestimation.assessment.eligibility import (
    CONTRACT_VERSION, AssessmentDeclaration, AssessmentTrack,
    ExposureStatus, TrustedVerificationEvidence, VerificationFinding,
    VerificationStatus,
)


def _internal(status="available", winner="LSTM"):
    return {
        "assessment_version": "m19_d3_b2_selection_consistency_v1",
        "scope": "internal_expanding_window",
        "status": status,
        "dataset_id": "battery-410",
        "dataset_fingerprint": "fingerprint-410",
        "fold_count": 3,
        "model_count": 2,
        "fold_validation_dates": [["2023-03-01"], ["2023-06-01"], ["2023-09-01"]],
        "baseline_top_models": [winner] if winner else ["ARIMA", "LSTM"],
        "baseline_recommended_model": winner,
        "rank_stability_fraction": None if status != "available" else 2 / 3,
        "winner_set_stability_fraction": None if status != "available" else 1.0,
        "recommendation_consistency_fraction": None if status != "available" or not winner else 1.0,
        "consistency_relationship_verified": True,
        "selection_authorized": False,
    }


def _patch(monkeypatch, report):
    monkeypatch.setattr(module, "assess_internal_selection_consistency", lambda *args, **kwargs: report)


def _eligibility(dataset_id="battery-410", exposure=ExposureStatus.UNKNOWN):
    declaration = AssessmentDeclaration(
        contract_version=CONTRACT_VERSION,
        assessment_id="assessment-1", assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id=dataset_id, development_cutoff="2022-12-01",
        forecast_dates=("2023-03-01",),
    )
    unknown = VerificationFinding(status=VerificationStatus.UNKNOWN)
    verified = VerificationFinding(
        status=VerificationStatus.VERIFIED, evidence_ref="exposure-ref", verifier_id="verifier-1"
    )
    evidence = TrustedVerificationEvidence(
        assessment_id=declaration.assessment_id, dataset_id=dataset_id,
        declaration_digest=declaration.digest,
        technical_integrity=unknown, protocol_compliance=unknown,
        target_exposure=exposure,
        exposure_evidence=verified if exposure is not ExposureStatus.UNKNOWN else unknown,
        forecast_freeze=unknown, target_availability=unknown,
        chronology=unknown, frozen_artifact_integrity=unknown,
    )
    return declaration, evidence


def test_available_is_deterministic_and_diagnostic(monkeypatch):
    _patch(monkeypatch, _internal())
    a = module.build_consolidated_assessment({}, validation_context={})
    b = module.build_consolidated_assessment({}, validation_context={})
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["status"] == "available"
    assert a["scientific_eligibility"] is None
    assert a["scientific_boundaries"] == {
        "selection_authorized": False, "prospective_certification_issued": False,
        "protected_holdout_accessed": False, "optimization_executed": False,
    }


@pytest.mark.parametrize("status", ["unavailable_insufficient_folds", "unavailable_insufficient_models"])
def test_unavailable_is_not_zero(monkeypatch, status):
    _patch(monkeypatch, _internal(status=status))
    report = module.build_consolidated_assessment({}, validation_context={})
    assert report["status"] == status
    assert report["internal_assessment"]["rank_stability_fraction"] is None
    assert report["internal_assessment"]["recommendation_consistency_fraction"] is None


def test_tied_winner_remains_ambiguous(monkeypatch):
    _patch(monkeypatch, _internal(winner=None))
    report = module.build_consolidated_assessment({}, validation_context={})
    assert report["internal_assessment"]["baseline_recommended_model"] is None
    assert report["internal_assessment"]["recommendation_consistency_fraction"] is None


def test_detached_snapshot(monkeypatch):
    source = _internal()
    _patch(monkeypatch, source)
    report = module.build_consolidated_assessment({}, validation_context={})
    source["baseline_top_models"].append("ARIMA")
    assert report["internal_assessment"]["baseline_top_models"] == ["LSTM"]


@pytest.mark.parametrize("key,value", [
    ("assessment_version", "forged"), ("scope", "protected_holdout"),
    ("status", "unrecognized"), ("selection_authorized", True),
    ("consistency_relationship_verified", False),
])
def test_rejects_bad_upstream_contract(monkeypatch, key, value):
    source = _internal()
    source[key] = value
    _patch(monkeypatch, source)
    with pytest.raises(ValueError):
        module.build_consolidated_assessment({}, validation_context={})


def test_rejects_nonfinite_upstream_report(monkeypatch):
    source = _internal()
    source["rank_stability_fraction"] = float("nan")
    _patch(monkeypatch, source)
    with pytest.raises(ValueError):
        module.build_consolidated_assessment({}, validation_context={})


def test_rejects_unpaired_eligibility(monkeypatch):
    _patch(monkeypatch, _internal())
    declaration, _ = _eligibility()
    with pytest.raises(ValueError, match="together"):
        module.build_consolidated_assessment({}, validation_context={}, declaration=declaration)


def test_optional_eligibility_is_conditional(monkeypatch):
    _patch(monkeypatch, _internal())
    declaration, evidence = _eligibility()
    report = module.build_consolidated_assessment(
        {}, validation_context={}, declaration=declaration, verification_evidence=evidence
    )
    assert report["scientific_eligibility"]["prospective_independence_certified"] is False
    assert report["evidence_relationship"]["dataset_id_match"] is True
    assert report["evidence_relationship"]["cross_track_identity_verified"] is False


def test_cross_track_mismatch_is_explicit_not_authenticated(monkeypatch):
    _patch(monkeypatch, _internal())
    declaration, evidence = _eligibility(dataset_id="other-dataset")
    report = module.build_consolidated_assessment(
        {}, validation_context={}, declaration=declaration, verification_evidence=evidence
    )
    assert report["evidence_relationship"]["dataset_id_match"] is False
    assert report["evidence_relationship"]["cross_track_identity_verified"] is False


def test_rejects_mismatched_evidence_binding(monkeypatch):
    _patch(monkeypatch, _internal())
    declaration, evidence = _eligibility()
    other, _ = _eligibility(dataset_id="other-dataset")
    with pytest.raises(ValueError):
        module.build_consolidated_assessment(
            {}, validation_context={}, declaration=other, verification_evidence=evidence
        )


def test_invalid_inputs_rejected():
    with pytest.raises(TypeError):
        module.build_consolidated_assessment([], validation_context={})
