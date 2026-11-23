"""M19-B2-I2-A: isolated temporal policy tests using M18 verifier seams.

The existing M18 test suite covers verifier internals; these tests cover I2
cross-contract behavior and ensure that verifier failures are not upgraded.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from zenerestimation.assessment.eligibility import (
    CONTRACT_VERSION, AssessmentDeclaration, AssessmentTrack,
    VerificationFinding, VerificationStatus,
)
from zenerestimation.assessment import temporal as mod


DATES = ("2025-03-01", "2025-06-01")


def finding(status=VerificationStatus.UNKNOWN):
    if status is VerificationStatus.VERIFIED:
        return VerificationFinding(status, "trusted.ref", "trusted.verifier")
    if status is VerificationStatus.FAILED:
        return VerificationFinding(status, reason_codes=("SOURCE_FAILED",))
    return VerificationFinding(status)


def setup(monkeypatch, *, plan_verified=True, artifact_verified=True,
          endpoint="2024-12-01", dates=DATES, horizon=2,
          development=None, freeze=None, availability=None, chronology=None):
    declaration = AssessmentDeclaration(
        CONTRACT_VERSION, "assessment.1", AssessmentTrack.PROSPECTIVE,
        "battery.410", "2024-12-01", DATES,
        {"plan": "plan.ref", "execution_artifact": "execution.ref",
         "development": "development.ref"},
    )
    request = mod.TemporalEligibilityRequest(
        declaration, "plan.ref", "execution.ref", "development.ref")
    plan = {"forecast_horizon": horizon, "forecast_dates": list(dates),
            "training_provenance": {"train_end": endpoint}}
    calls = []
    def verify_plan(value, **kwargs):
        calls.append("plan")
        return plan_verified
    def verify_artifact(value, *, plan, **kwargs):
        calls.append("artifact")
        return artifact_verified
    monkeypatch.setattr(mod, "verify_internal_forecast_plan", verify_plan)
    monkeypatch.setattr(mod, "verify_internal_forecast_execution_artifact", verify_artifact)
    args = dict(request=request, plan=plan, execution_artifact={},
                selection_artifact={}, results={}, validation_context={},
                registry=None, training_dataset=None, training_context={},
                forecast_horizon=horizon, forecast_dates=list(dates),
                development_boundary_evidence=development or finding(),
                forecast_freeze_evidence=freeze or finding(),
                target_availability_evidence=availability or finding(),
                chronology_evidence=chronology or finding())
    return args, calls


def evaluate(monkeypatch, **kwargs):
    args, calls = setup(monkeypatch, **kwargs)
    return mod.evaluate_temporal_eligibility(**args), calls


def test_n01_training_after_cutoff(monkeypatch):
    result, _ = evaluate(monkeypatch, endpoint="2025-03-01")
    assert result.training_cutoff.status is VerificationStatus.FAILED
    assert "TRAINING_AFTER_DEVELOPMENT_CUTOFF" in result.training_cutoff.reason_codes


def test_n02_plan_dates_mismatch(monkeypatch):
    result, _ = evaluate(monkeypatch, dates=("2025-06-01", "2025-09-01"))
    assert result.plan_binding.status is VerificationStatus.FAILED


def test_n03_horizon_mismatch(monkeypatch):
    result, _ = evaluate(monkeypatch, horizon=3)
    assert result.plan_binding.status is VerificationStatus.FAILED


def test_n04_plan_verification_failure(monkeypatch):
    result, calls = evaluate(monkeypatch, plan_verified=False)
    assert result.plan_binding.status is VerificationStatus.FAILED
    assert result.training_cutoff.status is VerificationStatus.UNKNOWN
    assert result.execution_integrity.status is VerificationStatus.UNKNOWN
    assert calls == ["plan"]


def test_n05_execution_verification_failure(monkeypatch):
    result, calls = evaluate(monkeypatch, artifact_verified=False)
    assert result.execution_integrity.status is VerificationStatus.FAILED
    assert calls == ["plan", "artifact"]


def test_n06_shifted_execution_dates_are_verifier_responsibility(monkeypatch):
    result, _ = evaluate(monkeypatch, artifact_verified=False)
    assert "EXECUTION_ARTIFACT_VERIFICATION_FAILED" in result.execution_integrity.reason_codes


def test_n07_fingerprint_mismatch_is_plan_failure(monkeypatch):
    result, _ = evaluate(monkeypatch, plan_verified=False)
    assert result.plan_binding.status is VerificationStatus.FAILED


def test_n08_unknown_development(monkeypatch):
    result, _ = evaluate(monkeypatch)
    assert result.development_boundary.status is VerificationStatus.UNKNOWN
    assert result.training_cutoff.status is VerificationStatus.VERIFIED


def test_n09_failed_development_evidence(monkeypatch):
    result, _ = evaluate(monkeypatch, development=finding(VerificationStatus.FAILED))
    assert result.development_boundary.status is VerificationStatus.FAILED


def test_n10_unknown_freeze(monkeypatch):
    result, _ = evaluate(monkeypatch)
    assert result.forecast_freeze.status is VerificationStatus.UNKNOWN


def test_n11_failed_chronology(monkeypatch):
    result, _ = evaluate(monkeypatch, chronology=finding(VerificationStatus.FAILED))
    assert result.chronology.status is VerificationStatus.FAILED


def test_n12_retrospective_not_certified(monkeypatch):
    args, _ = setup(monkeypatch)
    d = args["request"].declaration
    retro = AssessmentDeclaration(d.contract_version, d.assessment_id,
        AssessmentTrack.RETROSPECTIVE, d.dataset_id, d.development_cutoff,
        d.forecast_dates, d.evidence_refs)
    args["request"] = mod.TemporalEligibilityRequest(
        retro, "plan.ref", "execution.ref", "development.ref")
    result = mod.evaluate_temporal_eligibility(**args)
    assert "certified" not in result.to_dict()
    assert result.plan_binding.status is VerificationStatus.VERIFIED


def test_n13_all_supplied_verified_does_not_certify(monkeypatch):
    v = finding(VerificationStatus.VERIFIED)
    result, _ = evaluate(monkeypatch, development=v, freeze=v,
                         availability=v, chronology=v)
    assert all(getattr(result, name).status is VerificationStatus.VERIFIED
               for name in ("plan_binding", "execution_integrity", "training_cutoff",
                            "development_boundary", "forecast_freeze",
                            "target_availability", "chronology"))
    assert "certified" not in result.to_dict()


def test_n14_deterministic_serialization(monkeypatch):
    first, _ = evaluate(monkeypatch)
    second, _ = evaluate(monkeypatch)
    assert first.to_dict() == second.to_dict()


def test_declaration_ref_mismatch_rejected(monkeypatch):
    args, _ = setup(monkeypatch)
    with pytest.raises(ValueError, match="mismatch"):
        mod.TemporalEligibilityRequest(args["request"].declaration,
                                       "other.ref", "execution.ref", "development.ref")


def test_malformed_finding_rejected(monkeypatch):
    args, _ = setup(monkeypatch)
    args["chronology_evidence"] = "verified"
    with pytest.raises(TypeError, match="VerificationFinding"):
        mod.evaluate_temporal_eligibility(**args)


def test_request_is_frozen(monkeypatch):
    args, _ = setup(monkeypatch)
    with pytest.raises(FrozenInstanceError):
        args["request"].plan_ref = "other.ref"


def test_training_equal_cutoff_is_allowed(monkeypatch):
    result, _ = evaluate(monkeypatch, endpoint="2024-12-01")
    assert result.training_cutoff.status is VerificationStatus.VERIFIED


def test_no_execution_verification_on_bad_plan(monkeypatch):
    result, calls = evaluate(monkeypatch, plan_verified=False, artifact_verified=True)
    assert calls == ["plan"]
    assert result.execution_integrity.status is VerificationStatus.UNKNOWN
