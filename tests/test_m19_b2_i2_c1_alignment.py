"""M19-B2-I2-C1 alignment contract tests; M18 verifier is an isolated seam."""
from dataclasses import FrozenInstanceError, replace
from copy import deepcopy

import pytest

from zenerestimation.assessment.eligibility import (
    CONTRACT_VERSION, AssessmentDeclaration, AssessmentTrack,
    VerificationFinding, VerificationStatus,
)
from zenerestimation.assessment.observability import (
    TargetObservation, TargetObservationEvidence, TargetOrigin,
    TargetObservabilityReport, build_target_observability_report,
)
from zenerestimation.assessment import observed_scoring as mod

DATES = ("2025-03-01", "2025-06-01", "2025-09-01")


def finding(status):
    if status is VerificationStatus.VERIFIED:
        return VerificationFinding(status, "measurement.ref", "verifier.ref")
    return VerificationFinding(status)


def observation(date, origin=TargetOrigin.MEASURED, verified=True):
    missing = origin is TargetOrigin.MISSING
    return TargetObservationEvidence(
        TargetObservation("410", date, None if missing else 31.0,
                          origin, None if missing else "source.ref"),
        finding(VerificationStatus.VERIFIED if verified else VerificationStatus.UNKNOWN),
        finding(VerificationStatus.UNKNOWN),
        finding(VerificationStatus.UNKNOWN),
    )


@pytest.fixture
def case(monkeypatch):
    declaration = AssessmentDeclaration(
        CONTRACT_VERSION, "assessment.410", AssessmentTrack.RETROSPECTIVE,
        "410", "2024-12-01", DATES,
    )
    report = build_target_observability_report(declaration, (
        observation(DATES[0]), observation(DATES[1], verified=False),
        observation(DATES[2], origin=TargetOrigin.MISSING, verified=False),
    ))
    plan = {"forecast_horizon": 3, "forecast_dates": list(DATES)}
    execution = {
        "artifact_digest": "sha256:" + "a" * 64,
        "execution_record": {"forecast": {
            "horizon": 3, "dates": list(DATES),
            "values_microVolt": [30.0, 30.5, 31.0],
        }},
    }
    calls = []
    def verify(*args, **kwargs):
        calls.append(kwargs)
        return True
    monkeypatch.setattr(mod, "verify_internal_forecast_execution_artifact", verify)
    args = dict(declaration=declaration, observability_report=report,
                execution_artifact=execution, plan=plan, artifact={}, results={},
                validation_context={}, registry=None, training_dataset=None,
                training_context={})
    return args, calls


def test_partial_coverage_preserves_all_quarters(case):
    args, calls = case
    result = mod.align_verified_observations(**args)
    assert result.declared_horizon == 3
    assert result.coverage_mask == (True, False, False)
    assert (result.eligible_count, result.excluded_count) == (1, 2)
    assert [r.target_date for r in result.rows] == list(DATES)
    assert result.rows[0].observed_microVolt == 31.0
    assert result.rows[1].observed_microVolt is None
    assert result.rows[1].observation_evidence.observation.value_microVolt == 31.0
    assert result.rows[1].exclusion_reasons == ("PROVENANCE_NOT_VERIFIED",)
    assert calls[0]["forecast_horizon"] == 3
    assert calls[0]["forecast_dates"] == list(DATES)


def test_no_metrics_and_deterministic_serialization(case):
    args, _ = case
    a = mod.align_verified_observations(**args).to_dict()
    b = mod.align_verified_observations(**args).to_dict()
    assert a == b
    assert "rmse" not in str(a).lower()
    assert "mae" not in str(a).lower()


def test_zero_coverage_is_valid(case):
    args, _ = case
    args["observability_report"] = build_target_observability_report(
        args["declaration"], tuple(observation(d, TargetOrigin.MISSING, False) for d in DATES)
    )
    result = mod.align_verified_observations(**args)
    assert result.eligible_count == 0
    assert result.coverage_mask == (False, False, False)


def test_all_verified_is_valid(case):
    args, _ = case
    args["observability_report"] = build_target_observability_report(
        args["declaration"], tuple(observation(d) for d in DATES)
    )
    assert mod.align_verified_observations(**args).eligible_count == 3


def test_mismatched_digest_rejected_before_verifier(case):
    args, calls = case
    args["observability_report"] = replace(
        args["observability_report"], declaration_digest="sha256:" + "0" * 64
    )
    with pytest.raises(ValueError, match="binding"):
        mod.align_verified_observations(**args)
    assert calls == []


def test_wrong_report_calendar_rejected(case):
    args, calls = case
    changed = replace(args["observability_report"], observations=(
        observation(DATES[0]), observation(DATES[1]), observation("2025-12-01")
    ))
    args["observability_report"] = changed
    with pytest.raises(ValueError, match="dates must match"):
        mod.align_verified_observations(**args)
    assert calls == []


@pytest.mark.parametrize("change", [
    {"forecast_horizon": 2},
    {"forecast_dates": ["2025-06-01", "2025-09-01", "2025-12-01"]},
])
def test_plan_mismatch_rejected(case, change):
    args, calls = case
    args["plan"] = {**args["plan"], **change}
    with pytest.raises(ValueError, match="plan"):
        mod.align_verified_observations(**args)
    assert calls == []


def test_verifier_failure_is_fatal(case, monkeypatch):
    args, _ = case
    monkeypatch.setattr(mod, "verify_internal_forecast_execution_artifact", lambda *a, **k: False)
    with pytest.raises(ValueError, match="verification failed"):
        mod.align_verified_observations(**args)


@pytest.mark.parametrize("mutation", [
    lambda f: f.update(horizon=2),
    lambda f: f.update(dates=list(reversed(DATES))),
    lambda f: f.update(values_microVolt=[30.0]),
    lambda f: f.update(values_microVolt=[30.0, float("nan"), 31.0]),
    lambda f: f.update(values_microVolt=[30.0, True, 31.0]),
])
def test_bad_forecast_rejected_even_if_verifier_seam_says_true(case, mutation):
    args, _ = case
    args["execution_artifact"] = deepcopy(args["execution_artifact"])
    mutation(args["execution_artifact"]["execution_record"]["forecast"])
    with pytest.raises(ValueError, match="forecast"):
        mod.align_verified_observations(**args)


def test_frozen_result_and_rows(case):
    args, _ = case
    result = mod.align_verified_observations(**args)
    with pytest.raises(FrozenInstanceError):
        result.assessment_id = "changed"
    with pytest.raises(FrozenInstanceError):
        result.rows[0].forecast_microVolt = 99.0
    assert isinstance(result.rows, tuple)


def test_serialized_mutation_does_not_change_result(case):
    args, _ = case
    result = mod.align_verified_observations(**args)
    serialized = result.to_dict()
    serialized["rows"][0]["forecast_microVolt"] = 999.0
    assert result.rows[0].forecast_microVolt == 30.0
