"""M19-B2-I2-C3 real M18 verifier -> I2-A/B/C1/C2/C3 integration.

The M18 execution is synthetic. Successful verification proves consistency with
supplied inputs, not authentic measurements, development history, or independence.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

# Explicitly register transitive pytest fixtures from the M18 test modules.
from tests.test_m18_forecast_plan_contract import training as training_fixture
from tests.test_m18_forecast_execution_contract import prepared as execution_prepared
from tests.test_m19_b2_i2_c1_m18_integration import integrated_case, ProbeModel

from zenerestimation.assessment.eligibility import (
    VerificationFinding, VerificationStatus,
)
from zenerestimation.assessment.temporal import (
    TemporalEligibilityRequest, evaluate_temporal_eligibility,
)
from zenerestimation.assessment.observed_scoring import align_verified_observations
from zenerestimation.assessment.observed_metrics import score_verified_observations
from zenerestimation.assessment.evidence_integration import (
    TemporalSummaryStatus, integrate_assessment_evidence,
)


def _finding(status: VerificationStatus) -> VerificationFinding:
    if status is VerificationStatus.VERIFIED:
        return VerificationFinding(
            status, evidence_ref="supplied.chronology.evidence",
            verifier_id="synthetic.test.verifier",
        )
    if status is VerificationStatus.FAILED:
        return VerificationFinding(status, reason_codes=("CHRONOLOGY_FAILED",))
    return VerificationFinding(status)


def _temporal(case, *, chronology=VerificationStatus.UNKNOWN):
    declaration = case["declaration"]
    request = TemporalEligibilityRequest(
        declaration=declaration,
        plan_ref="m18.plan.fixture",
        execution_artifact_ref="m18.execution.fixture",
        development_evidence_ref="development.fixture",
    )
    unknown = _finding(VerificationStatus.UNKNOWN)
    return evaluate_temporal_eligibility(
        request,
        plan=case["plan"],
        execution_artifact=case["execution_artifact"],
        selection_artifact=case["artifact"],
        results=case["results"],
        validation_context=case["validation_context"],
        registry=case["registry"],
        training_dataset=case["training_dataset"],
        training_context=case["training_context"],
        forecast_horizon=case["forecast_horizon"],
        forecast_dates=case["forecast_dates"],
        development_boundary_evidence=unknown,
        forecast_freeze_evidence=unknown,
        target_availability_evidence=unknown,
        chronology_evidence=_finding(chronology),
    )


def _alignment(case):
    # C1 obtains forecast horizon/dates from the assessment declaration.
    return align_verified_observations(**{
        key: value for key, value in case.items()
        if key not in ("forecast_horizon", "forecast_dates")
    })


def _pipeline(case, *, chronology=VerificationStatus.UNKNOWN):
    temporal = _temporal(case, chronology=chronology)
    alignment = _alignment(case)
    metrics = score_verified_observations(alignment)
    integrated = integrate_assessment_evidence(
        declaration=case["declaration"],
        temporal_evidence=temporal,
        observability_report=case["observability_report"],
        alignment_result=alignment,
        metric_result=metrics,
    )
    return temporal, alignment, metrics, integrated


def test_c3_real_r1_verified_m18_chain_partial_coverage_and_metrics(integrated_case):
    temporal, alignment, metrics, integrated = _pipeline(integrated_case)

    assert temporal.plan_binding.status is VerificationStatus.VERIFIED
    assert temporal.execution_integrity.status is VerificationStatus.VERIFIED
    assert temporal.training_cutoff.status is VerificationStatus.VERIFIED
    assert alignment.coverage_mask == (True, False)
    assert integrated.coverage_status == "partial"
    assert integrated.score_available is True
    assert integrated.execution_artifact_digest == (
        integrated_case["execution_artifact"]["artifact_digest"]
    )
    assert metrics.rmse_microVolt == pytest.approx(0.5)
    assert metrics.mae_microVolt == pytest.approx(0.5)
    assert metrics.mape_percent == pytest.approx(100 * 0.5 / 29.9)
    assert integrated.to_dict()["evidence"]["observed_metrics"]["eligible_count"] == 1
    assert ProbeModel.events == []


def test_c3_real_r2_tampered_forecast_rejected_before_integration(integrated_case):
    case = dict(integrated_case)
    case["execution_artifact"] = deepcopy(case["execution_artifact"])
    case["execution_artifact"]["execution_record"]["forecast"]["values_microVolt"][0] = 999.0

    # Temporal policy records M18 failure as evidence; C1 rejects the artifact.
    temporal = _temporal(case)
    assert temporal.execution_integrity.status is VerificationStatus.FAILED
    with pytest.raises(ValueError, match="execution artifact verification failed"):
        _alignment(case)
    assert ProbeModel.events == []



def test_c3_real_r3_changed_training_fingerprint_rejected_before_integration(
    integrated_case,
):
    case = dict(integrated_case)
    case["training_context"] = deepcopy(case["training_context"])
    case["training_context"]["dataset_fingerprint"]["digest"] = "0" * 64

    # The real M18 plan verifier validates trusted training provenance
    # before execution-artifact verification.
    #
    # An invalid fingerprint raises ValueError immediately; I2-A does
    # not convert malformed trusted inputs into temporal findings.
    with pytest.raises(
        ValueError,
        match="training dataset does not match trusted fingerprint",
    ):
        _temporal(case)

    # C1 independently rejects the same invalid trusted input.
    with pytest.raises(ValueError):
        _alignment(case)

    # Verification and integration must not rerun the forecasting model.
    assert ProbeModel.events == []



def test_c3_real_r4_unknown_temporal_findings_remain_incomplete(integrated_case):
    temporal, _, metrics, integrated = _pipeline(integrated_case)
    assert temporal.chronology.status is VerificationStatus.UNKNOWN
    assert temporal.forecast_freeze.status is VerificationStatus.UNKNOWN
    assert integrated.temporal_status is TemporalSummaryStatus.INCOMPLETE
    assert integrated.score_available is True
    assert integrated.to_dict()["temporal_status"] == "incomplete"
    assert metrics.rmse_microVolt == pytest.approx(0.5)
    assert "certified" not in str(integrated.to_dict()).lower()
    assert ProbeModel.events == []


def test_c3_real_r5_failed_chronology_preserved_with_observed_score(integrated_case):
    temporal, _, metrics, integrated = _pipeline(
        integrated_case, chronology=VerificationStatus.FAILED
    )
    assert temporal.chronology.status is VerificationStatus.FAILED
    assert temporal.chronology.reason_codes == ("CHRONOLOGY_FAILED",)
    assert integrated.temporal_status is TemporalSummaryStatus.FAILED
    assert integrated.score_available is True
    assert integrated.to_dict()["evidence"]["temporal"]["chronology"]["status"] == "failed"
    assert metrics.rmse_microVolt == pytest.approx(0.5)
    assert ProbeModel.events == []


def test_c3_real_r6_no_model_rerun_during_verification_and_integration(integrated_case):
    assert ProbeModel.events == []  # The fixture clears initial synthetic execution.
    _pipeline(integrated_case)
    assert ProbeModel.events == []
