"""M19-B2-I2-D: focused adversarial cross-contract acceptance checks.

These tests distinguish consistency verification from external authentication.
D-01 intentionally demonstrates a trust limitation, not a security guarantee.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from math import sqrt

import pytest

# Register the complete fixture chain for the real M18 path.
from tests.test_m18_forecast_plan_contract import training as training_fixture
from tests.test_m18_forecast_execution_contract import prepared as execution_prepared
from tests.test_m19_b2_i2_c1_m18_integration import integrated_case, ProbeModel
# Pytest fixtures imported from other test modules must include their dependencies.
from tests.test_m19_b2_i2_c1_alignment import case as c1_case
from tests.test_m19_b2_i2_c3_evidence_integration import (
    integrated_args, finding, temporal_for, rebind_observations,
)
from tests.test_m19_b2_i2_c3_m18_real_path import _alignment

from zenerestimation.assessment.eligibility import VerificationStatus
from zenerestimation.assessment.observability import TargetOrigin
from zenerestimation.assessment.evidence_integration import (
    TemporalSummaryStatus, integrate_assessment_evidence,
)


def _integrate(args):
    return integrate_assessment_evidence(**args)


def _all_measured(args, *, values):
    """Construct an internally consistent B/C1/C2 trio for the same declaration.

    Deliberately does not reauthenticate measurement source or rerun M18.
    """
    old = args["observability_report"].observations
    assert len(old) == len(values)
    verified = finding(VerificationStatus.VERIFIED)
    observations = tuple(
        replace(item, provenance=verified,
                observation=replace(item.observation,
                    origin=TargetOrigin.MEASURED,
                    value_microVolt=float(value),
                    source_ref="synthetic.measurement.source"))
        for item, value in zip(old, values)
    )
    return rebind_observations(args, observations)


def test_d01_coordinated_observation_change_is_consistent_but_not_authenticated(integrated_args):
    """A consistent upstream rewrite can pass C3: document the trust boundary."""
    baseline = _integrate(integrated_args)
    old = integrated_args["observability_report"].observations
    rewritten = (
        replace(old[0], observation=replace(old[0].observation,
                                           value_microVolt=41.0,
                                           source_ref="different.claimed.source")),
        *old[1:],
    )
    changed = rebind_observations(integrated_args, rewritten)
    result = _integrate(changed)

    assert result.metric_result.rmse_microVolt != baseline.metric_result.rmse_microVolt
    assert result.metric_result.rmse_microVolt == pytest.approx(11.0)
    assert result.declaration_digest == baseline.declaration_digest
    assert result.execution_artifact_digest == baseline.execution_artifact_digest
    assert result.to_dict()["evidence"]["observability"] != baseline.to_dict()["evidence"]["observability"]
    assert "certified" not in str(result.to_dict()).lower()
    # Passing means *cross-contract consistency*, not measurement authentication.


def test_d02_all_verified_findings_do_not_claim_scientific_certification(integrated_args):
    result = _integrate(integrated_args)
    payload = result.to_dict()
    assert result.temporal_status is TemporalSummaryStatus.VERIFIED_FINDINGS
    assert payload["temporal_status"] == "verified_findings"
    assert "scientific_classification" not in payload
    assert "certified" not in str(payload).lower()
    assert all(
        payload["evidence"]["temporal"][name]["status"] == "verified"
        for name in (
            "plan_binding", "execution_integrity", "training_cutoff",
            "development_boundary", "forecast_freeze", "target_availability",
            "chronology",
        )
    )


def test_d03_invalid_real_m18_artifact_never_reaches_integrated_assessment(integrated_case):
    case = dict(integrated_case)
    case["execution_artifact"] = deepcopy(case["execution_artifact"])
    case["execution_artifact"]["execution_record"]["forecast"]["values_microVolt"][0] = 999.0

    # The real M18 verifier is used by C1; no C3 result may be built from it.
    with pytest.raises(ValueError, match="execution artifact verification failed"):
        _alignment(case)
    assert ProbeModel.events == []


@pytest.mark.parametrize("status,expected", [
    (VerificationStatus.UNKNOWN, TemporalSummaryStatus.INCOMPLETE),
    (VerificationStatus.FAILED, TemporalSummaryStatus.FAILED),
])
def test_d04_complete_observation_coverage_does_not_override_temporal_status(
    integrated_args, status, expected,
):
    args = _all_measured(integrated_args, values=(31.0, 32.0, 33.0))
    args["temporal_evidence"] = temporal_for(
        args["declaration"], {"chronology": status},
    )
    result = _integrate(args)
    assert result.coverage_status == "complete"
    assert result.score_available is True
    assert result.metric_result.eligible_count == 3
    assert result.temporal_status is expected
    assert result.to_dict()["temporal_status"] == expected.value


def test_d05_zero_actual_and_missing_target_keep_metric_denominators_separate(integrated_args):
    old = integrated_args["observability_report"].observations
    verified = finding(VerificationStatus.VERIFIED)
    observations = (
        replace(old[0], observation=replace(old[0].observation, value_microVolt=0.0)),
        replace(old[1], provenance=verified,
                observation=replace(old[1].observation, value_microVolt=32.0)),
        old[2],  # Missing quarter must remain excluded.
    )
    args = rebind_observations(integrated_args, observations)
    result = _integrate(args)
    metrics = result.metric_result
    assert result.coverage_status == "partial"
    assert metrics.coverage_mask == (True, True, False)
    assert (metrics.rmse_denominator, metrics.mae_denominator, metrics.mape_denominator) == (2, 2, 1)
    assert metrics.rmse_microVolt == pytest.approx(sqrt((30.0 ** 2 + 1.5 ** 2) / 2))
    assert metrics.mae_microVolt == pytest.approx((30.0 + 1.5) / 2)
    assert metrics.mape_percent == pytest.approx(100.0 * 1.5 / 32.0)
    assert metrics.mape_zero_actual_excluded_dates == (old[0].observation.target_date,)
    assert metrics.excluded_dates == (old[2].observation.target_date,)


def test_d06_sealed_serialization_unchanged_after_mutating_nested_metric_sources(integrated_args):
    result = _integrate(integrated_args)
    snapshot = result.to_dict()
    assert result.metric_result.alignment_rows
    result.metric_result.alignment_rows[0]["forecast_microVolt"] = -999.0
    assert result.to_dict() == snapshot
    detached = result.to_dict()
    detached["evidence"]["observed_metrics"]["alignment_rows"][0]["forecast_microVolt"] = -888.0
    assert result.to_dict() == snapshot
    # Typed sources remain inspectable but are not a deep-immutable evidence store.
    assert result.metric_result.alignment_rows[0]["forecast_microVolt"] == -999.0
