"""M19-B2-I2-C1 integration against the REAL M18 execution artifact verifier.

Uses the synthetic M18 execution fixture; no real measurement authentication or
independent execution attestation is implied.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

# Register the transitive fixture required by M18 prepared.
from tests.test_m18_forecast_plan_contract import training as training_fixture

from tests.test_m18_forecast_execution_contract import (
    ProbeModel,
    prepared as execution_prepared,
    execute,
)
from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration,
    AssessmentTrack,
    CONTRACT_VERSION,
    VerificationFinding,
    VerificationStatus,
)
from zenerestimation.assessment.observability import (
    TargetObservation,
    TargetObservationEvidence,
    TargetOrigin,
    build_target_observability_report,
)
from zenerestimation.assessment.observed_scoring import align_verified_observations
from zenerestimation.optimization.internal_forecast_execution_artifact import (
    build_internal_forecast_execution_artifact,
    verify_internal_forecast_execution_artifact,
)


@pytest.fixture
def integrated_case(execution_prepared):
    plan, kwargs = execution_prepared
    execution = execute(plan, kwargs)
    artifact = build_internal_forecast_execution_artifact(execution, plan=plan, **kwargs)
    # No execution should occur during either verification or alignment.
    ProbeModel.events = []
    dates = tuple(plan["forecast_dates"])
    declaration = AssessmentDeclaration(
        CONTRACT_VERSION,
        "m19.m18.integration",
        AssessmentTrack.RETROSPECTIVE,
        "battery.410",
        "2023-12-01",
        dates,
        {"execution_artifact": "m18.execution.fixture"},
    )
    verified = VerificationFinding(
        VerificationStatus.VERIFIED, "measurement.evidence", "trusted.verifier"
    )
    unknown = VerificationFinding(VerificationStatus.UNKNOWN)
    observations = (
        TargetObservationEvidence(
            TargetObservation("battery.410", dates[0], 29.9, TargetOrigin.MEASURED,
                              "measurement.source.1"),
            verified, unknown, unknown,
        ),
        TargetObservationEvidence(
            TargetObservation("battery.410", dates[1], None, TargetOrigin.MISSING),
            unknown, unknown, unknown,
        ),
    )
    report = build_target_observability_report(declaration, observations)
    align_kwargs = dict(
        declaration=declaration,
        observability_report=report,
        execution_artifact=artifact,
        plan=plan,
        **kwargs,
    )
    return align_kwargs


def test_real_m18_verifier_aligns_forecast_and_partial_observations(integrated_case):
    args = integrated_case
    assert verify_internal_forecast_execution_artifact(
        args["execution_artifact"],
        plan=args["plan"],
        **{k: args[k] for k in (
            "artifact", "results", "validation_context", "registry",
            "training_dataset", "training_context", "forecast_horizon", "forecast_dates"
        )},
    ) is True
    # The alignment entry point supplies horizon/dates from declaration itself.
    alignment_args = {k: v for k, v in args.items()
                      if k not in ("forecast_horizon", "forecast_dates")}
    result = align_verified_observations(**alignment_args)
    assert result.declared_horizon == 2
    assert result.coverage_mask == (True, False)
    assert (result.eligible_count, result.excluded_count) == (1, 1)
    assert tuple(row.target_date for row in result.rows) == tuple(args["plan"]["forecast_dates"])
    assert tuple(row.forecast_microVolt for row in result.rows) == (29.4, 29.5)
    assert result.rows[0].observed_microVolt == 29.9
    assert result.rows[1].observed_microVolt is None
    assert "ORIGIN_NOT_MEASURED" in result.rows[1].exclusion_reasons
    assert result.execution_artifact_digest == args["execution_artifact"]["artifact_digest"]
    assert "rmse" not in result.to_dict() and "mae" not in result.to_dict()
    assert ProbeModel.events == []


def test_real_m18_verifier_rejects_unresealed_forecast_tampering(integrated_case):
    args = dict(integrated_case)
    args["execution_artifact"] = deepcopy(args["execution_artifact"])
    args["execution_artifact"]["execution_record"]["forecast"]["values_microVolt"][0] = 999.0
    args.pop("forecast_horizon")
    args.pop("forecast_dates")
    with pytest.raises(ValueError, match="execution artifact verification failed"):
        align_verified_observations(**args)
    assert ProbeModel.events == []


def test_real_m18_verifier_rejects_changed_training_context(integrated_case):
    args = dict(integrated_case)
    args["training_context"] = deepcopy(args["training_context"])
    args["training_context"]["dataset_fingerprint"]["digest"] = "0" * 64
    args.pop("forecast_horizon")
    args.pop("forecast_dates")
    with pytest.raises(ValueError, match="execution artifact verification failed"):
        align_verified_observations(**args)
    assert ProbeModel.events == []
