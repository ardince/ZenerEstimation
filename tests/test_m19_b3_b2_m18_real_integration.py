"""M19-B3-B2 integration with genuine M18 execution and B1 eligibility.

Only the deterministic ProbeModel is used. Observation verification findings
are fixture assertions, not independently authenticated measurements.
"""
from copy import deepcopy

import pandas as pd
import pytest

from tests.test_m18_forecast_plan_contract import training as processed_training
from tests.test_m19_b3_b1_m18_real_integration import (
    declaration, full_frame, _real_case, _reset_probe,
)
from zenerestimation.assessment.eligibility import VerificationFinding, VerificationStatus
from zenerestimation.assessment.holdout_evaluation_eligibility import (
    establish_holdout_evaluation_eligibility,
)
from zenerestimation.assessment.holdout_target_disclosure import disclose_holdout_targets
from zenerestimation.assessment.observability import (
    TargetObservation, TargetObservationEvidence, TargetOrigin,
    build_target_observability_report,
)
from tests.test_m18_forecast_execution_contract import ProbeModel


def _report(declaration, isolated, *, verified=True):
    provenance = VerificationFinding(
        status=VerificationStatus.VERIFIED if verified else VerificationStatus.UNKNOWN,
        **({"evidence_ref": "fixture-measurement-log", "verifier_id": "fixture-verifier"}
           if verified else {}),
    )
    unknown = VerificationFinding(status=VerificationStatus.UNKNOWN)
    entries = tuple(
        TargetObservationEvidence(
            observation=TargetObservation(
                dataset_id=declaration.dataset_id,
                target_date=date,
                value_microVolt=float(value),
                origin=TargetOrigin.MEASURED,
                source_ref="fixture-measurement-log",
            ),
            provenance=provenance,
            availability=unknown,
            exposure=unknown,
        )
        for date, value in zip(
            declaration.forecast_dates,
            isolated.holdout["microVolt"].tolist(),
        )
    )
    return build_target_observability_report(declaration, entries)


def _case(processed_training, frame, declaration):
    isolated, contract, binding = _real_case(processed_training, frame, declaration)
    events = deepcopy(ProbeModel.events)
    eligible = establish_holdout_evaluation_eligibility(
        **contract, binding_report=binding,
    )
    assert ProbeModel.events == events  # eligibility must not fit or predict
    return isolated, eligible


def test_real_m18_b1_b2_discloses_only_after_eligibility(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    isolated, eligible = _case(processed_training, full_frame, declaration)
    report = _report(declaration, isolated)
    before = deepcopy(ProbeModel.events)
    disclosure = disclose_holdout_targets(
        declaration=declaration, eligibility_snapshot=eligible,
        isolated=isolated, observability_report=report,
    )
    assert disclosure["status"] == "TARGETS_DISCLOSED_FOR_EVALUATION"
    assert disclosure["observed_microVolt"] == [29.9, 30.506]
    assert disclosure["eligibility_digest"] == eligible["eligibility_digest"]
    assert disclosure["execution_artifact_digest"] == eligible["execution_artifact_digest"]
    assert disclosure["forecast_dates"] == list(declaration.forecast_dates)
    assert disclosure["metrics_included"] is False
    assert "predicted_microVolt" not in disclosure
    assert ProbeModel.events == before
    assert [event[0] for event in before] == ["construct", "fit", "predict"]


def test_holdout_target_perturbation_changes_only_disclosed_targets(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    isolated_a, eligible_a = _case(processed_training, full_frame, declaration)
    disclosure_a = disclose_holdout_targets(
        declaration=declaration, eligibility_snapshot=eligible_a,
        isolated=isolated_a, observability_report=_report(declaration, isolated_a),
    )
    changed = full_frame.copy(deep=True)
    changed.loc[changed["ds"] > pd.Timestamp(declaration.development_cutoff), "microVolt"] = [999.0, -999.0]
    _reset_probe()
    isolated_b, eligible_b = _case(processed_training, changed, declaration)
    disclosure_b = disclose_holdout_targets(
        declaration=declaration, eligibility_snapshot=eligible_b,
        isolated=isolated_b, observability_report=_report(declaration, isolated_b),
    )
    assert eligible_a == eligible_b
    assert disclosure_a["eligibility_digest"] == disclosure_b["eligibility_digest"]
    assert disclosure_a["observed_microVolt"] == [29.9, 30.506]
    assert disclosure_b["observed_microVolt"] == [999.0, -999.0]
    assert disclosure_a["disclosure_digest"] != disclosure_b["disclosure_digest"]
    assert disclosure_a["forecast_dates"] == disclosure_b["forecast_dates"]


def test_real_eligible_forecast_cannot_disclose_unverified_observations(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    isolated, eligible = _case(processed_training, full_frame, declaration)
    report = _report(declaration, isolated, verified=False)
    assert report.observed_count == 0
    before = deepcopy(ProbeModel.events)
    with pytest.raises(ValueError, match="coverage"):
        disclose_holdout_targets(
            declaration=declaration, eligibility_snapshot=eligible,
            isolated=isolated, observability_report=report,
        )
    assert ProbeModel.events == before
