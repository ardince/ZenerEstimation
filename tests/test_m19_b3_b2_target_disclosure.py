"""M19-B3-B2 focused contract tests; real B1 integration remains a separate gate."""
from copy import deepcopy
import pytest
import pandas as pd

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION,
    VerificationFinding, VerificationStatus,
)
from zenerestimation.assessment.holdout_isolation import isolate_declared_holdout
from zenerestimation.assessment.observability import (
    TargetObservation, TargetObservationEvidence, TargetOrigin,
    build_target_observability_report,
)
from zenerestimation.assessment.holdout_target_disclosure import (
    disclose_holdout_targets, _digest,
)
from zenerestimation.data.dataset import BatteryDataset


@pytest.fixture
def case():
    declaration = AssessmentDeclaration(
        contract_version=CONTRACT_VERSION, assessment_id="b3-b2-410",
        assessment_track=AssessmentTrack.RETROSPECTIVE, dataset_id="410",
        development_cutoff="2023-12-01",
        forecast_dates=("2024-03-01", "2024-06-01"),
    )
    frame = pd.DataFrame({
        "ds": pd.to_datetime(["2023-09-01", "2023-12-01", "2024-03-01", "2024-06-01"]),
        "microVolt": [29.2, 29.3, 29.9, 30.506],
    })
    isolated = isolate_declared_holdout(BatteryDataset(frame), declaration=declaration)
    verified = VerificationFinding(
        status=VerificationStatus.VERIFIED, evidence_ref="measurement-log",
        verifier_id="fixture-verifier",
    )
    unknown = VerificationFinding(status=VerificationStatus.UNKNOWN)
    entries = tuple(TargetObservationEvidence(
        observation=TargetObservation(
            dataset_id="410", target_date=date, value_microVolt=value,
            origin=TargetOrigin.MEASURED, source_ref="measurement-log",
        ),
        provenance=verified, availability=unknown, exposure=unknown,
    ) for date, value in zip(declaration.forecast_dates, [29.9, 30.506]))
    report = build_target_observability_report(declaration, entries)
    eligible = {
        "eligibility_version": "m19_b3_b1_evaluation_eligibility_v1",
        "status": "ELIGIBLE_FOR_HOLDOUT_DISCLOSURE",
        "assessment_id": declaration.assessment_id,
        "declaration_digest": declaration.digest,
        "dataset_id": "410", "development_cutoff": "2023-12-01",
        "selected_model": "GRU", "decision_digest": "decision",
        "plan_digest": "plan", "execution_artifact_digest": "execution",
        "training_fingerprint": {"digest": "fingerprint"},
        "forecast_horizon": 2, "forecast_dates": list(declaration.forecast_dates),
        "predicted_microVolt": [29.4, 29.5],
        "scientific_independence": "not_established", "target_values_included": False,
    }
    eligible["eligibility_digest"] = _digest(eligible)
    return dict(declaration=declaration, eligibility_snapshot=eligible,
                isolated=isolated, observability_report=report)


def test_disclosure_complete_measured_targets(case):
    output = disclose_holdout_targets(**case)
    assert output["status"] == "TARGETS_DISCLOSED_FOR_EVALUATION"
    assert output["observed_microVolt"] == [29.9, 30.506]
    assert output["forecast_dates"] == ["2024-03-01", "2024-06-01"]
    assert output["metrics_included"] is False
    assert "rmse" not in output and "predicted_microVolt" not in output
    assert output["disclosure_digest"].startswith("sha256:")


def test_deterministic_detached(case):
    first = disclose_holdout_targets(**case)
    second = disclose_holdout_targets(**case)
    assert first == second
    first["observed_microVolt"][0] = -100
    assert second["observed_microVolt"][0] == 29.9
    assert case["isolated"].holdout["microVolt"].tolist() == [29.9, 30.506]


@pytest.mark.parametrize("field,value", [
    ("status", "DENIED"), ("dataset_id", "110"),
    ("forecast_horizon", 3), ("forecast_dates", ["2024-03-01", "2024-09-01"]),
    ("target_values_included", True), ("scientific_independence", "established"),
])
def test_eligibility_mismatch_rejected_even_with_recomputed_digest(case, field, value):
    case["eligibility_snapshot"][field] = value
    snapshot = case["eligibility_snapshot"]
    snapshot["eligibility_digest"] = _digest({k: v for k, v in snapshot.items() if k != "eligibility_digest"})
    with pytest.raises(ValueError, match="B1 eligibility"):
        disclose_holdout_targets(**case)


def test_eligibility_digest_tampering_rejected(case):
    case["eligibility_snapshot"]["predicted_microVolt"][0] = 999
    with pytest.raises(ValueError, match="digest"):
        disclose_holdout_targets(**case)


def test_extra_eligibility_field_rejected(case):
    case["eligibility_snapshot"]["actual_microVolt"] = [29.9, 30.506]
    with pytest.raises(ValueError, match="schema"):
        disclose_holdout_targets(**case)


@pytest.mark.parametrize("field,value", [
    ("status", "UNVERIFIED"), ("dataset_id", "110"),
    ("holdout_dates", ["2024-03-01", "2024-09-01"]),
    ("holdout_rows", 3),
])
def test_isolation_evidence_mismatch_rejected(case, field, value):
    case["isolated"].evidence[field] = value
    with pytest.raises(ValueError, match="isolation"):
        disclose_holdout_targets(**case)


@pytest.mark.parametrize("values", [[29.9, float("nan")], [29.9, float("inf")],
                                     [29.9, None], [29.9, "invalid"], [29.9, True]])
def test_invalid_holdout_measurements_rejected(case, values):
    case["isolated"].holdout["microVolt"] = values
    with pytest.raises(ValueError, match="finite numeric"):
        disclose_holdout_targets(**case)


def test_mismatched_holdout_measurement_rejected(case):
    case["isolated"].holdout.loc[0, "microVolt"] = 888.0
    with pytest.raises(ValueError, match="differs"):
        disclose_holdout_targets(**case)


def test_holdout_calendar_mismatch_rejected(case):
    case["isolated"].holdout.loc[0, "ds"] = pd.Timestamp("2024-09-01")
    with pytest.raises(ValueError, match="calendar"):
        disclose_holdout_targets(**case)


def test_unverified_observation_coverage_rejected(case):
    entries = list(case["observability_report"].observations)
    old = entries[1]
    entries[1] = TargetObservationEvidence(
        observation=old.observation,
        provenance=VerificationFinding(status=VerificationStatus.UNKNOWN),
        availability=old.availability, exposure=old.exposure,
    )
    case["observability_report"] = build_target_observability_report(case["declaration"], entries)
    with pytest.raises(ValueError, match="coverage"):
        disclose_holdout_targets(**case)


def test_observability_target_mismatch_rejected(case):
    entries = list(case["observability_report"].observations)
    old = entries[0]
    entries[0] = TargetObservationEvidence(
        observation=TargetObservation(
            dataset_id="410", target_date="2024-03-01", value_microVolt=500.0,
            origin=TargetOrigin.MEASURED, source_ref="measurement-log",
        ), provenance=old.provenance, availability=old.availability, exposure=old.exposure,
    )
    case["observability_report"] = build_target_observability_report(case["declaration"], entries)
    with pytest.raises(ValueError, match="differs"):
        disclose_holdout_targets(**case)
