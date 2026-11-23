"""M19-B3-B3 focused contract tests (real-M18 integration is separate)."""
from copy import deepcopy
from math import isclose, sqrt
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
from zenerestimation.assessment.holdout_target_disclosure import disclose_holdout_targets, _digest
from zenerestimation.assessment.holdout_error_scoring import score_disclosed_holdout
from zenerestimation.data.dataset import BatteryDataset


@pytest.fixture
def case():
    declaration = AssessmentDeclaration(
        contract_version=CONTRACT_VERSION, assessment_id="b3-b3-410",
        assessment_track=AssessmentTrack.RETROSPECTIVE, dataset_id="410",
        development_cutoff="2023-12-01", forecast_dates=("2024-03-01", "2024-06-01"),
    )
    frame = pd.DataFrame({"ds": pd.to_datetime([
        "2023-09-01", "2023-12-01", "2024-03-01", "2024-06-01"]),
        "microVolt": [29.2, 29.3, 29.9, 30.506]})
    isolated = isolate_declared_holdout(BatteryDataset(frame), declaration=declaration)
    verified = VerificationFinding(status=VerificationStatus.VERIFIED,
                                   evidence_ref="measurement-log", verifier_id="fixture-verifier")
    unknown = VerificationFinding(status=VerificationStatus.UNKNOWN)
    entries = tuple(TargetObservationEvidence(
        observation=TargetObservation(dataset_id="410", target_date=date,
            value_microVolt=value, origin=TargetOrigin.MEASURED, source_ref="measurement-log"),
        provenance=verified, availability=unknown, exposure=unknown,
    ) for date, value in zip(declaration.forecast_dates, (29.9, 30.506)))
    report = build_target_observability_report(declaration, entries)
    eligible = {
        "eligibility_version": "m19_b3_b1_evaluation_eligibility_v1",
        "status": "ELIGIBLE_FOR_HOLDOUT_DISCLOSURE", "assessment_id": declaration.assessment_id,
        "declaration_digest": declaration.digest, "dataset_id": "410",
        "development_cutoff": "2023-12-01", "selected_model": "GRU",
        "decision_digest": "decision", "plan_digest": "plan",
        "execution_artifact_digest": "execution",
        "training_fingerprint": {"digest": "fingerprint"},
        "forecast_horizon": 2, "forecast_dates": list(declaration.forecast_dates),
        "predicted_microVolt": [29.4, 29.5],
        "scientific_independence": "not_established", "target_values_included": False,
    }
    eligible["eligibility_digest"] = _digest(eligible)
    disclosure = disclose_holdout_targets(declaration=declaration,
        eligibility_snapshot=eligible, isolated=isolated, observability_report=report)
    return {"declaration": declaration, "eligibility_snapshot": eligible,
            "disclosure_snapshot": disclosure}


def _redigest(snapshot, key):
    snapshot[key] = _digest({k: v for k, v in snapshot.items() if k != key})


def test_expected_metrics_and_sign_convention(case):
    result = score_disclosed_holdout(**case)
    errors = [-0.5, -1.006]
    assert result["status"] == "HOLDOUT_SCORED"
    assert result["error_convention"] == "forecast_minus_observed"
    assert isclose(result["metrics"]["rmse_microVolt"], sqrt(sum(e*e for e in errors)/2))
    assert isclose(result["metrics"]["mae_microVolt"], sum(abs(e) for e in errors)/2)
    assert isclose(result["metrics"]["bias_microVolt"], sum(errors)/2)
    assert result["denominators"] == {"rmse": 2, "mae": 2, "bias": 2, "mape": 2}
    assert [row["signed_error_microVolt"] for row in result["by_date"]] == pytest.approx(errors)


def test_deterministic_detached(case):
    first = score_disclosed_holdout(**case)
    second = score_disclosed_holdout(**case)
    assert first == second
    first["by_date"][0]["signed_error_microVolt"] = 100
    assert second["by_date"][0]["signed_error_microVolt"] != 100
    assert case["disclosure_snapshot"]["observed_microVolt"] == [29.9, 30.506]


def test_zero_actual_mape_policy(case):
    case["disclosure_snapshot"]["observed_microVolt"][0] = 0.0
    case["disclosure_snapshot"]["observation_evidence"][0]["observation"]["value_microVolt"] = 0.0
    _redigest(case["disclosure_snapshot"], "disclosure_digest")
    result = score_disclosed_holdout(**case)
    assert result["denominators"]["mape"] == 1
    assert result["mape_zero_actual_excluded_dates"] == ["2024-03-01"]


def test_all_zero_actuals_return_none_mape(case):
    for i in range(2):
        case["disclosure_snapshot"]["observed_microVolt"][i] = 0.0
        case["disclosure_snapshot"]["observation_evidence"][i]["observation"]["value_microVolt"] = 0.0
    _redigest(case["disclosure_snapshot"], "disclosure_digest")
    result = score_disclosed_holdout(**case)
    assert result["metrics"]["mape_percent"] is None
    assert result["denominators"]["mape"] == 0


@pytest.mark.parametrize("field,value", [
    ("status", "DENIED"), ("dataset_id", "110"),
    ("forecast_dates", ["2024-03-01", "2024-09-01"]),
    ("forecast_horizon", 3), ("metrics_included", True),
    ("scientific_independence", "established"),
    ("execution_artifact_digest", "other"),
])
def test_disclosure_binding_tampering_rejected(case, field, value):
    case["disclosure_snapshot"][field] = value
    _redigest(case["disclosure_snapshot"], "disclosure_digest")
    with pytest.raises(ValueError, match="B2 disclosure"):
        score_disclosed_holdout(**case)


def test_disclosure_digest_tampering_rejected(case):
    case["disclosure_snapshot"]["observed_microVolt"][0] = 999
    with pytest.raises(ValueError, match="digest"):
        score_disclosed_holdout(**case)


def test_eligibility_digest_tampering_rejected(case):
    case["eligibility_snapshot"]["predicted_microVolt"][0] = 999
    with pytest.raises(ValueError, match="digest"):
        score_disclosed_holdout(**case)


@pytest.mark.parametrize("field,value", [
    ("value_microVolt", 999), ("target_date", "2024-09-01"),
    ("origin", "interpolated"), ("source_ref", None),
])
def test_observation_mismatch_rejected_even_if_redigested(case, field, value):
    case["disclosure_snapshot"]["observation_evidence"][0]["observation"][field] = value
    _redigest(case["disclosure_snapshot"], "disclosure_digest")
    with pytest.raises(ValueError, match="observation"):
        score_disclosed_holdout(**case)


def test_unverified_provenance_rejected_even_if_redigested(case):
    case["disclosure_snapshot"]["observation_evidence"][0]["provenance"]["status"] = "unknown"
    _redigest(case["disclosure_snapshot"], "disclosure_digest")
    with pytest.raises(ValueError, match="provenance"):
        score_disclosed_holdout(**case)


def test_extra_disclosure_field_rejected(case):
    case["disclosure_snapshot"]["unexpected"] = True
    with pytest.raises(ValueError, match="schema"):
        score_disclosed_holdout(**case)


def test_nonfinite_disclosed_target_rejected(case):
    case["disclosure_snapshot"]["observed_microVolt"][0] = float("inf")
    with pytest.raises(ValueError, match="canonical JSON"):
        score_disclosed_holdout(**case)
