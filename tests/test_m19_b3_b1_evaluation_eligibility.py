"""M19-B3-B1 focused unit contract; upstream verifier seams are stubbed.

A separate real-M18 integration gate is required before B1 closure.
"""
from copy import deepcopy
import math
import pytest

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION,
)
from zenerestimation.assessment import holdout_evaluation_eligibility as gate


@pytest.fixture
def case(monkeypatch):
    declaration = AssessmentDeclaration(
        contract_version=CONTRACT_VERSION, assessment_id="b3-b1-410",
        assessment_track=AssessmentTrack.RETROSPECTIVE, dataset_id="410",
        development_cutoff="2023-12-01",
        forecast_dates=("2024-03-01", "2024-06-01"),
    )
    report = dict(
        binding_version="m19_b3_a1_holdout_binding_v1", status="BOUND",
        assessment_id=declaration.assessment_id,
        declaration_digest=declaration.digest, dataset_id="410",
        selected_model="GRU", decision_digest="decision", plan_digest="plan",
        execution_artifact_digest="execution", training_fingerprint={"digest": "fingerprint"},
        development_cutoff="2023-12-01", forecast_horizon=2,
        forecast_dates=["2024-03-01", "2024-06-01"],
        scientific_independence="not_established",
    )
    view = dict(status="VERIFIED", selected_model="GRU",
                forecast=dict(horizon=2, dates=list(declaration.forecast_dates),
                              values_microVolt=[29.4, 29.5]),
                provenance=dict(decision_digest="decision", plan_digest="plan",
                                execution_artifact_digest="execution",
                                training_fingerprint={"digest": "fingerprint"}))
    monkeypatch.setattr(gate, "verify_holdout_binding", lambda **kw: deepcopy(report))
    monkeypatch.setattr(gate, "consume_internal_forecast_execution_artifact", lambda *a, **kw: deepcopy(view))
    kwargs = dict(declaration=declaration, binding_report=deepcopy(report),
                  selection_artifact={}, results={}, validation_context={},
                  registry=object(), training_dataset=object(), training_context={},
                  plan={}, execution_artifact={}, trusted_dataset_id="410")
    return kwargs, report, view, monkeypatch


def test_target_blind_snapshot(case):
    kwargs, _, _, _ = case
    output = gate.establish_holdout_evaluation_eligibility(**kwargs)
    assert output["status"] == "ELIGIBLE_FOR_HOLDOUT_DISCLOSURE"
    assert output["predicted_microVolt"] == [29.4, 29.5]
    assert output["target_values_included"] is False
    assert "actual_microVolt" not in output and "rmse" not in output
    assert output["eligibility_digest"].startswith("sha256:")


def test_deterministic_and_detached(case):
    kwargs, _, _, _ = case
    first = gate.establish_holdout_evaluation_eligibility(**kwargs)
    second = gate.establish_holdout_evaluation_eligibility(**kwargs)
    assert first == second
    first["training_fingerprint"]["digest"] = "mutated"
    assert second["training_fingerprint"]["digest"] == "fingerprint"


@pytest.mark.parametrize("field,value", [
    ("dataset_id", "110"), ("forecast_horizon", 3),
    ("forecast_dates", ["2024-03-01", "2024-09-01"]),
    ("selected_model", "LSTM"), ("plan_digest", "tampered"),
    ("training_fingerprint", {"digest": "wrong"}),
])
def test_tampered_binding_rejected(case, field, value):
    kwargs, _, _, _ = case
    kwargs["binding_report"][field] = value
    with pytest.raises(ValueError, match="binding"):
        gate.establish_holdout_evaluation_eligibility(**kwargs)


def test_extra_binding_field_rejected(case):
    kwargs, _, _, _ = case
    kwargs["binding_report"]["actual_microVolt"] = [29.9, 30.5]
    with pytest.raises(ValueError, match="schema"):
        gate.establish_holdout_evaluation_eligibility(**kwargs)


@pytest.mark.parametrize("values", [
    [29.4], [29.4, float("nan")], [29.4, float("inf")],
    [29.4, "invalid"], [True, 29.5],
])
def test_invalid_forecast_values_rejected(case, values):
    kwargs, _, view, monkeypatch = case
    view["forecast"]["values_microVolt"] = values
    monkeypatch.setattr(gate, "consume_internal_forecast_execution_artifact", lambda *a, **kw: deepcopy(view))
    with pytest.raises(ValueError, match="forecast values"):
        gate.establish_holdout_evaluation_eligibility(**kwargs)


def test_wrong_execution_calendar_rejected(case):
    kwargs, _, view, monkeypatch = case
    view["forecast"]["dates"] = ["2024-03-01", "2024-09-01"]
    monkeypatch.setattr(gate, "consume_internal_forecast_execution_artifact", lambda *a, **kw: deepcopy(view))
    with pytest.raises(ValueError, match="calendar"):
        gate.establish_holdout_evaluation_eligibility(**kwargs)


def test_unverified_execution_rejected(case):
    kwargs, _, view, monkeypatch = case
    view["status"] = "FAILED"
    monkeypatch.setattr(gate, "consume_internal_forecast_execution_artifact", lambda *a, **kw: deepcopy(view))
    with pytest.raises(ValueError, match="independently verified"):
        gate.establish_holdout_evaluation_eligibility(**kwargs)
