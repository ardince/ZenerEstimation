"""M19-B3-A1 binding tests; upstream M18 verifiers are stubbed at this unit seam.

These tests do not replace end-to-end tests with real M18 artifacts.
"""
from copy import deepcopy

import pytest

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION,
)
from zenerestimation.assessment import holdout_binding as binding


@pytest.fixture
def case(monkeypatch):
    declaration = AssessmentDeclaration(
        contract_version=CONTRACT_VERSION,
        assessment_id="assessment-410",
        assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id="410",
        development_cutoff="2023-12-01",
        forecast_dates=("2024-03-01", "2024-06-01"),
    )
    fingerprint = {"algorithm": "sha256", "digest": "sha256:" + "a" * 64,
                   "canonicalization_version": "v1", "row_count": 10}
    plan = {
        "selected_model": "LSTM", "decision_digest": "decision-digest",
        "plan_digest": "plan-digest", "forecast_horizon": 2,
        "forecast_dates": ["2024-03-01", "2024-06-01"],
        "training_provenance": {**fingerprint, "train_start": "2020-03-01", "train_end": "2023-12-01"},
    }
    ready = {"status": "READY", "selected_model": "LSTM", "decision_digest": "decision-digest"}
    view = {"status": "VERIFIED", "selected_model": "LSTM",
            "forecast": {"horizon": 2, "dates": ["2024-03-01", "2024-06-01"],
                         "values_microVolt": [29.9, 30.5]},
            "provenance": {"decision_digest": "decision-digest", "plan_digest": "plan-digest",
                           "execution_artifact_digest": "artifact-digest",
                           "training_fingerprint": deepcopy(fingerprint)}}
    monkeypatch.setattr(binding, "consume_internal_selection_artifact", lambda *a, **k: deepcopy(ready))
    monkeypatch.setattr(binding, "verify_internal_forecast_plan", lambda *a, **k: True)
    monkeypatch.setattr(binding, "consume_internal_forecast_execution_artifact", lambda *a, **k: deepcopy(view))
    inputs = dict(declaration=declaration, selection_artifact={}, results={}, validation_context={},
                  registry=object(), training_dataset=object(), training_context={},
                  plan=plan, execution_artifact={"artifact_digest": "artifact-digest"},
                  trusted_dataset_id="410")
    return inputs, ready, view


def test_valid_binding_has_no_independence_claim(case):
    inputs, _, _ = case
    report = binding.verify_holdout_binding(**inputs)
    assert report["status"] == "BOUND"
    assert report["scientific_independence"] == "not_established"
    assert report["forecast_dates"] == list(inputs["declaration"].forecast_dates)


@pytest.mark.parametrize("field,value", [
    ("selected_model", "GRU"),
    ("decision_digest", "wrong"),
    ("forecast_horizon", 3),
    ("forecast_dates", ["2024-03-01", "2024-09-01"]),
])
def test_plan_mismatch_rejected(case, field, value):
    inputs, _, _ = case
    inputs["plan"][field] = value
    with pytest.raises(ValueError):
        binding.verify_holdout_binding(**inputs)


def test_wrong_dataset_rejected(case):
    inputs, _, _ = case
    inputs["trusted_dataset_id"] = "110"
    with pytest.raises(ValueError, match="dataset identity"):
        binding.verify_holdout_binding(**inputs)


def test_wrong_cutoff_rejected(case):
    inputs, _, _ = case
    inputs["plan"]["training_provenance"]["train_end"] = "2023-09-01"
    with pytest.raises(ValueError, match="development cutoff"):
        binding.verify_holdout_binding(**inputs)


def test_wrong_training_fingerprint_rejected(case):
    inputs, _, view = case
    view["provenance"]["training_fingerprint"]["digest"] = "tampered"
    # Replace verifier seam so the modified view is returned.
    original = binding.consume_internal_forecast_execution_artifact
    try:
        binding.consume_internal_forecast_execution_artifact = lambda *a, **k: deepcopy(view)
        with pytest.raises(ValueError, match="training fingerprint"):
            binding.verify_holdout_binding(**inputs)
    finally:
        binding.consume_internal_forecast_execution_artifact = original


def test_unverified_selection_rejected(case, monkeypatch):
    inputs, _, _ = case
    monkeypatch.setattr(binding, "consume_internal_selection_artifact",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("unverified")))
    with pytest.raises(ValueError, match="unverified"):
        binding.verify_holdout_binding(**inputs)


def test_unverified_plan_rejected(case, monkeypatch):
    inputs, _, _ = case
    monkeypatch.setattr(binding, "verify_internal_forecast_plan", lambda *a, **k: False)
    with pytest.raises(ValueError, match="forecast plan"):
        binding.verify_holdout_binding(**inputs)


def test_unverified_execution_rejected(case, monkeypatch):
    inputs, _, _ = case
    monkeypatch.setattr(binding, "consume_internal_forecast_execution_artifact",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("unverified execution")))
    with pytest.raises(ValueError, match="unverified execution"):
        binding.verify_holdout_binding(**inputs)


def test_report_is_detached(case):
    inputs, _, _ = case
    report = binding.verify_holdout_binding(**inputs)
    inputs["plan"]["forecast_dates"][0] = "2025-03-01"
    assert report["forecast_dates"][0] == "2024-03-01"
