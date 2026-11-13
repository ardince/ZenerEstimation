"""M18-C6-B2 RED contract: verified, non-executing forecast plans.

Install at tests/test_m18_forecast_plan_contract.py.
The production module internal_forecast_plan is deliberately absent at RED.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pandas as pd
import pytest

from tests.test_m18_selection_audit_contract import context, unique_results
from tests.test_m18_selection_adapter_contract import (
    artifact, registry, tied_results,
)
from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.fingerprint import fingerprint_dataset
from zenerestimation.optimization.internal_forecast_plan import (
    build_internal_forecast_plan,
    verify_internal_forecast_plan,
)

VERSION = "m18_c6_forecast_plan_v1"
FIELDS = {
    "plan_version", "status", "selected_model", "registry_model",
    "decision_digest", "model_params", "parameters_digest",
    "training_provenance", "forecast_horizon", "forecast_dates", "plan_digest",
}
DATES = ["2024-03-01", "2024-06-01"]


def canonical_digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


@pytest.fixture
def training(tmp_path):
    path = tmp_path / "processed_training.csv"
    pd.DataFrame({
        "ds": ["2023-03-01", "2023-06-01", "2023-09-01", "2023-12-01"],
        "microVolt": [29.0, 29.1, 29.2, 29.3],
        "is_observed": [True, True, True, True],
    }).to_csv(path, index=False)
    return BatteryDataset.from_processed_csv(path, battery="410")


def training_context(dataset):
    return {
        "scope": "final_training_processed",
        "dataset_fingerprint": fingerprint_dataset(dataset).to_dict(),
        "train_start": "2023-03-01",
        "train_end": "2023-12-01",
    }


def inputs(training):
    results = unique_results()
    trusted = context()
    return {
        "artifact": artifact(results=results, trusted=trusted),
        "results": results,
        "validation_context": trusted,
        "registry": registry(),
        "training_dataset": training,
        "training_context": training_context(training),
        "forecast_horizon": 2,
        "forecast_dates": list(DATES),
    }


def build(kwargs):
    return build_internal_forecast_plan(**kwargs)


def verify(plan, kwargs):
    return verify_internal_forecast_plan(plan, **kwargs)


def test_valid_plan_exact_schema_and_identity(training):
    kwargs = inputs(training)
    plan = build(kwargs)
    assert type(plan) is dict
    assert set(plan) == FIELDS
    assert plan["plan_version"] == VERSION
    assert plan["status"] == "PLANNED"
    assert plan["selected_model"] == "GRU"
    assert plan["registry_model"] == "gru"
    assert plan["decision_digest"] == kwargs["artifact"]["selection_decision_record"]["decision_digest"]
    assert plan["forecast_horizon"] == 2
    assert plan["forecast_dates"] == DATES
    assert plan["model_params"] == kwargs["results"]["GRU"].best_params
    assert plan["parameters_digest"] == canonical_digest(plan["model_params"])
    assert plan["plan_digest"] == canonical_digest({k: v for k, v in plan.items() if k != "plan_digest"})
    assert verify(plan, kwargs) is True


def test_training_provenance_exact_contract(training):
    plan = build(inputs(training))
    assert plan["training_provenance"] == training_context(training)["dataset_fingerprint"] | {
        "train_start": "2023-03-01", "train_end": "2023-12-01"
    }


def test_deterministic_and_json_safe(training):
    kwargs = inputs(training)
    first, second = build(kwargs), build(kwargs)
    assert first == second
    assert json.loads(json.dumps(first, allow_nan=False)) == first


def test_plan_is_detached_and_inputs_unchanged(training):
    kwargs = inputs(training)
    before_artifact = deepcopy(kwargs["artifact"])
    before_context = deepcopy(kwargs["validation_context"])
    before_training_context = deepcopy(kwargs["training_context"])
    before_results = {k: v.to_dict() for k, v in kwargs["results"].items()}
    before_registry = dict(kwargs["registry"]._models)
    before_data = training.data.copy(deep=True)
    plan = build(kwargs)
    assert kwargs["artifact"] == before_artifact
    assert kwargs["validation_context"] == before_context
    assert kwargs["training_context"] == before_training_context
    assert {k: v.to_dict() for k, v in kwargs["results"].items()} == before_results
    assert kwargs["registry"]._models == before_registry
    pd.testing.assert_frame_equal(training.data, before_data)
    plan["model_params"].clear()
    plan["forecast_dates"].append("2024-09-01")
    assert build(kwargs)["model_params"] == kwargs["results"]["GRU"].best_params
    assert build(kwargs)["forecast_dates"] == DATES


def test_pending_selection_rejected(training):
    kwargs = inputs(training)
    kwargs["artifact"] = artifact(results=kwargs["results"], trusted=kwargs["validation_context"], authorized=False)
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


@pytest.mark.parametrize("authorized", [False, True])
def test_tie_rejected(training, authorized):
    kwargs = inputs(training)
    kwargs["results"] = tied_results()
    kwargs["artifact"] = artifact(results=kwargs["results"], trusted=kwargs["validation_context"], authorized=authorized)
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


def test_tampered_selection_artifact_rejected(training):
    kwargs = inputs(training)
    kwargs["artifact"]["selection_decision_record"]["decision"]["selected_model"] = "LSTM"
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


def test_changed_trusted_results_rejected(training):
    from tests.test_m18_selection_audit_contract import result
    kwargs = inputs(training)
    kwargs["results"]["GRU"] = result("GRU", rmse=(0.8, 0.9))
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


def test_changed_validation_context_rejected(training):
    kwargs = inputs(training)
    kwargs["validation_context"] = context(dataset_id="other")
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


def test_unregistered_selected_model_rejected(training):
    kwargs = inputs(training)
    kwargs["registry"] = registry(include_gru=False, include_lstm=True)
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


@pytest.mark.parametrize("horizon", [0, -1, True, 2.0, "2", None])
def test_invalid_horizon_rejected(training, horizon):
    kwargs = inputs(training)
    kwargs["forecast_horizon"] = horizon
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


@pytest.mark.parametrize("dates", [
    [], ["2024-03-01"], ["2024-03-01", "2024-03-01"],
    ["2024-06-01", "2024-03-01"],
    ["2023-12-01", "2024-03-01"],
    ["2023-09-01", "2024-03-01"],
    ["2024-03-01", "bad-date"],
    ["2024-03-01T00:00:00+03:00", "2024-06-01"],
])
def test_invalid_calendar_rejected(training, dates):
    kwargs = inputs(training)
    kwargs["forecast_dates"] = dates
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


def test_trusted_training_context_required(training):
    kwargs = inputs(training)
    kwargs["training_context"] = None
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


@pytest.mark.parametrize("field,change", [
    ("scope", "untrusted"),
    ("train_start", "2023-06-01"),
    ("train_end", "2024-03-01"),
])
def test_training_context_mismatch_rejected(training, field, change):
    kwargs = inputs(training)
    kwargs["training_context"][field] = change
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


def test_training_fingerprint_mismatch_rejected(training):
    kwargs = inputs(training)
    kwargs["training_context"]["dataset_fingerprint"]["digest"] = "0" * 64
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


def test_training_measurement_changed_after_context_created_rejected(training):
    kwargs = inputs(training)
    training.data.loc[0, "microVolt"] += 1.0
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


def test_training_observation_flag_changed_rejected(training):
    kwargs = inputs(training)
    training.data.loc[0, "is_observed"] = False
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


def test_training_date_changed_rejected(training):
    kwargs = inputs(training)
    training.data.loc[0, "ds"] = pd.Timestamp("2023-02-01")
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


def test_raw_dataset_rejected(training):
    kwargs = inputs(training)
    kwargs["training_dataset"] = BatteryDataset(training.data.copy(deep=True))
    with pytest.raises((ValueError, TypeError)):
        build(kwargs)


@pytest.mark.parametrize("change", [
    lambda p: p.update(status="EXECUTED"),
    lambda p: p.update(plan_version="other"),
    lambda p: p.update(selected_model="LSTM"),
    lambda p: p.update(registry_model="lstm"),
    lambda p: p.update(model_params={"units": 999}),
    lambda p: p.update(parameters_digest="sha256:" + "0" * 64),
    lambda p: p.update(forecast_horizon=3),
    lambda p: p.update(forecast_dates=["2024-06-01", "2024-09-01"]),
    lambda p: p["training_provenance"].update(train_end="2024-03-01"),
    lambda p: p.update(plan_digest="sha256:" + "0" * 64),
    lambda p: p.update(unexpected="extra"),
    lambda p: p.pop("status"),
])
def test_modified_plan_rejected(training, change):
    kwargs = inputs(training)
    plan = build(kwargs)
    change(plan)
    assert verify(plan, kwargs) is False


def test_resigned_modified_plan_still_rejected(training):
    kwargs = inputs(training)
    plan = build(kwargs)
    plan["model_params"] = {"units": 999}
    plan["parameters_digest"] = canonical_digest(plan["model_params"])
    plan["plan_digest"] = canonical_digest({k: v for k, v in plan.items() if k != "plan_digest"})
    assert verify(plan, kwargs) is False


def test_verification_rechecks_trusted_training_inputs(training):
    kwargs = inputs(training)
    plan = build(kwargs)
    training.data.loc[0, "microVolt"] += 1.0
    with pytest.raises((ValueError, TypeError)):
        verify(plan, kwargs)


def test_no_forecast_execution_or_filesystem_writes(training, tmp_path, monkeypatch):
    from zenerestimation.forecasting.pipeline import ForecastPipeline
    def forbidden(*args, **kwargs):
        raise AssertionError("planning must never execute a forecast")
    monkeypatch.setattr(ForecastPipeline, "run", forbidden)
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    kwargs = inputs(training)
    plan = build(kwargs)
    assert verify(plan, kwargs) is True
    assert set(tmp_path.iterdir()) == before
    assert plan["status"] == "PLANNED"
    assert "forecast" not in plan and "model_instance" not in plan
