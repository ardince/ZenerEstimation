"""M18-C6-C2 RED: explicitly authorized, verified in-memory execution.

Install at tests/test_m18_forecast_execution_contract.py.
The production module internal_forecast_execution must be absent for the RED gate.
These tests use real C3-C6-B verification and deterministic dummy models.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from tests.test_m18_forecast_plan_contract import inputs, training as training_fixture
from zenerestimation.data.fingerprint import fingerprint_dataset
from zenerestimation.forecasting.base import BaseForecastModel
from zenerestimation.forecasting.models import ModelRegistry
from zenerestimation.forecasting.result import ForecastResult
from zenerestimation.optimization.internal_forecast_plan import build_internal_forecast_plan
from zenerestimation.optimization.internal_forecast_execution import (
    execute_internal_forecast_plan,
)

DATES = pd.DatetimeIndex(["2024-03-01", "2024-06-01"])


class ProbeModel(BaseForecastModel):
    events = []
    result_override = None
    fail_at = None
    mutate_training = False

    def __init__(self, **kwargs):
        type(self).events.append(("construct", deepcopy(kwargs)))
        if type(self).fail_at == "construct":
            raise RuntimeError("constructor failed")
        self.params = kwargs

    def fit(self, dataset):
        type(self).events.append(("fit", fingerprint_dataset(dataset).to_dict()))
        if type(self).mutate_training:
            dataset.data.loc[0, "microVolt"] += 2.0
        if type(self).fail_at == "fit":
            raise RuntimeError("fit failed")
        return self

    def predict(self, steps=1):
        type(self).events.append(("predict", steps))
        if type(self).fail_at == "predict":
            raise RuntimeError("predict failed")
        if type(self).result_override is not None:
            return type(self).result_override
        return valid_result()


class SubstituteModel(ProbeModel):
    pass


def valid_result(*, model="GRU", dates=None, values=None, horizon=2, index=None):
    dates = DATES if dates is None else pd.DatetimeIndex(dates)
    values = [29.4, 29.5] if values is None else values
    if index is None:
        # Allow malformed forecast lengths to reach the executor validator.
        # Keep the approved result dates unchanged for the mismatch test.
        index = dates[:len(values)] if len(values) != len(dates) else dates
    return ForecastResult(
        model=model,
        forecast=pd.Series(values, index=index),
        horizon=horizon,
        dates=dates,
    )


@pytest.fixture(autouse=True)
def reset_probe():
    for cls in (ProbeModel, SubstituteModel):
        cls.events = []
        cls.result_override = None
        cls.fail_at = None
        cls.mutate_training = False
    yield


@pytest.fixture
def prepared(training_fixture):
    kwargs = inputs(training_fixture)
    registry = ModelRegistry()
    registry.register("GRU", ProbeModel)
    kwargs["registry"] = registry
    plan = build_internal_forecast_plan(**kwargs)
    return plan, kwargs


def execute(plan, kwargs, *, authorized=True):
    return execute_internal_forecast_plan(
        plan, **kwargs, execution_authorized=authorized,
    )


def assert_not_executed():
    assert ProbeModel.events == []
    assert SubstituteModel.events == []


def test_successful_execution_uses_exact_winner_and_calendar(prepared):
    plan, kwargs = prepared
    output = execute(plan, kwargs)
    assert type(output) is dict
    assert set(output) == {
        "execution_version", "status", "selected_model", "plan_digest",
        "decision_digest", "forecast_result", "execution_evidence",
    }
    assert output["execution_version"] == "m18_c6_execution_v1"
    assert output["status"] == "EXECUTED"
    assert output["selected_model"] == "GRU"
    assert output["plan_digest"] == plan["plan_digest"]
    assert output["decision_digest"] == plan["decision_digest"]
    assert isinstance(output["forecast_result"], ForecastResult)
    assert output["forecast_result"].dates.equals(DATES)
    assert output["forecast_result"].forecast.tolist() == [29.4, 29.5]
    assert ProbeModel.events[0] == ("construct", plan["model_params"])
    assert [event[0] for event in ProbeModel.events] == ["construct", "fit", "predict"]
    assert ProbeModel.events[2] == ("predict", 2)


def test_execution_evidence_is_bound_to_verified_plan(prepared):
    plan, kwargs = prepared
    evidence = execute(plan, kwargs)["execution_evidence"]
    assert type(evidence) is dict
    assert evidence["execution_authorized"] is True
    assert evidence["model_params_digest"] == plan["parameters_digest"]
    assert evidence["training_fingerprint"] == kwargs["training_context"]["dataset_fingerprint"]
    assert evidence["forecast_horizon"] == 2
    assert evidence["forecast_dates"] == plan["forecast_dates"]
    assert evidence["result_validated"] is True
    assert "human_identity_verified" not in evidence


@pytest.mark.parametrize("authorization", [False, None, 1, 0, "True", [], object()])
def test_invalid_authorization_fails_before_any_model_operation(prepared, authorization):
    plan, kwargs = prepared
    with pytest.raises((TypeError, ValueError, PermissionError)):
        execute(plan, kwargs, authorized=authorization)
    assert_not_executed()


def test_omitted_authorization_is_denied_by_default(prepared):
    plan, kwargs = prepared
    with pytest.raises((TypeError, ValueError, PermissionError)):
        execute_internal_forecast_plan(plan, **kwargs)
    assert_not_executed()


@pytest.mark.parametrize("change", [
    lambda plan: plan.update(status="EXECUTED"),
    lambda plan: plan.update(selected_model="LSTM"),
    lambda plan: plan.update(plan_digest="sha256:" + "0" * 64),
    lambda plan: plan["model_params"].update(window=999),
    lambda plan: plan["forecast_dates"].reverse(),
    lambda plan: plan.update(extra="not allowed"),
])
def test_tampered_plan_never_executes(prepared, change):
    plan, kwargs = prepared
    change(plan)
    with pytest.raises((TypeError, ValueError, PermissionError)):
        execute(plan, kwargs)
    assert_not_executed()


def test_modified_trusted_training_data_never_executes(prepared):
    plan, kwargs = prepared
    kwargs["training_dataset"].data.loc[0, "microVolt"] += 0.5
    with pytest.raises((TypeError, ValueError, PermissionError)):
        execute(plan, kwargs)
    assert_not_executed()


def test_modified_trusted_training_context_never_executes(prepared):
    plan, kwargs = prepared
    kwargs["training_context"]["dataset_fingerprint"]["digest"] = "0" * 64
    with pytest.raises((TypeError, ValueError, PermissionError)):
        execute(plan, kwargs)
    assert_not_executed()


def test_modified_trusted_selection_evidence_never_executes(prepared):
    plan, kwargs = prepared
    kwargs["artifact"]["selection_decision_record"]["decision"]["selected_model"] = "LSTM"
    with pytest.raises((TypeError, ValueError, PermissionError)):
        execute(plan, kwargs)
    assert_not_executed()


def test_unregistered_model_never_executes(prepared):
    plan, kwargs = prepared
    kwargs["registry"] = ModelRegistry()
    with pytest.raises((TypeError, ValueError, PermissionError)):
        execute(plan, kwargs)
    assert_not_executed()


def test_registry_class_substitution_is_not_claimed_as_pinned(prepared):
    """C6-C1 explicitly does not pin a registered class implementation."""
    plan, kwargs = prepared
    kwargs["registry"].register("GRU", SubstituteModel)
    output = execute(plan, kwargs)
    assert output["status"] == "EXECUTED"
    assert ProbeModel.events == []
    assert [event[0] for event in SubstituteModel.events] == ["construct", "fit", "predict"]


def test_constructor_exception_cannot_produce_success(prepared):
    plan, kwargs = prepared
    ProbeModel.fail_at = "construct"
    with pytest.raises(RuntimeError, match="constructor failed"):
        execute(plan, kwargs)
    assert [event[0] for event in ProbeModel.events] == ["construct"]


@pytest.mark.parametrize("stage", ["fit", "predict"])
def test_model_failure_cannot_produce_success(prepared, stage):
    plan, kwargs = prepared
    ProbeModel.fail_at = stage
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        execute(plan, kwargs)
    assert "predict" not in [event[0] for event in ProbeModel.events] if stage == "fit" else True


@pytest.mark.parametrize("bad_result", [
    "not a forecast result",
    valid_result(horizon=1),
    valid_result(model="LSTM"),
    valid_result(dates=["2024-06-01", "2024-09-01"]),
    valid_result(values=[29.4]),
    valid_result(values=[29.4, float("nan")]),
    valid_result(values=[29.4, float("inf")]),
    valid_result(values=[29.4, "invalid"]),
    valid_result(index=pd.DatetimeIndex(["2024-03-01", "2024-09-01"])),
    valid_result(index=pd.RangeIndex(2)),
])
def test_invalid_forecast_result_fails_closed(prepared, bad_result):
    plan, kwargs = prepared
    ProbeModel.result_override = bad_result
    with pytest.raises((TypeError, ValueError)):
        execute(plan, kwargs)
    assert [event[0] for event in ProbeModel.events] == ["construct", "fit", "predict"]


def test_caller_owned_training_is_not_modified_by_success(prepared):
    plan, kwargs = prepared
    original = kwargs["training_dataset"].data.copy(deep=True)
    original_fingerprint = fingerprint_dataset(kwargs["training_dataset"]).to_dict()
    execute(plan, kwargs)
    pd.testing.assert_frame_equal(kwargs["training_dataset"].data, original)
    assert fingerprint_dataset(kwargs["training_dataset"]).to_dict() == original_fingerprint


def test_model_training_mutation_is_detected_without_mutating_caller(prepared):
    plan, kwargs = prepared
    original = kwargs["training_dataset"].data.copy(deep=True)
    ProbeModel.mutate_training = True
    with pytest.raises((TypeError, ValueError)):
        execute(plan, kwargs)
    pd.testing.assert_frame_equal(kwargs["training_dataset"].data, original)
    assert "predict" not in [event[0] for event in ProbeModel.events]


def test_inputs_and_plan_unchanged_after_execution(prepared):
    plan, kwargs = prepared
    before_plan = deepcopy(plan)
    before_artifact = deepcopy(kwargs["artifact"])
    before_context = deepcopy(kwargs["training_context"])
    before_registry = dict(kwargs["registry"]._models)
    before_results = {k: v.to_dict() for k, v in kwargs["results"].items()}
    execute(plan, kwargs)
    assert plan == before_plan
    assert kwargs["artifact"] == before_artifact
    assert kwargs["training_context"] == before_context
    assert kwargs["registry"]._models == before_registry
    assert {k: v.to_dict() for k, v in kwargs["results"].items()} == before_results


def test_execution_does_not_call_pipeline_run(prepared, monkeypatch):
    from zenerestimation.forecasting.pipeline import ForecastPipeline
    monkeypatch.setattr(ForecastPipeline, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("pipeline must not be called")))
    plan, kwargs = prepared
    assert execute(plan, kwargs)["status"] == "EXECUTED"


def test_execution_does_not_write_files(prepared, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    plan, kwargs = prepared
    assert execute(plan, kwargs)["status"] == "EXECUTED"
    assert set(tmp_path.iterdir()) == before


def test_execution_is_in_memory_and_does_not_claim_persistent_proof(prepared):
    plan, kwargs = prepared
    output = execute(plan, kwargs)
    assert "artifact_path" not in output
    assert "execution_signature" not in output
    assert "human_identity_verified" not in output
    assert "external_holdout" not in output
