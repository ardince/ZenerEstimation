"""M19-B3-A3: real M18 plan/executor exercised against M19 isolated data.

Scope: deterministic fixture and an explicit call-site isolation guard. This does
not claim that all production callers enforce the guard, or that historical
model selection was free from holdout exposure.
"""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from tests.test_m18_forecast_plan_contract import training as processed_training, inputs
from tests.test_m18_forecast_execution_contract import ProbeModel
from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION,
)
from zenerestimation.assessment.holdout_isolation import (
    isolate_declared_holdout, require_isolated_training_dataset,
)
from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.fingerprint import fingerprint_dataset
from zenerestimation.forecasting.models import ModelRegistry
from zenerestimation.optimization.internal_forecast_plan import (
    build_internal_forecast_plan, verify_internal_forecast_plan,
)
from zenerestimation.optimization.internal_forecast_execution import execute_internal_forecast_plan


@pytest.fixture
def declaration():
    return AssessmentDeclaration(
        contract_version=CONTRACT_VERSION,
        assessment_id="m19-b3-a3-m18-real-path",
        assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id="410",
        development_cutoff="2023-12-01",
        forecast_dates=("2024-03-01", "2024-06-01"),
    )


@pytest.fixture
def full_frame(processed_training):
    development = processed_training.data[["ds", "microVolt"]].copy(deep=True)
    future = pd.DataFrame({
        "ds": pd.to_datetime(["2024-03-01", "2024-06-01"]),
        "microVolt": [29.9, 30.506],
    })
    return pd.concat([development, future], ignore_index=True)


def _prepare_execution(processed_training, full_frame, declaration):
    isolated = isolate_declared_holdout(BatteryDataset(full_frame), declaration=declaration)
    require_isolated_training_dataset(isolated.development, isolated=isolated)

    # A2 returns a raw dataset; M18 requires a separately processed dataset.
    # Check observation identity explicitly at this seam, rather than treating
    # the raw A2 output as an M18-compatible fingerprinted training object.
    expected = isolated.development.data[["ds", "microVolt"]].reset_index(drop=True)
    actual = processed_training.data[["ds", "microVolt"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert bool(processed_training.data["is_observed"].all())

    kwargs = inputs(processed_training)
    registry = ModelRegistry()
    registry.register("GRU", ProbeModel)
    kwargs["registry"] = registry
    assert kwargs["forecast_dates"] == list(declaration.forecast_dates)
    assert kwargs["training_context"]["train_end"] == declaration.development_cutoff
    assert kwargs["training_context"]["dataset_fingerprint"] == fingerprint_dataset(processed_training).to_dict()
    plan = build_internal_forecast_plan(**kwargs)
    assert verify_internal_forecast_plan(plan, **kwargs) is True
    return isolated, plan, kwargs


def _execute(plan, kwargs):
    return execute_internal_forecast_plan(
        plan, **kwargs, execution_authorized=True,
    )


def test_real_m18_execution_invariant_to_holdout_target_changes(
    processed_training, full_frame, declaration,
):
    ProbeModel.events = []
    ProbeModel.result_override = None
    ProbeModel.fail_at = None
    ProbeModel.mutate_training = False
    a, plan_a, kwargs_a = _prepare_execution(processed_training, full_frame, declaration)
    output_a = _execute(plan_a, kwargs_a)
    events_a = deepcopy(ProbeModel.events)

    changed = full_frame.copy(deep=True)
    changed.loc[changed["ds"] > pd.Timestamp(declaration.development_cutoff), "microVolt"] = [999.0, -999.0]
    ProbeModel.events = []
    b, plan_b, kwargs_b = _prepare_execution(processed_training, changed, declaration)
    output_b = _execute(plan_b, kwargs_b)
    events_b = deepcopy(ProbeModel.events)

    pd.testing.assert_frame_equal(a.development.data, b.development.data)
    assert a.holdout["microVolt"].tolist() != b.holdout["microVolt"].tolist()
    assert kwargs_a["training_context"]["dataset_fingerprint"] == kwargs_b["training_context"]["dataset_fingerprint"]
    assert plan_a == plan_b
    assert plan_a["selected_model"] == "GRU"
    assert plan_a["model_params"] == kwargs_a["results"]["GRU"].best_params
    assert output_a["plan_digest"] == output_b["plan_digest"]
    assert output_a["decision_digest"] == output_b["decision_digest"]
    assert output_a["execution_evidence"]["training_fingerprint"] == output_b["execution_evidence"]["training_fingerprint"]
    pd.testing.assert_series_equal(output_a["forecast_result"].forecast, output_b["forecast_result"].forecast)
    assert events_a == events_b
    assert [event[0] for event in events_a] == ["construct", "fit", "predict"]

    forecast = output_a["forecast_result"].forecast.to_numpy(dtype=float)
    rmse_a = float(np.sqrt(np.mean((a.holdout["microVolt"].to_numpy(dtype=float) - forecast) ** 2)))
    rmse_b = float(np.sqrt(np.mean((b.holdout["microVolt"].to_numpy(dtype=float) - forecast) ** 2)))
    assert rmse_a != rmse_b


def test_undeclared_future_date_rejected_before_real_m18_fit(
    processed_training, full_frame, declaration,
):
    ProbeModel.events = []
    extra = pd.DataFrame({"ds": [pd.Timestamp("2024-09-01")], "microVolt": [31.0]})
    altered = pd.concat([full_frame, extra], ignore_index=True)
    with pytest.raises(ValueError, match="holdout calendar"):
        _prepare_execution(processed_training, altered, declaration)
    assert ProbeModel.events == []


def test_changed_development_cannot_reuse_approved_m18_training(
    processed_training, full_frame, declaration,
):
    ProbeModel.events = []
    altered = full_frame.copy(deep=True)
    altered.loc[0, "microVolt"] += 1.0
    with pytest.raises(AssertionError):
        _prepare_execution(processed_training, altered, declaration)
    assert ProbeModel.events == []
