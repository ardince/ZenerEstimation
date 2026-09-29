"""Regression tests for standardized KalmanLSTM optimization artifacts."""

import json

import numpy as np
import pandas as pd

from zenerestimation.data import BatteryDataset

# Use the established neural import path in this repository.
from zenerestimation.forecasting.neural.lstm import LSTMForecaster
from zenerestimation.forecasting.hybrid import KalmanLSTMForecaster

from zenerestimation.optimization import (
    ExpandingWindowSplitter,
    GenericOptimizer,
    HybridNeuralParameterSpace,
    OptimizationArtifact,
    OptimizationEvaluator,
)


TEST_EPOCHS = 2
TEST_BATCH_SIZE = 4
TEST_SEED = 42

space = HybridNeuralParameterSpace(
    window=(4, 6),
    units=(4,),
)

splitter = ExpandingWindowSplitter(
    folds=2,
    validation_steps=3,
)


def _synthetic_dataset(
    n: int = 32,
) -> BatteryDataset:
    dates = pd.date_range(
        start="2017-03-01",
        periods=n,
        freq="QS-MAR",
    )

    t = np.arange(
        n,
        dtype=float,
    )

    values = (
        20.0
        + 0.15 * t
        + 0.05 * np.sin(t / 2.0)
    )

    dataframe = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": values,
        }
    )

    return BatteryDataset(
        dataframe=dataframe,
    )


def _model_factory(**params):
    window = params["window"]
    units = params["units"]

    lstm_model = LSTMForecaster(
        window=window,
        units=units,
        epochs=TEST_EPOCHS,
        batch_size=TEST_BATCH_SIZE,
        seed=TEST_SEED,
    )

    return KalmanLSTMForecaster(
        window=window,
        lstm_model=lstm_model,
    )

def _collect_keys(value):
    keys = set()

    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(key)
            keys.update(
                _collect_keys(item)
            )

    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(
                _collect_keys(item)
            )

    return keys


def test_hybrid_optimization_artifact_contract():
    dataset = _synthetic_dataset()

    space = HybridNeuralParameterSpace(
        window=(4, 6),
        units=(4,),
    )

    splitter = ExpandingWindowSplitter(
        folds=2,
        validation_steps=3,
    )

    evaluator = OptimizationEvaluator(
        splitter,
        metric="rmse",
    )

    optimizer = GenericOptimizer(
        evaluator,
        model="KalmanLSTM",
    )

    result = optimizer.optimize(
        dataset,
        _model_factory,
        space.candidates(),
    )

    artifact = OptimizationArtifact(
        result,
        benchmark_steps=3,
    )

    payload = artifact.to_dict()

    assert payload["status"] == "optimized"
    assert payload["model"] == "KalmanLSTM"
    assert payload["metric"] == "rmse"

    assert (
        payload["search"]["method"]
        == "explicit_candidates"
    )
    assert (
        payload["search"]["objective"]
        == "minimize"
    )

    assert (
        payload["search"]["candidate_count"]
        == 2
    )
    assert (
        payload["search"]["successful_count"]
        == 2
    )
    assert (
        payload["search"]["failed_count"]
        == 0
    )

    assert (
        payload["validation"]["method"]
        == "expanding_window"
    )
    assert (
        payload["validation"]["folds"]
        == 2
    )
    assert (
        payload["validation"]["validation_steps"]
        == 3
    )
    assert (
        payload["validation"]["benchmark_steps"]
        == 3
    )
    assert (
        payload["validation"]["horizon_aligned"]
        is True
    )

    assert (
        payload["best"]["params"]
        == result.best_params
    )

    assert np.isclose(
        payload["best"]["score"],
        result.best_score,
    )

    assert len(
        payload["candidates"]
    ) == 2

    # Neural training-protocol controls are intentionally
    # fixed outside the optimization parameter space.
    for candidate in payload["candidates"]:
        assert set(
            candidate["params"]
        ) == {
            "window",
            "units",
        }

    forbidden_keys = {
        "evaluation",
        "actual",
        "predicted",
        "forecast_dates",
        "benchmark_rmse",
        "benchmark_mae",
        "benchmark_mape",
    }

    assert forbidden_keys.isdisjoint(
        _collect_keys(payload)
    )

    serialized = json.dumps(
        payload
    )

    assert isinstance(
        serialized,
        str,
    )