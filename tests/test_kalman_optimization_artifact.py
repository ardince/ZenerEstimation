import json

import numpy as np
import pandas as pd

from zenerestimation.data import BatteryDataset
from zenerestimation.forecasting.kalman import KalmanForecaster
from zenerestimation.optimization import (
    ExpandingWindowSplitter,
    GenericOptimizer,
    KalmanParameterSpace,
    OptimizationArtifact,
    OptimizationEvaluator,
)


def _synthetic_dataset(
    n: int = 40,
) -> BatteryDataset:
    dates = pd.date_range(
        start="2015-03-01",
        periods=n,
        freq="QS-MAR",
    )

    t = np.arange(
        n,
        dtype=float,
    )

    values = (
        20.0
        + 0.18 * t
        + 0.08 * np.sin(t / 2.0)
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
    return KalmanForecaster(
        dt=0.25,
        adaptive=True,
        **params,
    )


def test_kalman_optimization_artifact_contract():
    dataset = _synthetic_dataset()

    space = KalmanParameterSpace(
        process_noise=(1e-4, 1e-3),
        drift_noise=(1e-5,),
        regime_factor=(2.0,),
        regime_multiplier=(5.0,),
    )

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=4,
    )

    evaluator = OptimizationEvaluator(
        splitter,
        metric="rmse",
    )

    optimizer = GenericOptimizer(
        evaluator,
        model="Kalman",
    )

    result = optimizer.optimize(
        dataset,
        _model_factory,
        space.candidates(),
    )

    artifact = OptimizationArtifact(
        result,
        benchmark_steps=4,
    )

    payload = artifact.to_dict()

    assert payload["status"] == "optimized"
    assert payload["model"] == "Kalman"
    assert payload["metric"] == "rmse"

    assert payload["search"]["method"] == "explicit_candidates"
    assert payload["search"]["objective"] == "minimize"
    assert payload["search"]["candidate_count"] == 2
    assert payload["search"]["successful_count"] == 2
    assert payload["search"]["failed_count"] == 0

    assert payload["validation"]["method"] == "expanding_window"
    assert payload["validation"]["folds"] == 3
    assert payload["validation"]["validation_steps"] == 4
    assert payload["validation"]["benchmark_steps"] == 4
    assert payload["validation"]["horizon_aligned"] is True

    assert payload["best"]["params"] == result.best_params
    assert np.isclose(
        payload["best"]["score"],
        result.best_score,
    )

    assert len(payload["candidates"]) == 2

    # The optimization artifact contains model-selection evidence only.
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

    # Must remain directly JSON serializable.
    serialized = json.dumps(payload)
    assert isinstance(serialized, str)


def _collect_keys(value):
    keys = set()

    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(key)
            keys.update(_collect_keys(item))

    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_collect_keys(item))

    return keys