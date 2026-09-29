"""Integration tests for Kalman optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.forecasting.kalman import KalmanForecaster
from zenerestimation.optimization import (
    ExpandingWindowSplitter,
    GenericOptimizer,
    KalmanParameterSpace,
    OptimizationEvaluator,
)


def _synthetic_dataset(
    n: int = 40,
) -> BatteryDataset:
    """Create a deterministic quarterly dataset for optimization tests."""

    dates = pd.date_range(
        start="2015-03-01",
        periods=n,
        freq="QS-MAR",
    )

    t = np.arange(
        n,
        dtype=float,
    )

    # Smooth trend plus deterministic low-amplitude variation.
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

    return BatteryDataset(dataframe)


def _parameter_space() -> KalmanParameterSpace:
    """Return a deliberately small Kalman search space."""

    return KalmanParameterSpace(
        process_noise=(
            1e-4,
            1e-3,
        ),
        drift_noise=(
            1e-5,
            1e-4,
        ),
        regime_factor=(
            2.0,
        ),
        regime_multiplier=(
            5.0,
            10.0,
        ),
    )


def _model_factory(**params):
    """Create a fresh adaptive Kalman forecaster."""

    return KalmanForecaster(
        dt=0.25,
        adaptive=True,
        **params,
    )


def _optimizer(
    *,
    metric: str = "rmse",
) -> GenericOptimizer:
    """Build the generic temporal optimizer."""

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=4,
    )

    evaluator = OptimizationEvaluator(
        splitter,
        metric=metric,
    )

    return GenericOptimizer(
        evaluator,
        model="Kalman",
    )


def test_kalman_optimization_runs_end_to_end():
    dataset = _synthetic_dataset()
    space = _parameter_space()
    optimizer = _optimizer()

    result = optimizer.optimize(
        dataset,
        _model_factory,
        space.candidates(),
    )

    assert result.model == "Kalman"
    assert result.metric == "rmse"

    assert result.candidate_count == 8
    assert result.successful_count == 8
    assert result.failed_count == 0

    assert len(result.candidates) == 8

    assert np.isfinite(
        result.best_score
    )

    assert set(result.best_params) == {
        "process_noise",
        "drift_noise",
        "regime_factor",
        "regime_multiplier",
    }


def test_kalman_optimization_preserves_candidate_order():
    dataset = _synthetic_dataset()
    space = _parameter_space()
    candidates = space.candidates()

    result = _optimizer().optimize(
        dataset,
        _model_factory,
        candidates,
    )

    persisted_params = tuple(
        candidate.params
        for candidate in result.candidates
    )

    assert persisted_params == candidates


def test_kalman_optimization_selects_lowest_rmse():
    dataset = _synthetic_dataset()

    result = _optimizer(
        metric="rmse",
    ).optimize(
        dataset,
        _model_factory,
        _parameter_space().candidates(),
    )

    successful_scores = [
        candidate.score
        for candidate in result.candidates
        if candidate.status == "evaluated"
    ]

    assert result.best_score == pytest.approx(
        min(successful_scores)
    )


def test_kalman_optimization_records_fold_metadata():
    dataset = _synthetic_dataset()

    result = _optimizer().optimize(
        dataset,
        _model_factory,
        _parameter_space().candidates(),
    )

    for candidate in result.candidates:
        assert candidate.status == "evaluated"

        assert (
            candidate.metadata["validation"]
            == "expanding_window"
        )

        assert candidate.metadata["folds"] == 3

        assert (
            candidate.metadata["validation_steps"]
            == 4
        )

        assert (
            candidate.metadata["selection_metric"]
            == "rmse"
        )

        fold_metrics = candidate.metadata[
            "fold_metrics"
        ]

        assert len(fold_metrics) == 3

        for metrics in fold_metrics:
            assert set(metrics) == {
                "rmse",
                "mae",
                "mape",
            }

            assert all(
                np.isfinite(value)
                for value in metrics.values()
            )


def test_kalman_optimization_aggregates_fold_metrics():
    dataset = _synthetic_dataset()

    result = _optimizer().optimize(
        dataset,
        _model_factory,
        _parameter_space().candidates(),
    )

    candidate = result.candidates[0]

    fold_metrics = candidate.metadata[
        "fold_metrics"
    ]

    expected_rmse = np.mean(
        [
            metrics["rmse"]
            for metrics in fold_metrics
        ]
    )

    expected_mae = np.mean(
        [
            metrics["mae"]
            for metrics in fold_metrics
        ]
    )

    expected_mape = np.mean(
        [
            metrics["mape"]
            for metrics in fold_metrics
        ]
    )

    assert candidate.metrics["rmse"] == pytest.approx(
        expected_rmse
    )

    assert candidate.metrics["mae"] == pytest.approx(
        expected_mae
    )

    assert candidate.metrics["mape"] == pytest.approx(
        expected_mape
    )

    assert candidate.score == pytest.approx(
        expected_rmse
    )


def test_kalman_optimization_supports_mae_selection():
    dataset = _synthetic_dataset()

    result = _optimizer(
        metric="mae",
    ).optimize(
        dataset,
        _model_factory,
        _parameter_space().candidates(),
    )

    successful_scores = [
        candidate.score
        for candidate in result.candidates
        if candidate.status == "evaluated"
    ]

    assert result.metric == "mae"

    assert result.best_score == pytest.approx(
        min(successful_scores)
    )

    best_candidate = next(
        candidate
        for candidate in result.candidates
        if candidate.params == result.best_params
        and candidate.score
        == pytest.approx(result.best_score)
    )

    assert best_candidate.score == pytest.approx(
        best_candidate.metrics["mae"]
    )


def test_kalman_optimization_creates_fresh_model_per_fold():
    dataset = _synthetic_dataset()

    created_models = []

    def tracking_factory(**params):
        model = KalmanForecaster(
            dt=0.25,
            adaptive=True,
            **params,
        )

        created_models.append(model)

        return model

    space = KalmanParameterSpace(
        process_noise=(
            1e-4,
            1e-3,
        ),
        drift_noise=(1e-4,),
        regime_factor=(2.5,),
        regime_multiplier=(10.0,),
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
        tracking_factory,
        space.candidates(),
    )

    # 2 candidates × 3 temporal folds.
    assert len(created_models) == 6

    assert len(
        {
            id(model)
            for model in created_models
        }
    ) == 6

    assert result.candidate_count == 2