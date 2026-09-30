"""Synthetic integration tests for optimized multi-model benchmarking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.temporal import TemporalPreprocessor
from zenerestimation.evaluation.benchmark import (
    BenchmarkModelSpec,
    OptimizedBenchmark,
)

from zenerestimation.evaluation.evaluator import ForecastEvaluator
from zenerestimation.forecasting.arima import ARIMAForecaster
from zenerestimation.forecasting.kalman import KalmanForecaster
from zenerestimation.forecasting.neural.lstm import LSTMForecaster
from zenerestimation.forecasting.neural.gru import GRUForecaster
from zenerestimation.forecasting.hybrid.linear_trend_lstm import (
    LinearTrendLSTMForecaster,
)
from zenerestimation.forecasting.hybrid.kalman_lstm import (
    KalmanLSTMForecaster,
)


# ---------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------


def _synthetic_dataset() -> BatteryDataset:
    """Create a small quarterly degradation series."""

    dates = pd.date_range(
        start="2015-03-01",
        periods=32,
        freq="QS-MAR",
    )

    index = np.arange(
        len(dates),
        dtype=float,
    )

    values = (
        20.0
        + 0.30 * index
        + 0.15 * np.sin(
            2.0 * np.pi * index / 4.0
        )
    )

    dataframe = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": values,
        }
    )

    return BatteryDataset(
        dataframe=dataframe
    )


# ---------------------------------------------------------------------
# Hybrid factories
# ---------------------------------------------------------------------


def _linear_trend_lstm_factory(
    window,
    units,
    epochs,
    batch_size,
    seed,
):
    """Create a fresh LinearTrendLSTM hybrid."""

    residual_model = LSTMForecaster(
        window=window,
        units=units,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )

    return LinearTrendLSTMForecaster(
        lstm_model=residual_model,
        window=window,
    )


def _kalman_lstm_factory(
    window,
    units,
    epochs,
    batch_size,
    seed,
):
    """Create a fresh KalmanLSTM hybrid."""

    residual_model = LSTMForecaster(
        window=window,
        units=units,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )

    return KalmanLSTMForecaster(
        window=window,
        lstm_model=residual_model,
    )


# ---------------------------------------------------------------------
# Frozen benchmark specifications
# ---------------------------------------------------------------------


def _benchmark_specs():
    """Return cheap frozen configurations for all model families."""

    return (
        BenchmarkModelSpec(
            name="ARIMA",
            factory=ARIMAForecaster,
            params={
                "order": (
                    2,
                    1,
                    0,
                ),
            },
        ),
        BenchmarkModelSpec(
            name="Kalman",
            factory=KalmanForecaster,
            params={
                "dt": 0.25,
                "process_noise": 1e-3,
                "drift_noise": 1e-4,
                "adaptive": True,
                "regime_factor": 2.5,
                "regime_multiplier": 10.0,
            },
        ),
        BenchmarkModelSpec(
            name="LSTM",
            factory=LSTMForecaster,
            params={
                "window": 4,
                "units": 4,
                "epochs": 2,
                "batch_size": 4,
                "seed": 42,
            },
        ),
        BenchmarkModelSpec(
            name="GRU",
            factory=GRUForecaster,
            params={
                "window": 4,
                "units": 4,
                "epochs": 2,
                "batch_size": 4,
                "seed": 42,
            },
        ),
        BenchmarkModelSpec(
            name="LinearTrendLSTM",
            factory=_linear_trend_lstm_factory,
            params={
                "window": 4,
                "units": 4,
                "epochs": 2,
                "batch_size": 4,
                "seed": 42,
            },
        ),
        BenchmarkModelSpec(
            name="KalmanLSTM",
            factory=_kalman_lstm_factory,
            params={
                "window": 4,
                "units": 4,
                "epochs": 2,
                "batch_size": 4,
                "seed": 42,
            },
        ),
    )


def _benchmark():
    """Create the standardized synthetic benchmark."""

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    return OptimizedBenchmark(
        evaluator=evaluator
    )


# ---------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------


def test_optimized_benchmark_runs_all_model_families():
    """All six forecasting families should complete one benchmark."""

    results = _benchmark().evaluate(
        dataset=_synthetic_dataset(),
        specs=_benchmark_specs(),
    )

    assert list(
        results.keys()
    ) == [
        "ARIMA",
        "Kalman",
        "LSTM",
        "GRU",
        "LinearTrendLSTM",
        "KalmanLSTM",
    ]


def test_optimized_benchmark_returns_evaluation_results():
    """Every benchmark output should expose standard metrics."""

    results = _benchmark().evaluate(
        dataset=_synthetic_dataset(),
        specs=_benchmark_specs(),
    )

    for result in results.values():
        assert hasattr(
            result,
            "rmse",
        )

        assert hasattr(
            result,
            "mae",
        )

        assert hasattr(
            result,
            "mape",
        )


def test_optimized_benchmark_metrics_are_finite():
    """Every model must produce finite benchmark metrics."""

    results = _benchmark().evaluate(
        dataset=_synthetic_dataset(),
        specs=_benchmark_specs(),
    )

    for result in results.values():
        assert np.isfinite(
            result.rmse
        )

        assert np.isfinite(
            result.mae
        )

        assert np.isfinite(
            result.mape
        )


def test_optimized_benchmark_uses_common_holdout_size():
    """Every model must be evaluated on the same holdout horizon."""

    results = _benchmark().evaluate(
        dataset=_synthetic_dataset(),
        specs=_benchmark_specs(),
    )

    for result in results.values():
        assert len(
            result.actual
        ) == 4

        assert len(
            result.predicted
        ) == 4


def test_optimized_benchmark_uses_common_holdout_values():
    """All model evaluations must share identical actual observations."""

    results = _benchmark().evaluate(
        dataset=_synthetic_dataset(),
        specs=_benchmark_specs(),
    )

    actuals = [
        np.asarray(
            result.actual,
            dtype=float,
        )
        for result in results.values()
    ]

    reference = actuals[0]

    for actual in actuals[1:]:
        np.testing.assert_allclose(
            actual,
            reference,
        )


def test_optimized_benchmark_predictions_are_finite():
    """Every model must produce finite holdout predictions."""

    results = _benchmark().evaluate(
        dataset=_synthetic_dataset(),
        specs=_benchmark_specs(),
    )

    for result in results.values():
        predicted = np.asarray(
            result.predicted,
            dtype=float,
        )

        assert np.all(
            np.isfinite(predicted)
        )


def test_optimized_benchmark_preserves_model_order():
    """Benchmark evidence must retain frozen specification order."""

    results = _benchmark().evaluate(
        dataset=_synthetic_dataset(),
        specs=_benchmark_specs(),
    )

    assert tuple(
        results
    ) == (
        "ARIMA",
        "Kalman",
        "LSTM",
        "GRU",
        "LinearTrendLSTM",
        "KalmanLSTM",
    )