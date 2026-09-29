"""Synthetic repeated-seed stability integration for hybrid forecasters."""

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data import BatteryDataset
from zenerestimation.forecasting.hybrid import (
    KalmanLSTMForecaster,
    LinearTrendLSTMForecaster,
)
from zenerestimation.forecasting.neural import LSTMForecaster
from zenerestimation.optimization import (
    ExpandingWindowSplitter,
    OptimizationEvaluator,
    StabilityEvaluator,
    StabilityResult,
)


# ---------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------


def _synthetic_dataset(
    n: int = 32,
) -> BatteryDataset:
    """Create a small quarterly degradation series."""

    dates = pd.date_range(
        start="2016-03-01",
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
        + 0.04 * np.sin(t / 2.0)
    )

    return BatteryDataset(
        dataframe=pd.DataFrame(
            {
                "ds": dates,
                "microVolt": values,
            }
        )
    )


# ---------------------------------------------------------------------
# Shared evaluation configuration
# ---------------------------------------------------------------------


def _optimization_evaluator() -> OptimizationEvaluator:
    """Construct a cheap temporal evaluator."""

    return OptimizationEvaluator(
        ExpandingWindowSplitter(
            folds=2,
            validation_steps=3,
        ),
        metric="rmse",
    )


def _hybrid_params() -> dict:
    """Frozen residual-neural configuration."""

    return {
        "window": 4,
        "units": 4,
        "epochs": 2,
        "batch_size": 4,
    }


# ---------------------------------------------------------------------
# Hybrid factories
# ---------------------------------------------------------------------


def _linear_trend_lstm_factory(**params):
    """Create a fresh LinearTrendLSTM hybrid."""

    window = params["window"]

    residual_lstm = LSTMForecaster(
        window=window,
        units=params["units"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        seed=params["seed"],
    )

    return LinearTrendLSTMForecaster(
        lstm_model=residual_lstm,
        window=window,
    )


def _kalman_lstm_factory(**params):
    """Create a fresh KalmanLSTM hybrid."""

    window = params["window"]

    residual_lstm = LSTMForecaster(
        window=window,
        units=params["units"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        seed=params["seed"],
    )

    return KalmanLSTMForecaster(
        window=window,
        lstm_model=residual_lstm,
    )


# ---------------------------------------------------------------------
# Common hybrid cases
# ---------------------------------------------------------------------


HYBRID_CASES = [
    (
        "LinearTrendLSTM",
        _linear_trend_lstm_factory,
    ),
    (
        "KalmanLSTM",
        _kalman_lstm_factory,
    ),
]


# ---------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name,model_factory",
    HYBRID_CASES,
)
def test_hybrid_stability_runs_end_to_end(
    model_name,
    model_factory,
):
    """Both hybrid architectures should support repeated-seed evaluation."""

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model=model_name,
    ).evaluate(
        _synthetic_dataset(),
        model_factory,
        params=_hybrid_params(),
        seeds=(0, 1),
    )

    assert isinstance(
        result,
        StabilityResult,
    )

    assert result.model == model_name
    assert result.metric == "rmse"
    assert result.run_count == 2

    assert tuple(
        seed_result.seed
        for seed_result in result.seeds
    ) == (
        0,
        1,
    )


# ---------------------------------------------------------------------
# Frozen configuration
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name,model_factory",
    HYBRID_CASES,
)
def test_hybrid_stability_preserves_frozen_configuration(
    model_name,
    model_factory,
):
    """Seed must remain separate from optimized hybrid parameters."""

    params = _hybrid_params()

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model=model_name,
    ).evaluate(
        _synthetic_dataset(),
        model_factory,
        params=params,
        seeds=(0, 1),
    )

    assert result.params == params
    assert "seed" not in result.params


# ---------------------------------------------------------------------
# Numerical results
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name,model_factory",
    HYBRID_CASES,
)
def test_hybrid_stability_produces_finite_metrics(
    model_name,
    model_factory,
):
    """Every hybrid seed replication should produce finite metrics."""

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model=model_name,
    ).evaluate(
        _synthetic_dataset(),
        model_factory,
        params=_hybrid_params(),
        seeds=(0, 1),
    )

    for seed_result in result.seeds:
        assert np.isfinite(
            seed_result.score
        )

        assert np.isfinite(
            seed_result.metrics["rmse"]
        )

        assert np.isfinite(
            seed_result.metrics["mae"]
        )

        assert np.isfinite(
            seed_result.metrics["mape"]
        )

        assert np.isclose(
            seed_result.score,
            seed_result.metrics["rmse"],
        )


# ---------------------------------------------------------------------
# Temporal validation evidence
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name,model_factory",
    HYBRID_CASES,
)
def test_hybrid_stability_preserves_fold_evidence(
    model_name,
    model_factory,
):
    """Hybrid seed results should retain temporal-validation evidence."""

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model=model_name,
    ).evaluate(
        _synthetic_dataset(),
        model_factory,
        params=_hybrid_params(),
        seeds=(0, 1),
    )

    for seed_result in result.seeds:
        validation = seed_result.metadata[
            "validation"
        ]

        assert validation[
            "validation"
        ] == "expanding_window"

        assert validation["folds"] == 2

        assert validation[
            "validation_steps"
        ] == 3

        assert validation[
            "selection_metric"
        ] == "rmse"

        assert len(
            validation["fold_metrics"]
        ) == 2


# ---------------------------------------------------------------------
# Stability metadata
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name,model_factory",
    HYBRID_CASES,
)
def test_hybrid_stability_records_repeated_seed_metadata(
    model_name,
    model_factory,
):
    """Both hybrids should expose the common stability contract."""

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model=model_name,
    ).evaluate(
        _synthetic_dataset(),
        model_factory,
        params=_hybrid_params(),
        seeds=(0, 1),
    )

    assert result.metadata[
        "analysis"
    ] == "repeated_seed"

    assert result.metadata[
        "seed_count"
    ] == 2

    assert result.metadata[
        "selection"
    ] == "none"


# ---------------------------------------------------------------------
# Stability summary
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name,model_factory",
    HYBRID_CASES,
)
def test_hybrid_stability_summary_matches_seed_results(
    model_name,
    model_factory,
):
    """Summary statistics must include all hybrid seed replications."""

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model=model_name,
    ).evaluate(
        _synthetic_dataset(),
        model_factory,
        params=_hybrid_params(),
        seeds=(0, 1),
    )

    scores = np.asarray(
        result.scores,
        dtype=float,
    )

    assert np.isclose(
        result.mean_score,
        np.mean(scores),
    )

    assert np.isclose(
        result.std_score,
        np.std(
            scores,
            ddof=0,
        ),
    )

    assert np.isclose(
        result.min_score,
        np.min(scores),
    )

    assert np.isclose(
        result.max_score,
        np.max(scores),
    )