"""Synthetic stability integration tests for LSTM and GRU."""

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data import BatteryDataset
from zenerestimation.forecasting.neural import (
    GRUForecaster,
    LSTMForecaster,
)
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
        + 0.04 * np.sin(
            t / 2.0
        )
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


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------


def _optimization_evaluator() -> OptimizationEvaluator:
    """Construct a cheap temporal evaluator."""

    splitter = ExpandingWindowSplitter(
        folds=2,
        validation_steps=3,
    )

    return OptimizationEvaluator(
        splitter,
        metric="rmse",
    )


def _neural_params() -> dict:
    """Frozen neural configuration excluding replication seed."""

    return {
        "window": 4,
        "units": 4,
        "epochs": 2,
        "batch_size": 4,
    }


# ---------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------


def _lstm_factory(**params):
    """Create a fresh real LSTM forecaster."""

    return LSTMForecaster(
        window=params["window"],
        units=params["units"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        seed=params["seed"],
    )


def _gru_factory(**params):
    """Create a fresh real GRU forecaster."""

    return GRUForecaster(
        window=params["window"],
        units=params["units"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        seed=params["seed"],
    )


# ---------------------------------------------------------------------
# LSTM integration
# ---------------------------------------------------------------------


def test_lstm_stability_runs_end_to_end():
    """Real LSTM should run through repeated-seed evaluation."""

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model="LSTM",
    ).evaluate(
        _synthetic_dataset(),
        _lstm_factory,
        params=_neural_params(),
        seeds=(0, 1),
    )

    assert isinstance(
        result,
        StabilityResult,
    )

    assert result.model == "LSTM"
    assert result.metric == "rmse"
    assert result.run_count == 2

    assert tuple(
        seed_result.seed
        for seed_result in result.seeds
    ) == (
        0,
        1,
    )


def test_lstm_stability_preserves_frozen_configuration():
    """Replication seed must not become an LSTM model parameter."""

    params = _neural_params()

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model="LSTM",
    ).evaluate(
        _synthetic_dataset(),
        _lstm_factory,
        params=params,
        seeds=(0, 1),
    )

    assert result.params == params
    assert "seed" not in result.params


def test_lstm_stability_produces_finite_metrics():
    """Every real LSTM seed should produce finite metrics."""

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model="LSTM",
    ).evaluate(
        _synthetic_dataset(),
        _lstm_factory,
        params=_neural_params(),
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


def test_lstm_stability_preserves_fold_evidence():
    """Real LSTM seed results should retain temporal fold evidence."""

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model="LSTM",
    ).evaluate(
        _synthetic_dataset(),
        _lstm_factory,
        params=_neural_params(),
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

        assert len(
            validation["fold_metrics"]
        ) == 2


# ---------------------------------------------------------------------
# GRU integration
# ---------------------------------------------------------------------


def test_gru_stability_runs_end_to_end():
    """Real GRU should run through repeated-seed evaluation."""

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model="GRU",
    ).evaluate(
        _synthetic_dataset(),
        _gru_factory,
        params=_neural_params(),
        seeds=(0, 1),
    )

    assert isinstance(
        result,
        StabilityResult,
    )

    assert result.model == "GRU"
    assert result.metric == "rmse"
    assert result.run_count == 2

    assert tuple(
        seed_result.seed
        for seed_result in result.seeds
    ) == (
        0,
        1,
    )


def test_gru_stability_preserves_frozen_configuration():
    """Replication seed must not become a GRU model parameter."""

    params = _neural_params()

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model="GRU",
    ).evaluate(
        _synthetic_dataset(),
        _gru_factory,
        params=params,
        seeds=(0, 1),
    )

    assert result.params == params
    assert "seed" not in result.params


def test_gru_stability_produces_finite_metrics():
    """Every real GRU seed should produce finite metrics."""

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model="GRU",
    ).evaluate(
        _synthetic_dataset(),
        _gru_factory,
        params=_neural_params(),
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


def test_gru_stability_preserves_fold_evidence():
    """Real GRU seed results should retain temporal fold evidence."""

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model="GRU",
    ).evaluate(
        _synthetic_dataset(),
        _gru_factory,
        params=_neural_params(),
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

        assert len(
            validation["fold_metrics"]
        ) == 2


# ---------------------------------------------------------------------
# Cross-model stability contract
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "model_name",
        "model_factory",
    ),
    [
        (
            "LSTM",
            _lstm_factory,
        ),
        (
            "GRU",
            _gru_factory,
        ),
    ],
)
def test_neural_stability_records_repeated_seed_metadata(
    model_name,
    model_factory,
):
    """Both neural architectures should expose common stability metadata."""

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model=model_name,
    ).evaluate(
        _synthetic_dataset(),
        model_factory,
        params=_neural_params(),
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


@pytest.mark.parametrize(
    (
        "model_name",
        "model_factory",
    ),
    [
        (
            "LSTM",
            _lstm_factory,
        ),
        (
            "GRU",
            _gru_factory,
        ),
    ],
)
def test_neural_stability_summary_matches_seed_results(
    model_name,
    model_factory,
):
    """Summary statistics should include all neural seed replications."""

    result = StabilityEvaluator(
        _optimization_evaluator(),
        model=model_name,
    ).evaluate(
        _synthetic_dataset(),
        model_factory,
        params=_neural_params(),
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