"""Integration tests for LinearTrendLSTM hybrid optimization."""

import numpy as np
import pandas as pd

from zenerestimation.data import BatteryDataset
from zenerestimation.forecasting.neural import LSTMForecaster
from zenerestimation.forecasting.hybrid import LinearTrendLSTMForecaster
from zenerestimation.optimization import (
    ExpandingWindowSplitter,
    GenericOptimizer,
    HybridNeuralParameterSpace,
    OptimizationEvaluator,
)


# ---------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------


def _synthetic_dataset(n: int = 32) -> BatteryDataset:
    """Create a small deterministic quarterly dataset."""

    dates = pd.date_range(
        start="2017-03-01",
        periods=n,
        freq="QS-MAR",
    )

    t = np.arange(n, dtype=float)

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


# ---------------------------------------------------------------------
# Lightweight hybrid configuration
# ---------------------------------------------------------------------


TEST_EPOCHS = 2
TEST_BATCH_SIZE = 4
TEST_SEED = 42


def _model_factory(**params):
    """Create a fresh lightweight LinearTrendLSTM hybrid model."""

    window = params["window"]
    units = params["units"]

    lstm_model = LSTMForecaster(
        window=window,
        units=units,
        epochs=TEST_EPOCHS,
        batch_size=TEST_BATCH_SIZE,
        seed=TEST_SEED,
    )

    return LinearTrendLSTMForecaster(
        lstm_model=lstm_model,
        window=window,
    )


def _parameter_space() -> HybridNeuralParameterSpace:
    """Return a deliberately small hybrid optimization search space."""

    return HybridNeuralParameterSpace(
        window=(4, 6),
        units=(4,),
    )


def _optimizer(
    metric: str = "rmse",
) -> GenericOptimizer:
    """Construct the standard generic optimization stack."""

    splitter = ExpandingWindowSplitter(
        folds=2,
        validation_steps=3,
    )

    evaluator = OptimizationEvaluator(
        splitter,
        metric=metric,
    )

    return GenericOptimizer(
        evaluator,
        model="LinearTrendLSTM",
    )


# ---------------------------------------------------------------------
# End-to-end optimization
# ---------------------------------------------------------------------


def test_linear_trend_lstm_optimization_runs_end_to_end():
    """LinearTrendLSTM should run through generic optimization."""

    dataset = _synthetic_dataset()
    space = _parameter_space()

    result = _optimizer().optimize(
        dataset,
        _model_factory,
        space.candidates(),
    )

    assert result.model == "LinearTrendLSTM"
    assert result.metric == "rmse"

    assert result.candidate_count == 2
    assert result.successful_count == 2
    assert result.failed_count == 0

    assert np.isfinite(result.best_score)

    assert set(result.best_params) == {
        "window",
        "units",
    }


# ---------------------------------------------------------------------
# Candidate ordering
# ---------------------------------------------------------------------


def test_linear_trend_lstm_optimization_preserves_candidate_order():
    """Optimization must preserve explicit candidate ordering."""

    dataset = _synthetic_dataset()
    space = _parameter_space()

    expected = space.candidates()

    result = _optimizer().optimize(
        dataset,
        _model_factory,
        expected,
    )

    actual = [
        candidate.params
        for candidate in result.candidates
    ]

    assert actual == expected


# ---------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------


def test_linear_trend_lstm_optimization_selects_lowest_rmse():
    """The selected hybrid candidate should have minimum RMSE."""

    dataset = _synthetic_dataset()

    result = _optimizer().optimize(
        dataset,
        _model_factory,
        _parameter_space().candidates(),
    )

    scores = [
        candidate.score
        for candidate in result.candidates
        if candidate.status == "evaluated"
    ]

    assert scores

    assert np.isclose(
        result.best_score,
        min(scores),
    )

    selected = [
        candidate
        for candidate in result.candidates
        if candidate.params == result.best_params
    ]

    assert selected

    assert any(
        np.isclose(
            candidate.score,
            result.best_score,
        )
        for candidate in selected
    )


# ---------------------------------------------------------------------
# Temporal validation metadata
# ---------------------------------------------------------------------


def test_linear_trend_lstm_optimization_records_fold_metadata():
    """Candidates should retain temporal-validation evidence."""

    dataset = _synthetic_dataset()

    result = _optimizer().optimize(
        dataset,
        _model_factory,
        _parameter_space().candidates(),
    )

    candidate = result.candidates[0]
    metadata = candidate.metadata

    assert metadata["validation"] == "expanding_window"
    assert metadata["folds"] == 2
    assert metadata["validation_steps"] == 3
    assert metadata["selection_metric"] == "rmse"

    fold_metrics = metadata["fold_metrics"]

    assert len(fold_metrics) == 2

    for metrics in fold_metrics:
        assert set(metrics) == {
            "rmse",
            "mae",
            "mape",
        }

        assert np.isfinite(metrics["rmse"])
        assert np.isfinite(metrics["mae"])
        assert np.isfinite(metrics["mape"])


# ---------------------------------------------------------------------
# Fold aggregation
# ---------------------------------------------------------------------


def test_linear_trend_lstm_optimization_aggregates_fold_metrics():
    """Candidate metrics should equal means of temporal-fold metrics."""

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

    for metric in (
        "rmse",
        "mae",
        "mape",
    ):
        expected = np.mean(
            [
                fold[metric]
                for fold in fold_metrics
            ]
        )

        assert np.isclose(
            candidate.metrics[metric],
            expected,
        )

    assert np.isclose(
        candidate.score,
        candidate.metrics["rmse"],
    )


# ---------------------------------------------------------------------
# Alternate selection metric
# ---------------------------------------------------------------------


def test_linear_trend_lstm_optimization_supports_mae_selection():
    """Generic optimization should support MAE selection."""

    dataset = _synthetic_dataset()

    result = _optimizer(
        metric="mae",
    ).optimize(
        dataset,
        _model_factory,
        _parameter_space().candidates(),
    )

    assert result.metric == "mae"

    scores = [
        candidate.score
        for candidate in result.candidates
        if candidate.status == "evaluated"
    ]

    assert scores

    assert np.isclose(
        result.best_score,
        min(scores),
    )

    best_candidates = [
        candidate
        for candidate in result.candidates
        if (
            candidate.status == "evaluated"
            and np.isclose(
                candidate.score,
                result.best_score,
            )
        )
    ]

    assert best_candidates


# ---------------------------------------------------------------------
# Fresh-model invariant
# ---------------------------------------------------------------------


def test_linear_trend_lstm_optimization_creates_fresh_model_per_fold():
    """Every candidate/fold must receive a fresh hybrid and residual LSTM."""

    dataset = _synthetic_dataset()

    created_models = []
    created_lstm_models = []

    def tracking_factory(**params):
        window = params["window"]
        units = params["units"]

        lstm_model = LSTMForecaster(
            window=window,
            units=units,
            epochs=TEST_EPOCHS,
            batch_size=TEST_BATCH_SIZE,
            seed=TEST_SEED,
        )

        model = LinearTrendLSTMForecaster(
            lstm_model=lstm_model,
            window=window,
        )

        created_lstm_models.append(lstm_model)
        created_models.append(model)

        return model

    result = _optimizer().optimize(
        dataset,
        tracking_factory,
        _parameter_space().candidates(),
    )

    assert result.candidate_count == 2

    # 2 candidates × 2 temporal folds.
    assert len(created_models) == 4
    assert len(created_lstm_models) == 4

    assert len(
        {
            id(model)
            for model in created_models
        }
    ) == 4

    assert len(
        {
            id(model)
            for model in created_lstm_models
        }
    ) == 4