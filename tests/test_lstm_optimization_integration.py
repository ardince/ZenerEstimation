"""Integration tests for LSTM optimization."""

import numpy as np
import pandas as pd

from zenerestimation.data import BatteryDataset
from zenerestimation.forecasting.neural.lstm import LSTMForecaster
from zenerestimation.optimization import (
    ExpandingWindowSplitter,
    GenericOptimizer,
    LSTMParameterSpace,
    OptimizationEvaluator,
)


# ---------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------


def _synthetic_dataset(
    n: int = 32,
) -> BatteryDataset:
    """Create a small deterministic quarterly dataset."""

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


# ---------------------------------------------------------------------
# Lightweight LSTM configuration
# ---------------------------------------------------------------------


TEST_EPOCHS = 2
TEST_BATCH_SIZE = 4
TEST_SEED = 42


def _model_factory(**params):
    """Create a fresh lightweight LSTM for integration testing."""

    return LSTMForecaster(
        epochs=TEST_EPOCHS,
        batch_size=TEST_BATCH_SIZE,
        seed=TEST_SEED,
        **params,
    )


def _parameter_space():
    """Return a deliberately small integration-test search space."""

    return LSTMParameterSpace(
        window=(4, 6),
        units=(4,),
    )


def _optimizer(
    metric: str = "rmse",
):
    """Construct the standard generic optimizer stack."""

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
        model="LSTM",
    )


# ---------------------------------------------------------------------
# End-to-end optimization
# ---------------------------------------------------------------------


def test_lstm_optimization_runs_end_to_end():
    """LSTM should run through the generic optimization architecture."""

    dataset = _synthetic_dataset()
    space = _parameter_space()

    result = _optimizer().optimize(
        dataset,
        _model_factory,
        space.candidates(),
    )

    assert result.model == "LSTM"
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


def test_lstm_optimization_preserves_candidate_order():
    """Generic optimization must preserve explicit candidate ordering."""

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
# Selection
# ---------------------------------------------------------------------


def test_lstm_optimization_selects_lowest_rmse():
    """The selected candidate should have the minimum RMSE."""

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
# Temporal-fold metadata
# ---------------------------------------------------------------------


def test_lstm_optimization_records_fold_metadata():
    """LSTM candidates should retain temporal validation evidence."""

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

        assert np.isfinite(
            metrics["rmse"]
        )
        assert np.isfinite(
            metrics["mae"]
        )
        assert np.isfinite(
            metrics["mape"]
        )


# ---------------------------------------------------------------------
# Fold aggregation
# ---------------------------------------------------------------------


def test_lstm_optimization_aggregates_fold_metrics():
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


def test_lstm_optimization_supports_mae_selection():
    """The generic evaluator should support MAE selection for LSTM."""

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


def test_lstm_optimization_creates_fresh_model_per_fold():
    """Every candidate/fold evaluation must receive a fresh LSTM."""

    dataset = _synthetic_dataset()

    created_models = []

    def tracking_factory(**params):
        model = LSTMForecaster(
            epochs=TEST_EPOCHS,
            batch_size=TEST_BATCH_SIZE,
            seed=TEST_SEED,
            **params,
        )

        created_models.append(model)

        return model

    space = _parameter_space()

    result = _optimizer().optimize(
        dataset,
        tracking_factory,
        space.candidates(),
    )

    assert result.candidate_count == 2

    # 2 candidates × 2 temporal folds.
    assert len(created_models) == 4

    assert len(
        {
            id(model)
            for model in created_models
        }
    ) == 4