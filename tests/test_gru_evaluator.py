"""
Integration tests for GRU evaluation.

These tests verify that GRUForecaster operates through
the standard ForecastEvaluator contract without a
model-specific evaluation path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data import BatteryDataset
from zenerestimation.data.temporal import (
    TemporalPreprocessor,
)
from zenerestimation.evaluation import (
    EvaluationResult,
    ForecastEvaluator,
)
from zenerestimation.forecasting.neural.gru import (
    GRUForecaster,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def build_gru():

    return GRUForecaster(
        window=4,
        units=8,
        epochs=1,
        batch_size=4,
        seed=42,
    )


@pytest.fixture
def evaluation_dataset():

    dates = pd.date_range(
        start="2020-03-01",
        periods=24,
        freq="QS-MAR",
    )

    training_values = np.linspace(
        10.0,
        20.0,
        20,
    )

    validation_values = np.array(
        [
            100.0,
            110.0,
            120.0,
            130.0,
        ]
    )

    values = np.concatenate(
        [
            training_values,
            validation_values,
        ]
    )

    return BatteryDataset(
        pd.DataFrame(
            {
                "ds": dates,
                "microVolt": values,
            }
        )
    )


# ---------------------------------------------------------
# Standard evaluator contract
# ---------------------------------------------------------


def test_gru_evaluator_returns_evaluation_result(
    evaluation_dataset,
):

    model = build_gru()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    result = evaluator.evaluate(
        dataset=evaluation_dataset,
        model=model,
    )

    assert isinstance(
        result,
        EvaluationResult,
    )

    assert result.model == "GRU"
    assert result.evaluation_steps == 4
    assert result.horizon == 4


def test_gru_evaluator_produces_metrics(
    evaluation_dataset,
):

    model = build_gru()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    result = evaluator.evaluate(
        dataset=evaluation_dataset,
        model=model,
    )

    assert np.isfinite(result.rmse)
    assert np.isfinite(result.mae)
    assert np.isfinite(result.mape)

    assert result.rmse >= 0.0
    assert result.mae >= 0.0
    assert result.mape >= 0.0


# ---------------------------------------------------------
# Holdout contract
# ---------------------------------------------------------


def test_gru_evaluator_preserves_validation_targets(
    evaluation_dataset,
):

    model = build_gru()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    result = evaluator.evaluate(
        dataset=evaluation_dataset,
        model=model,
    )

    assert result.actual == (
        100.0,
        110.0,
        120.0,
        130.0,
    )

    assert len(result.predicted) == 4


def test_gru_evaluator_dates_match_holdout(
    evaluation_dataset,
):

    model = build_gru()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    result = evaluator.evaluate(
        dataset=evaluation_dataset,
        model=model,
    )

    expected_dates = tuple(
        evaluation_dataset.data[
            "ds"
        ].iloc[-4:]
    )

    assert result.dates == expected_dates


# ---------------------------------------------------------
# Leakage regression
# ---------------------------------------------------------


def test_gru_evaluator_scaler_uses_training_only(
    evaluation_dataset,
):

    model = build_gru()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    evaluator.evaluate(
        dataset=evaluation_dataset,
        model=model,
    )

    scaler = model.scaler.scaler

    assert scaler.data_min_[0] == pytest.approx(
        10.0
    )

    assert scaler.data_max_[0] == pytest.approx(
        20.0
    )

    assert scaler.data_max_[0] != pytest.approx(
        130.0
    )


# ---------------------------------------------------------
# Evaluation metadata
# ---------------------------------------------------------


def test_gru_evaluator_metadata(
    evaluation_dataset,
):

    model = build_gru()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    result = evaluator.evaluate(
        dataset=evaluation_dataset,
        model=model,
    )

    assert result.metadata[
        "training_points"
    ] == 20

    assert result.metadata[
        "validation_points"
    ] == 4

    preprocessing = result.metadata[
        "preprocessing"
    ]

    assert preprocessing["enabled"] is True
    assert preprocessing[
        "name"
    ] == "TemporalPreprocessor"
    assert preprocessing[
        "method"
    ] == "linear"
    assert preprocessing[
        "fill_edges"
    ] is True


# ---------------------------------------------------------
# Explicit evaluation endpoint
# ---------------------------------------------------------


def test_gru_evaluator_respects_evaluation_end():

    dates = pd.date_range(
        start="2020-03-01",
        periods=28,
        freq="QS-MAR",
    )

    values = np.linspace(
        10.0,
        30.0,
        len(dates),
    )

    dataset = BatteryDataset(
        pd.DataFrame(
            {
                "ds": dates,
                "microVolt": values,
            }
        )
    )

    evaluation_end = dates[23]

    model = build_gru()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
        evaluation_end=evaluation_end,
    )

    result = evaluator.evaluate(
        dataset=dataset,
        model=model,
    )

    assert result.dates == tuple(
        dates[20:24]
    )

    assert result.metadata[
        "training_points"
    ] == 20

    assert result.metadata[
        "validation_points"
    ] == 4

    assert pd.Timestamp(
        result.metadata["evaluation_end"]
    ) == pd.Timestamp(
        evaluation_end
    )


# ---------------------------------------------------------
# Training missing values
# ---------------------------------------------------------


def test_gru_evaluator_interpolates_training_missing_values():

    dates = pd.date_range(
        start="2020-03-01",
        periods=24,
        freq="QS-MAR",
    )

    values = np.linspace(
        10.0,
        33.0,
        len(dates),
    )

    values[8] = np.nan

    dataset = BatteryDataset(
        pd.DataFrame(
            {
                "ds": dates,
                "microVolt": values,
            }
        )
    )

    model = build_gru()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    result = evaluator.evaluate(
        dataset=dataset,
        model=model,
    )

    assert isinstance(
        result,
        EvaluationResult,
    )

    assert len(model.dataset) == 20
    assert model.dataset.target.isna().sum() == 0

    assert model.dataset.data.loc[
        8,
        "microVolt",
    ] == pytest.approx(
        18.0
    )

    # Original dataset remains unchanged.
    assert pd.isna(
        dataset.data.loc[
            8,
            "microVolt",
        ]
    )


# ---------------------------------------------------------
# Validation missing values
# ---------------------------------------------------------


def test_gru_evaluator_rejects_missing_validation_target():

    dates = pd.date_range(
        start="2020-03-01",
        periods=24,
        freq="QS-MAR",
    )

    values = np.linspace(
        10.0,
        33.0,
        len(dates),
    )

    values[21] = np.nan

    dataset = BatteryDataset(
        pd.DataFrame(
            {
                "ds": dates,
                "microVolt": values,
            }
        )
    )

    model = build_gru()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    with pytest.raises(
        ValueError,
        match="validation",
    ):
        evaluator.evaluate(
            dataset=dataset,
            model=model,
        )


def test_gru_training_preprocessing_does_not_fill_validation_target():

    dates = pd.date_range(
        start="2020-03-01",
        periods=24,
        freq="QS-MAR",
    )

    values = np.linspace(
        10.0,
        33.0,
        len(dates),
    )

    values[8] = np.nan
    values[21] = np.nan

    dataset = BatteryDataset(
        pd.DataFrame(
            {
                "ds": dates,
                "microVolt": values,
            }
        )
    )

    model = build_gru()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    with pytest.raises(
        ValueError,
        match="validation",
    ):
        evaluator.evaluate(
            dataset=dataset,
            model=model,
        )

    assert pd.isna(
        dataset.data.loc[
            8,
            "microVolt",
        ]
    )

    assert pd.isna(
        dataset.data.loc[
            21,
            "microVolt",
        ]
    )