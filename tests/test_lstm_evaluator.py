"""
Integration tests for LSTM evaluation.

These tests verify that LSTMForecaster operates through
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
from zenerestimation.forecasting.neural.lstm import (
    LSTMForecaster,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def build_lstm():
    """
    Small LSTM suitable for integration tests.
    """

    return LSTMForecaster(
        window=4,
        units=8,
        epochs=1,
        batch_size=4,
        seed=42,
    )


@pytest.fixture
def evaluation_dataset():
    """
    Quarterly dataset with a deliberately different
    validation range.

    Training:
        20 observations from approximately 10 to 20.

    Validation:
        4 observations from 100 to 130.

    The large validation values make scaler leakage
    directly detectable.
    """

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

    data = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": values,
        }
    )

    return BatteryDataset(data)


# ---------------------------------------------------------
# Basic evaluator integration
# ---------------------------------------------------------


def test_lstm_evaluator_returns_evaluation_result(
    evaluation_dataset,
):

    model = build_lstm()

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

    assert result.model == "LSTM"

    assert result.evaluation_steps == 4
    assert result.horizon == 4


# ---------------------------------------------------------
# Standard metric contract
# ---------------------------------------------------------


def test_lstm_evaluator_produces_metrics(
    evaluation_dataset,
):

    model = build_lstm()

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
# Holdout values
# ---------------------------------------------------------


def test_lstm_evaluator_preserves_validation_targets(
    evaluation_dataset,
):

    model = build_lstm()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    result = evaluator.evaluate(
        dataset=evaluation_dataset,
        model=model,
    )

    expected = (
        100.0,
        110.0,
        120.0,
        130.0,
    )

    assert result.actual == expected

    assert len(result.predicted) == 4


# ---------------------------------------------------------
# Forecast dates = validation dates
# ---------------------------------------------------------


def test_lstm_evaluator_dates_match_holdout(
    evaluation_dataset,
):

    model = build_lstm()

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
# Critical leakage regression
# ---------------------------------------------------------


def test_lstm_evaluator_scaler_uses_training_only(
    evaluation_dataset,
):
    """
    Validation targets range from 100 to 130, while the
    training data ends at 20.

    If the holdout leaks into neural scaling, the fitted
    scaler maximum would become 130.

    Correct behaviour requires a maximum of 20.
    """

    model = build_lstm()

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


def test_lstm_evaluator_metadata(
    evaluation_dataset,
):

    model = build_lstm()

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


def test_lstm_evaluator_respects_evaluation_end():

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

    model = build_lstm()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
        evaluation_end=evaluation_end,
    )

    result = evaluator.evaluate(
        dataset=dataset,
        model=model,
    )

    expected_dates = tuple(
        dates[20:24]
    )

    assert result.dates == expected_dates

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
# Missing training values
# ---------------------------------------------------------


def test_lstm_evaluator_interpolates_training_missing_values():
    """
    Missing values inside the training partition may be
    repaired by TemporalPreprocessor.

    The validation partition must remain untouched.
    """

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

    # Missing value is safely inside the training partition.
    values[8] = np.nan

    dataset = BatteryDataset(
        pd.DataFrame(
            {
                "ds": dates,
                "microVolt": values,
            }
        )
    )

    model = build_lstm()

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

    assert result.evaluation_steps == 4

    # The model receives the preprocessed training
    # partition, which must contain no missing targets.
    assert model.dataset.target.isna().sum() == 0

    assert len(model.dataset) == 20

    # Original source dataset must not be mutated.
    assert pd.isna(
        dataset.data.loc[
            8,
            "microVolt",
        ]
    )

    assert model.dataset.data.loc[
        8,
        "microVolt",] == pytest.approx(18.0)


# ---------------------------------------------------------
# Missing validation targets
# ---------------------------------------------------------


def test_lstm_evaluator_rejects_missing_validation_target():
    """
    Missing targets in the holdout partition must never
    be interpolated, filled, or silently removed.
    """

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

    # Last four observations form the validation
    # partition. Introduce a missing validation target.
    values[21] = np.nan

    dataset = BatteryDataset(
        pd.DataFrame(
            {
                "ds": dates,
                "microVolt": values,
            }
        )
    )

    model = build_lstm()

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


# ---------------------------------------------------------
# Training repair must not hide validation failure
# ---------------------------------------------------------


def test_lstm_training_preprocessing_does_not_fill_validation_target():
    """
    A repairable training NaN must not cause preprocessing
    to spill across the holdout boundary and repair a
    validation NaN.
    """

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

    # Training partition
    values[8] = np.nan

    # Validation partition
    values[21] = np.nan

    dataset = BatteryDataset(
        pd.DataFrame(
            {
                "ds": dates,
                "microVolt": values,
            }
        )
    )

    model = build_lstm()

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

    # Source data remains unchanged.
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