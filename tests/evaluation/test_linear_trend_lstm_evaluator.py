import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.temporal import TemporalPreprocessor
from zenerestimation.evaluation.evaluator import ForecastEvaluator
from zenerestimation.evaluation.result import EvaluationResult
from zenerestimation.forecasting.neural.lstm import LSTMForecaster
from zenerestimation.forecasting.hybrid.linear_trend_lstm import (
    LinearTrendLSTMForecaster,
)

def build_dataset(
    *,
    training_nan=False,
    validation_nan=False,
):
    """
    Build 24 quarterly observations.

    First 20 points form the training region.
    Last 4 points are deliberately far outside the
    training range so leakage is easy to detect.
    """

    dates = pd.date_range(
        "2020-03-01",
        periods=24,
        freq="QS-MAR",
    )

    training = np.linspace(
        10.0,
        20.0,
        20,
    )

    validation = np.asarray(
        [100.0, 110.0, 120.0, 130.0],
        dtype=float,
    )

    values = np.concatenate(
        [training, validation]
    )

    if training_nan:
        values[8] = np.nan

    if validation_nan:
        values[21] = np.nan

    data = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": values,
        }
    )

    dataset = BatteryDataset(data)
    dataset._battery = "TEST-BATTERY"

    return dataset


def build_model():
    lstm = LSTMForecaster(
        window=4,
        units=8,
        epochs=1,
        batch_size=4,
        seed=42,
    )

    return LinearTrendLSTMForecaster(
        lstm_model=lstm,
    )


def build_evaluator(**kwargs):
    return ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
        **kwargs,
    )


def test_linear_trend_lstm_evaluator_returns_result():

    dataset = build_dataset()
    model = build_model()
    evaluator = build_evaluator()

    result = evaluator.evaluate(
        dataset=dataset,
        model=model,
    )

    assert isinstance(
        result,
        EvaluationResult,
    )


def test_linear_trend_lstm_evaluator_metrics_are_finite():

    result = build_evaluator().evaluate(
        dataset=build_dataset(),
        model=build_model(),
    )

    assert np.isfinite(result.rmse)
    assert np.isfinite(result.mae)
    assert np.isfinite(result.mape)


def test_linear_trend_lstm_evaluator_preserves_validation_targets():

    result = build_evaluator().evaluate(
        dataset=build_dataset(),
        model=build_model(),
    )

    np.testing.assert_allclose(
        result.actual,
        [100.0, 110.0, 120.0, 130.0],
    )


def test_linear_trend_lstm_evaluator_uses_holdout_dates():

    dataset = build_dataset()

    result = build_evaluator().evaluate(
        dataset=dataset,
        model=build_model(),
    )

    expected_dates = pd.DatetimeIndex(
        dataset.data["ds"].iloc[-4:]
    )

    assert pd.DatetimeIndex(
        result.dates
    ).equals(expected_dates)


def test_linear_trend_lstm_residual_scaler_is_training_only():

    dataset = build_dataset()
    model = build_model()

    build_evaluator().evaluate(
        dataset=dataset,
        model=model,
    )

    scaler = model.lstm.scaler.scaler

    assert scaler.data_max_[0] < 100.0


def test_linear_trend_lstm_receives_training_partition_only():

    dataset = build_dataset()
    model = build_model()

    build_evaluator().evaluate(
        dataset=dataset,
        model=model,
    )

    assert len(model.dataset.data) == 20

    assert (
        model.dataset.data["microVolt"].max()
        <= 20.0
    )


def test_linear_trend_lstm_evaluator_metadata():

    result = build_evaluator().evaluate(
        dataset=build_dataset(),
        model=build_model(),
    )

    assert result.metadata[
        "training_points"
    ] == 20

    assert result.metadata[
        "validation_points"
    ] == 4

    assert result.metadata[
        "preprocessing"
    ]["enabled"] is True

    assert result.metadata[
        "preprocessing"
    ]["name"] == "TemporalPreprocessor"


def test_linear_trend_lstm_evaluator_respects_evaluation_end():

    dataset = build_dataset()

    evaluation_end = dataset.data[
        "ds"
    ].iloc[-2]

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
        evaluation_end=evaluation_end,
    )

    result = evaluator.evaluate(
        dataset=dataset,
        model=build_model(),
    )

    assert pd.Timestamp(
        result.metadata["evaluation_end"]
    ) == pd.Timestamp(evaluation_end)


def test_linear_trend_lstm_training_missing_value_is_preprocessed():

    dataset = build_dataset(
        training_nan=True,
    )

    model = build_model()

    build_evaluator().evaluate(
        dataset=dataset,
        model=model,
    )

    assert (
        model.dataset.data["microVolt"].isna().sum()
        == 0
    )


def test_linear_trend_lstm_validation_missing_value_is_rejected():

    dataset = build_dataset(
        validation_nan=True,
    )

    with pytest.raises(
        ValueError,
        match="validation",
    ):
        build_evaluator().evaluate(
            dataset=dataset,
            model=build_model(),
        )


def test_linear_trend_lstm_preprocessing_does_not_fill_validation():

    dataset = build_dataset(
        validation_nan=True,
    )

    with pytest.raises(
        ValueError,
        match="validation",
    ):
        build_evaluator().evaluate(
            dataset=dataset,
            model=build_model(),
        )

    assert pd.isna(
        dataset.data[
            "microVolt"
        ].iloc[21]
    )