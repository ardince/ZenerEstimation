"""
Tests for KalmanLSTMForecaster.

These tests verify construction and neural residual-model
configuration ownership.

Forecasting and evaluator integration are tested separately.
"""

from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from zenerestimation.forecasting.result import ForecastResult

from zenerestimation.forecasting.hybrid.kalman_lstm import (
    KalmanLSTMForecaster,
)
from zenerestimation.forecasting.neural.lstm import (
    LSTMForecaster,
)

from zenerestimation.data import (
    BatteryDataset,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def build_provenance_dataset():
    """
    Build a small BatteryDataset carrying provenance
    metadata for residual-dataset tests.
    """

    dates = pd.date_range(
        "2020-03-01",
        periods=12,
        freq="QS-MAR",
    )

    data = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": np.linspace(
                10.0,
                21.0,
                12,
            ),
            "is_observed": [
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
        }
    )

    dataset = BatteryDataset(data)

    dataset._battery = "TEST-BATTERY"
    dataset._source_type = "processed"
    dataset._source_path = (
        "processed/TEST-BATTERY.csv"
    )

    dataset.metadata = {
        "frequency": "QS-MAR",
        "test_marker": "residual-provenance",
    }

    return dataset


def prepare_kalman_lstm_combination_state(
    model,
):
    """
    Provide the minimal fitted state required by the
    current KalmanLSTM combination implementation.
    """

    dates = pd.date_range(
        "2020-03-01",
        periods=4,
        freq="QS-MAR",
    )

    data = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": [
                10.0,
                11.0,
                12.0,
                13.0,
            ],
        }
    )

    model.dataset = BatteryDataset(data)

    model._trend = np.asarray(
        [9.5, 10.5, 11.5, 12.5],
        dtype=float,
    )

    model._residual = np.asarray(
        [0.5, 0.5, 0.5, 0.5],
        dtype=float,
    )

def build_forecast_result(
    model,
    dates,
    values,
):
    """
    Build a minimal ForecastResult for hybrid
    component-combination tests.
    """


    dates = pd.DatetimeIndex(dates)

    forecast = pd.Series(
        np.asarray(values, dtype=float),
        index=dates,
    )

    return ForecastResult(
        model=model,
        horizon=len(values),
        dates=dates,
        forecast=forecast,
    )


def build_lstm(
    window=6,
):

    return LSTMForecaster(
        window=window,
        units=8,
        epochs=1,
        batch_size=4,
        seed=42,
    )


# ---------------------------------------------------------
# Default construction
# ---------------------------------------------------------


def test_kalman_lstm_construction():

    model = KalmanLSTMForecaster()

    assert model.filter is not None
    assert model.lstm is not None
    assert model.residual_model is model.lstm


# ---------------------------------------------------------
# Default LSTM window
# ---------------------------------------------------------


def test_kalman_lstm_default_window_configures_lstm():

    model = KalmanLSTMForecaster(
        window=12,
    )

    assert model.window == 12
    assert model.lstm.window == 12


def test_kalman_lstm_summary_uses_lstm_window():

    model = KalmanLSTMForecaster(
        window=10,
    )

    summary = model.summary()

    assert summary["window"] == 10
    assert summary["window"] == model.lstm.window


# ---------------------------------------------------------
# Custom LSTM ownership
# ---------------------------------------------------------


def test_kalman_lstm_accepts_custom_lstm():

    lstm = build_lstm(
        window=10,
    )

    model = KalmanLSTMForecaster(
        lstm_model=lstm,
    )

    assert model.lstm is lstm
    assert model.residual_model is lstm
    assert model.window == 10


def test_kalman_lstm_custom_lstm_owns_window():

    lstm = build_lstm(
        window=10,
    )

    model = KalmanLSTMForecaster(
        window=6,
        lstm_model=lstm,
    )

    # The explicit window argument is only used when
    # constructing the default LSTM. A supplied residual
    # model owns its own neural configuration.
    assert model.window == 10
    assert model.lstm.window == 10
    assert model.lstm is lstm


def test_kalman_lstm_summary_uses_custom_lstm_window():

    lstm = build_lstm(
        window=12,
    )

    model = KalmanLSTMForecaster(
        window=6,
        lstm_model=lstm,
    )

    summary = model.summary()

    assert summary["window"] == 12
    assert summary["window"] == lstm.window

    assert summary["window"] != 6


def test_kalman_lstm_combines_matching_components():

    model = KalmanLSTMForecaster()


    prepare_kalman_lstm_combination_state(
        model
    )

    dates = pd.date_range(
        "2026-03-01",
        periods=3,
        freq="QS-MAR",
    )

    trend = build_forecast_result(
        model="AdaptiveKalmanFilter",
        dates=dates,
        values=[10.0, 11.0, 12.0],
    )

    residual = build_forecast_result(
        model="LSTM",
        dates=dates,
        values=[0.5, -0.2, 0.3],
    )

    combined = model.combine_forecasts(
        trend,
        residual,
    )

    np.testing.assert_allclose(
        combined.forecast.to_numpy(),
        [10.5, 10.8, 12.3],
    )

    assert combined.forecast.index.equals(
        dates
    )

    assert combined.fitted is None


def test_kalman_lstm_rejects_horizon_mismatch():

    model = KalmanLSTMForecaster()

    trend_dates = pd.date_range(
        "2026-03-01",
        periods=3,
        freq="QS-MAR",
    )

    residual_dates = pd.date_range(
        "2026-03-01",
        periods=2,
        freq="QS-MAR",
    )

    trend = build_forecast_result(
        model="AdaptiveKalmanFilter",
        dates=trend_dates,
        values=[10.0, 11.0, 12.0],
    )

    residual = build_forecast_result(
        model="LSTM",
        dates=residual_dates,
        values=[0.5, -0.2],
    )

    with pytest.raises(
        ValueError,
        match="horizon",
    ):
        model.combine_forecasts(
            trend,
            residual,
        )


def test_kalman_lstm_rejects_date_mismatch():

    model = KalmanLSTMForecaster()

    trend_dates = pd.date_range(
        "2026-03-01",
        periods=3,
        freq="QS-MAR",
    )

    residual_dates = pd.date_range(
        "2026-06-01",
        periods=3,
        freq="QS-MAR",
    )

    trend = build_forecast_result(
        model="AdaptiveKalmanFilter",
        dates=trend_dates,
        values=[10.0, 11.0, 12.0],
    )

    residual = build_forecast_result(
        model="LSTM",
        dates=residual_dates,
        values=[0.5, -0.2, 0.3],
    )

    with pytest.raises(
        ValueError,
        match="dates",
    ):
        model.combine_forecasts(
            trend,
            residual,
        )


def test_kalman_lstm_rejects_single_date_mismatch():

    model = KalmanLSTMForecaster()

    trend_dates = pd.DatetimeIndex([
        "2026-03-01",
        "2026-06-01",
        "2026-09-01",
    ])

    residual_dates = pd.DatetimeIndex([
        "2026-03-01",
        "2026-06-01",
        "2026-12-01",
    ])

    trend = build_forecast_result(
        model="AdaptiveKalmanFilter",
        dates=trend_dates,
        values=[10.0, 11.0, 12.0],
    )

    residual = build_forecast_result(
        model="LSTM",
        dates=residual_dates,
        values=[0.5, -0.2, 0.3],
    )

    with pytest.raises(
        ValueError,
        match="dates",
    ):
        model.combine_forecasts(
            trend,
            residual,
        )


def test_kalman_lstm_residual_preserves_dates():

    dataset = build_provenance_dataset()

    model = KalmanLSTMForecaster()

    residual = model.prepare_residuals(
        dataset
    )

    pd.testing.assert_series_equal(
        residual.data["ds"],
        dataset.data["ds"],
        check_names=True,
    )


def test_kalman_lstm_residual_preserves_battery():

    dataset = build_provenance_dataset()

    model = KalmanLSTMForecaster()

    residual = model.prepare_residuals(
        dataset
    )

    assert residual.battery == dataset.battery
    assert residual.battery == "TEST-BATTERY"


def test_kalman_lstm_residual_preserves_source():

    dataset = build_provenance_dataset()

    model = KalmanLSTMForecaster()

    residual = model.prepare_residuals(
        dataset
    )

    assert (
        residual.source_type
        == dataset.source_type
    )

    assert (
        residual.source_path
        == dataset.source_path
    )

    assert residual.source_type == "processed"

    assert (
        residual.source_path
        == "processed/TEST-BATTERY.csv"
    )


def test_kalman_lstm_residual_preserves_observed_mask():

    dataset = build_provenance_dataset()

    model = KalmanLSTMForecaster()

    residual = model.prepare_residuals(
        dataset
    )

    pd.testing.assert_series_equal(
        residual.data["is_observed"],
        dataset.data["is_observed"],
        check_names=True,
    )


def test_kalman_lstm_residual_copies_metadata():

    dataset = build_provenance_dataset()

    model = KalmanLSTMForecaster()

    residual = model.prepare_residuals(
        dataset
    )

    assert residual.metadata == dataset.metadata

    assert (
        residual.metadata
        is not dataset.metadata
    )


# ---------------------------------------------------------
# Historical fitted-value contract
# ---------------------------------------------------------


def test_kalman_lstm_result_contains_fitted_values():

    dataset = build_provenance_dataset()

    model = KalmanLSTMForecaster(
        lstm_model=build_lstm(
            window=4,
        ),
    )

    model.fit(dataset)

    result = model.predict(
        steps=3,
    )

    assert result.fitted is not None


def test_kalman_lstm_fitted_matches_training_length():

    dataset = build_provenance_dataset()

    model = KalmanLSTMForecaster(
        lstm_model=build_lstm(
            window=4,
        ),
    )

    model.fit(dataset)

    result = model.predict(
        steps=3,
    )

    assert len(result.fitted) == len(
        dataset
    )


def test_kalman_lstm_fitted_uses_training_dates():

    dataset = build_provenance_dataset()

    model = KalmanLSTMForecaster(
        lstm_model=build_lstm(
            window=4,
        ),
    )

    model.fit(dataset)

    result = model.predict(
        steps=3,
    )

    expected_dates = pd.DatetimeIndex(
        dataset.data["ds"]
    )

    pd.testing.assert_index_equal(
        result.fitted.index,
        expected_dates,
    )


def test_kalman_lstm_fitted_preserves_neural_warmup():

    window = 4

    dataset = build_provenance_dataset()

    model = KalmanLSTMForecaster(
        lstm_model=build_lstm(
            window=window,
        ),
    )

    model.fit(dataset)

    result = model.predict(
        steps=3,
    )

    assert result.fitted.iloc[
        :window
    ].isna().all()

    assert result.fitted.iloc[
        window:
    ].notna().all()