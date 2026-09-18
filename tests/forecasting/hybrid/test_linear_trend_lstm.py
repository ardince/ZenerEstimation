"""
Tests for LinearTrendLSTMForecaster.

These tests verify construction and neural residual-model
configuration ownership.

Forecasting and evaluator integration are tested separately.
"""

from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from zenerestimation.forecasting.result import ForecastResult

from zenerestimation.forecasting.hybrid.linear_trend_lstm import (
    LinearTrendLSTMForecaster,
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
# Construction
# ---------------------------------------------------------


def test_linear_trend_lstm_construction():

    lstm = build_lstm(
        window=6,
    )

    model = LinearTrendLSTMForecaster(
        lstm_model=lstm,
    )

    assert model.lstm is lstm
    assert model.residual_model is lstm


# ---------------------------------------------------------
# Window ownership
# ---------------------------------------------------------


def test_linear_trend_lstm_window_comes_from_lstm():

    lstm = build_lstm(
        window=12,
    )

    model = LinearTrendLSTMForecaster(
        lstm_model=lstm,
    )

    assert model.window == 12
    assert model.lstm.window == 12


def test_linear_trend_lstm_summary_uses_lstm_window():

    lstm = build_lstm(
        window=10,
    )

    model = LinearTrendLSTMForecaster(
        lstm_model=lstm,
    )

    summary = model.summary()

    assert summary["window"] == 10
    assert summary["window"] == model.lstm.window


# ---------------------------------------------------------
# Backward-compatible window argument
# ---------------------------------------------------------


def test_linear_trend_lstm_accepts_matching_window():

    lstm = build_lstm(
        window=8,
    )

    model = LinearTrendLSTMForecaster(
        lstm_model=lstm,
        window=8,
    )

    assert model.window == 8
    assert model.lstm.window == 8


def test_linear_trend_lstm_rejects_conflicting_window():

    lstm = build_lstm(
        window=12,
    )

    with pytest.raises(
        ValueError,
        match="window must match",
    ):
        LinearTrendLSTMForecaster(
            lstm_model=lstm,
            window=6,
        )


def test_linear_trend_lstm_window_is_optional_for_generic_residual():
    """
    Generic residual models without neural window
    metadata remain supported.
    """

    class GenericResidualModel:

        def fit(
            self,
            dataset,
        ):
            return self

        def predict(
            self,
            steps=1,
        ):
            return None

    residual = GenericResidualModel()

    model = LinearTrendLSTMForecaster(
        lstm_model=residual,
    )

    assert model.window is None

    summary = model.summary()

    assert "window" not in summary


def test_linear_trend_lstm_combines_matching_components():

    lstm = build_lstm()

    model = LinearTrendLSTMForecaster(
        lstm_model=lstm,
    )

    dates = pd.date_range(
        "2026-03-01",
        periods=3,
        freq="QS-MAR",
    )

    trend = build_forecast_result(
        model="LinearTrend",
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

    assert combined.fitted is None

    np.testing.assert_allclose(
        combined.forecast.to_numpy(),
        [10.5, 10.8, 12.3],
    )

    assert combined.forecast.index.equals(
        dates
    )


def test_linear_trend_lstm_rejects_horizon_mismatch():

    lstm = build_lstm()

    model = LinearTrendLSTMForecaster(
        lstm_model=lstm,
    )

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
        model="LinearTrend",
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


def test_linear_trend_lstm_rejects_date_mismatch():

    lstm = build_lstm()

    model = LinearTrendLSTMForecaster(
        lstm_model=lstm,
    )

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
        model="LinearTrend",
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


def test_linear_trend_lstm_rejects_single_date_mismatch():

    lstm = build_lstm()

    model = LinearTrendLSTMForecaster(
        lstm_model=lstm,
    )

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
        model="LinearTrend",
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
    dataset._source_path = "processed/TEST-BATTERY.csv"

    dataset.metadata = {
        "frequency": "QS-MAR",
        "test_marker": "residual-provenance",
    }

    return dataset


def test_linear_trend_lstm_residual_preserves_dates():

    dataset = build_provenance_dataset()

    model = LinearTrendLSTMForecaster(
        lstm_model=build_lstm(),
    )

    residual = model.prepare_residuals(
        dataset
    )

    pd.testing.assert_series_equal(
        residual.data["ds"],
        dataset.data["ds"],
        check_names=True,
    )


def test_linear_trend_lstm_residual_preserves_battery():

    dataset = build_provenance_dataset()

    model = LinearTrendLSTMForecaster(
        lstm_model=build_lstm(),
    )

    residual = model.prepare_residuals(
        dataset
    )

    assert residual.battery == dataset.battery
    assert residual.battery == "TEST-BATTERY"


def test_linear_trend_lstm_residual_preserves_source():

    dataset = build_provenance_dataset()

    model = LinearTrendLSTMForecaster(
        lstm_model=build_lstm(),
    )

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


def test_linear_trend_lstm_residual_preserves_observed_mask():

    dataset = build_provenance_dataset()

    model = LinearTrendLSTMForecaster(
        lstm_model=build_lstm(),
    )

    residual = model.prepare_residuals(
        dataset
    )

    pd.testing.assert_series_equal(
        residual.data["is_observed"],
        dataset.data["is_observed"],
        check_names=True,
    )


def test_linear_trend_lstm_residual_copies_metadata():

    dataset = build_provenance_dataset()

    model = LinearTrendLSTMForecaster(
        lstm_model=build_lstm(),
    )

    residual = model.prepare_residuals(
        dataset
    )

    assert residual.metadata == dataset.metadata

    assert (
        residual.metadata
        is not dataset.metadata
    )


def test_linear_trend_lstm_residual_preserves_source():

    dataset = build_provenance_dataset()

    model = LinearTrendLSTMForecaster(
        lstm_model=build_lstm(),
    )

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


# ---------------------------------------------------------
# Historical fitted values
# ---------------------------------------------------------

def test_linear_trend_lstm_result_contains_fitted_values():

    dataset = build_provenance_dataset()

    lstm = build_lstm(
        window=4,
    )

    model = LinearTrendLSTMForecaster(
        lstm_model=lstm,
    )

    model.fit(dataset)

    result = model.predict(
        steps=3,
    )

    assert result.fitted is not None


def test_linear_trend_lstm_fitted_matches_training_length():

    dataset = build_provenance_dataset()

    lstm = build_lstm(
        window=4,
    )

    model = LinearTrendLSTMForecaster(
        lstm_model=lstm,
    )

    model.fit(dataset)

    result = model.predict(
        steps=3,
    )

    assert len(result.fitted) == len(dataset)


def test_linear_trend_lstm_fitted_uses_training_dates():

    dataset = build_provenance_dataset()

    lstm = build_lstm(
        window=4,
    )

    model = LinearTrendLSTMForecaster(
        lstm_model=lstm,
    )

    model.fit(dataset)

    result = model.predict(
        steps=3,
    )

    expected_dates = pd.DatetimeIndex(
        dataset.data["ds"]
    )

    fitted_dates = pd.DatetimeIndex(
        result.fitted.index
    )

    result.fitted.index == dataset.data["ds"]

    assert fitted_dates.equals(
        expected_dates
    )


def test_linear_trend_lstm_fitted_preserves_neural_warmup():

    dataset = build_provenance_dataset()

    window = 4

    lstm = build_lstm(
        window=window,
    )

    model = LinearTrendLSTMForecaster(
        lstm_model=lstm,
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


# ---------------------------------------------------------
# Diagnostics decomposition contract
# ---------------------------------------------------------


def test_linear_trend_lstm_exposes_diagnostic_trend():

    dataset = build_provenance_dataset()

    model = LinearTrendLSTMForecaster(
        lstm_model=build_lstm(
            window=4,
        ),
    )

    model.fit(dataset)

    assert hasattr(
        model,
        "_trend",
    )

    assert model._trend is not None

    assert len(model._trend) == len(dataset)


def test_linear_trend_lstm_exposes_diagnostic_residual():

    dataset = build_provenance_dataset()

    model = LinearTrendLSTMForecaster(
        lstm_model=build_lstm(
            window=4,
        ),
    )

    model.fit(dataset)

    assert hasattr(
        model,
        "_residual",
    )

    assert model._residual is not None

    assert len(model._residual) == len(dataset)


def test_linear_trend_lstm_diagnostic_decomposition_reconstructs_signal():

    dataset = build_provenance_dataset()

    model = LinearTrendLSTMForecaster(
        lstm_model=build_lstm(
            window=4,
        ),
    )

    model.fit(dataset)

    reconstructed = (
        model._trend
        +
        model._residual
    )

    np.testing.assert_allclose(
        reconstructed,
        dataset.target.to_numpy(
            dtype=float,
        ),
        rtol=0.0,
        atol=1e-10,
    )


def test_linear_trend_lstm_supports_hybrid_diagnostics():

    dataset = build_provenance_dataset()

    model = LinearTrendLSTMForecaster(
        lstm_model=build_lstm(
            window=4,
        ),
    )

    model.fit(dataset)

    diagnostics = model.diagnostics()

    assert diagnostics.verify_decomposition()

    result = diagnostics.result()

    assert result.quality_score is not None
    assert result.quality_grade is not None

    assert isinstance(
        result.recommendations,
        list,
    )