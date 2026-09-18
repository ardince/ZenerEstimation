"""
Tests for the baseline GRU forecaster.

These tests verify the public forecasting contract
and Sprint 13 forecast-date standardization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data import BatteryDataset
from zenerestimation.forecasting import ForecastResult
from zenerestimation.forecasting.neural.gru import (
    GRUForecaster,
)


# ---------------------------------------------------------
# Test dataset
# ---------------------------------------------------------


@pytest.fixture
def quarterly_dataset():
    """
    Create a small quarterly dataset suitable for
    neural forecasting tests.
    """

    dates = pd.date_range(
        start="2022-03-01",
        periods=16,
        freq="QS-MAR",
    )

    values = np.linspace(
        20.0,
        23.0,
        len(dates),
    )

    data = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": values,
        }
    )

    return BatteryDataset(data)


# ---------------------------------------------------------
# Construction
# ---------------------------------------------------------


def test_gru_construction():

    model = GRUForecaster(
        window=4,
        units=8,
        epochs=1,
        batch_size=4,
        seed=42,
    )

    assert model.window == 4
    assert model.units == 8
    assert model.epochs == 1
    assert model.batch_size == 4
    assert model.seed == 42


# ---------------------------------------------------------
# Prediction before fit
# ---------------------------------------------------------


def test_gru_predict_before_fit_raises():

    model = GRUForecaster(
        window=4,
        units=8,
        epochs=1,
    )

    with pytest.raises(
        RuntimeError,
        match="Model has not been trained",
    ):
        model.predict(
            steps=2
        )


# ---------------------------------------------------------
# Fit contract
# ---------------------------------------------------------


def test_gru_fit_returns_self(
    quarterly_dataset,
):

    model = GRUForecaster(
        window=4,
        units=8,
        epochs=1,
        batch_size=4,
        seed=42,
    )

    fitted_model = model.fit(
        quarterly_dataset
    )

    assert fitted_model is model

    assert model.model is not None
    assert model.history is not None
    assert model.fitted is not None


# ---------------------------------------------------------
# ForecastResult contract
# ---------------------------------------------------------


def test_gru_predict_returns_forecast_result(
    quarterly_dataset,
):

    model = GRUForecaster(
        window=4,
        units=8,
        epochs=1,
        batch_size=4,
        seed=42,
    )

    model.fit(
        quarterly_dataset
    )

    result = model.predict(
        steps=3
    )

    assert isinstance(
        result,
        ForecastResult,
    )

    assert result.model == "GRU"
    assert result.horizon == 3
    assert len(result.forecast) == 3
    assert len(result.dates) == 3


# ---------------------------------------------------------
# Central forecast-date contract
# ---------------------------------------------------------


def test_gru_uses_dataset_forecast_dates(
    quarterly_dataset,
):

    model = GRUForecaster(
        window=4,
        units=8,
        epochs=1,
        batch_size=4,
        seed=42,
    )

    model.fit(
        quarterly_dataset
    )

    steps = 3

    expected_dates = (
        quarterly_dataset.forecast_dates(
            steps
        )
    )

    result = model.predict(
        steps=steps
    )

    assert list(result.dates) == list(
        expected_dates
    )


# ---------------------------------------------------------
# Forecast Series index contract
# ---------------------------------------------------------


def test_gru_forecast_index_matches_dates(
    quarterly_dataset,
):

    model = GRUForecaster(
        window=4,
        units=8,
        epochs=1,
        batch_size=4,
        seed=42,
    )

    model.fit(
        quarterly_dataset
    )

    result = model.predict(
        steps=3
    )

    assert list(
        result.forecast.index
    ) == list(
        result.dates
    )


# ---------------------------------------------------------
# Quarterly date regression
# ---------------------------------------------------------


def test_gru_quarterly_forecast_dates(
    quarterly_dataset,
):

    model = GRUForecaster(
        window=4,
        units=8,
        epochs=1,
        batch_size=4,
        seed=42,
    )

    model.fit(
        quarterly_dataset
    )

    result = model.predict(
        steps=3
    )

    expected = pd.DatetimeIndex(
        [
            "2026-03-01",
            "2026-06-01",
            "2026-09-01",
        ]
    )

    assert list(result.dates) == list(
        expected
    )

    assert list(
        result.forecast.index
    ) == list(
        expected
    )


# ---------------------------------------------------------
# Fitted-value alignment
# ---------------------------------------------------------


def test_gru_fitted_values_align_with_dataset(
    quarterly_dataset,
):

    window = 4

    model = GRUForecaster(
        window=window,
        units=8,
        epochs=1,
        batch_size=4,
        seed=42,
    )

    model.fit(
        quarterly_dataset
    )

    assert len(model.fitted) == len(
        quarterly_dataset
    )

    assert model.fitted.iloc[
        :window
    ].isna().all()

    assert model.fitted.iloc[
        window:
    ].notna().all()


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------


def test_gru_summary_contains_configuration():

    model = GRUForecaster(
        window=4,
        units=8,
        epochs=2,
        batch_size=4,
        seed=123,
    )

    summary = model.summary()

    assert summary["window"] == 4
    assert summary["units"] == 8
    assert summary["epochs"] == 2
    assert summary["batch_size"] == 4
    assert summary["seed"] == 123