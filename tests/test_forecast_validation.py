"""Unit tests for the shared forecast alignment validator."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from zenerestimation.evaluation.forecast_validation import validate_forecast_alignment


@pytest.fixture
def dates():
    return pd.DatetimeIndex(["2024-03-01", "2024-06-01", "2024-09-01"])


def result(dates, values=(29.9, 30.5, 30.7), horizon=3):
    return SimpleNamespace(
        forecast=pd.Series(values), dates=pd.DatetimeIndex(dates), horizon=horizon
    )


def test_valid_forecast(dates):
    actual = validate_forecast_alignment(result(dates), dates)
    np.testing.assert_allclose(actual, [29.9, 30.5, 30.7])


@pytest.mark.parametrize("shift", [-3, 3])
def test_shifted_quarter_fails(dates, shift):
    shifted = dates + pd.DateOffset(months=shift)
    with pytest.raises(ValueError, match="forecast dates do not match"):
        validate_forecast_alignment(result(shifted), dates)


def test_forecast_length_mismatch(dates):
    with pytest.raises(ValueError, match="forecast length"):
        validate_forecast_alignment(result(dates, values=[1, 2]), dates)


def test_horizon_mismatch(dates):
    with pytest.raises(ValueError, match="forecast horizon"):
        validate_forecast_alignment(result(dates, horizon=2), dates)


def test_date_count_mismatch(dates):
    with pytest.raises(ValueError, match="forecast date count"):
        validate_forecast_alignment(result(dates[:2]), dates)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_forecast_fails(dates, bad):
    with pytest.raises(ValueError, match="nonfinite"):
        validate_forecast_alignment(result(dates, values=[1, bad, 3]), dates)


def test_nondefault_series_index_is_positional(dates):
    forecast = result(dates)
    forecast.forecast.index = [10, 20, 30]
    np.testing.assert_allclose(
        validate_forecast_alignment(forecast, dates), [29.9, 30.5, 30.7]
    )


@pytest.mark.parametrize("which", ["forecast", "validation"])
def test_missing_timestamp_fails(dates, which):
    bad_dates = pd.DatetimeIndex([dates[0], pd.NaT, dates[2]])
    forecast_dates = bad_dates if which == "forecast" else dates
    validation_dates = bad_dates if which == "validation" else dates
    with pytest.raises(ValueError, match="missing timestamps"):
        validate_forecast_alignment(result(forecast_dates), validation_dates)


def test_multidimensional_forecast_fails(dates):
    forecast = result(dates)
    forecast.forecast = np.array([[1], [2], [3]])
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_forecast_alignment(forecast, dates)


def test_empty_validation_dates_fails():
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_forecast_alignment(result([], values=[], horizon=0), [])


def test_duplicate_validation_dates_fails(dates):
    duplicates = pd.DatetimeIndex([dates[0], dates[0], dates[2]])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_forecast_alignment(result(duplicates), duplicates)


def test_boolean_horizon_fails(dates):
    with pytest.raises(ValueError, match="forecast horizon"):
        validate_forecast_alignment(result(dates, horizon=True), dates)
