"""Section D.1 legacy result compatibility and strictness tests."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from zenerestimation.evaluation.forecast_validation import validate_forecast_alignment

@pytest.fixture
def dates():
    return pd.DatetimeIndex(["2024-03-01", "2024-06-01", "2024-09-01"])

@pytest.mark.parametrize("attributes", [{}, {"horizon": 3}, {"dates": "valid"}])
def test_legacy_missing_optional_fields(dates, attributes):
    attributes = {k: dates if v == "valid" else v for k, v in attributes.items()}
    result = SimpleNamespace(forecast=pd.Series([1., 2., 3.]), **attributes)
    np.testing.assert_array_equal(validate_forecast_alignment(result, dates), [1, 2, 3])

@pytest.mark.parametrize("bad", [None, True, 2, 3.0, "3"])
def test_explicit_invalid_horizon_rejected(dates, bad):
    with pytest.raises(ValueError, match="forecast horizon"):
        validate_forecast_alignment(SimpleNamespace(forecast=[1, 2, 3], horizon=bad), dates)

@pytest.mark.parametrize("shift", [-3, 3])
def test_explicit_shifted_dates_rejected(dates, shift):
    with pytest.raises(ValueError, match="forecast dates do not match"):
        validate_forecast_alignment(SimpleNamespace(forecast=[1, 2, 3], dates=dates + pd.DateOffset(months=shift)), dates)

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_legacy_nonfinite_rejected(dates, bad):
    with pytest.raises(ValueError, match="nonfinite"):
        validate_forecast_alignment(SimpleNamespace(forecast=[1, bad, 3]), dates)

def test_legacy_length_rejected(dates):
    with pytest.raises(ValueError, match="forecast length"):
        validate_forecast_alignment(SimpleNamespace(forecast=[1, 2]), dates)

def test_explicit_missing_dates_rejected(dates):
    with pytest.raises((ValueError, TypeError)):
        validate_forecast_alignment(SimpleNamespace(forecast=[1, 2, 3], dates=None), dates)

def test_legacy_multidimensional_rejected(dates):
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_forecast_alignment(SimpleNamespace(forecast=np.ones((3, 1))), dates)
