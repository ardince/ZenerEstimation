"""Fast, no-training checks for Sprint 15.1.7 diagnostic evidence validation."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from examples.optimization.run_110_historical_fit_diagnostics import (
    extract_fitted,
    reconcile_forecasts,
)


def test_extract_fitted_preserves_neural_warmup():
    dates = pd.date_range("2022-03-01", periods=4, freq="QS-MAR")
    forecasts = {
        "ARIMA": SimpleNamespace(model="ARIMA", fitted=pd.Series([1., 2., 3., 4.])),
        "LSTM": SimpleNamespace(model="LSTM", fitted=pd.Series([np.nan, np.nan, 3., 4.])),
    }
    result = extract_fitted(forecasts, dates, ("ARIMA", "LSTM"))
    assert len(result) == 4
    assert result["LSTM"].isna().tolist() == [True, True, False, False]


def test_extract_fitted_rejects_internal_missing_values():
    dates = pd.date_range("2022-03-01", periods=4, freq="QS-MAR")
    forecasts = {"LSTM": SimpleNamespace(model="LSTM", fitted=pd.Series([np.nan, 2., np.nan, 4.]))}
    with pytest.raises(ValueError, match="after warm-up"):
        extract_fitted(forecasts, dates, ("LSTM",))


def test_reconcile_forecasts_records_difference_without_replacing_frozen():
    dates = pd.to_datetime(["2023-03-01", "2023-06-01"])
    frozen = pd.DataFrame({"date": dates, "measured_microVolt": [10., 11.], "ARIMA": [10.1, 11.1]})
    evaluations = {"ARIMA": SimpleNamespace(dates=tuple(dates), actual=(10., 11.), predicted=(10.2, 11.2))}
    forecasts = {"ARIMA": SimpleNamespace(dates=dates, forecast=pd.Series([10.2, 11.2]))}
    differences, summary = reconcile_forecasts(evaluations, forecasts, frozen, ("ARIMA",))
    np.testing.assert_allclose(differences["ARIMA"], [0.1, 0.1])
    np.testing.assert_allclose(frozen["ARIMA"], [10.1, 11.1])
    assert not summary["ARIMA"]["exactly_equal_to_saved_values"]
