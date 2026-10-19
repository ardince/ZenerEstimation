"""Validate forecast values and available temporal alignment metadata."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_forecast_alignment(forecast_result, validation_dates) -> np.ndarray:
    """Validate forecast values, plus horizon/dates when provided.

    Legacy results may omit horizon or dates. Missing dates cannot prove
    temporal alignment; they are accepted for backward compatibility only.
    """
    expected_dates = pd.DatetimeIndex(validation_dates)
    predicted = np.asarray(forecast_result.forecast, dtype=float)
    expected_steps = len(expected_dates)

    if expected_steps == 0:
        raise ValueError("validation dates cannot be empty")
    if predicted.ndim != 1:
        raise ValueError("forecast must be one-dimensional")
    if len(predicted) != expected_steps:
        raise ValueError("forecast length does not match validation horizon")
    if expected_dates.hasnans:
        raise ValueError("forecast and validation dates must not contain missing timestamps")
    if not expected_dates.is_monotonic_increasing or not expected_dates.is_unique:
        raise ValueError("validation dates must be strictly increasing and unique")

    if hasattr(forecast_result, "horizon"):
        horizon = forecast_result.horizon
        if (isinstance(horizon, (bool, np.bool_))
                or not isinstance(horizon, (int, np.integer))
                or horizon != expected_steps):
            raise ValueError("forecast horizon does not match validation horizon")

    if hasattr(forecast_result, "dates"):
        predicted_dates = pd.DatetimeIndex(forecast_result.dates)
        if len(predicted_dates) != expected_steps:
            raise ValueError("forecast date count does not match validation horizon")
        if predicted_dates.hasnans:
            raise ValueError("forecast and validation dates must not contain missing timestamps")
        if not predicted_dates.equals(expected_dates):
            raise ValueError("forecast dates do not match validation dates")

    if not np.all(np.isfinite(predicted)):
        raise ValueError("forecast contains nonfinite values")
    return predicted
