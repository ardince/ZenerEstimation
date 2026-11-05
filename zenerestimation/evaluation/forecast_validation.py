"""Validate forecast values and retain explicit temporal alignment evidence."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastAlignmentEvidence:
    """Evidence from a successful forecast alignment validation.

    ``verified`` requires explicit model dates matching validation dates.
    ``positional_only`` means dates were absent, so temporal alignment
    could not be independently established.
    """

    status: str
    validation_steps: int
    forecast_horizon_provided: bool
    forecast_dates_provided: bool
    validation_dates: tuple[str, ...]
    forecast_dates: tuple[str, ...] | None
    timestamp_match: bool | None

    def __post_init__(self) -> None:
        if self.status not in {"verified", "positional_only"}:
            raise ValueError("status must be 'verified' or 'positional_only'")
        if not isinstance(self.validation_steps, int) or isinstance(self.validation_steps, bool) or self.validation_steps <= 0:
            raise ValueError("validation_steps must be a positive integer")
        if len(self.validation_dates) != self.validation_steps:
            raise ValueError("validation_dates length must match validation_steps")
        if self.forecast_dates_provided:
            if self.status != "verified" or self.timestamp_match is not True:
                raise ValueError("provided matching dates require verified evidence")
            if self.forecast_dates is None or len(self.forecast_dates) != self.validation_steps:
                raise ValueError("forecast_dates length must match validation_steps")
            if self.forecast_dates != self.validation_dates:
                raise ValueError("verified forecast dates must match validation dates")
        elif self.status != "positional_only" or self.forecast_dates is not None or self.timestamp_match is not None:
            raise ValueError("missing dates require positional_only evidence")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible alignment evidence."""
        return {
            "status": self.status,
            "validation_steps": self.validation_steps,
            "forecast_horizon_provided": self.forecast_horizon_provided,
            "forecast_dates_provided": self.forecast_dates_provided,
            "validation_dates": list(self.validation_dates),
            "forecast_dates": list(self.forecast_dates) if self.forecast_dates is not None else None,
            "timestamp_match": self.timestamp_match,
        }


def validate_forecast_alignment_with_evidence(
    forecast_result, validation_dates
) -> tuple[np.ndarray, ForecastAlignmentEvidence]:
    """Validate a forecast and return values with temporal evidence.

    Legacy forecasts without dates or horizon remain acceptable. A missing
    date attribute yields ``positional_only`` evidence, not verification.
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

    horizon_provided = hasattr(forecast_result, "horizon")
    if horizon_provided:
        horizon = forecast_result.horizon
        if (isinstance(horizon, (bool, np.bool_))
                or not isinstance(horizon, (int, np.integer))
                or horizon != expected_steps):
            raise ValueError("forecast horizon does not match validation horizon")

    dates_provided = hasattr(forecast_result, "dates")
    predicted_dates = None
    if dates_provided:
        predicted_dates = pd.DatetimeIndex(forecast_result.dates)
        if len(predicted_dates) != expected_steps:
            raise ValueError("forecast date count does not match validation horizon")
        if predicted_dates.hasnans:
            raise ValueError("forecast and validation dates must not contain missing timestamps")
        if not predicted_dates.equals(expected_dates):
            raise ValueError("forecast dates do not match validation dates")

    if not np.all(np.isfinite(predicted)):
        raise ValueError("forecast contains nonfinite values")

    evidence = ForecastAlignmentEvidence(
        status="verified" if dates_provided else "positional_only",
        validation_steps=expected_steps,
        forecast_horizon_provided=horizon_provided,
        forecast_dates_provided=dates_provided,
        validation_dates=tuple(date.isoformat() for date in expected_dates),
        forecast_dates=(tuple(date.isoformat() for date in predicted_dates)
                        if predicted_dates is not None else None),
        timestamp_match=True if dates_provided else None,
    )
    return predicted, evidence


def validate_forecast_alignment(forecast_result, validation_dates) -> np.ndarray:
    """Validate forecast values, plus horizon/dates when provided.

    Preserves the original NumPy-array return contract. Legacy results may
    omit horizon or dates; missing dates do not establish alignment.
    """
    predicted, _ = validate_forecast_alignment_with_evidence(
        forecast_result, validation_dates
    )
    return predicted
