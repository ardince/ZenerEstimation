"""Milestone 16.2-A: forecast alignment evidence contract."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from zenerestimation.evaluation.forecast_validation import (
    ForecastAlignmentEvidence,
    validate_forecast_alignment,
    validate_forecast_alignment_with_evidence,
)


DATES = pd.to_datetime(["2024-03-01", "2024-06-01"])


def test_explicit_dates_are_verified():
    forecast = SimpleNamespace(forecast=[1.0, 2.0], dates=DATES, horizon=2)
    values, evidence = validate_forecast_alignment_with_evidence(forecast, DATES)
    np.testing.assert_array_equal(values, [1.0, 2.0])
    assert evidence.status == "verified"
    assert evidence.forecast_horizon_provided is True
    assert evidence.forecast_dates_provided is True
    assert evidence.timestamp_match is True
    assert evidence.forecast_dates == evidence.validation_dates
    assert evidence.to_dict()["forecast_dates"] == [d.isoformat() for d in DATES]


def test_missing_dates_are_positional_only():
    forecast = SimpleNamespace(forecast=[1.0, 2.0], horizon=2)
    _, evidence = validate_forecast_alignment_with_evidence(forecast, DATES)
    assert evidence.status == "positional_only"
    assert evidence.timestamp_match is None
    assert evidence.forecast_dates is None
    assert evidence.forecast_horizon_provided is True
    assert evidence.to_dict()["forecast_dates"] is None


def test_missing_horizon_is_reported():
    forecast = SimpleNamespace(forecast=[1.0, 2.0], dates=DATES)
    _, evidence = validate_forecast_alignment_with_evidence(forecast, DATES)
    assert evidence.status == "verified"
    assert evidence.forecast_horizon_provided is False


def test_missing_both_attributes_is_positional_only():
    _, evidence = validate_forecast_alignment_with_evidence(
        SimpleNamespace(forecast=[1.0, 2.0]), DATES
    )
    assert evidence.status == "positional_only"
    assert not evidence.forecast_horizon_provided


@pytest.mark.parametrize("bad_dates", [
    pd.to_datetime(["2024-06-01", "2024-09-01"]),
    pd.to_datetime(["2024-06-01", "2024-03-01"]),
])
def test_shifted_or_reversed_forecast_dates_rejected(bad_dates):
    with pytest.raises(ValueError, match="forecast dates do not match"):
        validate_forecast_alignment_with_evidence(
            SimpleNamespace(forecast=[1.0, 2.0], dates=bad_dates), DATES
        )


def test_nonfinite_values_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        validate_forecast_alignment_with_evidence(
            SimpleNamespace(forecast=[1.0, np.nan], dates=DATES), DATES
        )


def test_legacy_validator_still_returns_ndarray():
    result = validate_forecast_alignment(
        SimpleNamespace(forecast=[1.0, 2.0]), DATES
    )
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1.0, 2.0])


def test_evidence_rejects_inconsistent_status():
    with pytest.raises(ValueError, match="provided matching dates"):
        ForecastAlignmentEvidence(
            status="positional_only",
            validation_steps=2,
            forecast_horizon_provided=True,
            forecast_dates_provided=True,
            validation_dates=("2024-03-01", "2024-06-01"),
            forecast_dates=("2024-03-01", "2024-06-01"),
            timestamp_match=True,
        )
