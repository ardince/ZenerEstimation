"""
Feature engineering utilities.

This module provides common statistical features derived from battery
voltage measurements, including first differences, percentage changes,
rolling statistics, and other transformations used by forecasting
algorithms.
"""

from __future__ import annotations

import pandas as pd


def delta(series: pd.Series) -> pd.Series:
    """
    First difference.
    """
    return series.diff()


def percentage_change(series: pd.Series) -> pd.Series:
    """
    Percentage change.
    """
    return series.pct_change()


def rolling_mean(
    series: pd.Series,
    window: int = 4,
) -> pd.Series:
    """
    Rolling mean.
    """
    return series.rolling(window).mean()


def rolling_std(
    series: pd.Series,
    window: int = 4,
) -> pd.Series:
    """
    Rolling standard deviation.
    """
    return series.rolling(window).std()