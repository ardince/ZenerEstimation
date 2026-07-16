"""
Forecast result container.

Provides a common return type for every
forecasting algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ForecastResult:
    """
    Result produced by a forecasting model.
    """

    model: str

    forecast: pd.Series

    horizon: int

    dates: pd.DatetimeIndex

    metadata: dict = field(default_factory=dict)


    @property
    def first_forecast(self):
        """
        Return the first forecast value.
        """
        return self.forecast.iloc[0]

    @property
    def last_forecast(self):
        """
        Return the last forecast value.
        """
        return self.forecast.iloc[-1]


    def __len__(self):

        return len(self.forecast)

    def to_dataframe(self):
        """
        Convert forecast to a DataFrame.
        """

        return pd.DataFrame(
            {
                "ds": self.dates,
                "forecast": self.forecast,
            }
        )

    def summary(self):
        """
        Return summary information.
        """

        return {
            "model": self.model,
            "horizon": self.horizon,
            "points": len(self.forecast),
            "start": self.dates.min(),
            "end": self.dates.max(),
        }

    def __repr__(self):

        return (
            f"ForecastResult("
            f"model='{self.model}', "
            f"horizon={self.horizon}, "
            f"points={len(self.forecast)})"
        )
    

    def __post_init__(self):

        if len(self.forecast) != len(self.dates):
            raise ValueError(
            "forecast and dates must have the same length."
            )