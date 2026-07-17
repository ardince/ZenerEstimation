"""
Forecast result container.

Provides a common return type for every
forecasting algorithm.
"""

from __future__ import annotations

import json
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

    def __len__(self):

        return len(self.forecast)

    @property
    def first_forecast(self):
        """
        Return the first forecast value.
        """
        return float(self.forecast.iloc[0])

    @property
    def last_forecast(self):
        """
        Return the last forecast value.
        """
        return float(self.forecast.iloc[-1])

    def to_dataframe(self):
        """
        Convert forecast to a pandas DataFrame.
        """

        return pd.DataFrame(
            {
                "ds": self.dates,
                "forecast": self.forecast,
            }
        )

    def to_dict(self):
        """
        Return a JSON-serializable dictionary.
        """

        return {
            "model": self.model,
            "horizon": self.horizon,
            "dates": [
                d.strftime("%Y-%m-%d")
                for d in self.dates
            ],
            "forecast": [
                float(v)
                for v in self.forecast
            ],
            "metadata": self.metadata,
        }

    def save_json(
        self,
        filename,
    ):
        """
        Save forecast as JSON.
        """

        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.to_dict(),
                f,
                indent=4,
                ensure_ascii=False,
            )

    def summary(self):
        """
        Return summary information.
        """

        summary = {
            "model": self.model,
            "horizon": self.horizon,
            "points": len(self.forecast),
            "start": self.dates.min().strftime("%Y-%m-%d"),
            "end": self.dates.max().strftime("%Y-%m-%d"),
        }

        summary.update(self.metadata)

        return summary

    def __repr__(self):

        return (
            f"ForecastResult("
            f"model='{self.model}', "
            f"horizon={self.horizon}, "
            f"points={len(self.forecast)})"
        )