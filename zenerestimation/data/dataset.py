"""
Core dataset representation.

Every forecasting algorithm in ZenerEstimation operates on a
BatteryDataset instead of directly manipulating pandas DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class BatteryDataset:
    """
    Central dataset object.

    Parameters
    ----------
    dataframe
        Input pandas DataFrame.

    date_column
        Name of datetime column.

    value_column
        Name of measurement column.
    """

    dataframe: pd.DataFrame

    date_column: str = "Date"

    value_column: str = "Value"

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dates(self) -> pd.Series:
        """Return date column."""
        return self.dataframe[self.date_column]

    @property
    def values(self) -> pd.Series:
        """Return measurement column."""
        return self.dataframe[self.value_column]

    @property
    def n_samples(self) -> int:
        """Number of observations."""
        return len(self.dataframe)

    @property
    def start_date(self):
        """First measurement."""
        return self.dates.iloc[0]

    @property
    def end_date(self):
        """Last measurement."""
        return self.dates.iloc[-1]

    @property
    def duration(self):
        """Measurement duration."""
        return self.end_date - self.start_date

    def summary(self) -> dict:
        """
        Return dataset summary.
        """

        return {
            "samples": self.n_samples,
            "start": self.start_date,
            "end": self.end_date,
            "duration": self.duration,
            "value_column": self.value_column,
            "date_column": self.date_column,
            "metadata": self.metadata,
        }

    def copy(self) -> "BatteryDataset":
        """
        Deep copy.
        """

        return BatteryDataset(
            dataframe=self.dataframe.copy(),
            date_column=self.date_column,
            value_column=self.value_column,
            metadata=self.metadata.copy(),
        )

    def __repr__(self) -> str:

        return (
            f"BatteryDataset("
            f"samples={self.n_samples}, "
            f"date='{self.date_column}', "
            f"value='{self.value_column}')"
        )