from __future__ import annotations

"""
Core dataset representation.

Every forecasting algorithm in ZenerEstimation operates on a
BatteryDataset instead of directly manipulating pandas DataFrames.
"""

"""
Dataset handling utilities.

Sprint 2
"""

from pathlib import Path

import pandas as pd


class BatteryDataset:
    """
    Container for battery degradation datasets.

    Expected columns
    ----------------
    ds
        Measurement date.

    microVolt
        Measured battery voltage.
    """

    REQUIRED_COLUMNS = ["ds", "microVolt"]

    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe.copy()

    # ---------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------

    @classmethod
    def from_csv(cls, filename):
        """
        Load a BatteryDataset from a CSV file.
        """

        path = Path(filename)

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        return cls(df)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validate(self):
        """
        Validate required columns.
        """

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in self.data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    # ---------------------------------------------------------
    # Preparation
    # ---------------------------------------------------------

    def prepare(self):
        """
        Prepare dataset for forecasting.
        """

        self.validate()

        self.data["ds"] = pd.to_datetime(
            self.data["ds"],
            dayfirst=True,
        )

        self.data = (
            self.data
            .sort_values("ds")
            .drop_duplicates(subset="ds")
            .reset_index(drop=True)
        )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def summary(self):
        """
        Return dataset summary.
        """

        self.prepare()

        return {
            "rows": len(self.data),
            "columns": len(self.data.columns),
            "missing": int(self.data.isna().sum().sum()),
            "start": self.data["ds"].min(),
            "end": self.data["ds"].max(),
        }

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __len__(self):

        return len(self.data)

    def __repr__(self):

        return (
            f"BatteryDataset("
            f"rows={len(self.data)}, "
            f"columns={list(self.data.columns)})"
        )