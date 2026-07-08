"""
Dataset handling utilities.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class BatteryDataset:
    """
    Container for battery degradation datasets.
    """

    REQUIRED_COLUMNS = ["ds", "microVolt"]

    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe.copy()

    # --------------------------------------------------
    # Constructors
    # --------------------------------------------------

    @classmethod
    def from_csv(cls, filename):
        """
        Load a dataset from a CSV file.
        """

        path = Path(filename)

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        return cls(df)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validate(self):
        """
        Validate required columns.
        """

        missing = [
            c for c in self.REQUIRED_COLUMNS
            if c not in self.data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    # --------------------------------------------------
    # Cleaning
    # --------------------------------------------------

    def prepare(self):
        """
        Prepare dataset.
        """

        self.validate()

        self.data["ds"] = pd.to_datetime(
            self.data["ds"],
            dayfirst=True
        )

        self.data = self.data.sort_values("ds")

        self.data = self.data.drop_duplicates(
            subset="ds"
        )

        self.data.reset_index(
            drop=True,
            inplace=True
        )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    def summary(self):

        self.prepare()

        return {
            "rows": len(self.data),
            "columns": len(self.data.columns),
            "missing": int(self.data.isna().sum().sum()),
            "start": self.data["ds"].min(),
            "end": self.data["ds"].max(),
        }