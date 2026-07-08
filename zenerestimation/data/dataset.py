from __future__ import annotations

"""
Dataset handling utilities.

Sprint 2
"""

from pathlib import Path

import pandas as pd

from ..exceptions import DatasetValidationError


class BatteryDataset:
    """
    Container for battery degradation datasets.
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
        Load dataset from CSV.
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
        Validate dataset integrity.
        """

        # Empty dataframe
        if self.data.empty:
            raise DatasetValidationError(
                "Dataset is empty."
            )

        # Required columns
        missing = [
            c
            for c in self.REQUIRED_COLUMNS
            if c not in self.data.columns
        ]

        if missing:
            raise DatasetValidationError(
                f"Missing required columns: {missing}"
            )

        # Missing values
        if self.data[self.REQUIRED_COLUMNS].isna().any().any():
            raise DatasetValidationError(
                "Dataset contains missing values."
            )

        # Numeric voltage
        try:
            self.data["microVolt"] = pd.to_numeric(
                self.data["microVolt"]
            )
        except Exception:
            raise DatasetValidationError(
                "Column 'microVolt' must be numeric."
            )

        # Datetime parsing
        try:
            self.data["ds"] = pd.to_datetime(
                self.data["ds"],
                dayfirst=True,
            )
        except Exception:
            raise DatasetValidationError(
                "Column 'ds' contains invalid dates."
            )

    # ---------------------------------------------------------
    # Preparation
    # ---------------------------------------------------------

    def prepare(self):
        """
        Prepare dataset for forecasting.
        """

        self.validate()

        self.data = (
            self.data
            .sort_values("ds")
            .reset_index(drop=True)
        )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def summary(self):

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