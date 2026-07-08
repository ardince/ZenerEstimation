from __future__ import annotations

"""
Dataset handling utilities.

Sprint 2
"""

from pathlib import Path

import pandas as pd

from ..exceptions import DatasetValidationError

from .preprocessing import (
    remove_duplicates,
    sort_by_date,
)


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
        Load dataset from CSV.
        """

        path = Path(filename)

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        return cls(df)

    
    def clean(self):
        """
        Clean dataset before forecasting.
        """

        self.prepare()

        self.data = remove_duplicates(self.data)

        self.data = sort_by_date(self.data)

        return self

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
            .drop_duplicates(subset="ds")
            .reset_index(drop=True)
        )

        return self

    # ---------------------------------------------------------
    # Frequency detection
    # ---------------------------------------------------------

    def sampling_days(self):
        """
        Median sampling interval in days.
        """

        self.prepare()

        delta = self.data["ds"].diff().dropna()

        if len(delta) == 0:
            return None

        return float(delta.dt.days.median())


    def detect_frequency(self):
        """
        Detect the dataset sampling frequency.

        Returns
        -------
        Monthly
        Quarterly
        Semiannual
        Annual
        Irregular
        """

        days = self.sampling_days()

        if days is None:
            return "Unknown"

        if 25 <= days <= 35:
            return "Monthly"

        if 80 <= days <= 100:
            return "Quarterly"

        if 170 <= days <= 190:
            return "Semiannual"

        if 350 <= days <= 380:
            return "Annual"

        return "Irregular"

    # ---------------------------------------------------------
    # Missing period detection
    # ---------------------------------------------------------

    def expected_index(self):
        """
        Construct the complete expected datetime index.
        """

        self.prepare()

        freq = self.detect_frequency()

        mapping = {
            "Monthly": "MS",
            "Quarterly": "QS",
            "Semiannual": "2QS",
            "Annual": "YS",
        }

        if freq not in mapping:
            return None

        return pd.date_range(
            start=self.data["ds"].min(),
            end=self.data["ds"].max(),
            freq=mapping[freq],
        )


    def missing_periods(self):

        """
        Detect missing quarterly measurements.
        """

        self.prepare()

        full_index = pd.date_range(
            start=self.data["ds"].min(),
            end=self.data["ds"].max(),
            freq="3MS",
        )

        existing = pd.DatetimeIndex(self.data["ds"])

        missing = full_index.difference(existing)

        return missing

    # ---------------------------------------------------------
    # Missing-data handling
    # ---------------------------------------------------------

    def missing_count(self):
        """
        Return total number of missing values.
        """

        return int(self.data.isna().sum().sum())

    def has_missing(self):
        """
        True if dataset contains missing values.
        """

        return self.missing_count() > 0

    def interpolate(self):
        """
        Linear interpolation of voltage values.
        """

        self.data["microVolt"] = (
            self.data["microVolt"]
            .interpolate(method="linear")
        )

        return self

    def forward_fill(self):
        """
        Forward-fill voltage values.
        """

        self.data["microVolt"] = (
            self.data["microVolt"]
            .ffill()
        )

        return self

    def backward_fill(self):
        """
        Backward-fill voltage values.
        """

        self.data["microVolt"] = (
            self.data["microVolt"]
            .bfill()
        )

        return self

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
            "missing": self.missing_count(),
            "start": self.data["ds"].min(),
            "end": self.data["ds"].max(),
            "frequency": self.detect_frequency(),
            "missing_periods": len(self.missing_periods()),
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