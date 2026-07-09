from __future__ import annotations

"""
Battery dataset abstraction.

This module defines the BatteryDataset class, which provides a unified
interface for loading, validating, cleaning, resampling, interpolating,
and analyzing battery degradation datasets.

All forecasting algorithms in ZenerEstimation operate on this class
rather than directly manipulating pandas DataFrames.
"""

from pathlib import Path

import pandas as pd
import numpy as np

from ..exceptions import DatasetValidationError

from .preprocessing import (
    remove_duplicates,
    sort_by_date,
)

from .diagnostics import dataset_report

from .features import (
    delta,
    percentage_change,
    rolling_mean,
    rolling_std,
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

    
    def report(self):
        """
        Return a diagnostics report.
        """

        self.prepare()

        return dataset_report(self.data)

    
    def add_delta(self):
        """
        Add first-difference feature.
        """

        self.data["delta"] = delta(
        self.data["microVolt"]
        )

        return self

    
    def add_pct_change(self):
        """
        Add percentage-change feature.
        """

        self.data["pct_change"] = percentage_change(
        self.data["microVolt"]
        )

        return self

    
    def add_rolling_mean(
        self,
        window=4,
    ):
        """
        Add rolling mean.
        """

        self.data["rolling_mean"] = rolling_mean(
        self.data["microVolt"],
        window,
        )

        return self

    def add_rolling_std(
        self,
        window=4,
    ):
        """
        Add rolling standard deviation.
        """

        self.data["rolling_std"] = rolling_std(
        self.data["microVolt"],
        window,
        )

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
    # Resampling
    # ---------------------------------------------------------

    def resample(
        self,
        frequency="QS",
        method="mean",
    ):
        """
        Resample the dataset.

        Parameters
        ----------
        frequency
        Pandas frequency string
        ("QS", "MS", "W", "D", ...)

        method
        Aggregation method.

        Returns
        -------
        BatteryDataset
        """

        self.prepare()

        df = self.data.copy()

        df = df.set_index("ds")

        if method == "mean":
            df = df.resample(frequency).mean(numeric_only=True)

        elif method == "median":
            df = df.resample(frequency).median(numeric_only=True)

        elif method == "max":
            df = df.resample(frequency).max(numeric_only=True)

        elif method == "min":
            df = df.resample(frequency).min(numeric_only=True)

        elif method == "first":
            df = df.resample(frequency).first()

        elif method == "last":
            df = df.resample(frequency).last()

        else:
            raise ValueError(
                f"Unknown aggregation method: {method}"
            )

        df = df.reset_index()

        return BatteryDataset(df)

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
    # Exploratory statistics
    # ---------------------------------------------------------

    def statistics(self):
        """
        Return descriptive statistics for the voltage series.
        """

        self.prepare()

        values = self.data["microVolt"]

        return {
            "count": len(values),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "variance": float(values.var()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "range": float(values.max() - values.min()),
        }


    def percentiles(self):
        """
        Return common percentiles.
        """

        self.prepare()

        values = self.data["microVolt"]

        return {
            "q25": float(values.quantile(0.25)),
            "q50": float(values.quantile(0.50)),
            "q75": float(values.quantile(0.75)),
        }


    def coefficient_of_variation(self):
        """
        Return coefficient of variation.
        """

        self.prepare()

        values = self.data["microVolt"]

        return float(values.std() / values.mean())


    def trend(self):
        """
        Estimate linear trend.
        """

        self.prepare()

        y = self.data["microVolt"].to_numpy()

        x = np.arange(len(y))

        slope = np.polyfit(x, y, 1)[0]

        return float(slope)

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