"""
Battery dataset analysis.

Provides statistical summaries and forecasting-oriented diagnostics.
"""

from __future__ import annotations

import numpy as np


class DatasetAnalyzer:

    def __init__(self, dataset):
        self.dataset = dataset

    def report(self):

        self.dataset.prepare()

        df = self.dataset.data

        values = df["microVolt"].to_numpy()

        drift = (
            np.mean(np.diff(values))
            if len(values) > 1
            else 0.0
        )

        return {
            "rows": len(df),
            "columns": len(df.columns),

            "start_date": df["ds"].min(),
            "end_date": df["ds"].max(),

            "minimum_voltage": float(values.min()),
            "maximum_voltage": float(values.max()),

            "mean_voltage": float(values.mean()),
            "median_voltage": float(np.median(values)),
            "std_voltage": float(values.std()),

            "estimated_drift": float(drift),

            "missing_values": int(df.isna().sum().sum()),
            "duplicate_dates": int(df.duplicated("ds").sum()),
        }