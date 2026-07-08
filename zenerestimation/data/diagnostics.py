"""
Dataset diagnostics utilities.
"""

import pandas as pd


def dataset_report(df: pd.DataFrame) -> dict:
    """
    Generate a diagnostics report for a dataset.
    """

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_dates": int(df["ds"].duplicated().sum()),
        "start": df["ds"].min(),
        "end": df["ds"].max(),
        "min_voltage": float(df["microVolt"].min()),
        "max_voltage": float(df["microVolt"].max()),
        "mean_voltage": float(df["microVolt"].mean()),
        "std_voltage": float(df["microVolt"].std()),
    }