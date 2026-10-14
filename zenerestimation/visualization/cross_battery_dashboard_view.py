"""Pure, read-only presentation selection for Sprint 15.3 dashboard.

No Streamlit import, disk I/O, or forecasting code. This module deliberately
keeps the two distinct benchmark protocols visible in every metrics view.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from zenerestimation.visualization.cross_battery_comparison import CrossBatteryComparison

BATTERY_IDS = ("732B-5610410", "732B-5610110")
METRIC_LABELS = {"rmse": "RMSE (µV)", "mae": "MAE (µV)", "mape": "MAPE (%)"}


@dataclass(frozen=True)
class DashboardView:
    summary: pd.DataFrame
    metrics: pd.DataFrame
    evidence: pd.DataFrame


def dashboard_view(
    comparison: CrossBatteryComparison,
    battery_ids: Sequence[str],
    metric: str = "rmse",
) -> DashboardView:
    """Return independent filtered copies; do not alter validated adapter frames."""
    if metric not in METRIC_LABELS:
        raise ValueError(f"Unsupported metric: {metric}")
    if not battery_ids or len(set(battery_ids)) != len(battery_ids):
        raise ValueError("Select distinct battery identifiers")
    if not set(battery_ids).issubset(BATTERY_IDS):
        raise ValueError("Unknown battery identifier")

    selected = set(battery_ids)
    summary = comparison.summary.loc[
        comparison.summary["battery_id"].isin(selected)
    ].copy(deep=True)
    metrics = comparison.model_metrics.loc[
        comparison.model_metrics["battery_id"].isin(selected),
        ["battery_id", "benchmark_type", "model", metric],
    ].copy(deep=True)
    metrics = metrics.sort_values(["battery_id", metric, "model"], kind="stable").reset_index(drop=True)
    evidence = comparison.evidence.loc[
        comparison.evidence["battery_id"].isin(selected)
    ].copy(deep=True)
    return DashboardView(summary.reset_index(drop=True), metrics, evidence.reset_index(drop=True))
