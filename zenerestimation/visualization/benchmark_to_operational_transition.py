"""Presentation-only transition diagnostics for frozen dataset-410 evidence."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BenchmarkToOperationalTransitionPlotter:
    """Draw existing measured, benchmark and operational evidence; never fit models."""

    def __init__(self, benchmark_run, operational_run, *, development_end="2023-12-01", operational_cutoff="2025-06-01", operational_dates=None):
        self.benchmark = benchmark_run
        self.operational = operational_run
        self.development_end = pd.Timestamp(development_end)
        self.operational_cutoff = pd.Timestamp(operational_cutoff)
        if operational_dates is None:
            operational_dates = pd.date_range("2025-09-01", periods=6, freq="QS-MAR")
        self.operational_dates = pd.DatetimeIndex(pd.to_datetime(operational_dates))
        self.figure = None

    def plot(self):
        evaluations = self.benchmark.evaluations
        operational = self.operational.forecasts
        if not evaluations or set(evaluations) != set(operational):
            raise ValueError("Benchmark and operational model sets must be identical and nonempty")
        if self.operational_dates.empty or self.operational_dates[0] <= self.operational_cutoff:
            raise ValueError("Operational dates must start after the operational cutoff")
        if not self.operational_dates.is_monotonic_increasing or self.operational_dates.has_duplicates:
            raise ValueError("Operational dates must be strictly increasing")
        history = self.benchmark.dataset.data.copy()
        history["ds"] = pd.to_datetime(history["ds"])
        history = history.loc[history["ds"] <= self.operational_cutoff].sort_values("ds")
        if history.empty or history["ds"].duplicated().any():
            raise ValueError("Measured history is empty or has duplicate dates")
        if pd.Timestamp(history["ds"].iloc[-1]) != self.operational_cutoff:
            raise ValueError("Measured operational-cutoff observation is missing")
        first = next(iter(evaluations.values()))
        benchmark_dates = pd.DatetimeIndex(pd.to_datetime(first.dates))
        actual = np.asarray(first.actual, dtype=float).reshape(-1)
        if len(actual) != len(benchmark_dates) or not np.isfinite(actual).all():
            raise ValueError("Invalid frozen benchmark actuals")
        if benchmark_dates[0] <= self.development_end or benchmark_dates[-1] != self.operational_cutoff:
            raise ValueError("Benchmark dates do not match the declared boundaries")
        if not benchmark_dates.is_monotonic_increasing or benchmark_dates.has_duplicates:
            raise ValueError("Benchmark dates must be strictly increasing")
        measured_holdout = history.set_index("ds")["microVolt"].reindex(benchmark_dates)
        if measured_holdout.isna().any() or not np.allclose(measured_holdout.to_numpy(dtype=float), actual, rtol=0, atol=1e-9):
            raise ValueError("Benchmark actuals disagree with measured history")
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(history["ds"], history["microVolt"], color="black", linewidth=2.2, marker=".", markersize=4, label="Measured voltage (through 2025-06)", zorder=5)
        colors = plt.get_cmap("tab10")
        for index, (name, result) in enumerate(evaluations.items()):
            dates = pd.DatetimeIndex(pd.to_datetime(result.dates))
            predicted = np.asarray(result.predicted, dtype=float).reshape(-1)
            future = np.asarray(operational[name], dtype=float).reshape(-1)
            if not dates.equals(benchmark_dates) or len(predicted) != len(dates) or not np.isfinite(predicted).all():
                raise ValueError(f"{name}: invalid frozen benchmark predictions")
            if len(future) != len(self.operational_dates) or not np.isfinite(future).all():
                raise ValueError(f"{name}: invalid operational predictions")
            color = colors(index % 10)
            ax.plot(dates, predicted, color=color, linestyle="--", linewidth=1.5, marker="o", markersize=3.5, alpha=0.9, label=f"{name} · frozen holdout")
            ax.plot(self.operational_dates, future, color=color, linestyle="-.", linewidth=1.7, marker="s", markersize=3.5, alpha=0.9, label=f"{name} · operational (unvalidated)")
        ax.axvline(self.development_end, color="gray", linestyle=":", linewidth=1.5, label="Development end · 2023-12")
        ax.axvline(self.operational_cutoff, color="dimgray", linestyle="-", linewidth=1.2, label="Operational refit cutoff · 2025-06")
        ax.set_xlim(pd.Timestamp("2022-01-01"), self.operational_dates[-1] + pd.DateOffset(months=2))
        ax.set_title("410 — Frozen benchmark vs refitted operational extrapolation")
        ax.set_xlabel("Date")
        ax.set_ylabel("Voltage (µV)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=2, loc="best")
        fig.autofmt_xdate()
        fig.tight_layout()
        self.figure = fig
        return fig, ax

    def save(self, path):
        if self.figure is None:
            self.plot()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(path, dpi=250, bbox_inches="tight")
        plt.close(self.figure)
        self.figure = None
        return path
