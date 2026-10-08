
"""Historical fitted-value visualization for optimized benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zenerestimation.evaluation.result import EvaluationResult


class BenchmarkHistoryPlotter:
    """
    Visualize measured history, fitted values and holdout forecasts.

    Historical fitted values are in-sample calculations.
    Benchmark forecasts are out-of-sample predictions.

    This class is presentation-only. It never fits models,
    generates forecasts, computes metrics, or selects models.

    Parameters
    ----------
    dataset:
        Original benchmark dataset.

    evaluations:
        Mapping of model names to EvaluationResult objects.

    forecasts:
        Mapping of model names to existing ForecastResult objects.

    training_dates:
        Dates of the prepared training observations, in the
        exact order used for model fitting.

        When omitted, training dates are inferred from the
        original dataset before the benchmark start. This
        fallback preserves the previous constructor behavior,
        but explicit dates are recommended.
    """

    def __init__(
        self,
        dataset,
        evaluations: Mapping[str, EvaluationResult],
        forecasts: Mapping,
        training_dates=None,
    ):
        # -----------------------------------------------------
        # Evaluation evidence
        # -----------------------------------------------------

        if not isinstance(evaluations, Mapping):
            raise TypeError(
                "evaluations must be a mapping"
            )

        if not evaluations:
            raise ValueError(
                "evaluations must contain at least one model"
            )

        copied_evaluations = dict(evaluations)

        for name, result in copied_evaluations.items():
            if not isinstance(result, EvaluationResult):
                raise TypeError(
                    "all evaluations must be EvaluationResult objects"
                )

            if name != result.model:
                raise ValueError(
                    "evaluation key must match result.model"
                )

        reference = next(iter(copied_evaluations.values()))

        for result in copied_evaluations.values():
            if result.evaluation_steps != reference.evaluation_steps:
                raise ValueError(
                    "inconsistent evaluation_steps"
                )

            if tuple(result.dates) != tuple(reference.dates):
                raise ValueError(
                    "inconsistent benchmark dates"
                )

            if not np.array_equal(
                np.asarray(result.actual),
                np.asarray(reference.actual),
            ):
                raise ValueError(
                    "inconsistent benchmark actual values"
                )

        # -----------------------------------------------------
        # Forecast evidence
        # -----------------------------------------------------

        if not isinstance(forecasts, Mapping):
            raise TypeError(
                "forecasts must be a mapping"
            )

        if set(forecasts) != set(copied_evaluations):
            raise ValueError(
                "forecast and evaluation model names must match"
            )

        copied_forecasts = dict(forecasts)

        for name, evaluation in copied_evaluations.items():
            forecast = copied_forecasts[name]

            if getattr(forecast, "model", None) != name:
                raise ValueError(
                    "forecast model name must match mapping key"
                )

            for attribute in ("forecast", "dates", "fitted"):
                if not hasattr(forecast, attribute):
                    raise TypeError(
                        f"forecast evidence must provide {attribute}"
                    )

            forecast_dates = pd.DatetimeIndex(
                pd.to_datetime(forecast.dates)
            )

            evaluation_dates = pd.DatetimeIndex(
                pd.to_datetime(evaluation.dates)
            )

            if not forecast_dates.equals(evaluation_dates):
                raise ValueError(
                    "forecast dates must match evaluation dates"
                )

            predicted = np.asarray(
                forecast.forecast,
                dtype=float,
            )

            if predicted.shape != (
                evaluation.evaluation_steps,
            ):
                raise ValueError(
                    "forecast horizon must match evaluation_steps"
                )

            if not np.array_equal(
                predicted,
                np.asarray(
                    evaluation.predicted,
                    dtype=float,
                ),
            ):
                raise ValueError(
                    "forecast predictions must match evaluation"
                )

        # -----------------------------------------------------
        # Original reference dataset
        # -----------------------------------------------------

        data = dataset.data.copy(deep=True)

        if not {"ds", "microVolt"}.issubset(data.columns):
            raise ValueError(
                "dataset must contain ds and microVolt"
            )

        data["ds"] = pd.to_datetime(data["ds"])

        data = (
            data.sort_values("ds")
            .reset_index(drop=True)
        )

        if (
            data["ds"].isna().any()
            or data["ds"].duplicated().any()
        ):
            raise ValueError(
                "dataset contains invalid or duplicate dates"
            )

        benchmark_dates = pd.DatetimeIndex(
            pd.to_datetime(reference.dates)
        )

        if (
            benchmark_dates.hasnans
            or not benchmark_dates.is_unique
            or not benchmark_dates.is_monotonic_increasing
        ):
            raise ValueError(
                "invalid benchmark dates"
            )

        benchmark_start = benchmark_dates[0]
        benchmark_end = benchmark_dates[-1]

        development = data.loc[
            data["ds"] < benchmark_start
        ].copy()

        if development.empty:
            raise ValueError(
                "dataset contains no development history"
            )

        # Do not display observations beyond the frozen
        # benchmark endpoint (important for dataset 110).
        reference_data = data.loc[
            data["ds"] <= benchmark_end
        ].copy()

        # -----------------------------------------------------
        # Training timeline
        # -----------------------------------------------------

        if training_dates is None:
            dates = pd.DatetimeIndex(
                development["ds"]
            )
        else:
            dates = pd.DatetimeIndex(
                pd.to_datetime(training_dates)
            )

        if (
            len(dates) == 0
            or dates.hasnans
            or not dates.is_unique
            or not dates.is_monotonic_increasing
        ):
            raise ValueError(
                "training_dates must be nonempty, unique "
                "and chronologically ordered"
            )

        if dates[-1] >= benchmark_start:
            raise ValueError(
                "training_dates must end before benchmark"
            )

        if training_dates is None:
            # The inferred dates come directly from the
            # original development partition.
            pass
        else:
            # For the current temporal preprocessor, training
            # timestamps are preserved exactly. We require
            # this correspondence rather than silently
            # introducing or removing dates.
            reference_training_dates = pd.DatetimeIndex(
                development["ds"]
            )

            if not dates.equals(reference_training_dates):
                raise ValueError(
                    "training_dates do not match the "
                    "reference development timeline"
                )

        # -----------------------------------------------------
        # Historical fitted-value alignment
        # -----------------------------------------------------

        fitted_values = {}

        for name, forecast in copied_forecasts.items():
            fitted = forecast.fitted

            if fitted is None:
                raise ValueError(
                    f"{name} has no historical fitted values"
                )

            values = np.asarray(
                fitted,
                dtype=float,
            )

            if values.ndim != 1:
                raise ValueError(
                    f"{name} fitted values must be one-dimensional"
                )

            if len(values) != len(dates):
                raise ValueError(
                    f"{name} fitted length does not match "
                    "training_dates"
                )

            if np.isinf(values).any():
                raise ValueError(
                    f"{name} fitted values contain infinity"
                )

            # Preserve unavailable warm-up observations.
            # Reject unexplained gaps after the first
            # available fitted value.
            finite_positions = np.flatnonzero(
                np.isfinite(values)
            )

            if len(finite_positions) == 0:
                raise ValueError(
                    f"{name} has no finite fitted values"
                )

            first_valid = finite_positions[0]

            if np.isnan(values[first_valid:]).any():
                raise ValueError(
                    f"{name} has missing fitted values "
                    "after its warm-up period"
                )

            # -------------------------------------------------
            # Explicit fitted-index verification
            # -------------------------------------------------

            if isinstance(fitted, pd.Series):
                index = fitted.index

                if isinstance(index, pd.DatetimeIndex):
                    if not index.equals(dates):
                        raise ValueError(
                            f"{name} fitted calendar index "
                            "does not match training_dates"
                        )

                else:
                    # Existing ARIMA/Kalman/neural models
                    # expose positional indexes.
                    #
                    # Accept either the original development
                    # index or a fresh zero-based RangeIndex.
                    # Arbitrary indexes are not accepted.

                    original_index = development.index

                    positional_index = pd.RangeIndex(
                        len(dates)
                    )

                    if not (
                        index.equals(original_index)
                        or index.equals(positional_index)
                    ):
                        raise ValueError(
                            f"{name} fitted positional index "
                            "does not match training sequence"
                        )

            fitted_values[name] = values.copy()

        # -----------------------------------------------------
        # Benchmark actual-value verification
        # -----------------------------------------------------

        indexed = data.set_index("ds")

        if not benchmark_dates.isin(indexed.index).all():
            raise ValueError(
                "benchmark dates are missing from dataset"
            )

        dataset_actual = indexed.loc[
            benchmark_dates,
            "microVolt",
        ].to_numpy(dtype=float)

        if not np.array_equal(
            dataset_actual,
            np.asarray(reference.actual, dtype=float),
        ):
            raise ValueError(
                "dataset benchmark actuals do not match evaluation"
            )

        self.dataset = dataset

        self._reference_data = reference_data
        self._development = development
        self._training_dates = dates

        self._evaluations = copied_evaluations
        self._forecasts = copied_forecasts
        self._fitted_values = fitted_values

        self._figure = None
        self._axes = None

    # ---------------------------------------------------------
    # Public properties
    # ---------------------------------------------------------

    @property
    def evaluations(self):
        return dict(self._evaluations)

    @property
    def results(self):
        """Alias for stored benchmark evaluations."""
        return self.evaluations

    @property
    def forecasts(self):
        return dict(self._forecasts)

    @property
    def training_dates(self):
        """Return a defensive copy of the training timeline."""
        return self._training_dates.copy()

    @property
    def models(self):
        return list(self._evaluations)

    @property
    def evaluation_steps(self):
        reference = next(iter(self._evaluations.values()))
        return reference.evaluation_steps

    @property
    def figure(self):
        return self._figure

    @property
    def axes(self):
        return self._axes

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------

    def plot(
        self,
        title="Optimized Multi-Model Benchmark — Historical Fits",
    ):
        """Plot historical fits and untouched holdout evidence."""

        reference = next(iter(self._evaluations.values()))

        benchmark_dates = pd.DatetimeIndex(
            pd.to_datetime(reference.dates)
        )

        benchmark_actual = np.asarray(
            reference.actual,
            dtype=float,
        )

        boundary_date = self._training_dates[-1]

        if self._figure is not None:
            plt.close(self._figure)

        fig, ax = plt.subplots(
            figsize=(14, 7)
        )

        # -----------------------------------------------------
        # Original measured reference
        # -----------------------------------------------------

        ax.plot(
            self._reference_data["ds"],
            self._reference_data["microVolt"],
            color="black",
            linewidth=1.8,
            alpha=0.8,
            label="Measured reference",
            zorder=3,
        )

        # -----------------------------------------------------
        # Historical fitted values
        # -----------------------------------------------------

        model_colors = {}

        for name, values in self._fitted_values.items():
            line, = ax.plot(
                self._training_dates,
                values,
                linewidth=1.3,
                alpha=0.85,
                label=f"{name} fitted (in-sample)",
            )

            model_colors[name] = line.get_color()

        # -----------------------------------------------------
        # Actual benchmark observations
        # -----------------------------------------------------

        ax.plot(
            benchmark_dates,
            benchmark_actual,
            color="black",
            marker="o",
            markersize=5,
            linestyle="none",
            label="Benchmark actual",
            zorder=5,
        )

        # -----------------------------------------------------
        # Frozen out-of-sample forecasts
        # -----------------------------------------------------

        for name, forecast in self._forecasts.items():
            ax.plot(
                pd.to_datetime(forecast.dates),
                np.asarray(
                    forecast.forecast,
                    dtype=float,
                ),
                color=model_colors[name],
                marker="s",
                markersize=4,
                linestyle="--",
                linewidth=1.6,
                label=f"{name} forecast",
            )

        # -----------------------------------------------------
        # Training / benchmark boundary
        # -----------------------------------------------------

        ax.axvline(
            boundary_date,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label="Forecast boundary",
        )

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Voltage (µV)")

        ax.grid(True, alpha=0.3)

        ax.legend(
            loc="best",
            fontsize=8,
            ncol=2,
        )

        fig.autofmt_xdate()
        fig.tight_layout()

        self._figure = fig
        self._axes = ax

        return fig

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def save(
        self,
        filename,
        dpi=300,
    ) -> Path:
        """Save the historical benchmark figure."""

        filename = Path(filename)

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self._figure is None:
            self.plot()

        self._figure.savefig(
            filename,
            dpi=dpi,
            bbox_inches="tight",
        )

        return filename

    def __repr__(self):
        return (
            "BenchmarkHistoryPlotter("
            f"models={self.models!r}, "
            f"evaluation_steps={self.evaluation_steps}"
            ")"
        )
