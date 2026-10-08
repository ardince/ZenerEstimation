"""Best-versus-worst historical-fit diagnostic for optimized benchmarks."""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class BenchmarkBestWorstHistoryPlotter:
    """Plot measured history and best/worst historical fits selected by RMSE."""
    TITLE = "Best and Worst Multi-Model Benchmark – Historical Fits"

    def __init__(self, dataset, evaluations: Mapping, forecasts: Mapping,
                 training_dates=None):
        if not evaluations:
            raise ValueError("evaluations must not be empty")
        if set(evaluations) != set(forecasts):
            raise ValueError("evaluations and forecasts must contain identical model names")
        self.dataset = dataset
        self.evaluations = dict(evaluations)
        self.forecasts = dict(forecasts)
        self.training_dates = None if training_dates is None else pd.DatetimeIndex(
            pd.to_datetime(training_dates)
        )
        self.best_model = min(
            self.evaluations, key=lambda n: float(self.evaluations[n].rmse)
        )
        self.worst_model = max(
            self.evaluations, key=lambda n: float(self.evaluations[n].rmse)
        )
        if self.best_model == self.worst_model:
            raise ValueError("at least two benchmark models are required")
        self.figure = None
        self.axes = None

    @staticmethod
    def _history_frame(dataset):
        frame = dataset.data.copy()
        if "ds" not in frame or "microVolt" not in frame:
            raise ValueError("dataset must contain ds and microVolt columns")
        frame["ds"] = pd.to_datetime(frame["ds"])
        return frame.sort_values("ds").reset_index(drop=True)

    def _training_dates(self, history):
        if self.training_dates is not None:
            return self.training_dates
        first = next(iter(self.evaluations.values()))
        evaluation_dates = pd.DatetimeIndex(pd.to_datetime(first.dates))
        if len(evaluation_dates) == 0:
            raise ValueError("evaluation dates must not be empty")
        return pd.DatetimeIndex(
            history.loc[history["ds"] < evaluation_dates[0], "ds"]
        )

    @staticmethod
    def _aligned_fitted(fitted, training_dates):
        if fitted is None:
            raise ValueError("historical fitted values are required")
        values = np.asarray(fitted, dtype=float).reshape(-1)
        if len(values) != len(training_dates):
            raise ValueError("historical fitted length must equal training-date length")
        return pd.Series(values, index=training_dates)

    def plot(self):
        history = self._history_frame(self.dataset)
        training_dates = self._training_dates(history)

        first_result = next(
            iter(self.evaluations.values())
        )

        holdout_dates = pd.DatetimeIndex(
            pd.to_datetime(first_result.dates)
        )

        holdout_actual = np.asarray(
            first_result.actual,
            dtype=float,
        ).reshape(-1)

        if len(holdout_dates) != len(holdout_actual):
            raise ValueError(
                "holdout actual values must align "
                "with evaluation dates"
            )

        evaluation_start = holdout_dates[0]

        # ---------------------------------------------------------
        # Measured development history
        # ---------------------------------------------------------
        #
        # Preserve raw measurement dates. Do not manufacture
        # measurements at quarters introduced by preprocessing.
        #
        measured_development = history.loc[
            history["ds"] < evaluation_start,
            ["ds", "microVolt"],
        ].copy()

        if measured_development.empty:
            raise ValueError(
                "measured development history must not be empty"
            )

        # ---------------------------------------------------------
        # Append the REAL measured holdout
        # ---------------------------------------------------------

        measured_holdout = pd.DataFrame(
            {
                "ds": holdout_dates,
                "microVolt": holdout_actual,
            }
        )

        measured_all = (
            pd.concat(
                [
                    measured_development,
                    measured_holdout,
                ],
                ignore_index=True,
            )
            .drop_duplicates(
                subset="ds",
                keep="last",
            )
            .sort_values("ds")
        )

        # ---------------------------------------------------------
        # Figure
        # ---------------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(13, 7)
        )

        # Principal line 1:
        # raw measured development + real measured holdout.
        ax.plot(
            measured_all["ds"],
            measured_all["microVolt"].astype(float),
            linewidth=1.8,
            label="Measured history + holdout",
        )

        # ---------------------------------------------------------
        # Best and worst model trajectories
        # ---------------------------------------------------------
        #
        # Historical part:
        #     ForecastResult.fitted
        #
        # Benchmark part:
        #     EvaluationResult.predicted
        #
        # No model execution occurs here.
        # ---------------------------------------------------------

        for role, name in (
            ("Best", self.best_model),
            ("Worst", self.worst_model),
        ):

            fitted = self._aligned_fitted(
                self.forecasts[name].fitted,
                training_dates,
            )

            result = self.evaluations[name]

            forecast_dates = pd.DatetimeIndex(
                pd.to_datetime(result.dates)
            )

            forecast_values = np.asarray(
                result.predicted,
                dtype=float,
            ).reshape(-1)

            if len(forecast_dates) != len(
                forecast_values
            ):
                raise ValueError(
                    f"{name}: frozen benchmark predictions "
                    "must align with evaluation dates"
                )

            if not np.isfinite(
                forecast_values
            ).all():
                raise ValueError(
                    f"{name}: frozen benchmark predictions "
                    "must be finite"
                )

            if (
                len(fitted) > 0
                and fitted.index[-1]
                >= forecast_dates[0]
            ):
                raise ValueError(
                    f"{name}: historical fitted period "
                    "must end before benchmark forecasts"
                )

            # Historical fitted trajectory followed by the
            # ORIGINAL frozen benchmark predictions.
            model_dates = (
                fitted.index.append(
                    forecast_dates
                )
            )

            model_values = np.concatenate(
                [
                    fitted.to_numpy(
                        dtype=float
                    ),
                    forecast_values,
                ]
            )

            ax.plot(
                model_dates,
                model_values,
                linewidth=1.5,
                label=(
                    f"{role}: {name} "
                    f"(RMSE="
                    f"{result.rmse:.6f})"
                ),
            )

        # ---------------------------------------------------------
        # Frozen development boundary
        # ---------------------------------------------------------

        development_end = (
            training_dates[-1]
        )

        ax.axvline(
            development_end,
            linestyle=":",
            linewidth=1.2,
            label=(
                "Development end: "
                f"{development_end:%Y-%m-%d}"
            ),
        )

        ax.set_title(
            self.TITLE
        )

        ax.set_xlabel(
            "Date"
        )

        ax.set_ylabel(
            "microVolt (µV)"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend()

        fig.tight_layout()

        self.figure = fig
        self.axes = ax

        return fig, ax


    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.figure is None:
            self.plot()
        self.figure.savefig(path, dpi=160, bbox_inches="tight")
        return path

    def __repr__(self):
        return (
            "BenchmarkBestWorstHistoryPlotter("
            f"best={self.best_model!r}, worst={self.worst_model!r})"
        )
