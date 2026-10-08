"""Visualization of standardized multi-model benchmark evidence."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from zenerestimation.evaluation import (
    EvaluationResult,
)


class BenchmarkPlotter:
    """Plot stored benchmark actuals and model predictions.

    BenchmarkPlotter is a presentation layer around existing
    EvaluationResult objects.

    It does not fit models, generate forecasts, calculate metrics,
    rank models, select models, optimize parameters, or select seeds.
    """

    def __init__(
        self,
        results: Mapping[
            str,
            EvaluationResult,
        ],
    ) -> None:

        if not isinstance(
            results,
            Mapping,
        ):
            raise TypeError(
                "results must be a mapping"
            )

        if not results:
            raise ValueError(
                "at least one benchmark result is required"
            )

        normalized = dict(
            results
        )

        for name, result in (
            normalized.items()
        ):
            if not isinstance(
                result,
                EvaluationResult,
            ):
                raise TypeError(
                    "all benchmark results must be "
                    "EvaluationResult instances"
                )

            if name != result.model:
                raise ValueError(
                    "benchmark mapping key must match "
                    "EvaluationResult model"
                )

        self._validate_common_holdout(
            normalized
        )

        self._results = normalized

        self._figure = None
        self._axes = None

    @staticmethod
    def _validate_common_holdout(
        results: dict[
            str,
            EvaluationResult,
        ],
    ) -> None:
        """Require all results to describe one benchmark holdout."""

        values = tuple(
            results.values()
        )

        reference = values[0]

        for result in values[1:]:

            if (
                result.evaluation_steps
                != reference.evaluation_steps
            ):
                raise ValueError(
                    "all benchmark results must have "
                    "identical evaluation_steps"
                )

            if tuple(
                result.dates
            ) != tuple(
                reference.dates
            ):
                raise ValueError(
                    "all benchmark results must have "
                    "identical dates"
                )

            if not np.array_equal(
                np.asarray(
                    result.actual,
                    dtype=float,
                ),
                np.asarray(
                    reference.actual,
                    dtype=float,
                ),
            ):
                raise ValueError(
                    "all benchmark results must have "
                    "identical actual values"
                )

    @property
    def results(
        self,
    ) -> dict[str, EvaluationResult]:
        """Return a defensive copy of benchmark results."""

        return deepcopy(
            self._results
        )

    @property
    def models(
        self,
    ) -> list[str]:
        """Return benchmark model names in stored order."""

        return list(
            self._results.keys()
        )

    @property
    def evaluation_steps(
        self,
    ) -> int:
        """Return the common benchmark horizon."""

        first = next(
            iter(
                self._results.values()
            )
        )

        return first.evaluation_steps

    @property
    def figure(self):
        """Return the current matplotlib Figure."""

        return self._figure

    @property
    def axes(self):
        """Return the current matplotlib Axes."""

        return self._axes

    def plot(
        self,
        title: str = (
            "Optimized Multi-Model Benchmark"
        ),
    ):
        """Plot actual holdout values and stored model predictions."""

        first = next(
            iter(
                self._results.values()
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        self._figure = fig
        self._axes = ax

        ax.plot(
            first.dates,
            first.actual,
            marker="o",
            linewidth=2,
            label="Actual",
        )

        for model, result in (
            self._results.items()
        ):
            ax.plot(
                result.dates,
                result.predicted,
                marker="s",
                linestyle="--",
                linewidth=1.8,
                label=model,
            )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            "Date"
        )

        ax.set_ylabel(
            "Voltage (µV)"
        )

        ax.grid(
            True
        )

        ax.legend(
            loc="best"
        )

        fig.autofmt_xdate()

        return fig

    def save(
        self,
        filename: str | Path,
        *,
        dpi: int = 300,
    ) -> Path:
        """Save the benchmark forecast figure."""

        filename = Path(
            filename
        )

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

    def __repr__(
        self,
    ) -> str:

        return (
            "BenchmarkPlotter("
            f"models={len(self._results)}, "
            f"steps={self.evaluation_steps}"
            ")"
        )