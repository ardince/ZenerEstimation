"""Human-readable reporting for multi-model benchmark evidence."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import numpy as np
import pandas as pd

from zenerestimation.evaluation.result import EvaluationResult

from zenerestimation.comparison import (
    ComparisonResult,
)


class BenchmarkReport:
    """Render stored multi-model benchmark evidence.

    BenchmarkReport is a presentation layer around ComparisonResult.

    It does not evaluate models, recompute metrics, create rankings,
    select models, optimize parameters, or select neural seeds.
    """

    def __init__(
        self,
        result: ComparisonResult,
        evaluations: Mapping[str, EvaluationResult] | None = None,
    ) -> None:

        if not isinstance(
            result,
            ComparisonResult,
        ):
            raise TypeError(
                "result must be a ComparisonResult"
            )

        if evaluations is not None:
            if not isinstance(evaluations, Mapping):
                raise TypeError("evaluations must be a mapping")
            if set(evaluations) != set(result.models):
                raise ValueError("evaluations must match comparison models")
            for name, evaluation in evaluations.items():
                if not isinstance(evaluation, EvaluationResult):
                    raise TypeError("evaluations must contain EvaluationResult")
                if evaluation.model != name:
                    raise ValueError("evaluation model name mismatch")
                dates = pd.DatetimeIndex(pd.to_datetime(evaluation.dates))
                n = evaluation.evaluation_steps
                if (n < 1 or len(dates) != n or
                    len(evaluation.actual) != n or
                    len(evaluation.predicted) != n):
                    raise ValueError("evaluation forecast lengths mismatch")
                if (dates.hasnans or not dates.is_unique or
                    not dates.is_monotonic_increasing):
                    raise ValueError("invalid evaluation dates")
                if not (np.isfinite(np.asarray(evaluation.actual, dtype=float)).all()
                        and np.isfinite(np.asarray(evaluation.predicted, dtype=float)).all()):
                    raise ValueError("nonfinite evaluation values")
            first = evaluations[result.models[0]]
            for name in result.models[1:]:
                other = evaluations[name]
                if (tuple(pd.to_datetime(other.dates)) != tuple(pd.to_datetime(first.dates))
                    or not np.array_equal(np.asarray(other.actual, dtype=float),
                                          np.asarray(first.actual, dtype=float))):
                    raise ValueError("inconsistent benchmark observations")
        self._evaluations = None if evaluations is None else dict(evaluations)
        self._result = result

    @property
    def result(
        self,
    ) -> ComparisonResult:
        """Return the underlying comparison result."""

        return self._result

    def render(
        self,
    ) -> str:
        """Return a human-readable benchmark report."""

        lines: list[str] = []

        lines.extend(
            self._header_lines()
        )

        lines.append("")

        lines.extend(
            self._metric_lines()
        )

        lines.append("")

        lines.extend(
            self._best_model_lines()
        )

        if self._evaluations is not None:
            lines.append("")
            lines.extend(self._forecast_lines())

        lines.append("")

        lines.extend(
            self._metadata_lines()
        )

        return "\n".join(lines) + "\n"

    def _header_lines(
        self,
    ) -> list[str]:
        """Build report heading."""

        return [
            "OPTIMIZED MULTI-MODEL BENCHMARK",
            "=" * 31,
            f"Battery: {self._result.battery}",
        ]

    def _metric_lines(
        self,
    ) -> list[str]:
        """Build standardized benchmark metric table."""

        lines = [
            "METRICS",
            "",
            (
                f"{'Model':<20}"
                f"{'RMSE':>12}"
                f"{'MAE':>12}"
                f"{'MAPE (%)':>14}"
            ),
            "-" * 58,
        ]

        metrics = self._result.metrics

        for model in self._result.models:

            values = metrics[
                model
            ]

            lines.append(
                f"{model:<20}"
                f"{values['rmse']:>12.6f}"
                f"{values['mae']:>12.6f}"
                f"{values['mape']:>14.6f}"
            )

        return lines

    def _best_model_lines(
        self,
    ) -> list[str]:
        """Render best-model fields already stored in the result."""

        best_models = (
            self._result.best_models
        )

        lines = [
            "BEST BY STORED METRIC",
        ]

        for metric in (
            "rmse",
            "mae",
            "mape",
        ):
            if metric not in best_models:
                continue

            lines.append(
                f"{metric.upper()}: "
                f"{best_models[metric]}"
            )

        return lines

    def _forecast_lines(self) -> list[str]:
        """Render stored holdout predictions and signed errors."""
        lines = [
            "DETAILED HOLDOUT FORECASTS",
            "Delta (uV) = Actual - Predicted; positive means underprediction.",
        ]
        for name in self._result.models:
            evidence = self._evaluations[name]
            lines.extend([
                "",
                f"{name}:",
                f"{'Step':<6}{'Date':<14}{'Actual (uV)':>16}"
                f"{'Predicted (uV)':>18}{'Delta (uV)':>18}",
                "-" * 72,
            ])
            for step, (date, actual, predicted) in enumerate(
                zip(evidence.dates, evidence.actual, evidence.predicted, strict=True),
                start=1,
            ):
                actual, predicted = float(actual), float(predicted)
                lines.append(
                    f"{step:<6}{pd.Timestamp(date):%Y-%m-%d}    "
                    f"{actual:>16.6f}{predicted:>18.6f}"
                    f"{actual - predicted:>+18.6f}"
                )
        return lines

    def _metadata_lines(
        self,
    ) -> list[str]:
        """Render stored comparison metadata."""

        lines = [
            "METADATA",
        ]

        metadata = (
            self._result.metadata
        )

        if not metadata:
            lines.append(
                "None"
            )

            return lines

        for key, value in metadata.items():

            lines.append(
                f"{key}: {value}"
            )

        return lines

    def save(
        self,
        filename: str | Path,
    ) -> Path:
        """Write the rendered report to disk."""

        filename = Path(
            filename
        )

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename.write_text(
            self.render(),
            encoding="utf-8",
        )

        return filename

    def __repr__(
        self,
    ) -> str:

        return (
            "BenchmarkReport("
            f"battery={self._result.battery!r}, "
            f"models={len(self._result.models)}"
            ")"
        )