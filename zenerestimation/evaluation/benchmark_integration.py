"""Integration helpers for optimized benchmark evidence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from zenerestimation.comparison import (
    ForecastComparison,
)
from zenerestimation.evaluation.result import (
    EvaluationResult,
)
from zenerestimation.utils.result_loader import (
    ResultPackage,
)


def comparison_from_evaluations(
    battery: str,
    results: Mapping[
        str,
        EvaluationResult,
    ],
):
    """Build ComparisonResult from stored benchmark evaluations.

    This adapter performs no forecasting and calculates no metrics.
    EvaluationResult metrics are transferred into the established
    ResultPackage contract and comparison is delegated entirely to
    ForecastComparison.
    """

    if (
        not isinstance(battery, str)
        or not battery.strip()
    ):
        raise ValueError(
            "battery must be a non-empty string"
        )

    if not isinstance(
        results,
        Mapping,
    ):
        raise TypeError(
            "results must be a mapping"
        )

    if not results:
        raise ValueError(
            "at least one evaluation result is required"
        )

    runs = []

    for index, (
        model,
        result,
    ) in enumerate(
        results.items(),
        start=1,
    ):
        if not isinstance(
            result,
            EvaluationResult,
        ):
            raise TypeError(
                "all results must be EvaluationResult instances"
            )

        if model != result.model:
            raise ValueError(
                "result mapping key must match "
                "EvaluationResult model"
            )

        runs.append(
            ResultPackage(
                battery=battery,
                model=model,
                timestamp="benchmark",
                run_number=index,
                directory=Path(
                    "benchmark"
                )
                / model,
                forecast=None,
                evaluation={
                    "rmse": result.rmse,
                    "mae": result.mae,
                    "mape": result.mape,
                },
                experiment=None,
                report=None,
                figure=None,
                log=None,
            )
        )

    comparison = ForecastComparison(
        runs
    )

    return comparison.compare()