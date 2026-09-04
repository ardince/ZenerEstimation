"""
Evaluation result container.

This module defines the standardized result object used by the
forecast evaluation framework.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvaluationResult:
    """
    Immutable container for forecast evaluation results.

    Parameters
    ----------
    model:
        Name of the forecasting model.

    evaluation_steps:
        Number of observations used for holdout evaluation.

    rmse:
        Root Mean Squared Error.

    mae:
        Mean Absolute Error.

    mape:
        Mean Absolute Percentage Error, expressed as a percentage.

    actual:
        Observed holdout values.

    predicted:
        Forecast values corresponding to ``actual``.

    dates:
        Dates associated with the holdout observations.

    metadata:
        Optional additional evaluation metadata.
    """

    model: str
    evaluation_steps: int
    rmse: float
    mae: float
    mape: float
    actual: tuple[float, ...]
    predicted: tuple[float, ...]
    dates: tuple[Any, ...]
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """
        Validate and normalize the evaluation result.
        """

        if not self.model:
            raise ValueError(
                "model must be a non-empty string"
            )

        if self.evaluation_steps <= 0:
            raise ValueError(
                "evaluation_steps must be greater than zero"
            )

        if len(self.actual) != self.evaluation_steps:
            raise ValueError(
                "actual length must match evaluation_steps"
            )

        if len(self.predicted) != self.evaluation_steps:
            raise ValueError(
                "predicted length must match evaluation_steps"
            )

        if len(self.dates) != self.evaluation_steps:
            raise ValueError(
                "dates length must match evaluation_steps"
            )

        for name, value in (
            ("rmse", self.rmse),
            ("mae", self.mae),
            ("mape", self.mape),
        ):
            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite"
                )

        # Defensive copy because the dataclass is frozen but the
        # dictionary supplied by the caller may still be mutable.
        object.__setattr__(
            self,
            "metadata",
            deepcopy(self.metadata)
            if self.metadata is not None
            else {},
        )

    @property
    def horizon(self) -> int:
        """
        Alias for the holdout evaluation length.
        """

        return self.evaluation_steps

    def summary(self) -> dict[str, Any]:
        """
        Return a compact summary of the evaluation.
        """

        return {
            "model": self.model,
            "evaluation_steps": self.evaluation_steps,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the complete evaluation result to a dictionary.
        """

        return {
            "status": "evaluated",
            "method": "holdout",
            "model": self.model,
            "evaluation_steps": self.evaluation_steps,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "actual": list(self.actual),
            "predicted": list(self.predicted),
            "dates": [
                str(date)
                for date in self.dates
            ],
            "metadata": deepcopy(self.metadata),
        }

    def __repr__(self) -> str:
        return (
            "EvaluationResult("
            f"model={self.model!r}, "
            f"steps={self.evaluation_steps}, "
            f"rmse={self.rmse:.6f}, "
            f"mae={self.mae:.6f}, "
            f"mape={self.mape:.6f}"
            ")"
        )