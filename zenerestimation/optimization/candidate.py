"""
Optimization candidate result.

Defines the immutable result produced by evaluating
one forecasting-model parameter configuration.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateResult:
    """Result of evaluating one optimization candidate."""

    params: dict[str, Any]
    score: float | None
    metrics: dict[str, float]
    status: str = "evaluated"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate and defensively copy candidate result data."""

        # ---------------------------------------------------------
        # Common validation
        # ---------------------------------------------------------

        if not isinstance(self.params, dict):
            raise TypeError(
                "params must be a dictionary"
            )

        if not self.params:
            raise ValueError(
                "params must not be empty"
            )

        if not isinstance(self.metrics, dict):
            raise TypeError(
                "metrics must be a dictionary"
            )

        if not isinstance(self.status, str):
            raise TypeError(
                "status must be a string"
            )

        if not self.status:
            raise ValueError(
                "status must not be empty"
            )

        if self.status not in {
            "evaluated",
            "failed",
        }:
            raise ValueError(
                "status must be 'evaluated' or 'failed'"
            )

        if (
            self.metadata is not None
            and not isinstance(
                self.metadata,
                dict,
            )
        ):
            raise TypeError(
                "metadata must be a dictionary or None"
            )

        # ---------------------------------------------------------
        # Evaluated candidate contract
        # ---------------------------------------------------------

        if self.status == "evaluated":

            if self.score is None:
                raise ValueError(
                    "evaluated candidate requires a score"
                )

            try:
                score = float(
                    self.score
                )
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "score must be numeric"
                ) from exc

            if not np.isfinite(score):
                raise ValueError(
                    "score must be finite"
                )

            if not self.metrics:
                raise ValueError(
                    "evaluated candidate requires metrics"
                )

            for name, value in self.metrics.items():

                if (
                    not isinstance(name, str)
                    or not name
                ):
                    raise ValueError(
                        "metric names must be "
                        "non-empty strings"
                    )

                try:
                    metric_value = float(
                        value
                    )
                except (
                    TypeError,
                    ValueError,
                ) as exc:
                    raise TypeError(
                        "metric values must be numeric"
                    ) from exc

                if not np.isfinite(
                    metric_value
                ):
                    raise ValueError(
                        "metric values must be finite"
                    )

        # ---------------------------------------------------------
        # Failed candidate contract
        # ---------------------------------------------------------

        else:

            if self.score is not None:
                raise ValueError(
                    "failed candidate score must be None"
                )

            if self.metrics:
                raise ValueError(
                    "failed candidate metrics must be empty"
                )

        # ---------------------------------------------------------
        # Defensive copies
        #
        # frozen=True prevents normal assignment, so use
        # object.__setattr__ during initialization.
        # ---------------------------------------------------------

        object.__setattr__(
            self,
            "params",
            deepcopy(
                self.params
            ),
        )

        object.__setattr__(
            self,
            "metrics",
            deepcopy(
                self.metrics
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            (
                {}
                if self.metadata is None
                else deepcopy(
                    self.metadata
                )
            ),
        )

    @classmethod
    def failed(
        cls,
        *,
        params: dict[str, Any],
        error: Exception,
        metadata: dict[str, Any] | None = None,
    ) -> "CandidateResult":
        """Create a failed candidate result."""

        failure_metadata = (
            {}
            if metadata is None
            else deepcopy(
                metadata
            )
        )

        failure_metadata.update(
            {
                "error_type": (
                    type(error).__name__
                ),
                "error_message": str(
                    error
                ),
            }
        )

        return cls(
            params=params,
            score=None,
            metrics={},
            status="failed",
            metadata=failure_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable candidate representation."""

        return {
            "status": self.status,
            "params": deepcopy(
                self.params
            ),
            "score": (
                float(self.score)
                if self.score is not None
                else None
            ),
            "metrics": deepcopy(
                self.metrics
            ),
            "metadata": deepcopy(
                self.metadata
            ),
        }

    def __repr__(self) -> str:
        return (
            "CandidateResult("
            f"status={self.status!r}, "
            f"params={self.params!r}, "
            f"score={self.score!r}"
            ")"
        )