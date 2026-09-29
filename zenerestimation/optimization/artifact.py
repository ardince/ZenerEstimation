"""Standardized optimization artifact representation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .candidate import CandidateResult
from .result import OptimizationResult


class OptimizationArtifact:
    """Build a standardized artifact from an optimization result.

    The artifact contains model-selection evidence only.

    Final benchmark metrics, predictions, actual values, and errors
    belong to EvaluationResult and must not be included here.
    """

    def __init__(
        self,
        result: OptimizationResult,
        *,
        benchmark_steps: int | None = None,
    ) -> None:
        if not isinstance(result, OptimizationResult):
            raise TypeError(
                "result must be an OptimizationResult"
            )

        if benchmark_steps is not None:
            if (
                isinstance(benchmark_steps, bool)
                or not isinstance(benchmark_steps, int)
            ):
                raise TypeError(
                    "benchmark_steps must be an integer or None"
                )

            if benchmark_steps <= 0:
                raise ValueError(
                    "benchmark_steps must be positive"
                )

        self._result = result
        self._benchmark_steps = benchmark_steps

    @property
    def result(self) -> OptimizationResult:
        """Return the underlying optimization result."""

        return self._result

    @property
    def benchmark_steps(self) -> int | None:
        """Return the optional final benchmark horizon."""

        return self._benchmark_steps

    @property
    def best_candidate(self) -> CandidateResult:
        """Return the evaluated candidate selected as best."""

        for candidate in self._result.candidates:
            if (
                candidate.status == "evaluated"
                and candidate.params == self._result.best_params
                and candidate.score is not None
                #and candidate.score == self._result.best_score
                and np.isclose(
                    candidate.score,
                    self._result.best_score,
                )
            ):
                return candidate

        raise RuntimeError(
            "best candidate could not be resolved "
            "from OptimizationResult"
        )

    @property
    def validation_steps(self) -> int | None:
        """Return the internal temporal validation horizon."""

        metadata = self.best_candidate.metadata

        value = metadata.get(
            "validation_steps"
        )

        if value is None:
            return None

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                "candidate validation_steps metadata "
                "must be a positive integer"
            )

        return value

    @property
    def horizon_aligned(self) -> bool | None:
        """Return whether internal and benchmark horizons match."""

        validation_steps = self.validation_steps

        if (
            validation_steps is None
            or self._benchmark_steps is None
        ):
            return None

        return (
            validation_steps
            == self._benchmark_steps
        )

    def _search_payload(self) -> dict[str, Any]:
        """Build standardized search metadata."""

        metadata = self._result.metadata

        return {
            "method": metadata.get(
                "search",
                "unknown",
            ),
            "objective": metadata.get(
                "objective",
                "unknown",
            ),
            "candidate_count": (
                self._result.candidate_count
            ),
            "successful_count": (
                self._result.successful_count
            ),
            "failed_count": (
                self._result.failed_count
            ),
        }

    def _validation_payload(
        self,
    ) -> dict[str, Any]:
        """Build standardized validation metadata."""

        metadata = self.best_candidate.metadata

        return {
            "method": metadata.get(
                "validation"
            ),
            "folds": metadata.get(
                "folds"
            ),
            "validation_steps": (
                self.validation_steps
            ),
            "benchmark_steps": (
                self._benchmark_steps
            ),
            "horizon_aligned": (
                self.horizon_aligned
            ),
        }

    def _best_payload(self) -> dict[str, Any]:
        """Build standardized best-candidate information."""

        candidate = self.best_candidate

        return {
            "params": deepcopy(
                candidate.params
            ),
            "score": float(
                candidate.score
            ),
            "metrics": deepcopy(
                candidate.metrics
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the standardized optimization artifact."""

        payload = {
            "status": "optimized",
            "model": self._result.model,
            "metric": self._result.metric,
            "search": self._search_payload(),
            "validation": (
                self._validation_payload()
            ),
            "best": self._best_payload(),
            "candidates": [
                candidate.to_dict()
                for candidate
                in self._result.candidates
            ],
            "metadata": deepcopy(
                self._result.metadata
            ),
        }

        return deepcopy(payload)

    def __repr__(self) -> str:
        return (
            "OptimizationArtifact("
            f"model={self._result.model!r}, "
            f"metric={self._result.metric!r}, "
            f"candidates="
            f"{self._result.candidate_count}, "
            f"benchmark_steps="
            f"{self._benchmark_steps!r}"
            ")"
        )