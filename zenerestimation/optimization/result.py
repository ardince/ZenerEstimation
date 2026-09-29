"""
Optimization result container.

Defines the immutable standardized result produced by a
forecasting-model optimization search.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .candidate import CandidateResult


@dataclass(frozen=True)
class OptimizationResult:
    """
    Complete result of a model optimization search.

    Parameters
    ----------
    model:
        Human-readable forecasting model name.

    metric:
        Metric used to select the best candidate.

    best_params:
        Parameter configuration selected by the search.

    best_score:
        Best candidate score.

    candidates:
        Candidate results produced by the search.

    metadata:
        Optional optimization metadata.
    """

    model: str
    metric: str
    best_params: dict[str, Any]
    best_score: float
    candidates: tuple[CandidateResult, ...]
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """
        Validate and normalize the optimization result.
        """

        if not isinstance(self.model, str) or not self.model:
            raise ValueError(
                "model must be a non-empty string"
            )

        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError(
                "metric must be a non-empty string"
            )

        if not isinstance(self.best_params, dict):
            raise TypeError(
                "best_params must be a dictionary"
            )

        if not self.best_params:
            raise ValueError(
                "best_params cannot be empty"
            )

        if not np.isfinite(self.best_score):
            raise ValueError(
                "best_score must be finite"
            )

        if not isinstance(self.candidates, tuple):
            raise TypeError(
                "candidates must be a tuple"
            )

        if not self.candidates:
            raise ValueError(
                "candidates cannot be empty"
            )

        if not all(
            isinstance(candidate, CandidateResult)
            for candidate in self.candidates
        ):
            raise TypeError(
                "candidates must contain only CandidateResult objects"
            )

        if (
            self.metadata is not None
            and not isinstance(self.metadata, dict)
        ):
            raise TypeError(
                "metadata must be a dictionary or None"
            )

        matching_candidates = [
            candidate
            for candidate in self.candidates
            if (
                candidate.status
                == "evaluated"
                and candidate.params
                == self.best_params
                and candidate.score
                is not None
                and np.isclose(
                    candidate.score,
                    self.best_score,
                )
            )
        ]

        if not matching_candidates:
            raise ValueError(
                "best_params and best_score must match "
                "one of the candidates"
            )

        object.__setattr__(
            self,
            "best_params",
            deepcopy(self.best_params),
        )

        object.__setattr__(
            self,
            "metadata",
            deepcopy(self.metadata)
            if self.metadata is not None
            else {},
        )

    @property
    def candidate_count(self) -> int:
        """
        Number of evaluated candidates.
        """

        return len(self.candidates)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the complete optimization result to a dictionary.
        """

        return {
            "status": "optimized",
            "model": self.model,
            "metric": self.metric,
            "best_params": deepcopy(self.best_params),
            "best_score": float(self.best_score),
            "candidate_count": self.candidate_count,
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "metadata": deepcopy(self.metadata),
        }

    def __repr__(self) -> str:
        return (
            "OptimizationResult("
            f"model={self.model!r}, "
            f"metric={self.metric!r}, "
            f"best_score={self.best_score:.6f}, "
            f"candidates={self.candidate_count}"
            ")"
        )


    @property
    def successful_count(
        self,
    ) -> int:

        return sum(
            candidate.status == "evaluated"
            for candidate in self.candidates
        )


    @property
    def failed_count(
        self,
    ) -> int:

        return sum(
            candidate.status == "failed"
            for candidate in self.candidates
        )