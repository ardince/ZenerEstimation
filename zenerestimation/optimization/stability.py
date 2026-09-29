from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SeedResult:
    """Evaluation result for one successful repeated neural seed."""

    seed: int
    score: float
    metrics: dict[str, float]
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(
            self.seed,
            int,
        ):
            raise TypeError("seed must be an integer")

        if isinstance(self.score, bool) or not isinstance(
            self.score,
            (int, float),
        ):
            raise TypeError("score must be numeric")

        if not np.isfinite(self.score):
            raise ValueError("score must be finite")

        if not isinstance(self.metrics, dict):
            raise TypeError(
                "metrics must be a dictionary"
            )

        if not self.metrics:
            raise ValueError(
                "metrics must not be empty"
            )

        normalized_metrics = {}

        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "metric names must be non-empty strings"
                )

            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    "metric values must be numeric"
                )

            if not np.isfinite(value):
                raise ValueError(
                    "metric values must be finite"
                )

            normalized_metrics[name] = float(value)

        if self.metadata is None:
            normalized_metadata = {}
        elif not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary or None"
            )
        else:
            normalized_metadata = deepcopy(
                self.metadata
            )

        object.__setattr__(
            self,
            "score",
            float(self.score),
        )

        object.__setattr__(
            self,
            "metrics",
            deepcopy(normalized_metrics),
        )

        object.__setattr__(
            self,
            "metadata",
            normalized_metadata,
        )

    @property
    def status(self) -> str:
        """Successful repeated-seed status."""

        return "evaluated"

    def to_dict(self) -> dict[str, Any]:
        """Serialize one successful seed result."""

        return {
            "seed": self.seed,
            "status": self.status,
            "score": self.score,
            "metrics": deepcopy(self.metrics),
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class FailedSeedResult:
    """Evidence for a repeated-seed run that could not be evaluated."""

    seed: int
    error_type: str
    error_message: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(
            self.seed,
            int,
        ):
            raise TypeError(
                "seed must be an integer"
            )

        if not isinstance(self.error_type, str):
            raise TypeError(
                "error_type must be a string"
            )

        if not self.error_type.strip():
            raise ValueError(
                "error_type must be non-empty"
            )

        if not isinstance(
            self.error_message,
            str,
        ):
            raise TypeError(
                "error_message must be a string"
            )

        if self.metadata is None:
            normalized_metadata = {}
        elif not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary or None"
            )
        else:
            normalized_metadata = deepcopy(
                self.metadata
            )

        object.__setattr__(
            self,
            "metadata",
            normalized_metadata,
        )

    @property
    def status(self) -> str:
        """Failed repeated-seed status."""

        return "failed"

    def to_dict(self) -> dict[str, Any]:
        """Serialize one failed seed result."""

        return {
            "seed": self.seed,
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class StabilityResult:
    """Repeated-seed stability evidence for a frozen configuration."""

    model: str
    metric: str
    params: dict[str, Any]
    seeds: tuple[
        SeedResult | FailedSeedResult,
        ...,
    ]
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # -------------------------------------------------------------
        # Model
        # -------------------------------------------------------------

        if not isinstance(self.model, str):
            raise TypeError(
                "model must be a string"
            )

        if not self.model.strip():
            raise ValueError(
                "model must be a non-empty string"
            )

        # -------------------------------------------------------------
        # Metric
        # -------------------------------------------------------------

        if not isinstance(self.metric, str):
            raise TypeError(
                "metric must be a string"
            )

        if not self.metric.strip():
            raise ValueError(
                "metric must be a non-empty string"
            )

        # -------------------------------------------------------------
        # Frozen model parameters
        # -------------------------------------------------------------

        if not isinstance(self.params, dict):
            raise TypeError(
                "params must be a dictionary"
            )

        if not self.params:
            raise ValueError(
                "params must not be empty"
            )

        # Seed is the replication dimension, not a frozen
        # hyperparameter.
        if "seed" in self.params:
            raise ValueError(
                "params must not contain seed"
            )

        # -------------------------------------------------------------
        # Seed results
        # -------------------------------------------------------------

        if not isinstance(self.seeds, tuple):
            raise TypeError(
                "seeds must be a tuple"
            )

        if not self.seeds:
            raise ValueError(
                "seeds must not be empty"
            )

        if not all(
            isinstance(
                result,
                (
                    SeedResult,
                    FailedSeedResult,
                ),
            )
            for result in self.seeds
        ):
            raise TypeError(
                "all seed entries must be SeedResult "
                "or FailedSeedResult instances"
            )

        seed_values = [
            result.seed
            for result in self.seeds
        ]

        if len(seed_values) != len(
            set(seed_values)
        ):
            raise ValueError(
                "duplicate seeds are not allowed"
            )

        # -------------------------------------------------------------
        # Metadata
        # -------------------------------------------------------------

        if self.metadata is None:
            normalized_metadata = {}
        elif not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary or None"
            )
        else:
            normalized_metadata = deepcopy(
                self.metadata
            )

        # -------------------------------------------------------------
        # Defensive copies
        # -------------------------------------------------------------

        object.__setattr__(
            self,
            "params",
            deepcopy(self.params),
        )

        object.__setattr__(
            self,
            "metadata",
            normalized_metadata,
        )

    @property
    def run_count(self) -> int:
        """Total number of attempted seed replications."""

        return len(self.seeds)

    @property
    def successful_seeds(
        self,
    ) -> tuple[SeedResult, ...]:
        """Successful seed evaluations in execution order."""

        return tuple(
            result
            for result in self.seeds
            if isinstance(
                result,
                SeedResult,
            )
        )

    @property
    def failed_seeds(
        self,
    ) -> tuple[FailedSeedResult, ...]:
        """Failed seed evaluations in execution order."""

        return tuple(
            result
            for result in self.seeds
            if isinstance(
                result,
                FailedSeedResult,
            )
        )

    @property
    def successful_count(self) -> int:
        """Number of successful seed evaluations."""

        return len(
            self.successful_seeds
        )

    @property
    def failed_count(self) -> int:
        """Number of failed seed evaluations."""

        return len(
            self.failed_seeds
        )

    @property
    def scores(self) -> tuple[float, ...]:
        """Successful seed scores in execution order."""

        return tuple(
            result.score
            for result in self.successful_seeds
        )

    def _require_successful_scores(
        self,
    ) -> tuple[float, ...]:
        """Return successful scores or reject an empty summary."""

        scores = self.scores

        if not scores:
            raise RuntimeError(
                "stability result contains no "
                "successful seed evaluations"
            )

        return scores

    @property
    def mean_score(self) -> float:
        """Mean score across successful seed evaluations."""

        return float(
            np.mean(
                self._require_successful_scores()
            )
        )

    @property
    def std_score(self) -> float:
        """Population standard deviation across successful seeds."""

        return float(
            np.std(
                self._require_successful_scores(),
                ddof=0,
            )
        )

    @property
    def min_score(self) -> float:
        """Minimum successful seed score."""

        return float(
            np.min(
                self._require_successful_scores()
            )
        )

    @property
    def max_score(self) -> float:
        """Maximum successful seed score."""

        return float(
            np.max(
                self._require_successful_scores()
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the repeated-seed stability evidence."""

        return {
            "status": "evaluated",
            "model": self.model,
            "metric": self.metric,
            "params": deepcopy(
                self.params
            ),
            "run_count": self.run_count,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "summary": {
                "mean": self.mean_score,
                "std": self.std_score,
                "min": self.min_score,
                "max": self.max_score,
            },
            "seeds": [
                result.to_dict()
                for result in self.seeds
            ],
            "metadata": deepcopy(
                self.metadata
            ),
        }