"""
Prognostic result container.

Stores the output of Remaining Useful Life (RUL)
and future prognostic analyses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PrognosticResult:
    """
    Result returned by a prognostic algorithm.
    """

    model: str

    threshold: float

    samples: np.ndarray

    mean: float

    median: float

    p10: float

    p90: float

    metadata: dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------

    @property
    def sample_count(self) -> int:

        return len(self.samples)

    @property
    def spread(self) -> float:

        return self.p90 - self.p10

    # -----------------------------------------------------
    # Serialization
    # -----------------------------------------------------

    def summary(self) -> dict:

        return {

            "model": self.model,

            "threshold": float(self.threshold),

            "mean": float(self.mean),

            "median": float(self.median),

            "p10": float(self.p10),

            "p90": float(self.p90),

            "sample_count": self.sample_count,

            "spread": float(self.spread),

            "metadata": self.metadata,

        }