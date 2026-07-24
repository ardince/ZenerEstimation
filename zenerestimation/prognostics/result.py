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

    @property
    def simulations(self):
        return len(self.samples)
    
    @property
    def minimum(self):
        return float(np.min(self.samples))
    
    @property
    def maximum(self):
        return float(np.max(self.samples))

    # -----------------------------------------------------
    # Serialization
    # -----------------------------------------------------

    def summary(self):

        return {

            "model": self.model,

            "threshold": self.threshold,

            "mean": self.mean,

            "median": self.median,

            "p10": self.p10,

            "p90": self.p90,

            "minimum": float(self.samples.min()),

            "maximum": float(self.samples.max()),

            "simulations": len(self.samples),

            **self.metadata,

        }
    

    def to_dict(self):

        return {

            "model": self.model,

            "threshold": self.threshold,

            "mean": self.mean,

            "median": self.median,

            "p10": self.p10,

            "p90": self.p90,

            "samples": self.samples.tolist(),

            "metadata": self.metadata,

        }
    

    def __str__(self):

        return (
            f"PrognosticResult("
            f"mean={self.mean:.2f} years, "
            f"median={self.median:.2f}, "
            f"P10={self.p10:.2f}, "
            f"P90={self.p90:.2f})"
        )
