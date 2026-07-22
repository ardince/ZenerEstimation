"""
High-level Remaining Useful Life analysis.

Combines threshold estimation and Monte Carlo simulation
into a single user-facing API.
"""

from __future__ import annotations

from .montecarlo import MonteCarloRUL
from .threshold import ThresholdEstimator


class RULAnalyzer:
    """
    High-level Remaining Useful Life estimator.

    This class is the recommended public interface for
    prognostic analysis.
    """

    def __init__(
        self,
        simulator=None,
    ):

        self.simulator = simulator or MonteCarloRUL()

    def analyze(
        self,
        *,
        current_value,
        current_drift,
        sigma,
        values,
        threshold=None,
        model="Unknown",
        metadata=None,
    ):

        if threshold is None:

            threshold = ThresholdEstimator.default(values)

        return self.simulator.run(

            current_value=current_value,

            current_drift=current_drift,

            sigma=sigma,

            threshold=threshold,

            model=model,

            metadata=metadata,

        )