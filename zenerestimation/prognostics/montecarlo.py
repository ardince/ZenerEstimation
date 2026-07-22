"""
Generic Monte Carlo Remaining Useful Life engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .result import PrognosticResult


@dataclass
class MonteCarloRUL:
    """
    Generic Monte Carlo Remaining Useful Life estimator.

    The simulator is intentionally independent from any
    forecasting algorithm (ARIMA, Kalman, LSTM, ETS, ...).
    """

    simulations: int = 3000

    dt: float = 0.25

    max_years: float = 100.0

    fast_probability: float = 0.20

    fast_noise: float = 0.001

    slow_noise: float = 0.0001

    random_seed: int | None = None

    def run(
        self,
        *,
        current_value: float,
        current_drift: float,
        sigma: float,
        threshold: float,
        model: str = "Unknown",
        metadata: dict | None = None,
    ) -> PrognosticResult:

        rng = np.random.default_rng(self.random_seed)

        samples = np.empty(self.simulations)

        for i in range(self.simulations):

            value = float(current_value)
            drift = float(current_drift)

            years = 0.0

            while value < threshold and years < self.max_years:

                if rng.random() < self.fast_probability:
                    drift += rng.normal(0.0, self.fast_noise)
                else:
                    drift += rng.normal(0.0, self.slow_noise)

                value += drift * self.dt

                value += rng.normal(0.0, sigma)

                years += self.dt

            samples[i] = years

        return PrognosticResult(

            model=model,

            threshold=float(threshold),

            samples=samples,

            mean=float(np.mean(samples)),

            median=float(np.median(samples)),

            p10=float(np.percentile(samples, 10)),

            p90=float(np.percentile(samples, 90)),

            metadata=metadata or {},

        )