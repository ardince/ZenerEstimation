"""
Hybrid forecasting models.

Hybrid forecasters combine two independent forecasting models:

- Trend model: captures the deterministic long-term behaviour of the time series
- Residual model: learns the nonlinear forecasting error

Both components expose the same public forecasting API.
"""

from .base import BaseHybridForecaster

__all__ = [
    "BaseHybridForecaster",
]