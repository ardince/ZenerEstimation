"""
Hybrid forecasting models.

Hybrid forecasters combine two independent forecasting models:

- Trend model: captures the deterministic long-term behaviour of the time series
- Residual model: learns the nonlinear forecasting error

Both components expose the same public forecasting API.
"""

from .base import BaseHybridForecaster
from .trend import LinearTrendModel
from .linear_trend_lstm import LinearTrendLSTMForecaster
from .kalman_lstm import KalmanLSTMForecaster

__all__ = [

    "BaseHybridForecaster",

    "LinearTrendModel",

    "LinearTrendLSTMForecaster",

    "KalmanLSTMForecaster"

]