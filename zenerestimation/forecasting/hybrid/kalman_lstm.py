"""
Kalman + LSTM hybrid forecaster.
"""

from __future__ import annotations

from .base import BaseHybridForecaster


class KalmanLSTMForecaster(BaseHybridForecaster):
    """
    Hybrid forecasting model combining

    Adaptive Kalman Filter
    +
    Long Short-Term Memory network.
    """

    def __init__(
        self,
        kalman_model,
        lstm_model,
    ):

        super().__init__(

            trend_model=kalman_model,

            residual_model=lstm_model,

        )

    def prepare_residuals(
        self,
        dataset,
    ):

        raise NotImplementedError

    def combine_forecasts(
        self,
        trend_result,
        residual_result,
    ):

        raise NotImplementedError

    def summary_metadata(self):

        return {

            "architecture": "Kalman-LSTM",

        }