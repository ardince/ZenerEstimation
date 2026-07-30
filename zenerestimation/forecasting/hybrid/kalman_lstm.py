"""
Kalman + LSTM hybrid forecaster.
"""

from __future__ import annotations
#from curses import window

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.forecasting.neural.lstm import LSTMForecaster

from zenerestimation.forecasting import AdaptiveKalmanFilter

from .base import BaseHybridForecaster

#import pytest

#@pytest.mark.skip(reason="KalmanLSTMForecaster not implemented yet")


class KalmanLSTMForecaster(BaseHybridForecaster):
    """
    Hybrid forecasting model combining

    Adaptive Kalman Filter
    +
    Long Short-Term Memory network.
    """

    def __init__(
        self,
        window=6,
        kalman_model=None,
        lstm_model=None,
    ):

        if kalman_model is None:

            kalman_model = AdaptiveKalmanFilter()

        if lstm_model is None:

            lstm_model = LSTMForecaster(
            window=window,
    )

        super().__init__(

            residual_model=lstm_model,

        )

        self.filter = kalman_model

        self.lstm = lstm_model

        self.window = window

        self.trend = None

        self.residual = None


    # ---------------------------------------------------------
    # Residual preparation
    # ---------------------------------------------------------

    def prepare_residuals(
        self,
        dataset,
    ):
        """
        residual = measurement − Kalman trend
        """

        self.filter.fit(dataset)

        self.trend = self.filter.smooth()

        values = dataset.target.to_numpy(dtype=float)

        residual = values - self.trend

        self.residual = residual

        residual_df = dataset.data.copy()

        residual_df["microVolt"] = residual

        residual_dataset = BatteryDataset(
        residual_df,
        )

        residual_dataset.metadata = dataset.metadata.copy()

        return residual_dataset


    def forecast_trend(
        self,
        steps,
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

            "architecture": "KalmanLSTM",

            "trend": "Adaptive Kalman Filter",

        }