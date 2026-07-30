"""
Kalman + LSTM hybrid forecaster.
"""

from __future__ import annotations
from importlib.metadata import metadata

from matplotlib import dates

from zenerestimation.forecasting import ForecastResult

import pandas as pd


from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.forecasting.hybrid import trend
from zenerestimation.forecasting.neural.lstm import LSTMForecaster

from zenerestimation.forecasting import AdaptiveKalmanFilter

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
        """
        Forecast the trend component using
        the Adaptive Kalman Filter.
        """

        trend = self.filter.forecast(steps)

        dates = self.dataset.forecast_dates(steps)

        return ForecastResult(

            model="AdaptiveKalmanFilter",

            forecast=pd.Series(
                trend,
                index=dates,
            ),

            horizon=steps,

            dates=dates,

            metadata={

                "component": "trend",

                "filter": "AdaptiveKalmanFilter",

            },

        )


    def combine_forecasts(
        self,
        trend_result,
        residual_result,
    ):

        forecast = (
            trend_result.forecast
            + residual_result.forecast

        )

        metadata = dict(residual_result.metadata)

        metadata.update({

            "architecture": "KalmanLSTM",

            "trend": "Adaptive Kalman Filter",

            "residual": "LSTM",

        })

        return ForecastResult(

            model="KalmanLSTM",

            forecast=forecast,

            horizon=trend_result.horizon,

            dates=trend_result.dates,

            metadata=metadata,

        )
    

    def summary_metadata(self):

        return {

            "architecture": "KalmanLSTM",

            "trend": "Adaptive Kalman Filter",

        }