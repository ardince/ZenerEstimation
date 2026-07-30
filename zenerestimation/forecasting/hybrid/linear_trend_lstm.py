"""
Linear Trend + LSTM hybrid forecaster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zenerestimation.data import BatteryDataset
from zenerestimation.forecasting import ForecastResult

from .base import BaseHybridForecaster


class LinearTrendLSTMForecaster(BaseHybridForecaster):
    """
    Hybrid forecaster based on

        Linear Trend
              +
            LSTM
    """

    def __init__(
        self,
        lstm_model,
        window=6,
    ):

        super().__init__(
            residual_model=lstm_model,
        )

        self.window = window

        self.lstm = lstm_model

        # fitted linear trend
        self.trend_coef = None
        self.trend = None

    # ---------------------------------------------------------
    # Residual preparation
    # ---------------------------------------------------------

    def prepare_residuals(
        self,
        dataset,
    ):
        """
        residual = measured − linear trend
        """

        values = dataset.target.values.astype(float)

        t = np.arange(len(values))

        self.trend_coef = np.polyfit(
            t,
            values,
            1,
        )

        self.trend = np.polyval(
            self.trend_coef,
            t,
        )

        residual = values - self.trend

        residual_df = dataset.data.copy()

        residual_df["microVolt"] = residual

        residual_dataset = BatteryDataset(
            residual_df,
        )

        residual_dataset.metadata = dataset.metadata.copy()

        return residual_dataset

    # ---------------------------------------------------------
    # Trend forecast
    # ---------------------------------------------------------

    def forecast_trend(
        self,
        steps,
    ):
        """
        Forecast the fitted linear trend.
        """

        n = len(self.dataset)

        future_t = np.arange(
            n,
            n + steps,
        )

        forecast = np.polyval(
            self.trend_coef,
            future_t,
        )

        dates = self.dataset.forecast_dates(
            steps,
        )

        return ForecastResult(

            model="LinearTrend",

            forecast=pd.Series(
                forecast,
                index=dates,
            ),

            horizon=steps,

            dates=dates,

            metadata={

                "component": "trend",

            },

        )

    # ---------------------------------------------------------
    # Combination
    # ---------------------------------------------------------

    def combine_forecasts(
        self,
        trend_result,
        residual_result,
    ):

        forecast = (
            trend_result.forecast.values
            +
            residual_result.forecast.values
        )

        return ForecastResult(

            model="LinearTrendLSTM",

            forecast=pd.Series(
                forecast,
                index=trend_result.dates,
            ),

            horizon=len(forecast),

            dates=trend_result.dates,

            metadata={

                "architecture": "LinearTrendLSTM",

            },

        )

    # ---------------------------------------------------------

    def summary_metadata(self):

        return {

            "architecture": "LinearTrendLSTM",

            "trend": "Linear Regression",

            "degree": 1,

            "window": self.window,

        }