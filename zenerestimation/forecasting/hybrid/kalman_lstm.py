"""
Kalman + LSTM hybrid forecaster.
"""

from __future__ import annotations
from importlib.metadata import metadata

from zenerestimation.forecasting import ForecastResult

import pandas as pd
import numpy as np

from zenerestimation.data.dataset import BatteryDataset

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

        # Step 1: Add internal attributes to store the trend and residuals
        self._trend = None
        self._residual = None
        self._residual_dataset = None

        self._trend_forecast = None

        self._trend_prediction = None
        self._residual_prediction = None
        self._hybrid_prediction = None


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

        # Step 2: Rewrite the dataset with residuals
        self.filter.fit(dataset)

        self.dataset = dataset

        self._trend = self.filter.smooth()

        values = dataset.target.to_numpy(dtype=float)

        residual = values - self._trend

        self._residual = residual

        # Step 3: Build the residual dataset
        residual_df = dataset.data.copy()

        residual_df["microVolt"] = residual

        self._residual_dataset = BatteryDataset(residual_df)

        return self._residual_dataset


    def trend(self):
        return self._trend

    def residuals(self):
        return self._residual

    def residual_dataset(self):
        return self._residual_dataset


    def verify_decomposition(self, atol=1e-8):

        values = self.dataset.target.to_numpy(dtype=float)

        reconstructed = self._trend + self._residual

        return np.allclose(
            values,
            reconstructed,
            atol=atol,
        )


    def forecast_trend(
        self,
        steps,
    ):
        """
        Forecast the trend component using
        the Adaptive Kalman Filter.
        """

        dates = self.dataset.forecast_dates(steps)

        trend_values = self.filter.forecast(steps)

        result = ForecastResult(

            model="AdaptiveKalmanFilter",

            forecast=pd.Series(
                trend_values,
                index=dates,
            ),

            horizon=steps,

            dates=dates,

            metadata={

                "component": "trend",

                "filter": "AdaptiveKalmanFilter",

            },

        )

        # Cache the ForecastResult
        self._trend_forecast = result

        return result


    def trend_forecast(self):
        """
        Return the cached Kalman trend forecast.
        """

        return self._trend_forecast


    def combine_forecasts(
        self,
        trend_result,
        residual_result,
    ):

        """
        Combine the Kalman trend forecast with the
        LSTM residual forecast.

        The hybrid forecast is

            hybrid = trend + residual

        Both component forecasts are cached for later
        diagnostics and uncertainty estimation.
        """

        # ---------------------------------------------------------
        # Extract forecast values
        # ---------------------------------------------------------

        trend_values = np.asarray(
            trend_result.forecast.values,
            dtype=float,
        )

        residual_values = np.asarray(
            residual_result.forecast.values,
            dtype=float,
        )

        # ---------------------------------------------------------
        # Validate dimensions
        # ---------------------------------------------------------

        if len(trend_values) != len(residual_values):

            raise ValueError(

                "Trend and residual forecasts must "
                "have identical forecast horizons."

            )

        # ---------------------------------------------------------
        # Hybrid forecast
        # ---------------------------------------------------------

        hybrid_values = (

            trend_values

                +

             residual_values

        )

        # ---------------------------------------------------------
        # Cache future predictions
        # ---------------------------------------------------------

        self._trend_prediction = trend_values.copy()

        self._residual_prediction = residual_values.copy()

        self._hybrid_prediction = hybrid_values.copy()

        # ---------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------

        metadata = dict(residual_result.metadata)

        metadata.update(

            {

                "architecture": "KalmanLSTM",

                "trend_model": self.filter.__class__.__name__,

                "residual_model": self.lstm.__class__.__name__,

                "combination": "additive",

                "window": self.window,

            }

        )

        # ---------------------------------------------------------
        # Result
        # ---------------------------------------------------------

        return ForecastResult(

            model="KalmanLSTM",

            forecast=pd.Series(

                hybrid_values,

                index=trend_result.dates,

            ),

            fitted=pd.Series(
                self._trend + self._residual,
                index=self.dataset.data["ds"],
            ),

            horizon=trend_result.horizon,

            dates=trend_result.dates,

            metadata=metadata,

        )

    

    def summary_metadata(self):

        return {

            "architecture": "KalmanLSTM",

            "trend": "Adaptive Kalman Filter",

        }