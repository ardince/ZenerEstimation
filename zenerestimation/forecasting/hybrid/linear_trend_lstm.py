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
        window=None,
    ):

        super().__init__(
            residual_model=lstm_model,
        )

        self.lstm = lstm_model

        residual_window = getattr(
            self.lstm,
            "window",
            None,
        )


        if (
            window is not None
            and residual_window is not None
            and int(window) != residual_window
        ):
            raise ValueError(
                "window must match the residual "
                "LSTM model window"
            )


        # fitted linear trend
        self.trend_coef = None
        self.trend = None

        # decomposition state used by hybrid diagnostics
        self._trend = None
        self._residual = None

    # ---------------------------------------------------------
    # Residual preparation
    # ---------------------------------------------------------

    @property
    def window(self):
        """
        Neural residual-model window size.

        Returns None when the supplied residual model does
        not expose a window configuration.
        """

        
        return getattr(
            self.lstm,
            "window",
            None,
        )
    

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


        self._trend = np.asarray(
            self.trend,
            dtype=float,
        ).copy()

        self._residual = np.asarray(
            residual,
            dtype=float,
        ).copy()


        residual_df = dataset.data.copy()

        residual_df["microVolt"] = residual

        residual_dataset = BatteryDataset(
            residual_df,
        )

        # Preserve dataset provenance.
        residual_dataset._battery = dataset.battery
        residual_dataset._source_type = dataset.source_type
        residual_dataset._source_path = dataset.source_path

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
    # Historical fitted values
    # ---------------------------------------------------------

    def fitted_values(self):
        """
        Return historical fitted values of the complete hybrid.

        The fitted hybrid is constructed as

            fitted linear trend
            +
            fitted residual LSTM

        Neural warm-up NaNs are intentionally preserved.

        Returns None when historical fitted state is not
        available, allowing component forecasts to be combined
        independently of a prior fit().
        """

        if self.dataset is None:
            return None

        if self.trend is None:
            return None

        residual_fitted = getattr(
            self.residual_model,
            "fitted",
            None,
        )

        if residual_fitted is None:
            return None

        trend_values = np.asarray(
            self.trend,
            dtype=float,
        ).reshape(-1)

        residual_values = np.asarray(
            residual_fitted,
            dtype=float,
        ).reshape(-1)

        if len(trend_values) != len(residual_values):
            raise ValueError(
                "historical hybrid component length mismatch: "
                f"trend={len(trend_values)}, "
                f"residual={len(residual_values)}"
            )

        fitted = (
            trend_values
            +
            residual_values
        )

        dates = pd.DatetimeIndex(
            self.dataset.data["ds"]
        )

        return pd.Series(
            fitted,
            index=dates,
            name="fitted",
        )


    # ---------------------------------------------------------
    # Combination
    # ---------------------------------------------------------

    def combine_forecasts(
        self,
        trend_result,
        residual_result,
    ):

        self.validate_component_forecasts(
            trend_result,
            residual_result,
        )

        forecast = (
            trend_result.forecast.values
            +
            residual_result.forecast.values
        )

        fitted = self.fitted_values()


        return ForecastResult(

            model="LinearTrendLSTM",

            forecast=pd.Series(
                forecast,
                index=trend_result.dates,
            ),

            fitted=fitted,

            horizon=len(forecast),

            dates=trend_result.dates,

            metadata={

                "architecture": "LinearTrendLSTM",

            },

        )

    # ---------------------------------------------------------

    def summary_metadata(self):
        """
        Hybrid-specific model metadata.

        Neural window metadata is included only when the
        supplied residual model exposes a window size.
        """

        metadata = {
            "architecture": "LinearTrendLSTM",
            "trend": "Linear Regression",
            "degree": 1,
        }

        if self.window is not None:
            metadata["window"] = self.window

        return metadata