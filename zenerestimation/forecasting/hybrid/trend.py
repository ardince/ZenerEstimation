"""
Trend models used by hybrid forecasters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zenerestimation.forecasting import ForecastResult


class LinearTrendModel:
    """
    First-order linear trend model.

    Fits

        y = ax + b

    using least squares.
    """

    def __init__(self):

        self.coefficients = None

        self.n_samples = 0

        self.dataset = None

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def fit(
        self,
        dataset,
    ):
        """
        Fit a first-order linear trend.
        """

        self.dataset = dataset

        values = dataset.target.values.astype(float)

        x = np.arange(len(values))

        self.coefficients = np.polyfit(
            x,
            values,
            1,
        )

        self.n_samples = len(values)

        return self

    # ---------------------------------------------------------
    # In-sample trend
    # ---------------------------------------------------------

    def fitted(self):
        """
        Return fitted trend values.
        """

        if self.coefficients is None:
            raise RuntimeError(
                "LinearTrendModel has not been fitted."
            )

        a, b = self.coefficients

        x = np.arange(self.n_samples)

        return np.polyval(
            (a, b),
            x,
        )

    # ---------------------------------------------------------
    # Forecast
    # ---------------------------------------------------------

    def predict(
        self,
        steps,
    ):
        """
        Forecast future trend values.
        """

        if self.coefficients is None:
            raise RuntimeError(
                "LinearTrendModel has not been fitted."
            )

        a, b = self.coefficients

        x = np.arange(
            self.n_samples,
            self.n_samples + steps,
        )

        values = np.polyval(
            (a, b),
            x,
        )

        dates = self.dataset.forecast_dates(
            steps
        )

        forecast = pd.Series(
            values,
            index=dates,
        )

        return ForecastResult(

            model="LinearTrend",

            forecast=forecast,

            horizon=steps,

            dates=dates,

            metadata={

                "trend_slope": self.slope,

            },

        )

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def slope(self):
        """
        Return fitted trend slope.

        Returns
        -------
        float | None
        """

        if self.coefficients is None:
            return None

        return float(
            self.coefficients[0]
        )

    @property
    def intercept(self):
        """
        Return fitted intercept.

        Returns
        -------
        float | None
        """

        if self.coefficients is None:
            return None

        return float(
            self.coefficients[1]
        )

    @property
    def fitted_(self):
        """
        True after fitting.
        """

        return self.coefficients is not None

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self):
        """
        Return trend model information.
        """

        return {

            "model": "LinearTrend",

            "fitted": self.fitted_,

            "slope": self.slope,

            "intercept": self.intercept,

            "samples": self.n_samples,

        }