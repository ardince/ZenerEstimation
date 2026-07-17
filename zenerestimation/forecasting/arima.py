from __future__ import annotations

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from zenerestimation.forecasting.base import BaseForecastModel
from zenerestimation.forecasting.result import ForecastResult

class ARIMAForecaster(BaseForecastModel):

    @property
    def summary(self):
        """
        Return the fitted ARIMA summary.
        """
        return self.result.summary()


    def __init__(self, order=(1, 1, 1)):

        self.order = order

        self.model = None

        self.result = None

        self.dataset = None

    
    def fit(self, dataset):

        dataset.prepare()

        self.dataset = dataset

        series = dataset.target

        self.model = ARIMA(
            series,
            order=self.order,
        )

        self.result = self.model.fit()

        return self
    

    def predict(self, steps=1):

        forecast = self.result.forecast(
            steps=steps
        )

        last_date = self.dataset.data["ds"].iloc[-1]

        #freq = self.dataset.frequency
        freq = pd.infer_freq(self.dataset.data["ds"])

        if freq is None:
            freq = "QS-JAN"

        dates = pd.date_range(
            start=last_date,
            periods=steps + 1,
            freq=freq,
        )[1:]

        return ForecastResult(
            model="ARIMA",
            forecast=forecast,
            horizon=steps,
            dates=dates,
            metadata={
            "order": list(self.order),
            "aic": float(self.result.aic),
            "bic": float(self.result.bic),
            },
        )
    

        