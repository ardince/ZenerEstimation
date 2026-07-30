from __future__ import annotations

import numpy as np
import pandas as pd

from examples.dataset_demo import summary
from zenerestimation.data import BatteryDataset, dataset
from zenerestimation.forecasting import ForecastResult
from zenerestimation.forecasting.hybrid import (
    LinearTrendLSTMForecaster,
)


# =========================================================
# Dummy residual model
# =========================================================

class DummyResidualModel:
    """
    Minimal forecasting model used for unit testing.

    It always predicts zero residuals.
    """

    def __init__(self):

        self.dataset = None

    def fit(self, dataset):

        self.dataset = dataset

        return self

    def predict(self, steps):

        dates = self.dataset.forecast_dates(steps)

        forecast = pd.Series(
            np.zeros(steps),
            index=dates,
        )

        return ForecastResult(

            model="DummyResidual",

            forecast=forecast,

            horizon=steps,

            dates=dates,

        )


# =========================================================
# Dataset fixture
# =========================================================

def make_dataset():

    dates = pd.date_range(

        "2020-01-01",

        periods=8,

        freq="QS",

    )

    values = np.array(

        [10, 11, 12, 13, 14, 15, 16, 17],

        dtype=float,

    )

    df = pd.DataFrame({

        "ds": dates,

        "microVolt": values,

    })

    dataset = BatteryDataset(df)

    dataset.metadata = {

        "battery_id": "TEST",

    }

    return dataset


# =========================================================
# Constructor
# =========================================================

def test_constructor():

    model = LinearTrendLSTMForecaster(

        DummyResidualModel()

    )

    assert model.lstm is not None

    assert model.residual_model is not None


# =========================================================
# Residual generation
# =========================================================

def test_prepare_residuals():

    dataset = make_dataset()

    model = LinearTrendLSTMForecaster(

        DummyResidualModel()

    )

    residual_dataset = model.prepare_residuals(

        dataset

    )

    reconstructed = (
        model.trend
        +
        residual_dataset.target.values
    )

    np.testing.assert_allclose(
        reconstructed,
        dataset.target.values,
        atol=1e-8,
    )


# =========================================================
# Metadata
# =========================================================

def test_summary():

    model = LinearTrendLSTMForecaster(

        DummyResidualModel()

    )

    summary = model.summary()
    
    assert summary["family"] == "Hybrid"

    assert summary["architecture"] == "LinearTrendLSTM"

    assert summary["trend"] == "Linear Regression"

    #assert summary["residual"] == "LSTM"


# =========================================================
# Fit
# =========================================================

def test_fit():

    dataset = make_dataset()

    model = LinearTrendLSTMForecaster(

        DummyResidualModel()

    )

    model.fit(dataset)

    assert model.dataset is dataset

    assert model.residual_model.dataset is not None

# =========================================================
# Forecast combination
# =========================================================

def test_combine_forecasts():

    dataset = make_dataset()

    model = LinearTrendLSTMForecaster(

        DummyResidualModel()

    )

    trend_dates = dataset.forecast_dates(2)

    trend = ForecastResult(

        model="Trend",

        forecast=pd.Series(

            [20.0, 21.0],

            index=trend_dates,

        ),

        horizon=2,

        dates=trend_dates,

    )

    residual = ForecastResult(

        model="Residual",

        forecast=pd.Series(

            [1.0, -1.0],

            index=trend_dates,

        ),

        horizon=2,

        dates=trend_dates,

    )

    result = model.combine_forecasts(

        trend,

        residual,

    )

    np.testing.assert_allclose(

        result.forecast.values,

        [21.0, 20.0],

    )


# =========================================================
# Fit + Predict
# =========================================================

def test_fit_predict():

    dataset = make_dataset()

    model = LinearTrendLSTMForecaster(

        DummyResidualModel()

    )

    model.fit(dataset)

    result = model.predict(3)

    assert len(result.forecast) == 3

    assert result.model == "LinearTrendLSTM"