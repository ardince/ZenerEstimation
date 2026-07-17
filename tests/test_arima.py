import pandas as pd

from zenerestimation.data import BatteryDataset
from zenerestimation.forecasting import ForecastResult
from zenerestimation.forecasting.arima import ARIMAForecaster


def create_dataset():

    df = pd.DataFrame(
        {
            "ds": [
                "01/01/2023",
                "01/04/2023",
                "01/07/2023",
                "01/10/2023",
                "01/01/2024",
                "01/04/2024",
                "01/07/2024",
                "01/10/2024",
            ],
            "microVolt":
            [
                100.0,
                101.2,
                100.9,
                102.5,
                103.0,
                104.1,
                103.8,
                105.4,
            ]
        }
    )

    return BatteryDataset(df)


def test_fit():

    dataset = create_dataset()

    model = ARIMAForecaster()

    model.fit(dataset)

    assert model.result is not None


def test_predict():

    dataset = create_dataset()

    model = ARIMAForecaster()

    model.fit(dataset)

    result = model.predict(steps=4)

    assert isinstance(
        result,
        ForecastResult,
    )

    assert len(result) == 4


def test_fit_predict():

    dataset = create_dataset()

    model = ARIMAForecaster()

    result = model.fit_predict(
        dataset,
        steps=3,
    )

    assert isinstance(
        result,
        ForecastResult,
    )

    assert result.model == "ARIMA"

    assert result.horizon == 3


def test_metadata():

    dataset = create_dataset()

    model = ARIMAForecaster()

    result = model.fit_predict(
        dataset,
        steps=2,
    )

    assert "order" in result.metadata

    assert "aic" in result.metadata

    assert "bic" in result.metadata


def test_summary():

    dataset = create_dataset()

    model = ARIMAForecaster()

    model.fit(dataset)

    summary = model.summary

    assert summary is not None
