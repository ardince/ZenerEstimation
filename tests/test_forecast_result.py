import pandas as pd

from zenerestimation.forecasting import ForecastResult


def test_forecast_result():

    dates = pd.date_range(
        "2025-01-01",
        periods=3,
        freq="QS",
    )

    forecast = pd.Series(
        [170.1, 170.8, 171.4]
    )

    result = ForecastResult(
        model="Dummy",
        forecast=forecast,
        horizon=3,
        dates=dates,
    )

    assert len(result) == 3

    assert result.model == "Dummy"

    assert result.horizon == 3


def test_dataframe():

    dates = pd.date_range(
        "2025-01-01",
        periods=2,
        freq="QS",
    )

    result = ForecastResult(
        model="Dummy",
        forecast=pd.Series([1.0, 2.0]),
        horizon=2,
        dates=dates,
    )

    df = result.to_dataframe()

    assert list(df.columns) == [
        "ds",
        "forecast",
    ]

    assert len(df) == 2


def test_summary():

    dates = pd.date_range(
        "2025-01-01",
        periods=4,
        freq="QS",
    )

    result = ForecastResult(
        model="Dummy",
        forecast=pd.Series([1, 2, 3, 4]),
        horizon=4,
        dates=dates,
    )

    summary = result.summary()

    assert summary["model"] == "Dummy"

    assert summary["points"] == 4