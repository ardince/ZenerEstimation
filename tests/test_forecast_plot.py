import pandas as pd

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.forecasting.result import ForecastResult
from zenerestimation.visualization.forecast import ForecastPlot

import matplotlib
matplotlib.use("Agg")


def create_dataset():

    df = pd.DataFrame(
        {
            "ds": [
                "01/01/2024",
                "01/04/2024",
                "01/07/2024",
            ],
            "microVolt": [
                180.0,
                179.0,
                178.0,
            ],
        }
    )

    dataset = BatteryDataset(df)

    dataset.prepare()

    return dataset


def create_result():

    dates = pd.to_datetime(
        [
            "2024-10-01",
            "2025-01-01",
        ]
    )

    forecast = pd.Series(
        [
            177.5,
            177.0,
        ]
    )

    return ForecastResult(
        model="ARIMA",
        forecast=forecast,
        horizon=2,
        dates=dates,
        metadata={},
    )


def test_plot():

    dataset = create_dataset()

    result = create_result()

    plot = ForecastPlot(
        dataset,
        result,
    )

    fig = plot.plot()

    assert fig is not None


def test_figure_property():

    dataset = create_dataset()

    result = create_result()

    plot = ForecastPlot(
        dataset,
        result,
    )

    fig = plot.plot()

    assert plot.figure is fig


def test_axes_property():

    dataset = create_dataset()

    result = create_result()

    plot = ForecastPlot(
        dataset,
        result,
    )

    plot.plot()

    assert plot.axes is not None


def test_save(tmp_path):

    dataset = create_dataset()

    result = create_result()

    plot = ForecastPlot(
        dataset,
        result,
    )

    filename = tmp_path / "forecast.png"

    plot.save(filename)

    assert filename.exists()


import matplotlib.pyplot as plt


def test_show(monkeypatch):

    called = False

    def fake_show():
        nonlocal called
        called = True

    monkeypatch.setattr(
        plt,
        "show",
        fake_show,
    )

    dataset = create_dataset()

    result = create_result()

    plot = ForecastPlot(
        dataset,
        result,
    )

    plot.show()

    assert called


def test_initial_state():

    dataset = create_dataset()

    result = create_result()

    plot = ForecastPlot(
        dataset,
        result,
    )

    assert plot.figure is None

    assert plot.axes is None


