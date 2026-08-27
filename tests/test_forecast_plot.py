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


def test_forecast_plot_accepts_evaluation():

    dataset = create_dataset()

    result = create_result()

    evaluation = {
        "evaluation_steps": 6,
        "rmse": 0.42,
        "mae": 0.31,
        "mape": 1.08,
    }

    plot = ForecastPlot(
        dataset,
        result,
        evaluation=evaluation,
    )

    assert plot.evaluation == evaluation


def test_experiment_info_contains_metrics():

    dataset = create_dataset()

    result = create_result()

    evaluation = {
        "evaluation_steps": 6,
        "rmse": 0.42,
        "mae": 0.31,
        "mape": 1.08,
    }

    plot = ForecastPlot(
        dataset,
        result,
        evaluation=evaluation,
    )

    info = plot._build_experiment_info()

    assert "Holdout: 6 quarters" in info
    assert "RMSE: 0.4200" in info
    assert "Missing Periods:" in info

    assert "MAE" not in info
    assert "MAPE" not in info

def test_experiment_info_handles_missing_metrics():

    dataset = create_dataset()

    result = create_result()

    plot = ForecastPlot(
        dataset,
        result,
        evaluation={
            "rmse": None,
            "mae": None,
            "mape": None,
        },
    )

    info = plot._build_experiment_info()

    assert "RMSE" not in info
    assert "MAE" not in info
    assert "MAPE" not in info


def test_plot_without_evaluation():

    dataset = create_dataset()

    result = create_result()

    plot = ForecastPlot(
        dataset,
        result,
    )

    figure = plot.plot()

    assert figure is not None


def test_custom_title_is_used():

    dataset = create_dataset()
    result = create_result()

    plot = ForecastPlot(
        dataset,
        result,
    )

    plot.plot(
        title="Custom Forecast Title"
    )

    assert (
        plot.axes.get_title()
        == "Custom Forecast Title"
    )