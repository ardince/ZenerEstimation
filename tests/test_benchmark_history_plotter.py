
"""Tests for historical optimized benchmark visualization."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.evaluation.result import EvaluationResult
from zenerestimation.visualization.benchmark_history import (
    BenchmarkHistoryPlotter,
)


@dataclass
class DummyForecast:
    model: str
    fitted: pd.Series
    forecast: tuple
    dates: tuple


def create_dataset():
    frame = pd.DataFrame(
        {
            "ds": [
                "01/03/2022",
                "01/06/2022",
                "01/09/2022",
                "01/12/2022",
                "01/03/2023",
                "01/06/2023",
            ],
            "microVolt": [
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
            ],
        }
    )

    dataset = BatteryDataset(frame)
    dataset.prepare()

    return dataset


def create_evidence():
    dates = (
        pd.Timestamp("2023-03-01"),
        pd.Timestamp("2023-06-01"),
    )

    evaluations = {}
    forecasts = {}

    for name, fitted, predicted in [
        (
            "ARIMA",
            [10.0, 11.1, 12.1, 13.1],
            [13.8, 14.8],
        ),
        (
            "LSTM",
            [np.nan, np.nan, 12.2, 13.2],
            [13.7, 14.7],
        ),
    ]:
        actual = np.array([14.0, 15.0])
        predicted_array = np.array(predicted)

        evaluations[name] = EvaluationResult(
            model=name,
            evaluation_steps=2,
            rmse=float(
                np.sqrt(
                    np.mean(
                        (actual - predicted_array) ** 2
                    )
                )
            ),
            mae=float(
                np.mean(
                    np.abs(
                        actual - predicted_array
                    )
                )
            ),
            mape=float(
                np.mean(
                    np.abs(
                        (actual - predicted_array) / actual
                    )
                ) * 100
            ),
            actual=tuple(actual),
            predicted=tuple(predicted),
            dates=dates,
        )

        forecasts[name] = DummyForecast(
            model=name,
            fitted=pd.Series(fitted),
            forecast=tuple(predicted),
            dates=dates,
        )

    return evaluations, forecasts


@pytest.fixture
def plotter():
    evaluations, forecasts = create_evidence()

    instance = BenchmarkHistoryPlotter(
        create_dataset(),
        evaluations=evaluations,
        forecasts=forecasts,
    )

    yield instance

    if instance.figure is not None:
        plt.close(instance.figure)


def test_constructor_accepts_matching_evidence(plotter):
    assert plotter.models == ["ARIMA", "LSTM"]
    assert plotter.evaluation_steps == 2


def test_evaluations_property_is_defensive(plotter):
    copied = plotter.evaluations
    copied.clear()

    assert len(plotter.evaluations) == 2


def test_results_alias(plotter):
    assert plotter.results == plotter.evaluations


def test_forecasts_property_is_defensive(plotter):
    copied = plotter.forecasts
    copied.clear()

    assert len(plotter.forecasts) == 2


def test_plot_returns_figure(plotter):
    fig = plotter.plot()

    assert fig is plotter.figure
    assert plotter.axes is not None


def test_plot_contains_historical_fits(plotter):
    plotter.plot()

    labels = [
        line.get_label()
        for line in plotter.axes.lines
    ]

    assert "ARIMA fitted (in-sample)" in labels
    assert "LSTM fitted (in-sample)" in labels


def test_historical_fitted_values_are_not_recalculated(plotter):
    plotter.plot()

    line = next(
        line
        for line in plotter.axes.lines
        if line.get_label() == "ARIMA fitted (in-sample)"
    )

    np.testing.assert_array_equal(
        line.get_ydata(),
        [10.0, 11.1, 12.1, 13.1],
    )


def test_unavailable_neural_fits_remain_nan(plotter):
    plotter.plot()

    line = next(
        line
        for line in plotter.axes.lines
        if line.get_label() == "LSTM fitted (in-sample)"
    )

    values = np.asarray(
        line.get_ydata(),
        dtype=float,
    )

    assert np.isnan(values[:2]).all()

    np.testing.assert_array_equal(
        values[2:],
        [12.2, 13.2],
    )


def test_forecast_uses_stored_predictions(plotter):
    plotter.plot()

    line = next(
        line
        for line in plotter.axes.lines
        if line.get_label() == "ARIMA forecast"
    )

    np.testing.assert_array_equal(
        line.get_ydata(),
        [13.8, 14.8],
    )

    # No artificial measured-history anchor.
    assert len(line.get_ydata()) == 2


def test_measured_reference_includes_benchmark(plotter):
    plotter.plot()

    line = next(
        line
        for line in plotter.axes.lines
        if line.get_label() == "Measured reference"
    )

    assert len(line.get_ydata()) == 6


def test_forecast_mismatch_is_rejected():
    evaluations, forecasts = create_evidence()

    forecasts["ARIMA"] = DummyForecast(
        model="ARIMA",
        fitted=pd.Series(
            [10.0, 11.1, 12.1, 13.1]
        ),
        forecast=(99.0, 99.0),
        dates=forecasts["ARIMA"].dates,
    )

    with pytest.raises(
        ValueError,
        match="predictions must match",
    ):
        BenchmarkHistoryPlotter(
            create_dataset(),
            evaluations,
            forecasts,
        )


def test_fitted_length_mismatch_is_rejected():
    evaluations, forecasts = create_evidence()

    forecasts["LSTM"] = DummyForecast(
        model="LSTM",
        fitted=pd.Series([12.2, 13.2]),
        forecast=forecasts["LSTM"].forecast,
        dates=forecasts["LSTM"].dates,
    )

    with pytest.raises(
        ValueError,
        match="fitted length",
    ):
        BenchmarkHistoryPlotter(
            create_dataset(),
            evaluations,
            forecasts,
        )


def test_missing_forecast_model_is_rejected():
    evaluations, forecasts = create_evidence()

    del forecasts["LSTM"]

    with pytest.raises(
        ValueError,
        match="model names must match",
    ):
        BenchmarkHistoryPlotter(
            create_dataset(),
            evaluations,
            forecasts,
        )


def test_forecast_date_mismatch_is_rejected():
    evaluations, forecasts = create_evidence()

    forecasts["ARIMA"] = DummyForecast(
        model="ARIMA",
        fitted=forecasts["ARIMA"].fitted,
        forecast=forecasts["ARIMA"].forecast,
        dates=(
            pd.Timestamp("2023-06-01"),
            pd.Timestamp("2023-09-01"),
        ),
    )

    with pytest.raises(
        ValueError,
        match="dates must match",
    ):
        BenchmarkHistoryPlotter(
            create_dataset(),
            evaluations,
            forecasts,
        )


def test_save_creates_figure(plotter, tmp_path):
    destination = (
        tmp_path
        / "benchmark"
        / "benchmark_history.png"
    )

    saved = plotter.save(destination)

    assert saved == destination
    assert destination.exists()
    assert destination.stat().st_size > 0


def test_repr(plotter):
    representation = repr(plotter)

    assert "BenchmarkHistoryPlotter" in representation
    assert "ARIMA" in representation
    assert "LSTM" in representation
    assert "evaluation_steps=2" in representation
