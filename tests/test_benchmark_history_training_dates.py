
"""Regression tests for explicit fitted training dates."""

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


def make_evidence(
    *,
    calendar_index=False,
    extra_observation=False,
):
    dates = pd.date_range(
        "2022-03-01",
        periods=7 if extra_observation else 6,
        freq="QS-MAR",
    )

    frame = pd.DataFrame({
        "ds": dates,
        "microVolt": np.arange(
            10.0,
            10.0 + len(dates),
        ),
    })

    dataset = BatteryDataset(frame)
    dataset.prepare()

    training_dates = pd.DatetimeIndex(
        dates[:4]
    )

    benchmark_dates = tuple(dates[4:6])

    evaluations = {}
    forecasts = {}

    for name, fitted, predicted in (
        ("ARIMA", [10.0, 11.1, 12.1, 13.1], (13.8, 14.8)),
        ("LSTM", [np.nan, np.nan, 12.2, 13.2], (13.7, 14.7)),
    ):
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
                    np.abs(actual - predicted_array)
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
            predicted=predicted,
            dates=benchmark_dates,
        )

        fitted_index = (
            training_dates
            if calendar_index
            else pd.RangeIndex(4)
        )

        forecasts[name] = DummyForecast(
            model=name,
            fitted=pd.Series(
                fitted,
                index=fitted_index,
            ),
            forecast=predicted,
            dates=benchmark_dates,
        )

    return (
        dataset,
        evaluations,
        forecasts,
        training_dates,
    )


def test_explicit_training_dates_are_accepted():
    dataset, evaluations, forecasts, dates = make_evidence()

    plotter = BenchmarkHistoryPlotter(
        dataset,
        evaluations,
        forecasts,
        training_dates=dates,
    )

    assert plotter.training_dates.equals(dates)


def test_calendar_indexed_fits_are_accepted():
    dataset, evaluations, forecasts, dates = make_evidence(
        calendar_index=True
    )

    plotter = BenchmarkHistoryPlotter(
        dataset,
        evaluations,
        forecasts,
        training_dates=dates,
    )

    fig = plotter.plot()

    assert fig is plotter.figure

    plt.close(fig)


def test_fitted_dates_are_used_for_plotting():
    dataset, evaluations, forecasts, dates = make_evidence()

    plotter = BenchmarkHistoryPlotter(
        dataset,
        evaluations,
        forecasts,
        training_dates=dates,
    )

    try:
        plotter.plot()

        line = next(
            line
            for line in plotter.axes.lines
            if line.get_label()
            == "ARIMA fitted (in-sample)"
        )

        assert pd.DatetimeIndex(
            pd.to_datetime(line.get_xdata())
        ).equals(dates)

    finally:
        if plotter.figure is not None:
            plt.close(plotter.figure)


def test_wrong_calendar_index_is_rejected():
    dataset, evaluations, forecasts, dates = make_evidence(
        calendar_index=True
    )

    original = forecasts["ARIMA"]

    forecasts["ARIMA"] = DummyForecast(
        model="ARIMA",
        fitted=pd.Series(
            original.fitted.to_numpy(),
            index=dates + pd.Timedelta(days=1),
        ),
        forecast=original.forecast,
        dates=original.dates,
    )

    with pytest.raises(
        ValueError,
        match="calendar index",
    ):
        BenchmarkHistoryPlotter(
            dataset,
            evaluations,
            forecasts,
            training_dates=dates,
        )


def test_wrong_positional_index_is_rejected():
    dataset, evaluations, forecasts, dates = make_evidence()

    original = forecasts["ARIMA"]

    forecasts["ARIMA"] = DummyForecast(
        model="ARIMA",
        fitted=pd.Series(
            original.fitted.to_numpy(),
            index=pd.Index([5, 6, 7, 8]),
        ),
        forecast=original.forecast,
        dates=original.dates,
    )

    with pytest.raises(
        ValueError,
        match="positional index",
    ):
        BenchmarkHistoryPlotter(
            dataset,
            evaluations,
            forecasts,
            training_dates=dates,
        )


def test_training_dates_cannot_enter_holdout():
    dataset, evaluations, forecasts, dates = make_evidence()

    shifted_dates = dates + pd.DateOffset(months=3)

    with pytest.raises(
        ValueError,
        match="training_dates",
    ):
        BenchmarkHistoryPlotter(
            dataset,
            evaluations,
            forecasts,
            training_dates=shifted_dates,
        )


def test_extra_post_benchmark_observations_are_hidden():
    dataset, evaluations, forecasts, dates = make_evidence(
        extra_observation=True
    )

    plotter = BenchmarkHistoryPlotter(
        dataset,
        evaluations,
        forecasts,
        training_dates=dates,
    )

    try:
        plotter.plot()

        reference_line = next(
            line
            for line in plotter.axes.lines
            if line.get_label() == "Measured reference"
        )

        assert len(reference_line.get_ydata()) == 6

    finally:
        if plotter.figure is not None:
            plt.close(plotter.figure)


def test_default_training_dates_remain_supported():
    dataset, evaluations, forecasts, dates = make_evidence()

    plotter = BenchmarkHistoryPlotter(
        dataset,
        evaluations,
        forecasts,
    )

    assert plotter.training_dates.equals(dates)


def test_training_dates_property_is_defensive():
    dataset, evaluations, forecasts, dates = make_evidence()

    plotter = BenchmarkHistoryPlotter(
        dataset,
        evaluations,
        forecasts,
        training_dates=dates,
    )

    returned = plotter.training_dates

    assert returned is not plotter.training_dates
    assert returned.equals(dates)


def test_fitted_length_must_match_training_dates():
    dataset, evaluations, forecasts, dates = make_evidence()

    original = forecasts["LSTM"]

    forecasts["LSTM"] = DummyForecast(
        model="LSTM",
        fitted=pd.Series([12.2, 13.2]),
        forecast=original.forecast,
        dates=original.dates,
    )

    with pytest.raises(
        ValueError,
        match="fitted length",
    ):
        BenchmarkHistoryPlotter(
            dataset,
            evaluations,
            forecasts,
            training_dates=dates,
        )
