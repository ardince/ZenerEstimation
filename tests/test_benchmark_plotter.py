"""Tests for standardized multi-model benchmark visualization."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest

from zenerestimation.evaluation import (
    EvaluationResult,
)
from zenerestimation.visualization import (
    BenchmarkPlotter,
)


matplotlib.use(
    "Agg",
    force=True,
)


# ============================================================
# Helpers
# ============================================================


def make_result(
    model,
    *,
    actual=(10.0, 11.0, 12.0),
    predicted=(10.1, 10.9, 12.2),
    dates=(
        "2026-03-01",
        "2026-06-01",
        "2026-09-01",
    ),
):
    return EvaluationResult(
        model=model,
        evaluation_steps=3,
        rmse=0.2,
        mae=0.15,
        mape=1.0,
        actual=actual,
        predicted=predicted,
        dates=dates,
        metadata={},
    )


@pytest.fixture
def benchmark_results():
    return {
        "ARIMA": make_result(
            "ARIMA",
            predicted=(
                10.1,
                11.1,
                12.1,
            ),
        ),
        "Kalman": make_result(
            "Kalman",
            predicted=(
                10.0,
                11.0,
                12.0,
            ),
        ),
        "LSTM": make_result(
            "LSTM",
            predicted=(
                9.9,
                10.8,
                11.8,
            ),
        ),
    }


# ============================================================
# Construction / validation
# ============================================================


def test_requires_mapping():

    with pytest.raises(
        TypeError,
        match="mapping",
    ):
        BenchmarkPlotter(
            ["not", "a", "mapping"]
        )


def test_rejects_empty_results():

    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        BenchmarkPlotter({})


def test_requires_evaluation_results():

    with pytest.raises(
        TypeError,
        match="EvaluationResult",
    ):
        BenchmarkPlotter(
            {
                "ARIMA": "not a result",
            }
        )


def test_requires_key_to_match_model():

    result = make_result(
        "ARIMA"
    )

    with pytest.raises(
        ValueError,
        match="model",
    ):
        BenchmarkPlotter(
            {
                "WrongName": result,
            }
        )


def test_requires_same_evaluation_steps():

    first = make_result(
        "ARIMA"
    )

    second = EvaluationResult(
        model="Kalman",
        evaluation_steps=2,
        rmse=0.2,
        mae=0.1,
        mape=1.0,
        actual=(10.0, 11.0),
        predicted=(10.1, 11.1),
        dates=(
            "2026-03-01",
            "2026-06-01",
        ),
        metadata={},
    )

    with pytest.raises(
        ValueError,
        match="evaluation_steps",
    ):
        BenchmarkPlotter(
            {
                "ARIMA": first,
                "Kalman": second,
            }
        )


def test_requires_identical_dates():

    first = make_result(
        "ARIMA"
    )

    second = make_result(
        "Kalman",
        dates=(
            "2026-03-01",
            "2026-06-01",
            "2026-12-01",
        ),
    )

    with pytest.raises(
        ValueError,
        match="dates",
    ):
        BenchmarkPlotter(
            {
                "ARIMA": first,
                "Kalman": second,
            }
        )


def test_requires_identical_actual_values():

    first = make_result(
        "ARIMA"
    )

    second = make_result(
        "Kalman",
        actual=(
            10.0,
            11.0,
            99.0,
        ),
    )

    with pytest.raises(
        ValueError,
        match="actual",
    ):
        BenchmarkPlotter(
            {
                "ARIMA": first,
                "Kalman": second,
            }
        )


def test_preserves_model_order(
    benchmark_results,
):

    plotter = BenchmarkPlotter(
        benchmark_results
    )

    assert plotter.models == [
        "ARIMA",
        "Kalman",
        "LSTM",
    ]


# ============================================================
# Plotting
# ============================================================


def test_plot_returns_figure(
    benchmark_results,
):

    plotter = BenchmarkPlotter(
        benchmark_results
    )

    fig = plotter.plot()

    try:
        assert isinstance(
            fig,
            matplotlib.figure.Figure,
        )
    finally:
        plt.close(fig)


def test_plot_exposes_figure_and_axes(
    benchmark_results,
):

    plotter = BenchmarkPlotter(
        benchmark_results
    )

    fig = plotter.plot()

    try:
        assert plotter.figure is fig
        assert plotter.axes is not None
    finally:
        plt.close(fig)


def test_plot_contains_actual_and_all_models(
    benchmark_results,
):

    plotter = BenchmarkPlotter(
        benchmark_results
    )

    fig = plotter.plot()

    try:
        labels = [
            line.get_label()
            for line
            in plotter.axes.get_lines()
        ]

        assert labels == [
            "Actual",
            "ARIMA",
            "Kalman",
            "LSTM",
        ]
    finally:
        plt.close(fig)


def test_plot_preserves_actual_values(
    benchmark_results,
):

    plotter = BenchmarkPlotter(
        benchmark_results
    )

    fig = plotter.plot()

    try:
        actual_line = (
            plotter.axes.get_lines()[0]
        )

        assert tuple(
            actual_line.get_ydata()
        ) == (
            10.0,
            11.0,
            12.0,
        )
    finally:
        plt.close(fig)


def test_plot_preserves_predictions(
    benchmark_results,
):

    plotter = BenchmarkPlotter(
        benchmark_results
    )

    fig = plotter.plot()

    try:
        lines = {
            line.get_label(): line
            for line
            in plotter.axes.get_lines()
        }

        assert tuple(
            lines["ARIMA"].get_ydata()
        ) == (
            10.1,
            11.1,
            12.1,
        )

        assert tuple(
            lines["LSTM"].get_ydata()
        ) == (
            9.9,
            10.8,
            11.8,
        )
    finally:
        plt.close(fig)


def test_plot_uses_standard_labels(
    benchmark_results,
):

    plotter = BenchmarkPlotter(
        benchmark_results
    )

    fig = plotter.plot()

    try:
        assert (
            plotter.axes.get_xlabel()
            == "Date"
        )

        assert (
            plotter.axes.get_ylabel()
            == "Voltage (µV)"
        )

        assert (
            plotter.axes.get_title()
            == "Optimized Multi-Model Benchmark"
        )
    finally:
        plt.close(fig)


def test_custom_title(
    benchmark_results,
):

    plotter = BenchmarkPlotter(
        benchmark_results
    )

    fig = plotter.plot(
        title="Battery BAT001 Benchmark"
    )

    try:
        assert (
            plotter.axes.get_title()
            == "Battery BAT001 Benchmark"
        )
    finally:
        plt.close(fig)


# ============================================================
# Persistence
# ============================================================


def test_save_creates_figure(
    benchmark_results,
    tmp_path,
):

    filename = (
        tmp_path
        / "benchmark_forecast.png"
    )

    plotter = BenchmarkPlotter(
        benchmark_results
    )

    path = plotter.save(
        filename
    )

    try:
        assert path == filename
        assert path.exists()
        assert path.stat().st_size > 0
    finally:
        if plotter.figure is not None:
            plt.close(
                plotter.figure
            )


def test_save_creates_parent_directory(
    benchmark_results,
    tmp_path,
):

    filename = (
        tmp_path
        / "nested"
        / "figures"
        / "benchmark_forecast.png"
    )

    plotter = BenchmarkPlotter(
        benchmark_results
    )

    path = plotter.save(
        filename
    )

    try:
        assert path.exists()
    finally:
        if plotter.figure is not None:
            plt.close(
                plotter.figure
            )


def test_save_plots_automatically(
    benchmark_results,
    tmp_path,
):

    plotter = BenchmarkPlotter(
        benchmark_results
    )

    assert plotter.figure is None

    filename = (
        tmp_path
        / "benchmark_forecast.png"
    )

    plotter.save(
        filename
    )

    try:
        assert plotter.figure is not None
        assert filename.exists()
    finally:
        plt.close(
            plotter.figure
        )


# ============================================================
# Representation
# ============================================================


def test_repr_contains_useful_summary(
    benchmark_results,
):

    plotter = BenchmarkPlotter(
        benchmark_results
    )

    text = repr(
        plotter
    )

    assert "BenchmarkPlotter" in text
    assert "models=3" in text
    assert "steps=3" in text