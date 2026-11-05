"""Tests for BenchmarkBestWorstHistoryPlotter."""
from types import SimpleNamespace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest
from zenerestimation.visualization.benchmark_best_worst_history import (
    BenchmarkBestWorstHistoryPlotter,
)

class _Dataset:
    def __init__(self):
        self.data = pd.DataFrame({
            "ds": pd.to_datetime([
                "2022-03-01","2022-06-01","2022-09-01",
                "2022-12-01","2023-03-01","2023-06-01",
            ]),
            "microVolt": [10.,11.,12.,13.,14.,15.],
        })

def _evaluations():
    dates = (
        pd.Timestamp("2023-03-01"),
        pd.Timestamp("2023-06-01"),
    )

    actual = (
        14.0,
        15.0,
    )

    return {
        "Middle": SimpleNamespace(
            model="Middle",
            evaluation_steps=2,
            rmse=0.7,
            mae=0.6,
            mape=4.0,
            actual=actual,
            predicted=(
                13.8,
                14.8,
            ),
            dates=dates,
        ),
        "Worst": SimpleNamespace(
            model="Worst",
            evaluation_steps=2,
            rmse=1.4,
            mae=1.2,
            mape=8.0,
            actual=actual,
            predicted=(
                13.4,
                14.2,
            ),
            dates=dates,
        ),
        "Best": SimpleNamespace(
            model="Best",
            evaluation_steps=2,
            rmse=0.5,
            mae=0.4,
            mape=3.0,
            actual=actual,
            predicted=(
                13.9,
                14.9,
            ),
            dates=dates,
        ),
    }

def _forecasts():
    idx=pd.to_datetime(["2022-03-01","2022-06-01","2022-09-01","2022-12-01"])
    return {
        "Middle": SimpleNamespace(fitted=pd.Series([9.9,10.9,11.9,12.9],index=idx)),
        "Worst": SimpleNamespace(fitted=pd.Series([9.7,10.7,11.7,12.7],index=idx)),
        "Best": SimpleNamespace(fitted=pd.Series([10.,11.,12.,13.],index=idx)),
    }

def test_selects_best_and_worst_strictly_by_rmse():
    p=BenchmarkBestWorstHistoryPlotter(_Dataset(),_evaluations(),_forecasts())
    assert p.best_model=="Best"
    assert p.worst_model=="Worst"

def test_requires_identical_model_sets():
    f=_forecasts(); f.pop("Middle")
    with pytest.raises(ValueError):
        BenchmarkBestWorstHistoryPlotter(_Dataset(),_evaluations(),f)

def test_plot_has_three_data_lines_plus_development_boundary():
    p = BenchmarkBestWorstHistoryPlotter(
        _Dataset(),
        _evaluations(),
        _forecasts(),
    )

    fig, ax = p.plot()

    try:
        assert len(ax.lines) == 4

        labels = [
            line.get_label()
            for line in ax.lines
        ]

        assert labels[0] == (
            "Measured history + holdout"
        )

        assert labels[1].startswith(
            "Best: Best"
        )

        assert labels[2].startswith(
            "Worst: Worst"
        )

        assert labels[3].startswith(
            "Development end:"
        )

        assert ax.get_title() == p.TITLE

    finally:
        plt.close(fig)

def test_plot_does_not_mutate_existing_fitted_values():
    f=_forecasts()
    before={n:r.fitted.copy(deep=True) for n,r in f.items()}
    p=BenchmarkBestWorstHistoryPlotter(_Dataset(),_evaluations(),f)
    fig,_=p.plot()
    try:
        for n,expected in before.items():
            pd.testing.assert_series_equal(f[n].fitted,expected)
    finally:
        plt.close(fig)

def test_save_creates_figure(tmp_path):
    p=BenchmarkBestWorstHistoryPlotter(_Dataset(),_evaluations(),_forecasts())
    path=p.save(tmp_path/"nested"/"best_worst.png")
    assert path.exists() and path.stat().st_size>0
    plt.close(p.figure)

def test_rejects_fitted_length_mismatch():
    f=_forecasts(); f["Best"]=SimpleNamespace(fitted=pd.Series([1.,2.]))
    p=BenchmarkBestWorstHistoryPlotter(_Dataset(),_evaluations(),f)
    with pytest.raises(ValueError,match="fitted length"):
        p.plot()


def test_raw_history_may_omit_a_preprocessed_training_date():
    dataset = _Dataset()

    # Simulate a raw dataset with one missing development quarter.
    dataset.data = dataset.data.loc[
        dataset.data["ds"] != pd.Timestamp("2022-09-01")
    ].reset_index(drop=True)

    training_dates = pd.to_datetime(
        [
            "2022-03-01",
            "2022-06-01",
            "2022-09-01",
            "2022-12-01",
        ]
    )

    plotter = BenchmarkBestWorstHistoryPlotter(
        dataset,
        _evaluations(),
        _forecasts(),
        training_dates=training_dates,
    )

    fig, ax = plotter.plot()

    try:

        # Three principal data trajectories plus one
        # development-end reference line.
        assert len(ax.lines) == 4
        #assert len(ax.lines) == 3

        labels = [
            line.get_label()
            for line in ax.lines
        ]

        assert labels[0] == (
            "Measured history + holdout"
        )

        assert labels[1].startswith(
            "Best:"
        )

        assert labels[2].startswith(
            "Worst:"
        )

        assert labels[3].startswith(
            "Development end:"
        )

        measured_line = ax.lines[0]

        measured_dates = pd.DatetimeIndex(
            measured_line.get_xdata()
        )

        # Raw measurements must remain raw: the absent quarter
        # must not be manufactured by the visualization.
        assert pd.Timestamp("2022-09-01") not in measured_dates

        # Fitted histories remain aligned to the complete
        # preprocessed model-training timeline.
        best_dates = pd.DatetimeIndex(
            ax.lines[1].get_xdata()
        )
        worst_dates = pd.DatetimeIndex(
            ax.lines[2].get_xdata()
        )

        # Model trajectories contain:
        #
        #   complete preprocessed training timeline
        #       +
        #   already-computed frozen benchmark dates.
        #
        # The missing raw observation therefore remains present
        # in the model-training timeline, while no synthetic
        # measured observation is created.

        evaluation_dates = pd.DatetimeIndex(
            pd.to_datetime(
                next(
                    iter(
                        _evaluations().values()
                    )
                ).dates
            )
        )

        expected_model_dates = (
            pd.DatetimeIndex(training_dates)
            .append(evaluation_dates)
        )

        assert best_dates.equals(
            expected_model_dates
        )

        assert worst_dates.equals(
            expected_model_dates
        )

        # The preprocessed quarter missing from the raw
        # measurements must still exist in both fitted
        # model trajectories.
        assert pd.Timestamp(
            "2022-09-01"
        ) in best_dates

        assert pd.Timestamp(
            "2022-09-01"
        ) in worst_dates

        # Both model trajectories must extend through
        # the final frozen benchmark date.
        assert best_dates[-1] == evaluation_dates[-1]
        assert worst_dates[-1] == evaluation_dates[-1]

    finally:
        plt.close(fig)


def test_model_lines_include_frozen_benchmark_forecasts():
    plotter = BenchmarkBestWorstHistoryPlotter(
        _Dataset(),
        _evaluations(),
        _forecasts(),
    )

    fig, ax = plotter.plot()

    try:
        best_line = ax.lines[1]
        worst_line = ax.lines[2]

        best_result = (
            _evaluations()["Best"]
        )

        worst_result = (
            _evaluations()["Worst"]
        )

        best_dates = pd.DatetimeIndex(
            best_line.get_xdata()
        )

        worst_dates = pd.DatetimeIndex(
            worst_line.get_xdata()
        )

        # Both model trajectories must terminate
        # at the final frozen benchmark date.
        assert best_dates[-1] == pd.Timestamp(
            best_result.dates[-1]
        )

        assert worst_dates[-1] == pd.Timestamp(
            worst_result.dates[-1]
        )

        # Their final benchmark section must be the
        # already-computed frozen predictions.
        np.testing.assert_allclose(
            np.asarray(
                best_line.get_ydata(),
                dtype=float,
            )[-len(best_result.predicted):],
            np.asarray(
                best_result.predicted,
                dtype=float,
            ),
        )

        np.testing.assert_allclose(
            np.asarray(
                worst_line.get_ydata(),
                dtype=float,
            )[-len(worst_result.predicted):],
            np.asarray(
                worst_result.predicted,
                dtype=float,
            ),
        )

    finally:
        plt.close(fig)


def test_measured_line_includes_real_holdout_values():
    plotter = BenchmarkBestWorstHistoryPlotter(
        _Dataset(),
        _evaluations(),
        _forecasts(),
    )

    fig, ax = plotter.plot()

    try:
        measured_line = ax.lines[0]

        result = next(
            iter(
                _evaluations().values()
            )
        )

        measured_dates = pd.DatetimeIndex(
            measured_line.get_xdata()
        )

        measured_values = np.asarray(
            measured_line.get_ydata(),
            dtype=float,
        )

        assert measured_dates[-1] == pd.Timestamp(
            result.dates[-1]
        )

        np.testing.assert_allclose(
            measured_values[
                -len(result.actual):
            ],
            np.asarray(
                result.actual,
                dtype=float,
            ),
        )

    finally:
        plt.close(fig)