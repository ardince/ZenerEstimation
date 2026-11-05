"""F.6.3 transition figure: evidence-only, no forecasting calls."""
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from zenerestimation.visualization.benchmark_to_operational_transition import (
    BenchmarkToOperationalTransitionPlotter,
)
from examples.optimization import run_410_complete as complete


BENCH_DATES = pd.date_range("2024-03-01", periods=6, freq="QS-MAR")
FUTURE_DATES = pd.date_range("2025-09-01", periods=6, freq="QS-MAR")


def evidence():
    measured_dates = pd.date_range("2023-03-01", periods=10, freq="QS-MAR")
    measured = np.arange(10, dtype=float) + 24
    frame = pd.DataFrame({"ds": measured_dates, "microVolt": measured})
    actual = tuple(frame.set_index("ds").loc[BENCH_DATES, "microVolt"].to_numpy())
    evaluation = SimpleNamespace(dates=tuple(BENCH_DATES), actual=actual, predicted=tuple(np.arange(6) + 28.5))
    bench = SimpleNamespace(dataset=SimpleNamespace(data=frame), evaluations={"ARIMA": evaluation})
    oper = SimpleNamespace(forecasts={"ARIMA": np.arange(6, dtype=float) + 34})
    return bench, oper


def test_transition_line_segments_and_boundaries():
    bench, oper = evidence()
    plotter = BenchmarkToOperationalTransitionPlotter(bench, oper, operational_dates=FUTURE_DATES)
    fig, ax = plotter.plot()
    try:
        assert len(ax.lines) == 5  # measured, frozen, operational, two boundaries
        assert pd.DatetimeIndex(ax.lines[1].get_xdata()).equals(BENCH_DATES)
        assert pd.DatetimeIndex(ax.lines[2].get_xdata()).equals(FUTURE_DATES)
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), bench.evaluations["ARIMA"].predicted)
        np.testing.assert_array_equal(ax.lines[2].get_ydata(), oper.forecasts["ARIMA"])
        assert ax.lines[1].get_linestyle() != ax.lines[2].get_linestyle()
        assert "unvalidated" in ax.lines[2].get_label()
        assert pd.DatetimeIndex(ax.lines[0].get_xdata())[-1] == pd.Timestamp("2025-06-01")
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_transition_rejects_model_mismatch():
    bench, oper = evidence()
    oper.forecasts = {"GRU": np.ones(6)}
    with pytest.raises(ValueError, match="model sets"):
        BenchmarkToOperationalTransitionPlotter(bench, oper).plot()


def test_complete_runner_reuses_both_stages_once(monkeypatch, tmp_path):
    bench, oper = evidence()
    benchmark_fn = Mock(return_value=bench)
    operational_fn = Mock(return_value=oper)
    save_fn = Mock(return_value=tmp_path / "transition.png")
    monkeypatch.setattr(complete.benchmark_run, "run_benchmark", benchmark_fn)
    monkeypatch.setattr(complete.operational_run, "run_operational_extrapolation", operational_fn)
    monkeypatch.setattr(complete.BenchmarkToOperationalTransitionPlotter, "save", save_fn)
    complete.main()
    benchmark_fn.assert_called_once_with()
    operational_fn.assert_called_once_with()
    save_fn.assert_called_once()
