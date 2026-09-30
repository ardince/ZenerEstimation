"""
Regression tests for Sprint 14.8 optimized benchmark comparison.

These tests verify that frozen final-benchmark evidence from the
optimized multi-model benchmark can be consumed by the existing
ForecastComparison subsystem without rerunning forecasting models.

The numerical values below are final benchmark evidence produced by
the standardized benchmark runners for datasets 410 and 110.

No optimization, model fitting, prediction, parameter selection,
or seed selection is performed in this module.
"""

from copy import deepcopy
from pathlib import Path

import pytest

from zenerestimation.comparison import (
    ComparisonResult,
    ForecastComparison,
)
from zenerestimation.utils.result_loader import (
    ResultPackage,
)


# ============================================================
# Frozen benchmark evidence
# ============================================================


EXPECTED_410 = {
    "ARIMA": {
        "rmse": 0.849442,
        "mae": 0.762509,
        "mape": 2.427533,
    },
    "Kalman": {
        "rmse": 0.939829,
        "mae": 0.852141,
        "mape": 2.714113,
    },
    "LSTM": {
        "rmse": 1.307027,
        "mae": 1.176976,
        "mape": 3.746587,
    },
    "GRU": {
        "rmse": 1.395413,
        "mae": 1.260344,
        "mape": 4.012700,
    },
    "LinearTrendLSTM": {
        "rmse": 0.788173,
        "mae": 0.727231,
        "mape": 2.319916,
    },
    "KalmanLSTM": {
        "rmse": 0.672161,
        "mae": 0.570011,
        "mape": 1.860285,
    },
}


EXPECTED_110 = {
    "ARIMA": {
        "rmse": 0.447650,
        "mae": 0.353932,
        "mape": 0.209373,
    },
    "Kalman": {
        "rmse": 0.161591,
        "mae": 0.112413,
        "mape": 0.066398,
    },
    "LSTM": {
        "rmse": 0.260595,
        "mae": 0.230460,
        "mape": 0.136878,
    },
    "GRU": {
        "rmse": 0.246924,
        "mae": 0.181763,
        "mape": 0.107679,
    },
    "LinearTrendLSTM": {
        "rmse": 1.623236,
        "mae": 1.586931,
        "mape": 0.941381,
    },
    "KalmanLSTM": {
        "rmse": 5.292223,
        "mae": 5.279530,
        "mape": 3.134292,
    },
}


MODEL_ORDER = (
    "ARIMA",
    "Kalman",
    "LSTM",
    "GRU",
    "LinearTrendLSTM",
    "KalmanLSTM",
)


# ============================================================
# Helpers
# ============================================================


def make_benchmark_run(
    *,
    battery,
    model,
    metrics,
    run_number,
):
    """
    Construct a stored-result representation of one frozen
    optimized benchmark evaluation.

    This helper intentionally creates ResultPackage objects
    directly. No forecasting model is instantiated or executed.
    """

    return ResultPackage(
        battery=battery,
        model=model,
        timestamp="20260930_140000",
        run_number=run_number,
        directory=Path(
            f"results/{battery}/{model}/benchmark"
        ),
        forecast={
            "model": model,
        },
        evaluation=deepcopy(metrics),
        experiment={
            "battery": battery,
            "model": model,
            "analysis": "optimized_benchmark",
        },
        report=None,
        figure=None,
        log=None,
    )


def make_benchmark_runs(
    battery,
    evidence,
):
    """
    Convert frozen benchmark evidence into ResultPackage objects
    while preserving the standardized model order.
    """

    return [
        make_benchmark_run(
            battery=battery,
            model=model,
            metrics=evidence[model],
            run_number=index,
        )
        for index, model in enumerate(
            MODEL_ORDER,
            start=1,
        )
    ]


# ============================================================
# Dataset 410
# ============================================================


def test_410_benchmark_comparison_preserves_all_models():

    runs = make_benchmark_runs(
        "732B-5610410",
        EXPECTED_410,
    )

    comparison = ForecastComparison(
        runs
    )

    assert comparison.battery == "732B-5610410"

    assert comparison.models() == list(
        MODEL_ORDER
    )


def test_410_benchmark_metrics_are_preserved():

    runs = make_benchmark_runs(
        "732B-5610410",
        EXPECTED_410,
    )

    comparison = ForecastComparison(
        runs
    )

    table = comparison.metric_table()

    assert set(table) == set(MODEL_ORDER)

    for model in MODEL_ORDER:

        assert table[model]["rmse"] == pytest.approx(
            EXPECTED_410[model]["rmse"]
        )

        assert table[model]["mae"] == pytest.approx(
            EXPECTED_410[model]["mae"]
        )

        assert table[model]["mape"] == pytest.approx(
            EXPECTED_410[model]["mape"]
        )


def test_410_comparison_result_preserves_benchmark_evidence():

    runs = make_benchmark_runs(
        "732B-5610410",
        EXPECTED_410,
    )

    result = ForecastComparison(
        runs
    ).compare()

    assert isinstance(
        result,
        ComparisonResult,
    )

    assert result.battery == "732B-5610410"

    metrics = result.metrics

    for model in MODEL_ORDER:

        assert metrics[model]["rmse"] == pytest.approx(
            EXPECTED_410[model]["rmse"]
        )

        assert metrics[model]["mae"] == pytest.approx(
            EXPECTED_410[model]["mae"]
        )

        assert metrics[model]["mape"] == pytest.approx(
            EXPECTED_410[model]["mape"]
        )


# ============================================================
# Dataset 110
# ============================================================


def test_110_benchmark_comparison_preserves_all_models():

    runs = make_benchmark_runs(
        "732B-5610110",
        EXPECTED_110,
    )

    comparison = ForecastComparison(
        runs
    )

    assert comparison.battery == "732B-5610110"

    assert comparison.models() == list(
        MODEL_ORDER
    )


def test_110_benchmark_metrics_are_preserved():

    runs = make_benchmark_runs(
        "732B-5610110",
        EXPECTED_110,
    )

    comparison = ForecastComparison(
        runs
    )

    table = comparison.metric_table()

    assert set(table) == set(MODEL_ORDER)

    for model in MODEL_ORDER:

        assert table[model]["rmse"] == pytest.approx(
            EXPECTED_110[model]["rmse"]
        )

        assert table[model]["mae"] == pytest.approx(
            EXPECTED_110[model]["mae"]
        )

        assert table[model]["mape"] == pytest.approx(
            EXPECTED_110[model]["mape"]
        )


def test_110_comparison_result_preserves_benchmark_evidence():

    runs = make_benchmark_runs(
        "732B-5610110",
        EXPECTED_110,
    )

    result = ForecastComparison(
        runs
    ).compare()

    assert isinstance(
        result,
        ComparisonResult,
    )

    assert result.battery == "732B-5610110"

    metrics = result.metrics

    for model in MODEL_ORDER:

        assert metrics[model]["rmse"] == pytest.approx(
            EXPECTED_110[model]["rmse"]
        )

        assert metrics[model]["mae"] == pytest.approx(
            EXPECTED_110[model]["mae"]
        )

        assert metrics[model]["mape"] == pytest.approx(
            EXPECTED_110[model]["mape"]
        )


# ============================================================
# Evidence isolation
# ============================================================


def test_410_and_110_remain_separate_comparisons():

    runs_410 = make_benchmark_runs(
        "732B-5610410",
        EXPECTED_410,
    )

    runs_110 = make_benchmark_runs(
        "732B-5610110",
        EXPECTED_110,
    )

    comparison_410 = ForecastComparison(
        runs_410
    )

    comparison_110 = ForecastComparison(
        runs_110
    )

    assert comparison_410.battery == "732B-5610410"

    assert comparison_110.battery == "732B-5610110"

    assert (
        comparison_410.metric_table()
        != comparison_110.metric_table()
    )


def test_mixed_benchmark_batteries_are_rejected():

    run_410 = make_benchmark_run(
        battery="732B-5610410",
        model="ARIMA",
        metrics=EXPECTED_410["ARIMA"],
        run_number=1,
    )

    run_110 = make_benchmark_run(
        battery="732B-5610110",
        model="Kalman",
        metrics=EXPECTED_110["Kalman"],
        run_number=2,
    )

    with pytest.raises(ValueError):

        ForecastComparison(
            [
                run_410,
                run_110,
            ]
        )


# ============================================================
# No mutation / no execution
# ============================================================


def test_comparison_does_not_mutate_stored_benchmark_evidence():

    runs = make_benchmark_runs(
        "732B-5610410",
        EXPECTED_410,
    )

    before = [
        deepcopy(run.evaluation)
        for run in runs
    ]

    comparison = ForecastComparison(
        runs
    )

    comparison.metric_table()
    comparison.compare()

    after = [
        run.evaluation
        for run in runs
    ]

    assert after == before


def test_comparison_uses_stored_evidence_only():

    runs = make_benchmark_runs(
        "732B-5610410",
        EXPECTED_410,
    )

    comparison = ForecastComparison(
        runs
    )

    table = comparison.metric_table()

    # The comparison layer receives only ResultPackage objects.
    # No model factory, dataset, evaluator, fit(), or predict()
    # object is supplied to this operation.
    assert len(table) == 6

    assert set(table) == set(
        MODEL_ORDER
    )