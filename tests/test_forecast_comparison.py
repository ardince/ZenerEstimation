"""
Tests for ForecastComparison.
"""

import pytest

from pathlib import Path

from zenerestimation.comparison import (
    ForecastComparison,
    ComparisonResult,
)

from zenerestimation.utils.result_loader import (
    ResultPackage,
)


# ============================================================
# Helpers
# ============================================================


def make_run(
    battery="BAT001",
    model="arima",
    rmse=1.0,
    mae=1.0,
    mape=1.0,
    run_number=1,
):

    return ResultPackage(

        battery=battery,

        model=model,

        timestamp="20260819_160000",

        run_number=run_number,

        directory=Path(
            f"results/{battery}/{model}/run"
        ),

        forecast={
            "model": model,
        },

        evaluation={

            "rmse": rmse,

            "mae": mae,

            "mape": mape,

        },

        experiment={
            "battery": battery,
            "model": model,
        },

        report=None,

        figure=None,

        log=None,

    )


# ============================================================
# Constructor
# ============================================================


def test_constructor():

    runs = [
        make_run(),
    ]

    comparison = ForecastComparison(
        runs
    )

    assert comparison.battery == "BAT001"


# ============================================================
# Empty Input
# ============================================================


def test_empty_runs_raise():

    with pytest.raises(ValueError):

        ForecastComparison([])


# ============================================================
# Battery Validation
# ============================================================


def test_mixed_batteries_raise():

    runs = [

        make_run(
            battery="BAT001",
            model="arima",
        ),

        make_run(
            battery="BAT002",
            model="kalman",
        ),

    ]

    with pytest.raises(ValueError):

        ForecastComparison(
            runs
        )


# ============================================================
# Models
# ============================================================


def test_models():

    runs = [

        make_run(
            model="arima",
        ),

        make_run(
            model="kalman",
        ),

    ]

    comparison = ForecastComparison(
        runs
    )

    assert comparison.models() == [
        "arima",
        "kalman",
    ]


# ============================================================
# Metric Table
# ============================================================


def test_metric_table():

    runs = [

        make_run(
            model="arima",
            rmse=1.8,
            mae=1.4,
            mape=4.8,
        ),

        make_run(
            model="kalman",
            rmse=1.5,
            mae=1.2,
            mape=4.1,
        ),

    ]

    comparison = ForecastComparison(
        runs
    )

    metrics = (
        comparison.metric_table()
    )

    assert metrics["arima"]["rmse"] == 1.8

    assert metrics["kalman"]["mae"] == 1.2


# ============================================================
# Ranking
# ============================================================


def test_rank_rmse():

    runs = [

        make_run(
            model="arima",
            rmse=1.8,
        ),

        make_run(
            model="kalman",
            rmse=1.2,
        ),

        make_run(
            model="lstm",
            rmse=1.4,
        ),

    ]

    comparison = ForecastComparison(
        runs
    )

    ranking = comparison.rank(
        "rmse"
    )

    assert ranking[0]["model"] == "kalman"

    assert ranking[0]["rank"] == 1

    assert ranking[1]["model"] == "lstm"

    assert ranking[2]["model"] == "arima"


def test_rank_mae():

    runs = [

        make_run(
            model="arima",
            mae=1.3,
        ),

        make_run(
            model="kalman",
            mae=0.9,
        ),

    ]

    comparison = ForecastComparison(
        runs
    )

    ranking = comparison.rank(
        "mae"
    )

    assert ranking[0]["model"] == "kalman"


def test_rank_mape():

    runs = [

        make_run(
            model="arima",
            mape=4.0,
        ),

        make_run(
            model="kalman",
            mape=2.5,
        ),

    ]

    comparison = ForecastComparison(
        runs
    )

    ranking = comparison.rank(
        "mape"
    )

    assert ranking[0]["model"] == "kalman"


# ============================================================
# Unsupported Metric
# ============================================================


def test_unsupported_metric_raises():

    comparison = ForecastComparison(
        [
            make_run(),
        ]
    )

    with pytest.raises(ValueError):

        comparison.rank(
            "r2"
        )


# ============================================================
# Best Model
# ============================================================


def test_best():

    runs = [

        make_run(
            model="arima",
            rmse=1.8,
        ),

        make_run(
            model="kalman",
            rmse=1.1,
        ),

    ]

    comparison = ForecastComparison(
        runs
    )

    assert comparison.best(
        "rmse"
    ) == "kalman"


# ============================================================
# Missing Evaluation
# ============================================================


def test_missing_evaluation_is_skipped():

    run = make_run(
        model="arima"
    )

    run_without_evaluation = (
        ResultPackage(

            battery="BAT001",

            model="kalman",

            timestamp="20260819_160100",

            run_number=1,

            directory=Path(
                "results/BAT001/kalman/run"
            ),

            forecast=None,

            evaluation=None,

            experiment=None,

            report=None,

            figure=None,

            log=None,

        )
    )

    comparison = ForecastComparison(
        [
            run,
            run_without_evaluation,
        ]
    )

    metrics = (
        comparison.metric_table()
    )

    assert "arima" in metrics

    assert "kalman" not in metrics


# ============================================================
# Missing Individual Metric
# ============================================================


def test_missing_metric_is_skipped():

    run = make_run(
        model="arima"
    )

    run_partial = ResultPackage(

        battery="BAT001",

        model="kalman",

        timestamp="20260819_160100",

        run_number=1,

        directory=Path(
            "results/BAT001/kalman/run"
        ),

        forecast=None,

        evaluation={
            "rmse": 0.9,
        },

        experiment=None,

        report=None,

        figure=None,

        log=None,

    )

    comparison = ForecastComparison(
        [
            run,
            run_partial,
        ]
    )

    ranking = comparison.rank(
        "mae"
    )

    models = [
        item["model"]
        for item in ranking
    ]

    assert "arima" in models

    assert "kalman" not in models


# ============================================================
# Compare
# ============================================================


def test_compare():

    runs = [

        make_run(
            model="arima",
            rmse=1.8,
            mae=1.4,
            mape=4.8,
        ),

        make_run(
            model="kalman",
            rmse=1.2,
            mae=0.9,
            mape=3.2,
        ),

    ]

    comparison = ForecastComparison(
        runs
    )

    result = comparison.compare()

    assert isinstance(
        result,
        ComparisonResult,
    )

    assert result.battery == "BAT001"

    assert (
        result.best_models["rmse"]
        == "kalman"
    )


# ============================================================
# Summary
# ============================================================


def test_summary():

    comparison = ForecastComparison(
        [
            make_run(),
        ]
    )

    summary = comparison.summary()

    assert summary["battery"] == "BAT001"

    assert "metrics" in summary

    assert "rankings" in summary

    assert "best_models" in summary


# ============================================================
# Ranking Order
# ============================================================


def test_lower_metric_is_better():

    runs = [

        make_run(
            model="high",
            rmse=10.0,
        ),

        make_run(
            model="low",
            rmse=1.0,
        ),

    ]

    comparison = ForecastComparison(
        runs
    )

    ranking = comparison.rank(
        "rmse"
    )

    assert ranking[0]["model"] == "low"

    assert ranking[-1]["model"] == "high"