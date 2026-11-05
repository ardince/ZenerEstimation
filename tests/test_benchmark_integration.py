"""Tests for benchmark-to-comparison integration."""

from __future__ import annotations

import pytest

from zenerestimation.comparison import (
    ComparisonResult,
)
from zenerestimation.evaluation.benchmark_integration import (
    comparison_from_evaluations,
)
from zenerestimation.evaluation.result import (
    EvaluationResult,
)


def make_result(
    model,
    rmse,
    mae,
    mape,
):
    return EvaluationResult(
        model=model,
        evaluation_steps=3,
        rmse=rmse,
        mae=mae,
        mape=mape,
        actual=(
            10.0,
            11.0,
            12.0,
        ),
        predicted=(
            10.1,
            11.1,
            12.1,
        ),
        dates=(
            "2026-03-01",
            "2026-06-01",
            "2026-09-01",
        ),
        metadata={},
    )


@pytest.fixture
def results():
    return {
        "ARIMA": make_result(
            "ARIMA",
            0.8,
            0.6,
            0.4,
        ),
        "Kalman": make_result(
            "Kalman",
            0.5,
            0.4,
            0.3,
        ),
        "LSTM": make_result(
            "LSTM",
            0.7,
            0.5,
            0.35,
        ),
    }


def test_requires_non_empty_battery(
    results,
):

    with pytest.raises(
        ValueError,
        match="battery",
    ):
        comparison_from_evaluations(
            "",
            results,
        )


def test_requires_mapping():

    with pytest.raises(
        TypeError,
        match="mapping",
    ):
        comparison_from_evaluations(
            "BAT001",
            [],
        )


def test_requires_non_empty_results():

    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        comparison_from_evaluations(
            "BAT001",
            {},
        )


def test_requires_evaluation_results():

    with pytest.raises(
        TypeError,
        match="EvaluationResult",
    ):
        comparison_from_evaluations(
            "BAT001",
            {
                "ARIMA": "invalid",
            },
        )


def test_requires_matching_model_name():

    result = make_result(
        "ARIMA",
        0.8,
        0.6,
        0.4,
    )

    with pytest.raises(
        ValueError,
        match="model",
    ):
        comparison_from_evaluations(
            "BAT001",
            {
                "WrongName": result,
            },
        )


def test_returns_comparison_result(
    results,
):

    comparison = (
        comparison_from_evaluations(
            "BAT001",
            results,
        )
    )

    assert isinstance(
        comparison,
        ComparisonResult,
    )


def test_preserves_battery(
    results,
):

    comparison = (
        comparison_from_evaluations(
            "BAT001",
            results,
        )
    )

    assert comparison.battery == "BAT001"


def test_preserves_model_order(
    results,
):

    comparison = (
        comparison_from_evaluations(
            "BAT001",
            results,
        )
    )

    assert comparison.models == [
        "ARIMA",
        "Kalman",
        "LSTM",
    ]


def test_preserves_metrics(
    results,
):

    comparison = (
        comparison_from_evaluations(
            "BAT001",
            results,
        )
    )

    assert comparison.metrics == {
        "ARIMA": {
            "rmse": 0.8,
            "mae": 0.6,
            "mape": 0.4,
        },
        "Kalman": {
            "rmse": 0.5,
            "mae": 0.4,
            "mape": 0.3,
        },
        "LSTM": {
            "rmse": 0.7,
            "mae": 0.5,
            "mape": 0.35,
        },
    }


def test_comparison_ranking_is_delegated(
    results,
):

    comparison = (
        comparison_from_evaluations(
            "BAT001",
            results,
        )
    )

    assert [
        entry["model"]
        for entry
        in comparison.rankings["rmse"]
    ] == [
        "Kalman",
        "LSTM",
        "ARIMA",
    ]


def test_best_models_are_delegated(
    results,
):

    comparison = (
        comparison_from_evaluations(
            "BAT001",
            results,
        )
    )

    assert comparison.best_models == {
        "rmse": "Kalman",
        "mae": "Kalman",
        "mape": "Kalman",
    }


def test_does_not_mutate_evaluation_results(
    results,
):

    before = {
        model: result.to_dict()
        for model, result
        in results.items()
    }

    comparison_from_evaluations(
        "BAT001",
        results,
    )

    after = {
        model: result.to_dict()
        for model, result
        in results.items()
    }

    assert after == before