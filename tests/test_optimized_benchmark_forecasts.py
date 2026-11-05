"""Tests for OptimizedBenchmark forecast-evidence retention."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from zenerestimation.evaluation.benchmark import (
    BenchmarkModelSpec,
    OptimizedBenchmark,
)


@dataclass
class DummyEvaluation:
    """Minimal standardized evaluation evidence."""

    model: str


@dataclass
class DummyForecast:
    """Minimal forecast evidence."""

    model: str
    fitted: tuple[float, ...]
    forecast: tuple[float, ...]


class DummyModel:
    """Fresh benchmark model created by a specification."""

    def __init__(
        self,
        name,
        token=None,
    ):
        self.name = name
        self.token = token


class RecordingEvaluator:
    """
    Evaluator exposing both benchmark APIs and recording calls.
    """

    def __init__(self):
        self.evaluate_calls = []
        self.evaluate_with_forecast_calls = []

    def evaluate(
        self,
        dataset,
        model,
    ):
        self.evaluate_calls.append(
            (
                dataset,
                model,
            )
        )

        return DummyEvaluation(
            model=model.name
        )

    def evaluate_with_forecast(
        self,
        dataset,
        model,
    ):
        self.evaluate_with_forecast_calls.append(
            (
                dataset,
                model,
            )
        )

        evaluation = DummyEvaluation(
            model=model.name
        )

        forecast = DummyForecast(
            model=model.name,
            fitted=(
                1.0,
                2.0,
                3.0,
            ),
            forecast=(
                4.0,
                5.0,
            ),
        )

        return (
            evaluation,
            forecast,
        )


def _spec(
    name,
    token=None,
):
    return BenchmarkModelSpec(
        name=name,
        factory=DummyModel,
        params={
            "name": name,
            "token": token,
        },
    )


def test_existing_evaluate_contract_is_preserved():
    evaluator = RecordingEvaluator()

    benchmark = OptimizedBenchmark(
        evaluator
    )

    dataset = object()

    results = benchmark.evaluate(
        dataset,
        [
            _spec("ModelA"),
            _spec("ModelB"),
        ],
    )

    assert list(results) == [
        "ModelA",
        "ModelB",
    ]

    assert isinstance(
        results["ModelA"],
        DummyEvaluation,
    )

    assert isinstance(
        results["ModelB"],
        DummyEvaluation,
    )


def test_existing_evaluate_uses_only_standard_evaluator_method():
    evaluator = RecordingEvaluator()

    benchmark = OptimizedBenchmark(
        evaluator
    )

    benchmark.evaluate(
        object(),
        [
            _spec("ModelA"),
            _spec("ModelB"),
        ],
    )

    assert len(
        evaluator.evaluate_calls
    ) == 2

    assert (
        evaluator.evaluate_with_forecast_calls
        == []
    )


def test_evaluate_with_forecasts_returns_two_mappings():
    evaluator = RecordingEvaluator()

    benchmark = OptimizedBenchmark(
        evaluator
    )

    evaluations, forecasts = (
        benchmark.evaluate_with_forecasts(
            object(),
            [
                _spec("ModelA"),
                _spec("ModelB"),
            ],
        )
    )

    assert isinstance(
        evaluations,
        dict,
    )

    assert isinstance(
        forecasts,
        dict,
    )


def test_evaluate_with_forecasts_preserves_specification_order():
    evaluator = RecordingEvaluator()

    benchmark = OptimizedBenchmark(
        evaluator
    )

    evaluations, forecasts = (
        benchmark.evaluate_with_forecasts(
            object(),
            [
                _spec("ModelB"),
                _spec("ModelA"),
                _spec("ModelC"),
            ],
        )
    )

    expected = [
        "ModelB",
        "ModelA",
        "ModelC",
    ]

    assert list(
        evaluations
    ) == expected

    assert list(
        forecasts
    ) == expected


def test_evaluate_with_forecasts_returns_matching_evidence():
    evaluator = RecordingEvaluator()

    benchmark = OptimizedBenchmark(
        evaluator
    )

    evaluations, forecasts = (
        benchmark.evaluate_with_forecasts(
            object(),
            [
                _spec("ModelA"),
            ],
        )
    )

    assert (
        evaluations["ModelA"].model
        == "ModelA"
    )

    assert (
        forecasts["ModelA"].model
        == "ModelA"
    )

    assert (
        forecasts["ModelA"].fitted
        == (
            1.0,
            2.0,
            3.0,
        )
    )

    assert (
        forecasts["ModelA"].forecast
        == (
            4.0,
            5.0,
        )
    )


def test_evaluate_with_forecasts_calls_extended_evaluator_once_per_model():
    evaluator = RecordingEvaluator()

    benchmark = OptimizedBenchmark(
        evaluator
    )

    benchmark.evaluate_with_forecasts(
        object(),
        [
            _spec("ModelA"),
            _spec("ModelB"),
            _spec("ModelC"),
        ],
    )

    assert len(
        evaluator.evaluate_with_forecast_calls
    ) == 3

    assert (
        evaluator.evaluate_calls
        == []
    )


def test_evaluate_with_forecasts_uses_fresh_model_instances():
    evaluator = RecordingEvaluator()

    benchmark = OptimizedBenchmark(
        evaluator
    )

    benchmark.evaluate_with_forecasts(
        object(),
        [
            _spec(
                "ModelA",
                token="first",
            ),
            _spec(
                "ModelB",
                token="second",
            ),
        ],
    )

    first_model = (
        evaluator
        .evaluate_with_forecast_calls[0][1]
    )

    second_model = (
        evaluator
        .evaluate_with_forecast_calls[1][1]
    )

    assert first_model is not second_model

    assert first_model.token == "first"
    assert second_model.token == "second"


def test_evaluate_with_forecasts_passes_same_dataset_to_each_model():
    evaluator = RecordingEvaluator()

    benchmark = OptimizedBenchmark(
        evaluator
    )

    dataset = object()

    benchmark.evaluate_with_forecasts(
        dataset,
        [
            _spec("ModelA"),
            _spec("ModelB"),
        ],
    )

    assert (
        evaluator
        .evaluate_with_forecast_calls[0][0]
        is dataset
    )

    assert (
        evaluator
        .evaluate_with_forecast_calls[1][0]
        is dataset
    )


def test_evaluate_with_forecasts_requires_extended_evaluator_api():
    class StandardOnlyEvaluator:
        def evaluate(
            self,
            dataset,
            model,
        ):
            return DummyEvaluation(
                model=model.name
            )

    benchmark = OptimizedBenchmark(
        StandardOnlyEvaluator()
    )

    with pytest.raises(
        TypeError,
        match="evaluate_with_forecast",
    ):
        benchmark.evaluate_with_forecasts(
            object(),
            [
                _spec("ModelA"),
            ],
        )


def test_standard_only_evaluator_still_supports_existing_benchmark_api():
    class StandardOnlyEvaluator:
        def evaluate(
            self,
            dataset,
            model,
        ):
            return DummyEvaluation(
                model=model.name
            )

    benchmark = OptimizedBenchmark(
        StandardOnlyEvaluator()
    )

    results = benchmark.evaluate(
        object(),
        [
            _spec("ModelA"),
        ],
    )

    assert (
        results["ModelA"].model
        == "ModelA"
    )


def test_evaluate_with_forecasts_rejects_empty_specs():
    benchmark = OptimizedBenchmark(
        RecordingEvaluator()
    )

    with pytest.raises(
        ValueError,
        match="at least one benchmark",
    ):
        benchmark.evaluate_with_forecasts(
            object(),
            [],
        )


def test_evaluate_with_forecasts_rejects_duplicate_names():
    benchmark = OptimizedBenchmark(
        RecordingEvaluator()
    )

    with pytest.raises(
        ValueError,
        match="names must be unique",
    ):
        benchmark.evaluate_with_forecasts(
            object(),
            [
                _spec(
                    "ModelA",
                    token="first",
                ),
                _spec(
                    "ModelA",
                    token="second",
                ),
            ],
        )