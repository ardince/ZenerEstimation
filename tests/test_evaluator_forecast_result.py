"""Tests for ForecastEvaluator forecast-result retention."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.evaluation.evaluator import ForecastEvaluator
from zenerestimation.evaluation.result import EvaluationResult


@dataclass
class DummyForecastResult:
    """Minimal forecast result used by evaluator tests."""

    model: str
    forecast: tuple[float, ...]
    dates: tuple[pd.Timestamp, ...]
    fitted: pd.Series | None = None


class CountingForecastModel:
    """
    Minimal forecasting model that records fit/predict calls.

    This lets the tests verify that retaining ForecastResult
    evidence does not cause additional model execution.
    """

    def __init__(self):
        self.fit_calls = 0
        self.predict_calls = 0
        self.training_dataset = None

    def fit(self, dataset):
        self.fit_calls += 1
        self.training_dataset = dataset

        return self

    def predict(self, steps):
        self.predict_calls += 1

        if steps != 2:
            raise ValueError(
                "test model expects a two-step horizon"
            )

        fitted = pd.Series(
            [
                10.0,
                11.0,
                12.0,
                13.0,
            ],
            name="fitted",
        )

        return DummyForecastResult(
            model="CountingModel",
            forecast=(
                13.5,
                14.5,
            ),
            dates=(
                pd.Timestamp("2023-03-01"),
                pd.Timestamp("2023-06-01"),
            ),
            fitted=fitted,
        )


def create_dataset():
    """Create a prepared six-point quarterly test dataset."""

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

    dataset = BatteryDataset(
        frame
    )

    dataset.prepare()

    return dataset


def test_evaluate_preserves_original_return_contract():
    """
    evaluate() must continue returning only EvaluationResult.
    """

    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    model = CountingForecastModel()

    result = evaluator.evaluate(
        create_dataset(),
        model,
    )

    assert isinstance(
        result,
        EvaluationResult,
    )

    assert result.model == (
        "CountingModel"
    )

    assert result.evaluation_steps == 2

    assert result.actual == (
        14.0,
        15.0,
    )

    assert result.predicted == (
        13.5,
        14.5,
    )


def test_evaluate_runs_model_once():
    """
    Existing evaluate() must still perform exactly one fit
    and one prediction.
    """

    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    model = CountingForecastModel()

    evaluator.evaluate(
        create_dataset(),
        model,
    )

    assert model.fit_calls == 1
    assert model.predict_calls == 1


def test_evaluate_with_forecast_returns_evaluation_result():
    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    model = CountingForecastModel()

    evaluation, _ = (
        evaluator.evaluate_with_forecast(
            create_dataset(),
            model,
        )
    )

    assert isinstance(
        evaluation,
        EvaluationResult,
    )

    assert evaluation.model == (
        "CountingModel"
    )


def test_evaluate_with_forecast_returns_original_forecast_result():
    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    model = CountingForecastModel()

    _, forecast = (
        evaluator.evaluate_with_forecast(
            create_dataset(),
            model,
        )
    )

    assert isinstance(
        forecast,
        DummyForecastResult,
    )

    assert forecast.model == (
        "CountingModel"
    )

    assert forecast.forecast == (
        13.5,
        14.5,
    )


def test_evaluation_predictions_match_forecast_evidence():
    """
    EvaluationResult predictions must come from the same
    ForecastResult returned to the caller.
    """

    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    model = CountingForecastModel()

    evaluation, forecast = (
        evaluator.evaluate_with_forecast(
            create_dataset(),
            model,
        )
    )

    assert evaluation.predicted == tuple(
        float(value)
        for value in forecast.forecast
    )


def test_evaluate_with_forecast_runs_model_once():
    """
    Retaining forecast evidence must not trigger a second
    fit or prediction.
    """

    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    model = CountingForecastModel()

    evaluator.evaluate_with_forecast(
        create_dataset(),
        model,
    )

    assert model.fit_calls == 1
    assert model.predict_calls == 1


def test_forecast_result_retains_fitted_values():
    """
    Historical fitted values must remain available for
    downstream presentation layers.
    """

    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    model = CountingForecastModel()

    _, forecast = (
        evaluator.evaluate_with_forecast(
            create_dataset(),
            model,
        )
    )

    assert forecast.fitted is not None

    assert forecast.fitted.tolist() == [
        10.0,
        11.0,
        12.0,
        13.0,
    ]


def test_evaluate_with_forecast_uses_same_holdout_actuals():
    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    model = CountingForecastModel()

    evaluation, _ = (
        evaluator.evaluate_with_forecast(
            create_dataset(),
            model,
        )
    )

    assert evaluation.actual == (
        14.0,
        15.0,
    )

    assert tuple(
        pd.Timestamp(value)
        for value in evaluation.dates
    ) == (
        pd.Timestamp("2023-03-01"),
        pd.Timestamp("2023-06-01"),
    )


def test_evaluate_with_forecast_uses_training_partition_only():
    """
    The model must still be fitted only on observations
    preceding the two-point holdout.
    """

    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    model = CountingForecastModel()

    evaluator.evaluate_with_forecast(
        create_dataset(),
        model,
    )

    assert model.training_dataset is not None

    assert len(
        model.training_dataset
    ) == 4

    assert (
        model.training_dataset
        .data["microVolt"]
        .tolist()
        == [
            10.0,
            11.0,
            12.0,
            13.0,
        ]
    )


def test_evaluate_and_evaluate_with_forecast_have_same_metrics():
    """
    Both public APIs must implement the same scientific
    evaluation contract.

    Separate model instances are used because each call is an
    independent evaluation.
    """

    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    standard = evaluator.evaluate(
        create_dataset(),
        CountingForecastModel(),
    )

    extended, _ = (
        evaluator.evaluate_with_forecast(
            create_dataset(),
            CountingForecastModel(),
        )
    )

    assert extended.rmse == pytest.approx(
        standard.rmse
    )

    assert extended.mae == pytest.approx(
        standard.mae
    )

    assert extended.mape == pytest.approx(
        standard.mape
    )

    assert extended.actual == (
        standard.actual
    )

    assert extended.predicted == (
        standard.predicted
    )

    assert extended.dates == (
        standard.dates
    )


def test_evaluate_with_forecast_preserves_metadata():
    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    evaluation, _ = (
        evaluator.evaluate_with_forecast(
            create_dataset(),
            CountingForecastModel(),
        )
    )

    assert evaluation.metadata[
        "training_points"
    ] == 4

    assert evaluation.metadata[
        "validation_points"
    ] == 2

    assert evaluation.metadata[
        "preprocessing"
    ] == {
        "enabled": False,
    }

    assert evaluation.metadata[
        "evaluation_end"
    ] == "2023-06-01"