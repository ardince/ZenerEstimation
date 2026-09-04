"""
Tests for EvaluationResult.
"""

from datetime import datetime

import pytest

from zenerestimation.evaluation import EvaluationResult


def make_result():
    return EvaluationResult(
        model="ARIMA",
        evaluation_steps=3,
        rmse=0.42,
        mae=0.31,
        mape=1.08,
        actual=(30.0, 30.5, 31.0),
        predicted=(30.1, 30.4, 30.9),
        dates=(
            datetime(2024, 3, 1),
            datetime(2024, 6, 1),
            datetime(2024, 9, 1),
        ),
        metadata={
            "training_points": 100,
        },
    )


def test_evaluation_result_creation():
    result = make_result()

    assert result.model == "ARIMA"
    assert result.evaluation_steps == 3

    assert result.rmse == pytest.approx(0.42)
    assert result.mae == pytest.approx(0.31)
    assert result.mape == pytest.approx(1.08)


def test_horizon_alias():
    result = make_result()

    assert result.horizon == 3


def test_actual_values():
    result = make_result()

    assert result.actual == (
        30.0,
        30.5,
        31.0,
    )


def test_predicted_values():
    result = make_result()

    assert result.predicted == (
        30.1,
        30.4,
        30.9,
    )


def test_summary():
    result = make_result()

    summary = result.summary()

    assert summary == {
        "model": "ARIMA",
        "evaluation_steps": 3,
        "rmse": 0.42,
        "mae": 0.31,
        "mape": 1.08,
    }


def test_to_dict():
    result = make_result()

    data = result.to_dict()

    assert data["status"] == "evaluated"
    assert data["method"] == "holdout"

    assert data["model"] == "ARIMA"
    assert data["evaluation_steps"] == 3

    assert data["actual"] == [
        30.0,
        30.5,
        31.0,
    ]

    assert data["predicted"] == [
        30.1,
        30.4,
        30.9,
    ]


def test_dates_are_serialized():
    result = make_result()

    data = result.to_dict()

    assert data["dates"] == [
        "2024-03-01 00:00:00",
        "2024-06-01 00:00:00",
        "2024-09-01 00:00:00",
    ]


def test_metadata():
    result = make_result()

    assert result.metadata == {
        "training_points": 100,
    }


def test_metadata_is_defensively_copied():
    metadata = {
        "training_points": 100,
    }

    result = EvaluationResult(
        model="ARIMA",
        evaluation_steps=1,
        rmse=0.1,
        mae=0.1,
        mape=0.1,
        actual=(30.0,),
        predicted=(30.1,),
        dates=(
            datetime(2024, 3, 1),
        ),
        metadata=metadata,
    )

    metadata["training_points"] = 999

    assert (
        result.metadata["training_points"]
        == 100
    )


def test_to_dict_returns_metadata_copy():
    result = make_result()

    data = result.to_dict()

    data["metadata"]["training_points"] = 999

    assert (
        result.metadata["training_points"]
        == 100
    )


def test_empty_model_raises():
    with pytest.raises(
        ValueError,
        match="model must be a non-empty string",
    ):
        EvaluationResult(
            model="",
            evaluation_steps=1,
            rmse=0.1,
            mae=0.1,
            mape=0.1,
            actual=(1.0,),
            predicted=(1.0,),
            dates=(datetime(2024, 3, 1),),
        )


def test_zero_evaluation_steps_raises():
    with pytest.raises(
        ValueError,
        match="evaluation_steps must be greater than zero",
    ):
        EvaluationResult(
            model="ARIMA",
            evaluation_steps=0,
            rmse=0.1,
            mae=0.1,
            mape=0.1,
            actual=(),
            predicted=(),
            dates=(),
        )


def test_actual_length_mismatch_raises():
    with pytest.raises(
        ValueError,
        match="actual length must match evaluation_steps",
    ):
        EvaluationResult(
            model="ARIMA",
            evaluation_steps=2,
            rmse=0.1,
            mae=0.1,
            mape=0.1,
            actual=(1.0,),
            predicted=(1.0, 2.0),
            dates=(
                datetime(2024, 3, 1),
                datetime(2024, 6, 1),
            ),
        )


def test_predicted_length_mismatch_raises():
    with pytest.raises(
        ValueError,
        match="predicted length must match evaluation_steps",
    ):
        EvaluationResult(
            model="ARIMA",
            evaluation_steps=2,
            rmse=0.1,
            mae=0.1,
            mape=0.1,
            actual=(1.0, 2.0),
            predicted=(1.0,),
            dates=(
                datetime(2024, 3, 1),
                datetime(2024, 6, 1),
            ),
        )


def test_dates_length_mismatch_raises():
    with pytest.raises(
        ValueError,
        match="dates length must match evaluation_steps",
    ):
        EvaluationResult(
            model="ARIMA",
            evaluation_steps=2,
            rmse=0.1,
            mae=0.1,
            mape=0.1,
            actual=(1.0, 2.0),
            predicted=(1.0, 2.0),
            dates=(
                datetime(2024, 3, 1),
            ),
        )


@pytest.mark.parametrize(
    "metric",
    [
        "rmse",
        "mae",
        "mape",
    ],
)
def test_non_finite_metrics_raise(metric):
    kwargs = {
        "model": "ARIMA",
        "evaluation_steps": 1,
        "rmse": 0.1,
        "mae": 0.1,
        "mape": 0.1,
        "actual": (1.0,),
        "predicted": (1.0,),
        "dates": (
            datetime(2024, 3, 1),
        ),
    }

    kwargs[metric] = float("nan")

    with pytest.raises(
        ValueError,
        match=f"{metric} must be finite",
    ):
        EvaluationResult(**kwargs)


def test_result_is_frozen():
    result = make_result()

    with pytest.raises(
        AttributeError
    ):
        result.rmse = 100.0


def test_repr():
    result = make_result()

    text = repr(result)

    assert "EvaluationResult" in text
    assert "ARIMA" in text
    assert "steps=3" in text
    assert "rmse=0.420000" in text