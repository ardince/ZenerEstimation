import numpy as np
import pandas as pd

from zenerestimation.data import BatteryDataset
from zenerestimation.data.temporal import TemporalPreprocessor
from zenerestimation.evaluation import ForecastEvaluator
from zenerestimation.forecasting.kalman import KalmanForecaster

def make_kalman_dataset():
    dates = pd.date_range(
        start="2018-03-01",
        periods=24,
        freq="QS-MAR",
    )

    values = np.linspace(
        20.0,
        32.0,
        len(dates),
    )

    data = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": values,
            "is_observed": True,
        }
    )

    return BatteryDataset(data)


def test_kalman_evaluates_through_forecast_evaluator():

    dataset = make_kalman_dataset()

    model = KalmanForecaster()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    result = evaluator.evaluate(
        dataset,
        model,
    )

    assert result.model
    assert result.evaluation_steps == 4

    assert len(result.actual) == 4
    assert len(result.predicted) == 4
    assert len(result.dates) == 4

    assert np.isfinite(result.rmse)
    assert np.isfinite(result.mae)
    assert np.isfinite(result.mape)


def test_kalman_evaluation_supports_training_gap():

    dataset = make_kalman_dataset()

    dataset.data.loc[
        5,
        "microVolt",
    ] = np.nan

    dataset.data.loc[
        5,
        "is_observed",
    ] = False

    model = KalmanForecaster()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    result = evaluator.evaluate(
        dataset,
        model,
    )

    assert np.isfinite(result.rmse)
    assert np.isfinite(result.mae)
    assert np.isfinite(result.mape)


def test_kalman_evaluation_records_preprocessing():

    dataset = make_kalman_dataset()

    model = KalmanForecaster()

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    result = evaluator.evaluate(
        dataset,
        model,
    )

    preprocessing = (
        result.metadata[
            "preprocessing"
        ]
    )

    assert preprocessing[
        "enabled"
    ] is True

    assert (
        preprocessing["name"]
        == "TemporalPreprocessor"
    )


def test_kalman_evaluator_accepts_processed_dataset():

    dataset = make_kalman_dataset()

    dataset.data.loc[
        5,
        "microVolt",
    ] = np.nan

    dataset.data.loc[
        5,
        "is_observed",
    ] = False

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        preprocessor=TemporalPreprocessor(),
    )

    result = evaluator.evaluate(
        dataset,
        KalmanForecaster(),
    )

    assert result.metadata[
        "training_points"
    ] == len(dataset) - 4

    assert result.metadata[
        "validation_points"
    ] == 4