"""
Integration test for ForecastEvaluator with ARIMAForecaster.
"""

import numpy as np
import pandas as pd

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.evaluation import ForecastEvaluator
from zenerestimation.forecasting.arima import ARIMAForecaster


def make_dataset():
    """
    Create a deterministic quarterly dataset.
    """

    dates = pd.date_range(
        "2018-03-01",
        periods=20,
        freq="3MS",
    )

    values = (
        20.0
        + 0.5 * np.arange(20)
        + 0.05 * np.sin(
            np.arange(20)
        )
    )

    return BatteryDataset(
        pd.DataFrame(
            {
                "ds": dates,
                "microVolt": values,
            }
        )
    )


def test_forecast_evaluator_with_arima():
    dataset = make_dataset()

    evaluator = ForecastEvaluator(
        evaluation_steps=4
    )

    model = ARIMAForecaster()

    result = evaluator.evaluate(
        dataset,
        model,
    )

    assert result.model == "ARIMA"

    assert result.evaluation_steps == 4

    assert len(result.actual) == 4

    assert len(result.predicted) == 4

    assert len(result.dates) == 4

    assert result.rmse >= 0.0

    assert result.mae >= 0.0

    assert result.mape >= 0.0

    assert result.metadata[
        "training_points"
    ] == 16

    assert result.metadata[
        "validation_points"
    ] == 4