"""
Shared contract tests for neural forecasting models.

These tests verify that all public neural forecasters
follow the common ZenerEstimation forecasting contract.

Forecast accuracy is intentionally not tested here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data import BatteryDataset
from zenerestimation.forecasting import ForecastResult
from zenerestimation.forecasting.neural.lstm import (
    LSTMForecaster,
)
from zenerestimation.forecasting.neural.gru import (
    GRUForecaster,
)


# ---------------------------------------------------------
# Neural model classes
# ---------------------------------------------------------


NEURAL_MODELS = [
    LSTMForecaster,
    GRUForecaster,
]


# ---------------------------------------------------------
# Test dataset
# ---------------------------------------------------------


@pytest.fixture
def neural_dataset():
    """
    Small deterministic quarterly dataset suitable
    for neural contract tests.
    """

    dates = pd.date_range(
        start="2021-03-01",
        periods=20,
        freq="QS-MAR",
    )

    values = np.linspace(
        20.0,
        24.0,
        len(dates),
    )

    data = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": values,
        }
    )

    return BatteryDataset(data)


# ---------------------------------------------------------
# Model factory
# ---------------------------------------------------------


def build_model(
    model_class,
):

    return model_class(
        window=4,
        units=8,
        epochs=1,
        batch_size=4,
        seed=42,
    )


# ---------------------------------------------------------
# Public fit contract
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "model_class",
    NEURAL_MODELS,
)
def test_neural_fit_returns_self(
    model_class,
    neural_dataset,
):

    model = build_model(
        model_class
    )

    result = model.fit(
        neural_dataset
    )

    assert result is model


# ---------------------------------------------------------
# Dataset ownership
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "model_class",
    NEURAL_MODELS,
)
def test_neural_fit_stores_dataset(
    model_class,
    neural_dataset,
):

    model = build_model(
        model_class
    )

    model.fit(
        neural_dataset
    )

    assert model.dataset is neural_dataset


# ---------------------------------------------------------
# ForecastResult contract
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "model_class",
    NEURAL_MODELS,
)
def test_neural_predict_returns_forecast_result(
    model_class,
    neural_dataset,
):

    model = build_model(
        model_class
    )

    model.fit(
        neural_dataset
    )

    result = model.predict(
        steps=3
    )

    assert isinstance(
        result,
        ForecastResult,
    )

    assert result.horizon == 3

    assert len(result.forecast) == 3

    assert len(result.dates) == 3


# ---------------------------------------------------------
# Central forecast-date contract
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "model_class",
    NEURAL_MODELS,
)
def test_neural_uses_dataset_forecast_dates(
    model_class,
    neural_dataset,
):

    model = build_model(
        model_class
    )

    model.fit(
        neural_dataset
    )

    steps = 3

    expected_dates = (
        neural_dataset.forecast_dates(
            steps
        )
    )

    result = model.predict(
        steps=steps
    )

    assert list(result.dates) == list(
        expected_dates
    )


# ---------------------------------------------------------
# Forecast index contract
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "model_class",
    NEURAL_MODELS,
)
def test_neural_forecast_index_matches_dates(
    model_class,
    neural_dataset,
):

    model = build_model(
        model_class
    )

    model.fit(
        neural_dataset
    )

    result = model.predict(
        steps=3
    )

    assert list(
        result.forecast.index
    ) == list(
        result.dates
    )


# ---------------------------------------------------------
# Fitted-value alignment
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "model_class",
    NEURAL_MODELS,
)
def test_neural_fitted_values_align_with_dataset(
    model_class,
    neural_dataset,
):

    window = 4

    model = build_model(
        model_class
    )

    model.fit(
        neural_dataset
    )

    assert len(model.fitted) == len(
        neural_dataset
    )

    assert model.fitted.iloc[
        :window
    ].isna().all()

    assert model.fitted.iloc[
        window:
    ].notna().all()


# ---------------------------------------------------------
# Training-only scaler contract
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "model_class",
    NEURAL_MODELS,
)
def test_neural_scaler_is_fitted_from_training_dataset(
    model_class,
    neural_dataset,
):
    """
    Verify that the neural scaler is fitted from the
    dataset supplied directly to fit().

    ForecastEvaluator leakage isolation will be tested
    separately during Milestone 13.3.
    """

    model = build_model(
        model_class
    )

    model.fit(
        neural_dataset
    )

    expected_min = float(
        neural_dataset.target.min()
    )

    expected_max = float(
        neural_dataset.target.max()
    )

    scaler = model.scaler.scaler

    assert scaler.data_min_[0] == pytest.approx(
        expected_min
    )

    assert scaler.data_max_[0] == pytest.approx(
        expected_max
    )


# ---------------------------------------------------------
# Prediction-before-fit contract
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "model_class",
    NEURAL_MODELS,
)
def test_neural_predict_before_fit_raises(
    model_class,
):

    model = build_model(
        model_class
    )

    with pytest.raises(
        RuntimeError,
        match="Model has not been trained",
    ):
        model.predict(
            steps=2
        )


# ---------------------------------------------------------
# Metadata contract
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "model_class",
    NEURAL_MODELS,
)
def test_neural_summary_contains_common_metadata(
    model_class,
):

    model = build_model(
        model_class
    )

    summary = model.summary()

    assert summary["window"] == 4
    assert summary["units"] == 8
    assert summary["epochs"] == 1
    assert summary["batch_size"] == 4
    assert summary["seed"] == 42

    assert "model" in summary
    assert "framework" in summary

    assert "tensorflow" in summary[
        "framework"
    ]

    assert "keras" in summary[
        "framework"
    ]