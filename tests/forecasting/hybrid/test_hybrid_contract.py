import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.forecasting.result import ForecastResult
from zenerestimation.forecasting.neural.lstm import (
    LSTMForecaster,
)
from zenerestimation.forecasting.hybrid.linear_trend_lstm import (
    LinearTrendLSTMForecaster,
)
from zenerestimation.forecasting.hybrid.kalman_lstm import (
    KalmanLSTMForecaster,
)

def build_lstm():
    """
    Build a small residual LSTM for contract tests.
    """

    return LSTMForecaster(
        window=4,
        units=8,
        epochs=1,
        batch_size=4,
        seed=42,
    )


def build_hybrid(model_type):
    """
    Construct a hybrid forecaster for shared-contract
    regression tests.
    """

    if model_type == "linear":
        return LinearTrendLSTMForecaster(
            lstm_model=build_lstm(),
        )

    if model_type == "kalman":
        return KalmanLSTMForecaster(
            lstm_model=build_lstm(),
        )

    raise ValueError(
        f"Unknown hybrid model: {model_type}"
    )


def build_dataset():
    """
    Build a regular quarterly dataset for shared
    hybrid contract tests.
    """

    dates = pd.date_range(
        "2018-03-01",
        periods=28,
        freq="QS-MAR",
    )

    t = np.arange(
        len(dates),
        dtype=float,
    )

    values = (
        10.0
        + 0.25 * t
        + 0.15 * np.sin(t)
    )

    data = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": values,
        }
    )

    dataset = BatteryDataset(data)

    dataset._battery = "TEST-BATTERY"

    return dataset


HYBRID_TYPES = [
    "linear",
    "kalman",
]


@pytest.mark.parametrize(
    "model_type",
    HYBRID_TYPES,
)
def test_hybrid_residual_model_contract(
    model_type,
):
    model = build_hybrid(model_type)

    assert model.residual_model is model.lstm
    assert model.window == model.lstm.window
    assert model.window == 4


@pytest.mark.parametrize(
    "model_type",
    HYBRID_TYPES,
)
def test_hybrid_fit_returns_self(
    model_type,
):
    dataset = build_dataset()
    model = build_hybrid(model_type)

    result = model.fit(dataset)

    assert result is model


@pytest.mark.parametrize(
    "model_type",
    HYBRID_TYPES,
)
def test_hybrid_fit_retains_dataset(
    model_type,
):
    dataset = build_dataset()
    model = build_hybrid(model_type)

    model.fit(dataset)

    assert model.dataset is dataset


@pytest.mark.parametrize(
    "model_type",
    HYBRID_TYPES,
)
def test_hybrid_predict_returns_forecast_result(
    model_type,
):
    dataset = build_dataset()
    model = build_hybrid(model_type)

    model.fit(dataset)

    result = model.predict(3)

    assert isinstance(
        result,
        ForecastResult,
    )


@pytest.mark.parametrize(
    "model_type",
    HYBRID_TYPES,
)
def test_hybrid_predict_preserves_horizon(
    model_type,
):
    dataset = build_dataset()
    model = build_hybrid(model_type)

    model.fit(dataset)

    result = model.predict(3)

    assert result.horizon == 3
    assert len(result.forecast) == 3
    assert len(result.dates) == 3


@pytest.mark.parametrize(
    "model_type",
    HYBRID_TYPES,
)
def test_hybrid_uses_dataset_forecast_dates(
    model_type,
):
    dataset = build_dataset()
    model = build_hybrid(model_type)

    model.fit(dataset)

    result = model.predict(3)

    expected_dates = (
        dataset.forecast_dates(3)
    )

    assert pd.DatetimeIndex(
        result.dates
    ).equals(
        pd.DatetimeIndex(expected_dates)
    )


@pytest.mark.parametrize(
    "model_type",
    HYBRID_TYPES,
)
def test_hybrid_forecast_index_matches_dates(
    model_type,
):
    dataset = build_dataset()
    model = build_hybrid(model_type)

    model.fit(dataset)

    result = model.predict(3)

    assert result.forecast.index.equals(
        pd.DatetimeIndex(result.dates)
    )


@pytest.mark.parametrize(
    "model_type",
    HYBRID_TYPES,
)
def test_hybrid_summary_contract(
    model_type,
):
    model = build_hybrid(model_type)

    summary = model.summary()

    assert summary["family"] == "Hybrid"

    assert (
        summary["residual_model"]
        == model.lstm.__class__.__name__
    )

    assert summary["window"] == 4