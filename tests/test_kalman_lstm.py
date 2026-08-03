import numpy as np

from tests.helpers import make_dataset

from zenerestimation.forecasting.hybrid import (
    KalmanLSTMForecaster,
)

from zenerestimation.forecasting.kalman_filter import (
    AdaptiveKalmanFilter,
)


def make_model():

    return KalmanLSTMForecaster(

        kalman_model=AdaptiveKalmanFilter(),

    )


# ---------------------------------------------------------


def test_constructor():

    model = make_model()

    assert model.filter is not None

    assert model.lstm is not None


# ---------------------------------------------------------


def test_prepare_residuals():

    ds = make_dataset()

    model = make_model()

    residual_ds = model.prepare_residuals(ds)

    reconstructed = (

        model._trend

        +

        residual_ds.target.values

    )

    np.testing.assert_allclose(

        reconstructed,

        ds.target.values,

        atol=1e-8,

    )

    assert model.trend() is not None

    assert model.residuals() is not None

    assert model.verify_decomposition()


# ---------------------------------------------------------


def test_summary():

    model = make_model()

    summary = model.summary()

    assert summary["family"] == "Hybrid"

    assert summary["architecture"] == "KalmanLSTM"

    assert summary["trend"] == "Adaptive Kalman Filter"


def test_trend_forecast_cache():

    ds = make_dataset()

    model = make_model()

    model.fit(ds)

    trend = model.forecast_trend(6)

    assert trend is model.trend_forecast()


def test_hybrid_forecast_is_sum():

    ds = make_dataset()

    model = make_model()

    model.fit(ds)

    result = model.predict(6)

    np.testing.assert_allclose(

        model._hybrid_prediction,

        model._trend_prediction
        +
        model._residual_prediction,

        atol=1e-8,

    )


def test_prediction_cache():

    ds = make_dataset()

    model = make_model()

    model.fit(ds)

    model.predict(6)

    assert model._trend_prediction is not None

    assert model._residual_prediction is not None

    assert model._hybrid_prediction is not None


def test_prediction_metadata():

    ds = make_dataset()

    model = make_model()

    model.fit(ds)

    result = model.predict(6)

    assert result.metadata["architecture"] == "KalmanLSTM"

    assert result.metadata["combination"] == "additive"

    assert result.metadata["trend_model"] == "AdaptiveKalmanFilter"

    assert result.metadata["residual_model"] == "LSTMForecaster"


def test_prediction_lengths():

    ds = make_dataset()

    model = make_model()

    model.fit(ds)

    result = model.predict(6)

    assert len(model._trend_prediction) == 6

    assert len(model._residual_prediction) == 6

    assert len(model._hybrid_prediction) == 6

    assert len(result.forecast) == 6