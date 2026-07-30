import numpy as np

from tests.helpers import make_dataset

#from zenerestimation.forecasting.hybrid import (
 #   KalmanLSTMForecaster,
#)

from zenerestimation.forecasting.kalman_filter import (
    AdaptiveKalmanFilter,
)


def test_forecast_length():

    ds = make_dataset()

    model = AdaptiveKalmanFilter()

    model.fit(ds)

    forecast = model.forecast(6)

    assert len(forecast) == 6


def test_forecast_is_finite():

    ds = make_dataset()

    model = AdaptiveKalmanFilter()

    model.fit(ds)

    forecast = model.forecast(8)

    assert np.isfinite(forecast).all()


def test_forecast_starts_from_last_state():

    ds = make_dataset()

    model = AdaptiveKalmanFilter()

    model.fit(ds)

    trend = model.smooth()

    forecast = model.forecast(3)

    assert abs(forecast[0] - trend[-1]) < 5.0


def test_forecast_has_constant_drift():

    ds = make_dataset()

    model = AdaptiveKalmanFilter()

    model.fit(ds)

    forecast = model.forecast(6)

    diff = np.diff(forecast)

    np.testing.assert_allclose(
        diff,
        diff[0],
        atol=1e-8,
    )


def test_forecast_is_deterministic():

    ds = make_dataset()

    model = AdaptiveKalmanFilter()

    model.fit(ds)

    f1 = model.forecast(6)

    f2 = model.forecast(6)

    np.testing.assert_allclose(f1, f2)