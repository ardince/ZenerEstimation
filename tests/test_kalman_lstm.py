import numpy as np

from tests.helpers import make_dataset

from zenerestimation.forecasting.hybrid import (
    KalmanLSTMForecaster,
)

#from zenerestimation.forecasting.neural.testing import (
 #   DummyResidualModel,
#)

from zenerestimation.forecasting.kalman_filter import (
    AdaptiveKalmanFilter,
)


def make_model():

    return KalmanLSTMForecaster(

        kalman_model=AdaptiveKalmanFilter(),

        #lstm_model=DummyResidualModel(),

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

        model.trend

        +

        residual_ds.target.values

    )

    np.testing.assert_allclose(

        reconstructed,

        ds.target.values,

        atol=1e-8,

    )


# ---------------------------------------------------------


def test_summary():

    model = make_model()

    summary = model.summary()

    assert summary["family"] == "Hybrid"

    assert summary["architecture"] == "KalmanLSTM"

    assert summary["trend"] == "Adaptive Kalman Filter"