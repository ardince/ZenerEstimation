import numpy as np

from zenerestimation.forecasting.neural.scaler import (
    SequenceScaler,
)


def test_fit_transform():

    values = np.array(
        [10, 20, 30, 40],
        dtype=float,
    )

    scaler = SequenceScaler()

    scaled = scaler.fit_transform(values)

    assert scaled.min() == 0.0
    assert scaled.max() == 1.0


def test_inverse_transform():

    values = np.array(
        [5, 10, 15, 20],
        dtype=float,
    )

    scaler = SequenceScaler()

    scaled = scaler.fit_transform(values)

    recovered = scaler.inverse_transform(scaled)

    np.testing.assert_allclose(
        recovered,
        values,
        atol=1e-8,
    )


def test_transform_after_fit():

    train = np.array(
        [10, 20, 30],
        dtype=float,
    )

    scaler = SequenceScaler()

    scaler.fit(train)

    transformed = scaler.transform(
        np.array([15], dtype=float)
    )

    assert transformed.shape == (1,)