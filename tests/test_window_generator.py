import numpy as np

from zenerestimation.forecasting.neural.windows import (
    WindowGenerator,
)


def test_window_generator_shapes():

    values = np.arange(10)

    generator = WindowGenerator(window=3)

    X, y = generator.transform(values)

    assert X.shape == (7, 3, 1)
    assert y.shape == (7,)


def test_first_window():

    values = np.arange(10)

    generator = WindowGenerator(window=3)

    X, y = generator.transform(values)

    np.testing.assert_array_equal(
        X[0].flatten(),
        [0, 1, 2],
    )

    assert y[0] == 3


def test_last_window():

    values = np.arange(10)

    generator = WindowGenerator(window=3)

    X, y = generator.transform(values)

    np.testing.assert_array_equal(
        X[-1].flatten(),
        [6, 7, 8],
    )

    assert y[-1] == 9


def test_summary():

    generator = WindowGenerator(window=6)

    summary = generator.summary()

    assert summary["window"] == 6