import pandas as pd

import numpy as np
import pytest

from zenerestimation.forecasting.neural.windows import (
    WindowGenerator,
)

from zenerestimation.data import BatteryDataset


def test_window_generation():

    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=6, freq="QS"),
            "microVolt": [10, 20, 30, 40, 50, 60],
        }
    )

    dataset = BatteryDataset(df)

    X, y = dataset.windows(3)

    assert X.shape == (3, 3)
    assert y.shape == (3,)

    assert list(X[0]) == [10, 20, 30]
    assert y[0] == 40


def test_window_generator_rejects_zero_window():

    with pytest.raises(
        ValueError,
        match="window must be greater than zero",
    ):
        WindowGenerator(window=0)


def test_window_generator_rejects_negative_window():

    with pytest.raises(
        ValueError,
        match="window must be greater than zero",
    ):
        WindowGenerator(window=-1)


def test_window_generator_rejects_sequence_equal_to_window():

    generator = WindowGenerator(
        window=4,
    )

    values = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    with pytest.raises(
        ValueError,
        match="sequence length must be greater than",
    ):
        generator.transform(values)


def test_window_generator_rejects_sequence_shorter_than_window():

    generator = WindowGenerator(
        window=4,
    )

    values = np.array(
        [1.0, 2.0, 3.0]
    )

    with pytest.raises(
        ValueError,
        match="sequence length must be greater than",
    ):
        generator.transform(values)


def test_window_generator_creates_expected_windows():

    generator = WindowGenerator(
        window=3,
    )

    values = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    X, y = generator.transform(values)

    assert X.shape == (2, 3, 1)
    assert y.shape == (2,)

    np.testing.assert_array_equal(
        X[:, :, 0],
        np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
            ]
        ),
    )

    np.testing.assert_array_equal(
        y,
        np.array(
            [4.0, 5.0]
        ),
    )