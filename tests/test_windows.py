import pandas as pd

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