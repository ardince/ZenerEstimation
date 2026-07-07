import pandas as pd

from zenerestimation.data import BatteryDataset


def test_summary():

    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3),
            "Value": [10, 11, 12],
        }
    )

    dataset = BatteryDataset(df)

    summary = dataset.summary()

    assert summary["samples"] == 3