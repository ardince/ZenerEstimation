import pandas as pd

from zenerestimation.data import BatteryDataset


def test_dataset_creation():

    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=5),
            "Value": [1, 2, 3, 4, 5],
        }
    )

    dataset = BatteryDataset(df)

    assert dataset.n_samples == 5
    assert dataset.value_column == "Value"