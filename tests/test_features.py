import pandas as pd

from zenerestimation.data import BatteryDataset


def test_feature_engineering():

    df = pd.DataFrame(
        {
            "ds": [
                "01/01/2024",
                "01/04/2024",
                "01/07/2024",
                "01/10/2024",
            ],
            "microVolt": [
                180,
                179,
                178,
                177,
            ],
        }
    )

    dataset = BatteryDataset(df)

    dataset.add_delta()
    dataset.add_pct_change()
    dataset.add_rolling_mean()
    dataset.add_rolling_std()

    assert "delta" in dataset.data.columns
    assert "pct_change" in dataset.data.columns
    assert "rolling_mean" in dataset.data.columns
    assert "rolling_std" in dataset.data.columns