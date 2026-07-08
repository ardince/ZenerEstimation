import pandas as pd

from zenerestimation.data import BatteryDataset


def test_missing_quarter():

    df = pd.DataFrame(
        {
            "ds": [
                "01/01/2024",
                "01/04/2024",
                "01/10/2024",
            ],
            "microVolt": [
                180,
                179,
                177,
            ],
        }
    )

    dataset = BatteryDataset(df)

    missing = dataset.missing_periods()

    assert len(missing) == 1