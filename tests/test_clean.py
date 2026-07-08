import pandas as pd

from zenerestimation.data import BatteryDataset


def test_clean():

    df = pd.DataFrame(
        {
            "ds": [
                "01/04/2024",
                "01/01/2024",
                "01/01/2024",
            ],
            "microVolt": [
                179,
                180,
                180,
            ],
        }
    )

    dataset = BatteryDataset(df)

    dataset.clean()

    assert len(dataset) == 2

    assert dataset.data["ds"].iloc[0] < dataset.data["ds"].iloc[1]