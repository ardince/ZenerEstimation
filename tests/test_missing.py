import pandas as pd

from zenerestimation.data import BatteryDataset


def test_interpolation():

    df = pd.DataFrame(
        {
            "ds": [
                "01/01/2024",
                "01/04/2024",
                "01/07/2024",
            ],
            "microVolt": [
                180,
                None,
                178,
            ],
        }
    )

    dataset = BatteryDataset(df)

    dataset.interpolate()

    assert dataset.has_missing() is False