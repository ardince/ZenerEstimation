import pandas as pd

from zenerestimation.data import BatteryDataset


def test_statistics():

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

    stats = dataset.statistics()

    assert stats["count"] == 4
    assert stats["minimum"] == 177
    assert stats["maximum"] == 180