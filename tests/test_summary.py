import pandas as pd

from zenerestimation.data import BatteryDataset


def test_summary():

    df = pd.DataFrame(
        {
            "ds": [
                "01/01/2024",
                "01/04/2024",
                "01/07/2024",
            ],
            "microVolt": [
                182.3,
                181.8,
                181.1,
            ],
        }
    )

    dataset = BatteryDataset(df)

    summary = dataset.summary()

    assert summary["rows"] == 3
    assert summary["columns"] == 2
    assert summary["missing"] == 0