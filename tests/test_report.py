import pandas as pd

from zenerestimation.data import BatteryDataset


def test_report():

    df = pd.DataFrame(
        {
            "ds": [
                "01/01/2024",
                "01/04/2024",
                "01/07/2024",
            ],
            "microVolt": [
                180,
                179,
                178,
            ],
        }
    )

    dataset = BatteryDataset(df)

    report = dataset.report()

    assert report["rows"] == 3
    assert report["columns"] == 2
    assert report["missing_values"] == 0
    assert report["duplicate_dates"] == 0