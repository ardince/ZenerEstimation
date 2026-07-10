import pandas as pd

from zenerestimation.data import BatteryDataset


def test_dataset_report():

    df = pd.DataFrame(
        {
            "ds": pd.date_range(
                "2024-01-01",
                periods=4,
                freq="QS"
            ),
            "microVolt": [10, 11, 12, 13],
        }
    )

    dataset = BatteryDataset(df)

    report = dataset.report()

    assert report["rows"] == 4

    assert report["minimum_voltage"] == 10

    assert report["maximum_voltage"] == 13

    assert report["duplicate_dates"] == 0