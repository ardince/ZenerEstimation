import pandas as pd

from zenerestimation.data import BatteryDataset


def test_resample_monthly():

    df = pd.DataFrame(
        {
            "ds": [
                "01/01/2024",
                "15/01/2024",
                "01/02/2024",
            ],
            "microVolt": [
                180,
                182,
                179,
            ],
        }
    )

    dataset = BatteryDataset(df)

    monthly = dataset.resample("MS")

    assert len(monthly) == 2


def test_resample_returns_dataset():

    df = pd.DataFrame(
        {
            "ds": ["01/01/2024"],
            "microVolt": [180],
        }
    )

    dataset = BatteryDataset(df)

    result = dataset.resample()

    assert isinstance(result, BatteryDataset)