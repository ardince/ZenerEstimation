import pandas as pd

from zenerestimation.data import BatteryDataset


def test_dataset_creation():

    df = pd.DataFrame(
        {
            "ds": ["01/01/2024"],
            "microVolt": [180.0],
        }
    )

    dataset = BatteryDataset(df)

    assert len(dataset.data) == 1