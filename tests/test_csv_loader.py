import pandas as pd

from zenerestimation.data import BatteryDataset


def test_csv_loader(tmp_path):

    df = pd.DataFrame(
        {
            "ds": ["01/01/2024", "01/04/2024"],
            "microVolt": [182.5, 181.9],
        }
    )

    filename = tmp_path / "battery.csv"

    df.to_csv(filename, index=False)

    dataset = BatteryDataset.from_csv(filename)

    assert len(dataset.data) == 2