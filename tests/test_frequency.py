import pandas as pd

from zenerestimation.data import BatteryDataset


def test_quarterly_detection():

    df = pd.DataFrame(
        {
            "ds": [
                "01/01/2024",
                "01/04/2024",
                "01/07/2024",
                "01/10/2024",
            ],
            "microVolt": [
                182,
                181,
                180,
                179,
            ],
        }
    )

    dataset = BatteryDataset(df)

    assert dataset.detect_frequency() == "Quarterly"