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


def test_forecast_dates_uses_march_quarterly_fallback():

    data = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2024-03-01",
                    "2024-06-01",
                    "2024-12-01",
                    "2025-03-01",
                    "2025-06-01",
                ]
            ),
            "microVolt": [
                30.0,
                30.5,
                31.3,
                31.9,
                32.0,
            ],
        }
    )

    dataset = BatteryDataset(
        data
    )

    dates = dataset.forecast_dates(
        6
    )

    expected = pd.DatetimeIndex(
        [
            "2025-09-01",
            "2025-12-01",
            "2026-03-01",
            "2026-06-01",
            "2026-09-01",
            "2026-12-01",
        ]
    )

    assert dates.equals(
        expected
    )