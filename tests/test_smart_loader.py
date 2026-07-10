from pathlib import Path

import pandas as pd

from zenerestimation.data import BatteryDataset


def test_month_year_loader(tmp_path):

    df = pd.DataFrame(
        {
            "Month": [3, 6],
            "Year": [2024, 2024],
            "microVolt": [180.0, 179.5],
        }
    )

    filename = tmp_path / "732B-5610110.csv"

    df.to_csv(filename, index=False)

    dataset = BatteryDataset.from_csv(filename)

    assert "ds" in dataset.data.columns
    assert "microVolt" in dataset.data.columns

    assert dataset.metadata["format"] == "MONTH_YEAR"
    assert dataset.metadata["battery_id"] == "732B-5610110"

    assert len(dataset) == 2


def test_ds_loader(tmp_path):

    df = pd.DataFrame(
        {
            "ds": [
                "01/03/2024",
                "01/06/2024",
            ],
            "microVolt": [
                180.0,
                179.2,
            ],
        }
    )

    filename = tmp_path / "battery.csv"

    df.to_csv(filename, index=False)

    dataset = BatteryDataset.from_csv(filename)

    assert "ds" in dataset.data.columns
    assert "microVolt" in dataset.data.columns

    assert dataset.metadata["format"] == "DS"

    assert len(dataset) == 2


def test_ds_volt_loader(tmp_path):

    df = pd.DataFrame(
        {
            "ds": [
                "01/03/2024",
                "01/06/2024",
            ],
            "Volt": [
                10.00003,
                10.00004,
            ],
            "microVolt": [
                30.2,
                31.0,
            ],
        }
    )

    filename = tmp_path / "battery2.csv"

    df.to_csv(filename, index=False)

    dataset = BatteryDataset.from_csv(filename)

    assert "Volt" in dataset.data.columns
    assert "microVolt" in dataset.data.columns

    assert dataset.metadata["format"] == "DS_VOLT"

    assert len(dataset) == 2


def test_date_value_loader(tmp_path):

    df = pd.DataFrame(
        {
            "Date": [
                "2024-01-01",
                "2024-04-01",
            ],
            "Value": [
                12.5,
                13.1,
            ],
        }
    )

    filename = tmp_path / "example.csv"

    df.to_csv(filename, index=False)

    dataset = BatteryDataset.from_csv(filename)

    assert "ds" in dataset.data.columns
    assert "microVolt" in dataset.data.columns

    assert dataset.metadata["format"] == "DATE_VALUE"

    assert len(dataset) == 2


def test_unknown_format(tmp_path):

    df = pd.DataFrame(
        {
            "A": [1],
            "B": [2],
        }
    )

    filename = tmp_path / "unknown.csv"

    df.to_csv(filename, index=False)

    import pytest

    with pytest.raises(ValueError):

        BatteryDataset.from_csv(filename)


def test_loader_preserves_extra_columns(tmp_path):

    df = pd.DataFrame(
        {
            "ds": [
                "01/03/2024",
                "01/06/2024",
            ],
            "Volt": [
                10.000031,
                10.000032,
            ],
            "microVolt": [
                31.0,
                32.0,
            ],
            "Temperature": [
                23.0,
                24.0,
            ],
        }
    )

    filename = tmp_path / "battery.csv"

    df.to_csv(filename, index=False)

    dataset = BatteryDataset.from_csv(filename)

    assert "Volt" in dataset.data.columns
    assert "Temperature" in dataset.data.columns