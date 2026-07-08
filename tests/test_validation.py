import pandas as pd
import pytest

from zenerestimation.data import BatteryDataset
from zenerestimation.exceptions import DatasetValidationError


def test_empty_dataset():

    df = pd.DataFrame()

    dataset = BatteryDataset(df)

    with pytest.raises(DatasetValidationError):
        dataset.validate()


def test_missing_column():

    df = pd.DataFrame(
        {
            "ds": ["01/01/2024"]
        }
    )

    dataset = BatteryDataset(df)

    with pytest.raises(DatasetValidationError):
        dataset.validate()


def test_invalid_voltage():

    df = pd.DataFrame(
        {
            "ds": ["01/01/2024"],
            "microVolt": ["abc"]
        }
    )

    dataset = BatteryDataset(df)

    with pytest.raises(DatasetValidationError):
        dataset.validate()