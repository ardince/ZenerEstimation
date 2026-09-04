from pathlib import Path

import pandas as pd
import pytest

from zenerestimation.data import (
    BatteryDataset,
)

def make_processed_csv(
    tmp_path: Path,
) -> Path:

    filename = (
        tmp_path
        / "TEST-BATTERY.csv"
    )

    data = pd.DataFrame(
        {
            "ds": [
                "2023-03-01",
                "2023-06-01",
                "2023-09-01",
                "2023-12-01",
            ],
            "microVolt": [
                30.0,
                31.0,
                None,
                33.0,
            ],
            "is_observed": [
                True,
                True,
                False,
                True,
            ],
        }
    )

    data.to_csv(
        filename,
        index=False,
    )

    return filename


def test_from_processed_csv():

    # implemented below using tmp_path
    pass


def test_from_processed_csv(
    tmp_path,
):

    filename = make_processed_csv(
        tmp_path
    )

    dataset = (
        BatteryDataset
        .from_processed_csv(
            filename
        )
    )

    assert isinstance(
        dataset,
        BatteryDataset,
    )

    assert len(
        dataset.data
    ) == 4


def test_processed_columns_preserved(
    tmp_path,
):

    filename = make_processed_csv(
        tmp_path
    )

    dataset = (
        BatteryDataset
        .from_processed_csv(
            filename
        )
    )

    assert (
        "ds"
        in dataset.data.columns
    )

    assert (
        "microVolt"
        in dataset.data.columns
    )

    assert (
        "is_observed"
        in dataset.data.columns
    )


def test_processed_missing_value_is_preserved(
    tmp_path,
):

    filename = make_processed_csv(
        tmp_path
    )

    dataset = (
        BatteryDataset
        .from_processed_csv(
            filename
        )
    )

    value = dataset.data.loc[
        dataset.data[
            "ds"
        ]
        == pd.Timestamp(
            "2023-09-01"
        ),
        "microVolt",
    ].iloc[0]

    assert pd.isna(
        value
    )


def test_observation_flag_is_preserved(
    tmp_path,
):

    filename = make_processed_csv(
        tmp_path
    )

    dataset = (
        BatteryDataset
        .from_processed_csv(
            filename
        )
    )

    assert (
        dataset.observed_mask.tolist()
        == [
            True,
            True,
            False,
            True,
        ]
    )


def test_observed_rows(
    tmp_path,
):

    filename = make_processed_csv(
        tmp_path
    )

    dataset = (
        BatteryDataset
        .from_processed_csv(
            filename
        )
    )

    assert (
        dataset.observed_rows
        == 3
    )


def test_missing_period_count(
    tmp_path,
):

    filename = make_processed_csv(
        tmp_path
    )

    dataset = (
        BatteryDataset
        .from_processed_csv(
            filename
        )
    )

    assert (
        dataset.missing_period_count
        == 1
    )


def test_battery_inferred_from_filename(
    tmp_path,
):

    filename = make_processed_csv(
        tmp_path
    )

    dataset = (
        BatteryDataset
        .from_processed_csv(
            filename
        )
    )

    assert (
        dataset.battery
        == "TEST-BATTERY"
    )


def test_processed_battery_override(
    tmp_path,
):

    filename = make_processed_csv(
        tmp_path
    )

    dataset = (
        BatteryDataset
        .from_processed_csv(
            filename,
            battery="CUSTOM",
        )
    )

    assert (
        dataset.battery
        == "CUSTOM"
    )


def test_processed_source_metadata(
    tmp_path,
):

    filename = make_processed_csv(
        tmp_path
    )

    dataset = (
        BatteryDataset
        .from_processed_csv(
            filename
        )
    )

    assert (
        dataset.is_processed
        is True
    )

    assert (
        dataset.source_type
        == "processed"
    )

    assert (
        dataset.source_path
        == filename
    )


def test_processed_file_not_found(
    tmp_path,
):

    filename = (
        tmp_path
        / "missing.csv"
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            "processed dataset "
            "not found"
        ),
    ):

        BatteryDataset.from_processed_csv(
            filename
        )


@pytest.mark.parametrize(
    "column",
    [
        "ds",
        "microVolt",
        "is_observed",
    ],
)
def test_processed_required_columns(
    tmp_path,
    column,
):

    filename = make_processed_csv(
        tmp_path
    )

    data = pd.read_csv(
        filename
    )

    data = data.drop(
        columns=[column]
    )

    data.to_csv(
        filename,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="required columns",
    ):

        BatteryDataset.from_processed_csv(
            filename
        )


def test_observed_row_requires_target(
    tmp_path,
):

    filename = (
        tmp_path
        / "TEST.csv"
    )

    pd.DataFrame(
        {
            "ds": [
                "2023-03-01",
            ],
            "microVolt": [
                None,
            ],
            "is_observed": [
                True,
            ],
        }
    ).to_csv(
        filename,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            "observed rows cannot "
            "have missing"
        ),
    ):

        BatteryDataset.from_processed_csv(
            filename
        )


def test_inserted_row_requires_missing_target(
    tmp_path,
):

    filename = (
        tmp_path
        / "TEST.csv"
    )

    pd.DataFrame(
        {
            "ds": [
                "2023-03-01",
            ],
            "microVolt": [
                30.0,
            ],
            "is_observed": [
                False,
            ],
        }
    ).to_csv(
        filename,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            "inserted rows must "
            "have missing"
        ),
    ):

        BatteryDataset.from_processed_csv(
            filename
        )


def test_missing_periods_remains_callable(
    tmp_path,
):

    filename = make_processed_csv(
        tmp_path
    )

    dataset = (
        BatteryDataset
        .from_processed_csv(
            filename
        )
    )

    assert callable(
        dataset.missing_periods
    )