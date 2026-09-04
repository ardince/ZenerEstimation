"""
Tests for DatasetProcessingResult.
"""

import pandas as pd
import pytest

from zenerestimation.data.processing import (
    DatasetProcessingResult,
)


def make_data():

    return pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2024-03-01",
                    "2024-06-01",
                    "2024-09-01",
                    "2024-12-01",
                ]
            ),
            "microVolt": [
                30.0,
                30.5,
                float("nan"),
                31.0,
            ],
            "is_observed": [
                True,
                True,
                False,
                True,
            ],
        }
    )


def make_result():

    return DatasetProcessingResult(
        battery="TEST-001",
        data=make_data(),
        source_rows=3,
        processed_rows=4,
        missing_periods=1,
        frequency="QS-MAR",
        metadata={
            "source": "raw.csv",
        },
    )


def test_creation():

    result = make_result()

    assert result.battery == "TEST-001"

    assert result.source_rows == 3

    assert result.processed_rows == 4

    assert result.missing_periods == 1

    assert result.frequency == "QS-MAR"


def test_observed_rows():

    result = make_result()

    assert result.observed_rows == 3


def test_inserted_rows():

    result = make_result()

    assert result.inserted_rows == 1


def test_summary():

    result = make_result()

    assert result.summary() == {
        "battery": "TEST-001",
        "source_rows": 3,
        "processed_rows": 4,
        "observed_rows": 3,
        "missing_periods": 1,
        "frequency": "QS-MAR",
    }


def test_metadata_dict():

    result = make_result()

    metadata = result.metadata_dict()

    assert metadata[
        "battery"
    ] == "TEST-001"

    assert metadata[
        "processing_version"
    ] == "1.0"

    assert metadata[
        "target"
    ] == "microVolt"

    assert metadata[
        "source"
    ] == "raw.csv"


def test_dataframe_returns_copy():

    result = make_result()

    frame = result.dataframe()

    frame.loc[
        0,
        "microVolt",
    ] = 999.0

    assert (
        result.data.loc[
            0,
            "microVolt",
        ]
        == 30.0
    )


def test_input_dataframe_is_defensively_copied():

    data = make_data()

    result = DatasetProcessingResult(
        battery="TEST-001",
        data=data,
        source_rows=3,
        processed_rows=4,
        missing_periods=1,
        frequency="QS-MAR",
    )

    data.loc[
        0,
        "microVolt",
    ] = 999.0

    assert (
        result.data.loc[
            0,
            "microVolt",
        ]
        == 30.0
    )


def test_metadata_is_defensively_copied():

    metadata = {
        "source": "raw.csv",
    }

    result = DatasetProcessingResult(
        battery="TEST-001",
        data=make_data(),
        source_rows=3,
        processed_rows=4,
        missing_periods=1,
        frequency="QS-MAR",
        metadata=metadata,
    )

    metadata[
        "source"
    ] = "changed.csv"

    assert (
        result.metadata[
            "source"
        ]
        == "raw.csv"
    )


def test_empty_battery_raises():

    with pytest.raises(
        ValueError,
        match=(
            "battery must be a "
            "non-empty string"
        ),
    ):

        DatasetProcessingResult(
            battery="",
            data=make_data(),
            source_rows=3,
            processed_rows=4,
            missing_periods=1,
            frequency="QS-MAR",
        )


def test_invalid_data_type_raises():

    with pytest.raises(
        TypeError,
        match=(
            "data must be a "
            "pandas DataFrame"
        ),
    ):

        DatasetProcessingResult(
            battery="TEST",
            data=[1, 2, 3],
            source_rows=3,
            processed_rows=3,
            missing_periods=0,
            frequency="QS-MAR",
        )


@pytest.mark.parametrize(
    "column",
    [
        "ds",
        "microVolt",
        "is_observed",
    ],
)
def test_required_columns(
    column,
):

    data = make_data().drop(
        columns=column
    )

    with pytest.raises(
        ValueError,
        match=(
            "processed data is missing "
            "required columns"
        ),
    ):

        DatasetProcessingResult(
            battery="TEST",
            data=data,
            source_rows=3,
            processed_rows=4,
            missing_periods=1,
            frequency="QS-MAR",
        )


def test_zero_source_rows_raises():

    with pytest.raises(
        ValueError,
        match=(
            "source_rows must be "
            "greater than zero"
        ),
    ):

        DatasetProcessingResult(
            battery="TEST",
            data=make_data(),
            source_rows=0,
            processed_rows=4,
            missing_periods=4,
            frequency="QS-MAR",
        )


def test_processed_rows_must_match_data():

    with pytest.raises(
        ValueError,
        match=(
            "processed_rows must match "
            "data length"
        ),
    ):

        DatasetProcessingResult(
            battery="TEST",
            data=make_data(),
            source_rows=3,
            processed_rows=5,
            missing_periods=2,
            frequency="QS-MAR",
        )


def test_processed_rows_may_be_smaller_than_source_rows():

    data = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2023-03-01",
                    "2023-06-01",
                    "2023-09-01",
                ]
            ),
            "microVolt": [
                30.0,
                31.0,
                32.0,
            ],
            "is_observed": [
                True,
                True,
                True,
            ],
        }
    )

    result = DatasetProcessingResult(
        battery="TEST",
        data=data,
        source_rows=4,
        processed_rows=3,
        missing_periods=0,
        frequency="QS-MAR",
    )

    assert result.source_rows == 4
    assert result.processed_rows == 3
    assert result.observed_rows == 3
    assert result.duplicate_rows_resolved == 1


def test_negative_missing_periods_raises():

    data = make_data().iloc[
        :3
    ].copy()

    with pytest.raises(
        ValueError,
        match=(
            "missing_periods cannot "
            "be negative"
        ),
    ):

        DatasetProcessingResult(
            battery="TEST",
            data=data,
            source_rows=3,
            processed_rows=3,
            missing_periods=-1,
            frequency="QS-MAR",
        )


def test_missing_period_count_consistency():

    data = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2023-03-01",
                    "2023-06-01",
                    "2023-09-01",
                ]
            ),
            "microVolt": [
                30.0,
                float("nan"),
                32.0,
            ],
            "is_observed": [
                True,
                False,
                True,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="processed_rows - observed_rows",
    ):

        DatasetProcessingResult(
            battery="TEST",
            data=data,
            source_rows=2,
            processed_rows=3,
            missing_periods=0,
            frequency="QS-MAR",
        )


def test_empty_frequency_raises():

    with pytest.raises(
        ValueError,
        match=(
            "frequency must be a "
            "non-empty string"
        ),
    ):

        DatasetProcessingResult(
            battery="TEST",
            data=make_data(),
            source_rows=3,
            processed_rows=4,
            missing_periods=1,
            frequency="",
        )


def test_repr():

    result = make_result()

    text = repr(
        result
    )

    assert (
        "DatasetProcessingResult"
        in text
    )

    assert (
        "TEST-001"
        in text
    )

    assert (
        "missing_periods=1"
        in text
    )


def test_duplicate_rows_resolved_property():

    data = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2023-03-01",
                    "2023-06-01",
                    "2023-09-01",
                ]
            ),
            "microVolt": [
                30.0,
                31.0,
                32.0,
            ],
            "is_observed": [
                True,
                True,
                True,
            ],
        }
    )

    result = DatasetProcessingResult(
        battery="TEST",
        data=data,
        source_rows=5,
        processed_rows=3,
        missing_periods=0,
        frequency="QS-MAR",
    )

    assert (
        result.duplicate_rows_resolved
        == 2
    )


def test_missing_periods_use_observed_rows():

    data = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2023-03-01",
                    "2023-06-01",
                    "2023-09-01",
                    "2023-12-01",
                ]
            ),
            "microVolt": [
                30.0,
                float("nan"),
                32.0,
                float("nan"),
            ],
            "is_observed": [
                True,
                False,
                True,
                False,
            ],
        }
    )

    result = DatasetProcessingResult(
        battery="TEST",
        data=data,
        source_rows=3,
        processed_rows=4,
        missing_periods=2,
        frequency="QS-MAR",
    )

    assert result.observed_rows == 2
    assert result.missing_periods == 2