"""
Tests for DatasetProcessor.
"""

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.processing import (
    DatasetProcessingResult,
    DatasetProcessor,
)


# ============================================================
# Fixtures / helpers
# ============================================================


def make_complete_data():

    return pd.DataFrame(
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
                30.5,
                31.0,
                31.5,
            ],
        }
    )


def make_missing_data():

    return pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2023-03-01",
                    "2023-06-01",
                    # 2023-09-01 missing
                    "2023-12-01",
                    "2024-03-01",
                ]
            ),
            "microVolt": [
                30.0,
                30.5,
                31.5,
                32.0,
            ],
        }
    )


# ============================================================
# Construction
# ============================================================


def test_processor_creation():

    processor = DatasetProcessor()

    assert processor.frequency == "QS-MAR"


def test_custom_frequency():

    processor = DatasetProcessor(
        frequency="MS"
    )

    assert processor.frequency == "MS"


def test_empty_frequency_raises():

    with pytest.raises(
        ValueError,
        match=(
            "frequency must be a "
            "non-empty string"
        ),
    ):

        DatasetProcessor(
            frequency=""
        )


# ============================================================
# Basic processing
# ============================================================


def test_process_returns_result():

    processor = DatasetProcessor()

    result = processor.process(
        make_complete_data(),
        battery="TEST-001",
    )

    assert isinstance(
        result,
        DatasetProcessingResult,
    )


def test_complete_dataset_not_expanded():

    processor = DatasetProcessor()

    result = processor.process(
        make_complete_data(),
        battery="TEST-001",
    )

    assert result.source_rows == 4

    assert result.processed_rows == 4

    assert result.missing_periods == 0


def test_complete_dataset_all_observed():

    processor = DatasetProcessor()

    result = processor.process(
        make_complete_data(),
        battery="TEST-001",
    )

    assert (
        result.data[
            "is_observed"
        ].tolist()
        == [
            True,
            True,
            True,
            True,
        ]
    )


# ============================================================
# Missing periods
# ============================================================


def test_missing_period_is_inserted():

    processor = DatasetProcessor()

    result = processor.process(
        make_missing_data(),
        battery="TEST-001",
    )

    assert result.source_rows == 4

    assert result.processed_rows == 5

    assert result.missing_periods == 1


def test_inserted_period_date():

    processor = DatasetProcessor()

    result = processor.process(
        make_missing_data(),
        battery="TEST-001",
    )

    assert result.data.loc[
        2,
        "ds",
    ] == pd.Timestamp(
        "2023-09-01"
    )


def test_inserted_target_remains_nan():

    processor = DatasetProcessor()

    result = processor.process(
        make_missing_data(),
        battery="TEST-001",
    )

    assert pd.isna(
        result.data.loc[
            2,
            "microVolt",
        ]
    )


def test_inserted_period_not_observed():

    processor = DatasetProcessor()

    result = processor.process(
        make_missing_data(),
        battery="TEST-001",
    )

    assert (
        result.data.loc[
            2,
            "is_observed",
        ]
        is False
        or not result.data.loc[
            2,
            "is_observed",
        ]
    )


def test_real_periods_remain_observed():

    processor = DatasetProcessor()

    result = processor.process(
        make_missing_data(),
        battery="TEST-001",
    )

    observed = result.data[
        "is_observed"
    ].tolist()

    assert observed == [
        True,
        True,
        False,
        True,
        True,
    ]


# ============================================================
# Ordering
# ============================================================


def test_unsorted_data_is_sorted():

    data = make_complete_data().iloc[
        [3, 0, 2, 1]
    ].reset_index(
        drop=True
    )

    processor = DatasetProcessor()

    result = processor.process(
        data,
        battery="TEST-001",
    )

    assert result.data[
        "ds"
    ].is_monotonic_increasing


def test_values_follow_sorted_dates():

    data = make_complete_data().iloc[
        [3, 0, 2, 1]
    ].reset_index(
        drop=True
    )

    processor = DatasetProcessor()

    result = processor.process(
        data,
        battery="TEST-001",
    )

    assert result.data[
        "microVolt"
    ].tolist() == [
        30.0,
        30.5,
        31.0,
        31.5,
    ]


# ============================================================
# Input immutability
# ============================================================


def test_process_does_not_modify_input():

    data = make_missing_data()

    original = data.copy(
        deep=True
    )

    processor = DatasetProcessor()

    processor.process(
        data,
        battery="TEST-001",
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


# ============================================================
# Conversion
# ============================================================


def test_string_dates_are_converted():

    data = pd.DataFrame(
        {
            "ds": [
                "2023-03-01",
                "2023-06-01",
                "2023-09-01",
            ],
            "microVolt": [
                30.0,
                30.5,
                31.0,
            ],
        }
    )

    processor = DatasetProcessor()

    result = processor.process(
        data,
        battery="TEST",
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result.data["ds"]
    )


def test_numeric_strings_are_converted():

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
                "30.0",
                "30.5",
                "31.0",
            ],
        }
    )

    processor = DatasetProcessor()

    result = processor.process(
        data,
        battery="TEST",
    )

    assert result.data[
        "microVolt"
    ].tolist() == [
        30.0,
        30.5,
        31.0,
    ]


# ============================================================
# Invalid input
# ============================================================


def test_non_dataframe_raises():

    processor = DatasetProcessor()

    with pytest.raises(
        TypeError,
        match=(
            "data must be a "
            "pandas DataFrame"
        ),
    ):

        processor.process(
            [1, 2, 3],
            battery="TEST",
        )


def test_empty_dataframe_raises():

    processor = DatasetProcessor()

    with pytest.raises(
        ValueError,
        match="data cannot be empty",
    ):

        processor.process(
            pd.DataFrame(
                columns=[
                    "ds",
                    "microVolt",
                ]
            ),
            battery="TEST",
        )


def test_empty_battery_raises():

    processor = DatasetProcessor()

    with pytest.raises(
        ValueError,
        match=(
            "battery must be a "
            "non-empty string"
        ),
    ):

        processor.process(
            make_complete_data(),
            battery="",
        )


@pytest.mark.parametrize(
    "column",
    [
        "ds",
        "microVolt",
    ],
)
def test_missing_required_column_raises(
    column,
):

    data = (
        make_complete_data()
        .drop(
            columns=column
        )
    )

    processor = DatasetProcessor()

    with pytest.raises(
        ValueError,
        match=(
            "raw data is missing "
            "required columns"
        ),
    ):

        processor.process(
            data,
            battery="TEST",
        )


def test_invalid_date_raises():

    data = pd.DataFrame(
        {
            "ds": [
                "2023-03-01",
                "not-a-date",
                "2023-09-01",
            ],
            "microVolt": [
                30.0,
                30.5,
                31.0,
            ],
        }
    )

    processor = DatasetProcessor()

    with pytest.raises(
        ValueError,
        match=(
            "ds contains invalid "
            "or missing dates"
        ),
    ):

        processor.process(
            data,
            battery="TEST",
        )


def test_missing_raw_target_raises():

    data = make_complete_data()

    data.loc[
        1,
        "microVolt",
    ] = np.nan

    processor = DatasetProcessor()

    with pytest.raises(
        ValueError,
        match=(
            "raw microVolt values must "
            "be numeric and non-missing"
        ),
    ):

        processor.process(
            data,
            battery="TEST",
        )


def test_non_numeric_target_raises():

    data = pd.DataFrame(
        {
            "ds": [
                "2023-03-01",
                "2023-06-01",
                "2023-09-01",
            ],
            "microVolt": [
                "30.0",
                "invalid",
                "31.0",
            ],
        }
    )

    processor = DatasetProcessor()

    with pytest.raises(
        ValueError,
        match=(
            "raw microVolt values must "
            "be numeric and non-missing"
        ),
    ):

        processor.process(
            data,
            battery="TEST",
        )


def test_infinite_target_raises():

    data = make_complete_data()

    data.loc[
        1,
        "microVolt",
    ] = np.inf

    processor = DatasetProcessor()

    with pytest.raises(
        ValueError,
        match=(
            "raw microVolt values "
            "must be finite"
        ),
    ):

        processor.process(
            data,
            battery="TEST",
        )


# ============================================================
# Duplicate timestamps
# ============================================================


def test_duplicate_date_raises():

    data = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2023-03-01",
                    "2023-06-01",
                    "2023-06-01",
                    "2023-09-01",
                ]
            ),
            "microVolt": [
                30.0,
                30.5,
                30.6,
                31.0,
            ],
        }
    )

    processor = DatasetProcessor()

    with pytest.raises(
        ValueError,
        match=(
            "duplicate timestamps "
            "are not allowed"
        ),
    ):

        processor.process(
            data,
            battery="TEST",
        )


# ============================================================
# Frequency alignment
# ============================================================


def test_off_grid_timestamp_raises():

    data = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2023-03-01",
                    "2023-06-15",
                    "2023-09-01",
                ]
            ),
            "microVolt": [
                30.0,
                30.5,
                31.0,
            ],
        }
    )

    processor = DatasetProcessor()

    with pytest.raises(
        ValueError,
        match=(
            "timestamps are not aligned "
            "with frequency"
        ),
    ):

        processor.process(
            data,
            battery="TEST",
        )


# ============================================================
# Metadata
# ============================================================


def test_processor_metadata():

    processor = DatasetProcessor()

    result = processor.process(
        make_complete_data(),
        battery="TEST",
    )

    assert (
        result.metadata[
            "processor"
        ]
        == "DatasetProcessor"
    )


def test_custom_metadata():

    processor = DatasetProcessor()

    result = processor.process(
        make_complete_data(),
        battery="TEST",
        metadata={
            "source": (
                "datasets/raw/TEST.csv"
            ),
        },
    )

    assert result.metadata[
        "source"
    ] == (
        "datasets/raw/TEST.csv"
    )


# ============================================================
# Representation
# ============================================================


def test_repr():

    processor = DatasetProcessor(
        frequency="QS-MAR"
    )

    text = repr(
        processor
    )

    assert (
        "DatasetProcessor"
        in text
    )

    assert (
        "frequency='QS-MAR'"
        in text
    )

    assert (
        "dayfirst=True"
        in text
    )


def test_dayfirst_dates_are_parsed_correctly():

    data = pd.DataFrame(
        {
            "ds": [
                "01/03/2023",
                "01/06/2023",
                "01/09/2023",
                "01/12/2023",
            ],
            "microVolt": [
                30.0,
                30.5,
                31.0,
                31.5,
            ],
        }
    )

    processor = DatasetProcessor(
        dayfirst=True
    )

    result = processor.process(
        data,
        battery="TEST",
    )

    assert result.data[
        "ds"
    ].tolist() == [
        pd.Timestamp(
            "2023-03-01"
        ),
        pd.Timestamp(
            "2023-06-01"
        ),
        pd.Timestamp(
            "2023-09-01"
        ),
        pd.Timestamp(
            "2023-12-01"
        ),
    ]


def test_dayfirst_metadata():

    processor = DatasetProcessor(
        dayfirst=True
    )

    result = processor.process(
        make_complete_data(),
        battery="TEST",
    )

    assert (
        result.metadata[
            "date_dayfirst"
        ]
        is True
    )


def test_invalid_dayfirst_type_raises():

    with pytest.raises(
        TypeError,
        match=(
            "dayfirst must be "
            "a boolean"
        ),
    ):

        DatasetProcessor(
            dayfirst="yes"
        )


def test_iso_dates_remain_year_first():

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
                30.5,
                31.0,
                31.5,
            ],
        }
    )

    processor = DatasetProcessor(
        dayfirst=True
    )

    result = processor.process(
        data,
        battery="TEST",
    )

    assert result.data[
        "ds"
    ].tolist() == [
        pd.Timestamp("2023-03-01"),
        pd.Timestamp("2023-06-01"),
        pd.Timestamp("2023-09-01"),
        pd.Timestamp("2023-12-01"),
    ]


def test_iso_and_dayfirst_dates_can_coexist():

    data = pd.DataFrame(
        {
            "ds": [
                "2023-03-01",
                "01/06/2023",
                "2023-09-01",
                "01/12/2023",
            ],
            "microVolt": [
                30.0,
                30.5,
                31.0,
                31.5,
            ],
        }
    )

    processor = DatasetProcessor(
        dayfirst=True
    )

    result = processor.process(
        data,
        battery="TEST",
    )

    assert result.data[
        "ds"
    ].tolist() == [
        pd.Timestamp(
            "2023-03-01"
        ),
        pd.Timestamp(
            "2023-06-01"
        ),
        pd.Timestamp(
            "2023-09-01"
        ),
        pd.Timestamp(
            "2023-12-01"
        ),
    ]


def test_missing_periods_are_based_on_unique_observations():

    data = pd.DataFrame(
        {
            "ds": [
                "2023-03-01",
                "2023-03-01",
                "2023-09-01",
            ],
            "microVolt": [
                30.0,
                32.0,
                34.0,
            ],
        }
    )

    processor = DatasetProcessor(
        frequency="QS-MAR",
        duplicate_policy="mean",
    )

    result = processor.process(
        data,
        battery="TEST",
    )

    assert result.source_rows == 3
    assert result.observed_rows == 2
    assert result.processed_rows == 3

    assert result.missing_periods == 1


def test_duplicate_rows_resolved():

    data = pd.DataFrame(
        {
            "ds": [
                "2023-03-01",
                "2023-03-01",
                "2023-06-01",
            ],
            "microVolt": [
                30.0,
                32.0,
                33.0,
            ],
        }
    )

    processor = DatasetProcessor(
        duplicate_policy="mean",
    )

    result = processor.process(
        data,
        battery="TEST",
    )

    assert (
        result.duplicate_rows_resolved
        == 1
    )

    assert result.missing_periods == 0