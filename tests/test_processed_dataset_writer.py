"""
Tests for ProcessedDatasetWriter.
"""

import json

import pandas as pd
import pytest

from zenerestimation.data.processing import (
    DatasetProcessingResult,
    ProcessedDatasetPaths,
    ProcessedDatasetWriter,
)


# ============================================================
# Helpers
# ============================================================


def make_result():

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
                30.5,
                float("nan"),
                31.5,
            ],
            "is_observed": [
                True,
                True,
                False,
                True,
            ],
        }
    )

    return DatasetProcessingResult(
        battery="TEST-001",
        data=data,
        source_rows=3,
        processed_rows=4,
        missing_periods=1,
        frequency="QS-MAR",
        metadata={
            "source": (
                "datasets/raw/TEST-001.csv"
            ),
        },
    )


# ============================================================
# Construction
# ============================================================


def test_writer_creation(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    assert (
        writer.output_directory
        == tmp_path
    )


def test_paths_for():

    writer = ProcessedDatasetWriter(
        "datasets/processed"
    )

    paths = writer.paths_for(
        "TEST-001"
    )

    assert isinstance(
        paths,
        ProcessedDatasetPaths,
    )

    assert paths.data == (
        writer.output_directory
        / "TEST-001.csv"
    )

    assert paths.metadata == (
        writer.output_directory
        / "TEST-001.metadata.json"
    )


# ============================================================
# Persistence
# ============================================================


def test_save_creates_files(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    paths = writer.save(
        make_result()
    )

    assert paths.data.exists()

    assert paths.metadata.exists()


def test_save_creates_output_directory(
    tmp_path,
):

    destination = (
        tmp_path
        / "processed"
        / "datasets"
    )

    writer = ProcessedDatasetWriter(
        destination
    )

    writer.save(
        make_result()
    )

    assert destination.exists()


def test_saved_csv_columns(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    paths = writer.save(
        make_result()
    )

    data = pd.read_csv(
        paths.data
    )

    assert data.columns.tolist() == [
        "ds",
        "microVolt",
        "is_observed",
    ]


def test_saved_csv_row_count(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    paths = writer.save(
        make_result()
    )

    data = pd.read_csv(
        paths.data
    )

    assert len(
        data
    ) == 4


def test_saved_csv_missing_target(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    paths = writer.save(
        make_result()
    )

    data = pd.read_csv(
        paths.data
    )

    assert pd.isna(
        data.loc[
            2,
            "microVolt",
        ]
    )


def test_saved_csv_observation_flag(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    paths = writer.save(
        make_result()
    )

    data = pd.read_csv(
        paths.data
    )

    assert data[
        "is_observed"
    ].tolist() == [
        True,
        True,
        False,
        True,
    ]


def test_saved_csv_date_format(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    paths = writer.save(
        make_result()
    )

    first_line = (
        paths.data
        .read_text(
            encoding="utf-8"
        )
        .splitlines()[1]
    )

    assert first_line.startswith(
        "2023-03-01,"
    )


# ============================================================
# Metadata
# ============================================================


def test_metadata_is_valid_json(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    paths = writer.save(
        make_result()
    )

    with open(
        paths.metadata,
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    assert isinstance(
        metadata,
        dict,
    )


def test_metadata_identity(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    paths = writer.save(
        make_result()
    )

    metadata = json.loads(
        paths.metadata.read_text(
            encoding="utf-8"
        )
    )

    assert (
        metadata["battery"]
        == "TEST-001"
    )

    assert (
        metadata["target"]
        == "microVolt"
    )

    assert (
        metadata["frequency"]
        == "QS-MAR"
    )


def test_metadata_processing_counts(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    paths = writer.save(
        make_result()
    )

    metadata = json.loads(
        paths.metadata.read_text(
            encoding="utf-8"
        )
    )

    assert (
        metadata["source_rows"]
        == 3
    )

    assert (
        metadata["processed_rows"]
        == 4
    )

    assert (
        metadata["observed_rows"]
        == 3
    )

    assert (
        metadata["missing_periods"]
        == 1
    )


def test_metadata_contains_source(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    paths = writer.save(
        make_result()
    )

    metadata = json.loads(
        paths.metadata.read_text(
            encoding="utf-8"
        )
    )

    assert metadata[
        "source"
    ] == (
        "datasets/raw/TEST-001.csv"
    )


def test_metadata_schema(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    paths = writer.save(
        make_result()
    )

    metadata = json.loads(
        paths.metadata.read_text(
            encoding="utf-8"
        )
    )

    assert metadata[
        "schema"
    ] == (
        "zenerestimation.processed-dataset"
    )

    assert (
        metadata["schema_version"]
        == "1.0"
    )


def test_metadata_describes_missing_values(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    paths = writer.save(
        make_result()
    )

    metadata = json.loads(
        paths.metadata.read_text(
            encoding="utf-8"
        )
    )

    target = metadata[
        "columns"
    ][
        "microVolt"
    ]

    assert (
        target[
            "missing_representation"
        ]
        == "empty CSV field"
    )


# ============================================================
# Overwrite protection
# ============================================================


def test_existing_files_are_protected(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    result = make_result()

    writer.save(
        result
    )

    with pytest.raises(
        FileExistsError,
        match=(
            "processed dataset artifacts "
            "already exist"
        ),
    ):

        writer.save(
            result
        )


def test_overwrite_true_replaces_files(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    result = make_result()

    first = writer.save(
        result
    )

    first.data.write_text(
        "changed",
        encoding="utf-8",
    )

    writer.save(
        result,
        overwrite=True,
    )

    text = first.data.read_text(
        encoding="utf-8"
    )

    assert text.startswith(
        "ds,microVolt,is_observed"
    )


# ============================================================
# Safety
# ============================================================


@pytest.mark.parametrize(
    "battery",
    [
        "../TEST",
        "folder/TEST",
        r"folder\TEST",
    ],
)
def test_unsafe_battery_identifier_raises(
    tmp_path,
    battery,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match=(
            "battery must be a safe "
            "filename identifier"
        ),
    ):

        writer.paths_for(
            battery
        )


def test_invalid_result_raises(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    with pytest.raises(
        TypeError,
        match=(
            "result must be a "
            "DatasetProcessingResult"
        ),
    ):

        writer.save(
            {"battery": "TEST"}
        )


# ============================================================
# Immutability
# ============================================================


def test_save_does_not_modify_result(
    tmp_path,
):

    result = make_result()

    original = result.dataframe()

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    writer.save(
        result
    )

    pd.testing.assert_frame_equal(
        result.dataframe(),
        original,
    )


# ============================================================
# Temporary files
# ============================================================


def test_no_temporary_files_remain(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    writer.save(
        make_result()
    )

    temporary_files = list(
        tmp_path.glob(
            "*.tmp"
        )
    )

    assert temporary_files == []


# ============================================================
# Representation
# ============================================================


def test_repr(
    tmp_path,
):

    writer = ProcessedDatasetWriter(
        tmp_path
    )

    text = repr(
        writer
    )

    assert (
        "ProcessedDatasetWriter"
        in text
    )

    assert (
        "output_directory="
        in text
    )

    assert str(
        tmp_path.name
    ) in text