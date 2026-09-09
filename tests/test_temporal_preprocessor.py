import pandas as pd
import pytest

from zenerestimation.data import (
    BatteryDataset,
)

from zenerestimation.data.temporal import (
    TemporalPreprocessor,
)

def make_dataset():

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
                None,
                32.0,
                33.0,
            ],
            "is_observed": [
                True,
                False,
                True,
                True,
            ],
        }
    )

    dataset = BatteryDataset(
        data
    )

    dataset._battery = "TEST"

    return dataset


def test_create():

    preprocessor = (
        TemporalPreprocessor()
    )

    assert (
        preprocessor.method
        == "linear"
    )

    assert (
        preprocessor.fill_edges
        is True
    )


def test_interpolates_internal_missing_value():

    dataset = make_dataset()

    preprocessor = (
        TemporalPreprocessor()
    )

    result = (
        preprocessor.transform(
            dataset
        )
    )

    assert (
        result.data.loc[
            1,
            "microVolt",
        ]
        == pytest.approx(
            31.0
        )
    )


def test_target_has_no_missing_values():

    dataset = make_dataset()

    result = (
        TemporalPreprocessor()
        .transform(
            dataset
        )
    )

    assert not (
        result.data[
            "microVolt"
        ]
        .isna()
        .any()
    )


def test_observation_flag_is_preserved():

    dataset = make_dataset()

    result = (
        TemporalPreprocessor()
        .transform(
            dataset
        )
    )

    assert (
        result.data[
            "is_observed"
        ].tolist()
        == [
            True,
            False,
            True,
            True,
        ]
    )


def test_original_dataset_is_not_modified():

    dataset = make_dataset()

    TemporalPreprocessor().transform(
        dataset
    )

    assert pd.isna(
        dataset.data.loc[
            1,
            "microVolt",
        ]
    )


def test_leading_missing_value_is_filled():

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
                None,
                31.0,
                32.0,
            ],
            "is_observed": [
                False,
                True,
                True,
            ],
        }
    )

    dataset = BatteryDataset(
        data
    )

    result = (
        TemporalPreprocessor(
            fill_edges=True,
        )
        .transform(
            dataset
        )
    )

    assert (
        result.data.loc[
            0,
            "microVolt",
        ]
        == pytest.approx(
            31.0
        )
    )


def test_trailing_missing_value_is_filled():

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
                None,
            ],
            "is_observed": [
                True,
                True,
                False,
            ],
        }
    )

    dataset = BatteryDataset(
        data
    )

    result = (
        TemporalPreprocessor(
            fill_edges=True,
        )
        .transform(
            dataset
        )
    )

    assert (
        result.data.loc[
            2,
            "microVolt",
        ]
        == pytest.approx(
            31.0
        )
    )


def test_unresolved_edge_missing_raises():

    data = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2023-03-01",
                    "2023-06-01",
                ]
            ),
            "microVolt": [
                None,
                31.0,
            ],
            "is_observed": [
                False,
                True,
            ],
        }
    )

    dataset = BatteryDataset(
        data
    )

    preprocessor = (
        TemporalPreprocessor(
            fill_edges=False,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "could not resolve "
            "missing target values"
        ),
    ):

        preprocessor.transform(
            dataset
        )


def test_invalid_method_raises():

    with pytest.raises(
        ValueError,
        match=(
            "unsupported interpolation "
            "method"
        ),
    ):

        TemporalPreprocessor(
            method="invalid",
        )


def test_invalid_fill_edges_type_raises():

    with pytest.raises(
        TypeError,
        match=(
            "fill_edges must be bool"
        ),
    ):

        TemporalPreprocessor(
            fill_edges="yes",
        )


def test_invalid_dataset_type_raises():

    with pytest.raises(
        TypeError,
        match=(
            "dataset must be "
            "a BatteryDataset"
        ),
    ):

        TemporalPreprocessor().transform(
            object()
        )


def test_all_missing_target_raises():

    data = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2023-03-01",
                    "2023-06-01",
                ]
            ),
            "microVolt": [
                None,
                None,
            ],
            "is_observed": [
                False,
                False,
            ],
        }
    )

    dataset = BatteryDataset(
        data
    )

    with pytest.raises(
        ValueError,
        match=(
            "no observed target values"
        ),
    ):

        TemporalPreprocessor().transform(
            dataset
        )


def test_dataset_identity_is_preserved():

    dataset = make_dataset()

    dataset._source_type = (
        "processed"
    )

    dataset._source_path = (
        "datasets/processed/TEST.csv"
    )

    result = (
        TemporalPreprocessor()
        .transform(
            dataset
        )
    )

    assert (
        result.battery
        == "TEST"
    )

    assert (
        result.source_type
        == "processed"
    )

    assert (
        str(
            result.source_path
        )
        == (
            "datasets/processed/"
            "TEST.csv"
        )
    )


def test_fit_transform_matches_transform():

    dataset = make_dataset()

    preprocessor = (
        TemporalPreprocessor()
    )

    transformed = (
        preprocessor.transform(
            dataset
        )
    )

    fitted = (
        preprocessor.fit_transform(
            dataset
        )
    )

    pd.testing.assert_frame_equal(
        transformed.data,
        fitted.data,
    )