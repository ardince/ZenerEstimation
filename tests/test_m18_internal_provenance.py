
"""M18-B2a: development-dataset and fold provenance tests."""

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.optimization.validation import (
    ExpandingWindowSplitter,
)
from zenerestimation.optimization.internal_provenance import (
    build_internal_validation_context,
)


def dataset(rows=12, *, processed=False):
    df = pd.DataFrame({
        "ds": pd.date_range(
            "2020-03-01",
            periods=rows,
            freq="QS-MAR",
        ),
        "microVolt": [
            float(i) for i in range(rows)
        ],
    })

    if processed:
        df["is_observed"] = True

    return BatteryDataset(df)


def context(data=None, splitter=None):
    return build_internal_validation_context(
        dataset() if data is None else data,
        (
            ExpandingWindowSplitter(
                folds=2,
                validation_steps=2,
            )
            if splitter is None
            else splitter
        ),
        dataset_id="synthetic-dev",
    )


def test_identical_datasets_have_identical_fingerprint():
    assert (
        context()["dataset_fingerprint"]
        == context()["dataset_fingerprint"]
    )


@pytest.mark.parametrize("row", [0, 11])
def test_changing_measurement_changes_fingerprint(row):
    altered = dataset()
    altered.data.loc[row, "microVolt"] += 0.125

    assert (
        context(altered)["dataset_fingerprint"]
        != context()["dataset_fingerprint"]
    )


def test_observation_flag_changes_fingerprint():
    base = dataset(processed=True)
    altered = dataset(processed=True)

    altered.data.loc[0, "is_observed"] = False
    altered.data.loc[0, "microVolt"] = np.nan

    assert (
        context(base)["dataset_fingerprint"]
        != context(altered)["dataset_fingerprint"]
    )


def test_valid_inserted_training_period_is_allowed():
    altered = dataset(processed=True)

    altered.data.loc[0, "is_observed"] = False
    altered.data.loc[0, "microVolt"] = np.nan

    assert (
        context(altered)["scope"]
        == "internal_expanding_window"
    )


def test_unsorted_rows_rejected():
    altered = dataset()

    altered.data = (
        altered.data.iloc[::-1]
        .reset_index(drop=True)
    )

    with pytest.raises(ValueError, match="timestamps"):
        context(altered)


@pytest.mark.parametrize(
    "change",
    ["duplicate", "nat", "missing_column"],
)
def test_invalid_timestamp_or_structure_rejected(change):
    altered = dataset()

    if change == "duplicate":
        altered.data.loc[1, "ds"] = (
            altered.data.loc[0, "ds"]
        )
    elif change == "nat":
        altered.data.loc[1, "ds"] = pd.NaT
    else:
        altered.data = altered.data.drop(
            columns="microVolt"
        )

    with pytest.raises(ValueError):
        context(altered)


def test_insufficient_history_rejected():
    with pytest.raises(ValueError, match="enough rows"):
        context(dataset(4))


def test_fold_dates_and_training_cutoffs_match_splitter():
    result = context()

    assert result["fold_validation_dates"] == [
        ["2022-03-01", "2022-06-01"],
        ["2022-09-01", "2022-12-01"],
    ]

    assert result["fold_training_end_dates"] == [
        "2021-12-01",
        "2022-06-01",
    ]

    assert result["folds"] == 2
    assert result["validation_steps"] == 2

    assert result["dataset_fingerprint"].startswith(
        "sha256:"
    )


def test_no_mutation():
    data = dataset(processed=True)
    original = data.data.copy(deep=True)

    splitter = ExpandingWindowSplitter(
        folds=2,
        validation_steps=2,
    )

    before = (
        splitter.folds,
        splitter.validation_steps,
    )

    context(data, splitter)

    pd.testing.assert_frame_equal(
        data.data,
        original,
    )

    assert (
        splitter.folds,
        splitter.validation_steps,
    ) == before


@pytest.mark.parametrize(
    "value,observed",
    [
        (np.nan, True),
        (np.nan, False),
    ],
)
def test_invalid_validation_measurement_rejected(
    value, observed
):
    altered = dataset(processed=True)

    altered.data.loc[10, "microVolt"] = value
    altered.data.loc[10, "is_observed"] = observed

    with pytest.raises(ValueError):
        context(altered)


def test_splitter_configuration_changes_protocol():
    first = context()

    second = context(
        splitter=ExpandingWindowSplitter(
            folds=3,
            validation_steps=2,
        )
    )

    assert (
        first["dataset_fingerprint"]
        == second["dataset_fingerprint"]
    )

    assert (
        first["fold_validation_dates"]
        != second["fold_validation_dates"]
    )

    assert (
        first["fold_training_end_dates"]
        != second["fold_training_end_dates"]
    )


def test_invalid_observation_flag_rejected():
    altered = dataset(processed=True)

    altered.data["is_observed"] = (
        altered.data["is_observed"].astype(object)
    )
    altered.data.loc[0, "is_observed"] = "False"

    with pytest.raises(ValueError, match="booleans"):
        context(altered)
