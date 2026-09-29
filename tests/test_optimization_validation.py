import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.optimization import (
    ExpandingWindowSplitter,
    TemporalFold,
)


def make_dataset(rows=20):

    dates = pd.date_range(
        start="2020-03-01",
        periods=rows,
        freq="QS-MAR",
    )

    data = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": [
                float(value)
                for value in range(rows)
            ],
        }
    )

    return BatteryDataset(data)


def test_expanding_window_splitter_creates_requested_folds():

    dataset = make_dataset(20)

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=2,
    )

    folds = splitter.split(dataset)

    assert len(folds) == 3


def test_expanding_window_training_sizes_expand():

    dataset = make_dataset(20)

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=2,
    )

    folds = splitter.split(dataset)

    assert [
        fold.training_points
        for fold in folds
    ] == [
        14,
        16,
        18,
    ]


def test_expanding_window_validation_sizes_are_constant():

    dataset = make_dataset(20)

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=2,
    )

    folds = splitter.split(dataset)

    assert [
        fold.validation_points
        for fold in folds
    ] == [
        2,
        2,
        2,
    ]


def test_training_strictly_precedes_validation():

    dataset = make_dataset(20)

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=2,
    )

    folds = splitter.split(dataset)

    for fold in folds:

        assert (
            fold.train.data["ds"].iloc[-1]
            <
            fold.validation["ds"].iloc[0]
        )


def test_validation_blocks_do_not_overlap():

    dataset = make_dataset(20)

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=2,
    )

    folds = splitter.split(dataset)

    validation_dates = []

    for fold in folds:
        validation_dates.extend(
            fold.validation["ds"].tolist()
        )

    assert len(validation_dates) == len(
        set(validation_dates)
    )


def test_final_validation_fold_reaches_dataset_end():

    dataset = make_dataset(20)

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=2,
    )

    folds = splitter.split(dataset)

    assert (
        folds[-1].validation["ds"].iloc[-1]
        ==
        dataset.data["ds"].iloc[-1]
    )


def test_splitter_does_not_mutate_dataset():

    dataset = make_dataset(20)

    original = dataset.data.copy(
        deep=True
    )

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=2,
    )

    splitter.split(dataset)

    pd.testing.assert_frame_equal(
        dataset.data,
        original,
    )


def test_splitter_rejects_insufficient_history():

    dataset = make_dataset(6)

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=2,
    )

    with pytest.raises(ValueError):
        splitter.split(dataset)


def test_splitter_rejects_invalid_configuration():

    with pytest.raises(ValueError):
        ExpandingWindowSplitter(
            folds=0,
            validation_steps=2,
        )

    with pytest.raises(ValueError):
        ExpandingWindowSplitter(
            folds=3,
            validation_steps=0,
        )


def test_temporal_fold_rejects_overlap():

    dataset = make_dataset(10)

    train = BatteryDataset(
        dataset.data.iloc[:6].copy()
    )

    validation = (
        dataset.data
        .iloc[5:8]
        .copy()
    )

    with pytest.raises(ValueError):
        TemporalFold(
            index=0,
            train=train,
            validation=validation,
        )