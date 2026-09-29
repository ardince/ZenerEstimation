import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.optimization import (
    BenchmarkBoundary,
)

from zenerestimation.evaluation.evaluator import (
    ForecastEvaluator,
)


def make_dataset(
    rows=20,
):

    dates = pd.date_range(
        start="2020-03-01",
        periods=rows,
        freq="QS-MAR",
    )

    data = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": np.arange(
                1,
                rows + 1,
                dtype=float,
            ),
        }
    )

    return BatteryDataset(
        data
    )


def test_boundary_reserves_final_benchmark_rows():

    dataset = make_dataset(
        20
    )

    boundary = BenchmarkBoundary(
        evaluation_steps=4
    )

    development = (
        boundary.development_dataset(
            dataset
        )
    )

    benchmark = (
        boundary.benchmark_data(
            dataset
        )
    )

    assert len(
        development
    ) == 16

    assert len(
        benchmark
    ) == 4


def test_development_ends_before_benchmark():

    dataset = make_dataset()

    boundary = BenchmarkBoundary(
        evaluation_steps=4
    )

    development = (
        boundary.development_dataset(
            dataset
        )
    )

    benchmark = (
        boundary.benchmark_data(
            dataset
        )
    )

    assert (
        development.data["ds"].iloc[-1]
        <
        benchmark["ds"].iloc[0]
    )


def test_boundary_partitions_are_exact():

    dataset = make_dataset(
        20
    )

    boundary = BenchmarkBoundary(
        evaluation_steps=4
    )

    development = (
        boundary.development_dataset(
            dataset
        )
    )

    benchmark = (
        boundary.benchmark_data(
            dataset
        )
    )

    pd.testing.assert_frame_equal(
        development.data.reset_index(
            drop=True
        ),
        dataset.data.iloc[:16]
        .reset_index(drop=True),
    )

    pd.testing.assert_frame_equal(
        benchmark.reset_index(
            drop=True
        ),
        dataset.data.iloc[16:]
        .reset_index(drop=True),
    )


def test_boundary_respects_evaluation_end():

    dataset = make_dataset(
        20
    )

    evaluation_end = (
        dataset.data["ds"].iloc[15]
    )

    boundary = BenchmarkBoundary(
        evaluation_steps=4,
        evaluation_end=evaluation_end,
    )

    development = (
        boundary.development_dataset(
            dataset
        )
    )

    benchmark = (
        boundary.benchmark_data(
            dataset
        )
    )

    assert len(
        development
    ) == 12

    assert len(
        benchmark
    ) == 4

    assert (
        benchmark["ds"].iloc[-1]
        ==
        evaluation_end
    )


def test_rows_after_evaluation_end_are_excluded():

    dataset = make_dataset(
        20
    )

    evaluation_end = (
        dataset.data["ds"].iloc[15]
    )

    boundary = BenchmarkBoundary(
        evaluation_steps=4,
        evaluation_end=evaluation_end,
    )

    development = (
        boundary.development_dataset(
            dataset
        )
    )

    benchmark = (
        boundary.benchmark_data(
            dataset
        )
    )

    allowed_dates = set(
        development.data["ds"]
    ) | set(
        benchmark["ds"]
    )

    future_dates = set(
        dataset.data.loc[
            dataset.data["ds"]
            > evaluation_end,
            "ds",
        ]
    )

    assert allowed_dates.isdisjoint(
        future_dates
    )


def test_boundary_does_not_mutate_original_dataset():

    dataset = make_dataset()

    original = dataset.data.copy(
        deep=True
    )

    boundary = BenchmarkBoundary(
        evaluation_steps=4
    )

    boundary.development_dataset(
        dataset
    )

    boundary.benchmark_data(
        dataset
    )

    pd.testing.assert_frame_equal(
        dataset.data,
        original,
    )


@pytest.mark.parametrize(
    "steps",
    [
        0,
        -1,
    ],
)
def test_boundary_rejects_non_positive_steps(
    steps,
):

    with pytest.raises(
        ValueError
    ):
        BenchmarkBoundary(
            evaluation_steps=steps
        )


def test_boundary_rejects_boolean_steps():

    with pytest.raises(
        TypeError
    ):
        BenchmarkBoundary(
            evaluation_steps=True
        )


def test_boundary_rejects_insufficient_dataset():

    dataset = make_dataset(
        4
    )

    boundary = BenchmarkBoundary(
        evaluation_steps=4
    )

    with pytest.raises(
        ValueError,
        match="enough rows",
    ):
        boundary.development_dataset(
            dataset
        )


def test_boundary_rejects_evaluation_end_before_dataset():

    dataset = make_dataset()

    boundary = BenchmarkBoundary(
        evaluation_steps=4,
        evaluation_end="2010-03-01",
    )

    with pytest.raises(
        ValueError,
        match="precedes all",
    ):
        boundary.development_dataset(
            dataset
        )


def test_boundary_matches_forecast_evaluator_split():

    dataset = make_dataset(
        20
    )

    boundary = BenchmarkBoundary(
        evaluation_steps=4
    )

    development = (
        boundary.development_dataset(
            dataset
        )
    )

    benchmark = (
        boundary.benchmark_data(
            dataset
        )
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=4
    )

    evaluator_train, evaluator_validation = (
        evaluator.split_dataset(
            dataset
        )
    )

    pd.testing.assert_frame_equal(
        development.data.reset_index(
            drop=True
        ),
        evaluator_train.data.reset_index(
            drop=True
        ),
    )

    pd.testing.assert_frame_equal(
        benchmark.reset_index(
            drop=True
        ),
        evaluator_validation.reset_index(
            drop=True
        ),
    )



def test_boundary_matches_evaluator_with_evaluation_end():

    dataset = make_dataset(
        20
    )

    evaluation_end = (
        dataset.data["ds"].iloc[15]
    )

    boundary = BenchmarkBoundary(
        evaluation_steps=4,
        evaluation_end=evaluation_end,
    )

    development = (
        boundary.development_dataset(
            dataset
        )
    )

    benchmark = (
        boundary.benchmark_data(
            dataset
        )
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=4,
        evaluation_end=evaluation_end,
    )

    evaluator_train, evaluator_validation = (
        evaluator.split_dataset(
            dataset
        )
    )

    pd.testing.assert_frame_equal(
        development.data.reset_index(
            drop=True
        ),
        evaluator_train.data.reset_index(
            drop=True
        ),
    )

    pd.testing.assert_frame_equal(
        benchmark.reset_index(
            drop=True
        ),
        evaluator_validation.reset_index(
            drop=True
        ),
    )