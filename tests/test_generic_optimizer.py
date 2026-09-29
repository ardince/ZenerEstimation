import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.optimization import (
    ExpandingWindowSplitter,
    GenericOptimizer,
    OptimizationEvaluator,
    OptimizationResult,
)


class OffsetForecaster:

    def __init__(
        self,
        offset=0.0,
    ):
        self.offset = float(offset)
        self.dataset = None

    def fit(
        self,
        dataset,
    ):
        self.dataset = dataset
        return self

    def predict(
        self,
        steps,
    ):

        last_value = float(
            self.dataset.target.iloc[-1]
        )

        forecast = np.full(
            steps,
            last_value + self.offset,
            dtype=float,
        )

        return SimpleForecastResult(
            forecast
        )


class SimpleForecastResult:

    def __init__(
        self,
        forecast,
    ):
        self.forecast = forecast


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


def make_optimizer(
    metric="rmse",
):

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=2,
    )

    evaluator = OptimizationEvaluator(
        splitter,
        metric=metric,
    )

    return GenericOptimizer(
        evaluator,
        model="Offset",
    )


def test_generic_optimizer_returns_optimization_result():

    optimizer = make_optimizer()

    result = optimizer.optimize(
        dataset=make_dataset(),
        model_factory=OffsetForecaster,
        candidates=(
            {"offset": 0.0},
            {"offset": 1.0},
        ),
    )

    assert isinstance(
        result,
        OptimizationResult,
    )

    assert result.model == "Offset"
    assert result.metric == "rmse"
    assert result.candidate_count == 2


def test_generic_optimizer_selects_lowest_score():

    optimizer = make_optimizer()

    result = optimizer.optimize(
        dataset=make_dataset(),
        model_factory=OffsetForecaster,
        candidates=(
            {"offset": 0.0},
            {"offset": 1.0},
            {"offset": 5.0},
        ),
    )

    scores = [
        candidate.score
        for candidate in result.candidates
    ]

    assert result.best_score == pytest.approx(
        min(scores)
    )

    assert result.best_params == {
        "offset": 1.0,
    }


def test_generic_optimizer_preserves_candidate_order():

    optimizer = make_optimizer()

    candidates = (
        {"offset": 3.0},
        {"offset": 1.0},
        {"offset": 2.0},
    )

    result = optimizer.optimize(
        dataset=make_dataset(),
        model_factory=OffsetForecaster,
        candidates=candidates,
    )

    assert [
        candidate.params
        for candidate in result.candidates
    ] == list(candidates)


def test_generic_optimizer_evaluates_every_candidate():

    created = []

    def factory(
        offset=0.0,
    ):

        created.append(
            offset
        )

        return OffsetForecaster(
            offset=offset
        )

    optimizer = make_optimizer()

    result = optimizer.optimize(
        dataset=make_dataset(),
        model_factory=factory,
        candidates=(
            {"offset": 0.0},
            {"offset": 1.0},
            {"offset": 2.0},
        ),
    )

    assert result.candidate_count == 3

    # 3 candidates × 3 temporal folds
    assert len(created) == 9

    assert created == [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        2.0,
    ]


def test_generic_optimizer_resolves_tie_by_input_order():

    optimizer = make_optimizer()

    result = optimizer.optimize(
        dataset=make_dataset(),
        model_factory=OffsetForecaster,
        candidates=(
            {"offset": 1.0},
            {"offset": 1.0},
        ),
    )

    assert result.best_params == {
        "offset": 1.0,
    }

    assert result.best_score == pytest.approx(
        result.candidates[0].score
    )


def test_select_best_preserves_first_candidate_on_tie():

    from zenerestimation.optimization import CandidateResult

    first = CandidateResult(
        params={
            "candidate": "first",
        },
        score=0.25,
        metrics={
            "rmse": 0.25,
        },
    )

    second = CandidateResult(
        params={
            "candidate": "second",
        },
        score=0.25,
        metrics={
            "rmse": 0.25,
        },
    )

    selected = GenericOptimizer._select_best(
        [
            first,
            second,
        ]
    )

    assert selected is first


def test_generic_optimizer_rejects_empty_candidates():

    optimizer = make_optimizer()

    with pytest.raises(
        ValueError,
        match="candidates cannot be empty",
    ):

        optimizer.optimize(
            dataset=make_dataset(),
            model_factory=OffsetForecaster,
            candidates=(),
        )


def test_generic_optimizer_rejects_non_dictionary_candidate():

    optimizer = make_optimizer()

    with pytest.raises(
        TypeError,
        match="each candidate must be a dictionary",
    ):

        optimizer.optimize(
            dataset=make_dataset(),
            model_factory=OffsetForecaster,
            candidates=(
                {"offset": 0.0},
                "invalid",
            ),
        )


def test_generic_optimizer_rejects_empty_candidate_dictionary():

    optimizer = make_optimizer()

    with pytest.raises(
        ValueError,
        match="candidate parameter dictionaries cannot be empty",
    ):

        optimizer.optimize(
            dataset=make_dataset(),
            model_factory=OffsetForecaster,
            candidates=(
                {"offset": 0.0},
                {},
            ),
        )


def test_generic_optimizer_rejects_non_callable_factory():

    optimizer = make_optimizer()

    with pytest.raises(
        TypeError,
        match="model_factory must be callable",
    ):

        optimizer.optimize(
            dataset=make_dataset(),
            model_factory="invalid",
            candidates=(
                {"offset": 0.0},
            ),
        )


def test_generic_optimizer_records_search_metadata():

    optimizer = make_optimizer(
        metric="mae"
    )

    result = optimizer.optimize(
        dataset=make_dataset(),
        model_factory=OffsetForecaster,
        candidates=(
            {"offset": 0.0},
            {"offset": 1.0},
        ),
    )

    assert result.metric == "mae"

    assert result.metadata[
        "search"
    ] == "explicit_candidates"

    assert result.metadata[
        "objective"
    ] == "minimize"

    assert result.metadata[
        "candidate_count"
    ] == 2

    assert result.metadata[
        "validation"
    ] == "internal_temporal"