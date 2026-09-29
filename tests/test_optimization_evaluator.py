import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.optimization import (
    CandidateResult,
    ExpandingWindowSplitter,
    OptimizationEvaluator,
)


class ConstantForecaster:
    """
    Small deterministic test forecaster.
    """

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

        value = (
            float(
                self.dataset.target.iloc[-1]
            )
            + self.offset
        )

        forecast = np.full(
            steps,
            value,
            dtype=float,
        )

        return SimpleForecastResult(
            forecast=forecast,
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


def make_evaluator():

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=2,
    )

    return OptimizationEvaluator(
        splitter,
        metric="rmse",
    )


def test_optimization_evaluator_returns_candidate_result():

    evaluator = make_evaluator()

    result = evaluator.evaluate(
        dataset=make_dataset(),
        model_factory=ConstantForecaster,
        params={
            "offset": 0.0,
        },
    )

    assert isinstance(
        result,
        CandidateResult,
    )


def test_optimization_evaluator_preserves_params():

    evaluator = make_evaluator()

    result = evaluator.evaluate(
        dataset=make_dataset(),
        model_factory=ConstantForecaster,
        params={
            "offset": 1.0,
        },
    )

    assert result.params == {
        "offset": 1.0,
    }


def test_optimization_evaluator_aggregates_metrics():

    evaluator = make_evaluator()

    result = evaluator.evaluate(
        dataset=make_dataset(),
        model_factory=ConstantForecaster,
        params={
            "offset": 0.0,
        },
    )

    assert set(
        result.metrics
    ) == {
        "rmse",
        "mae",
        "mape",
    }

    assert all(
        np.isfinite(value)
        for value in result.metrics.values()
    )


def test_candidate_score_uses_selected_metric():

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=2,
    )

    evaluator = OptimizationEvaluator(
        splitter,
        metric="mae",
    )

    result = evaluator.evaluate(
        dataset=make_dataset(),
        model_factory=ConstantForecaster,
        params={
            "offset": 0.0,
        },
    )

    assert result.score == pytest.approx(
        result.metrics["mae"]
    )


def test_optimization_evaluator_records_fold_metadata():

    evaluator = make_evaluator()

    result = evaluator.evaluate(
        dataset=make_dataset(),
        model_factory=ConstantForecaster,
        params={
            "offset": 0.0,
        },
    )

    assert result.metadata["folds"] == 3

    assert (
        result.metadata[
            "validation_steps"
        ]
        == 2
    )

    assert len(
        result.metadata["fold_metrics"]
    ) == 3


def test_optimization_evaluator_rejects_unknown_metric():

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=2,
    )

    with pytest.raises(ValueError):

        OptimizationEvaluator(
            splitter,
            metric="unknown",
        )


def test_optimization_evaluator_rejects_non_callable_factory():

    evaluator = make_evaluator()

    with pytest.raises(TypeError):

        evaluator.evaluate(
            dataset=make_dataset(),
            model_factory="not-callable",
            params={
                "offset": 0.0,
            },
        )


def test_optimization_evaluator_rejects_empty_params():

    evaluator = make_evaluator()

    with pytest.raises(ValueError):

        evaluator.evaluate(
            dataset=make_dataset(),
            model_factory=ConstantForecaster,
            params={},
        )


def test_optimization_evaluator_rejects_missing_validation_target():

    dataset = make_dataset()

    dataset.data.loc[
        dataset.data.index[-1],
        "microVolt",
    ] = np.nan

    evaluator = make_evaluator()

    with pytest.raises(
        ValueError,
        match="missing target values",
    ):

        evaluator.evaluate(
            dataset=dataset,
            model_factory=ConstantForecaster,
            params={
                "offset": 0.0,
            },
        )


def test_optimization_evaluator_creates_fresh_model_per_fold():

    created_models = []

    def factory(
        offset=0.0,
    ):

        model = ConstantForecaster(
            offset=offset,
        )

        created_models.append(
            model
        )

        return model

    evaluator = make_evaluator()

    evaluator.evaluate(
        dataset=make_dataset(),
        model_factory=factory,
        params={
            "offset": 0.0,
        },
    )

    assert len(
        created_models
    ) == 3

    assert len(
        {
            id(model)
            for model in created_models
        }
    ) == 3


class StatefulPreprocessor:

    def __init__(self):
        self.calls = 0

    def fit_transform(
        self,
        dataset,
    ):
        self.calls += 1

        if self.calls != 1:
            raise RuntimeError(
                "preprocessor instance was reused"
            )

        return dataset


def test_preprocessor_is_not_reused_between_folds():

    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=2,
    )

    preprocessor = StatefulPreprocessor()

    evaluator = OptimizationEvaluator(
        splitter,
        metric="rmse",
        preprocessor=preprocessor,
    )

    result = evaluator.evaluate(
        dataset=make_dataset(),
        model_factory=ConstantForecaster,
        params={
            "offset": 0.0,
        },
    )

    assert isinstance(
        result,
        CandidateResult,
    )

    assert preprocessor.calls == 0