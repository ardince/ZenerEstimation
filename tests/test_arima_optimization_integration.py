import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.forecasting.arima import ARIMAForecaster
from zenerestimation.optimization import (
    ARIMAParameterSpace,
    CandidateResult,
    ExpandingWindowSplitter,
    GenericOptimizer,
    OptimizationEvaluator,
    OptimizationResult,
)


def make_quarterly_dataset(
    rows=40,
):
    """
    Deterministic quarterly series with trend and mild seasonality.
    """

    index = np.arange(
        rows,
        dtype=float,
    )

    values = (
        20.0
        + 0.35 * index
        + 0.40 * np.sin(
            2.0 * np.pi * index / 4.0
        )
    )

    dates = pd.date_range(
        start="2015-03-01",
        periods=rows,
        freq="QS-MAR",
    )

    data = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": values,
        }
    )

    return BatteryDataset(
        data
    )


def make_arima_optimizer(
    *,
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
        model="ARIMA",
    )


def make_small_arima_space():
    return ARIMAParameterSpace(
        p=(0, 1, 2),
        d=(0, 1),
        q=(0, 1),
    )


def test_arima_complete_optimization_chain():

    dataset = make_quarterly_dataset()

    space = make_small_arima_space()

    optimizer = make_arima_optimizer()

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            ConvergenceWarning,
        )

        result = optimizer.optimize(
            dataset=dataset,
            model_factory=ARIMAForecaster,
            candidates=space.candidates(),
        )

    assert isinstance(
        result,
        OptimizationResult,
    )

    assert result.model == "ARIMA"
    assert result.metric == "rmse"

    assert result.candidate_count == (
        space.candidate_count
    )

    assert len(
        result.candidates
    ) == 12


def test_arima_candidates_produce_finite_scores():

    dataset = make_quarterly_dataset()

    space = make_small_arima_space()

    optimizer = make_arima_optimizer()

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            ConvergenceWarning,
        )

        result = optimizer.optimize(
            dataset=dataset,
            model_factory=ARIMAForecaster,
            candidates=space.candidates(),
        )

    assert all(
        isinstance(
            candidate,
            CandidateResult,
        )
        for candidate in result.candidates
    )

    assert all(
        np.isfinite(
            candidate.score
        )
        for candidate in result.candidates
    )

    for candidate in result.candidates:

        assert set(
            candidate.metrics
        ) == {
            "rmse",
            "mae",
            "mape",
        }

        assert all(
            np.isfinite(value)
            for value
            in candidate.metrics.values()
        )


def test_arima_optimizer_selects_lowest_internal_score():

    dataset = make_quarterly_dataset()

    space = make_small_arima_space()

    optimizer = make_arima_optimizer()

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            ConvergenceWarning,
        )

        result = optimizer.optimize(
            dataset=dataset,
            model_factory=ARIMAForecaster,
            candidates=space.candidates(),
        )

    expected = min(
        result.candidates,
        key=lambda candidate: candidate.score,
    )

    assert result.best_score == pytest.approx(
        expected.score
    )

    assert result.best_params == (
        expected.params
    )


def test_arima_candidate_orders_are_preserved():

    dataset = make_quarterly_dataset()

    space = ARIMAParameterSpace(
        p=(0, 1),
        d=(1,),
        q=(0, 1),
    )

    optimizer = make_arima_optimizer()

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            ConvergenceWarning,
        )

        result = optimizer.optimize(
            dataset=dataset,
            model_factory=ARIMAForecaster,
            candidates=space.candidates(),
        )

    expected_orders = [
        candidate["order"]
        for candidate
        in space.candidates()
    ]

    actual_orders = [
        candidate.params["order"]
        for candidate
        in result.candidates
    ]

    assert actual_orders == (
        expected_orders
    )


def test_arima_candidates_record_temporal_validation_metadata():

    dataset = make_quarterly_dataset()

    space = ARIMAParameterSpace(
        p=(0, 1),
        d=(1,),
        q=(0,),
    )

    optimizer = make_arima_optimizer()

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            ConvergenceWarning,
        )

        result = optimizer.optimize(
            dataset=dataset,
            model_factory=ARIMAForecaster,
            candidates=space.candidates(),
        )

    for candidate in result.candidates:

        assert (
            candidate.metadata["validation"]
            == "expanding_window"
        )

        assert (
            candidate.metadata["folds"]
            == 3
        )

        assert (
            candidate.metadata[
                "validation_steps"
            ]
            == 2
        )

        assert (
            candidate.metadata[
                "selection_metric"
            ]
            == "rmse"
        )

        assert len(
            candidate.metadata[
                "fold_metrics"
            ]
        ) == 3


def test_arima_optimization_creates_fresh_model_per_fold():

    dataset = make_quarterly_dataset()

    space = ARIMAParameterSpace(
        p=(0, 1),
        d=(1,),
        q=(0,),
    )

    created_models = []

    def factory(
        **params,
    ):
        model = ARIMAForecaster(
            **params
        )

        created_models.append(
            model
        )

        return model

    optimizer = make_arima_optimizer()

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            ConvergenceWarning,
        )

        result = optimizer.optimize(
            dataset=dataset,
            model_factory=factory,
            candidates=space.candidates(),
        )

    expected_models = (
        space.candidate_count
        * 3
    )

    assert len(
        created_models
    ) == expected_models

    assert len(
        {
            id(model)
            for model in created_models
        }
    ) == expected_models

    assert result.candidate_count == 2


def test_arima_optimization_supports_mae_selection():

    dataset = make_quarterly_dataset()

    space = ARIMAParameterSpace(
        p=(0, 1),
        d=(0, 1),
        q=(0,),
    )

    optimizer = make_arima_optimizer(
        metric="mae"
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            ConvergenceWarning,
        )

        result = optimizer.optimize(
            dataset=dataset,
            model_factory=ARIMAForecaster,
            candidates=space.candidates(),
        )

    assert result.metric == "mae"

    expected = min(
        result.candidates,
        key=lambda candidate: (
            candidate.metrics["mae"]
        ),
    )

    assert result.best_score == pytest.approx(
        expected.metrics["mae"]
    )

    assert result.best_params == (
        expected.params
    )