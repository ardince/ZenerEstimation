"""Regression tests for optimization candidate failure handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.optimization import (
    ExpandingWindowSplitter,
    GenericOptimizer,
    OptimizationEvaluator,
)


class SyntheticForecastResult:
    """Minimal forecast result used by the synthetic forecaster."""

    def __init__(self, forecast: np.ndarray) -> None:
        self.forecast = forecast


class FailingForecaster:
    """Deterministic forecaster used to test candidate failure handling.

    The candidate with ``value == 999`` raises a LinAlgError during fit.
    Other candidates forecast the final training value plus ``value``.
    """

    def __init__(self, *, value: float) -> None:
        self.value = value
        self.dataset = None

    def fit(self, dataset: BatteryDataset):
        if self.value == 999:
            raise np.linalg.LinAlgError("synthetic failure")

        self.dataset = dataset
        return self

    def predict(self, steps: int = 1) -> SyntheticForecastResult:
        if self.dataset is None:
            raise RuntimeError("model must be fitted before prediction")

        last_value = float(
            self.dataset.data["microVolt"].iloc[-1]
        )

        forecast = np.full(
            steps,
            last_value + float(self.value),
            dtype=float,
        )

        return SyntheticForecastResult(forecast)


class AlwaysFailingForecaster:
    """Forecaster for testing the all-candidates-fail case."""

    def __init__(self, *, value: float) -> None:
        self.value = value

    def fit(self, dataset: BatteryDataset):
        raise np.linalg.LinAlgError("all candidates fail")

    def predict(self, steps: int = 1):
        raise AssertionError("predict must not be called")


class InterruptingForecaster:
    """Forecaster used to verify that KeyboardInterrupt is propagated."""

    def __init__(self, *, value: float) -> None:
        self.value = value

    def fit(self, dataset: BatteryDataset):
        raise KeyboardInterrupt("synthetic interrupt")

    def predict(self, steps: int = 1):
        raise AssertionError("predict must not be called")


@pytest.fixture
def dataset() -> BatteryDataset:
    """Return a deterministic quarterly dataset for optimization tests."""

    dates = pd.date_range(
        start="2020-03-01",
        periods=16,
        freq="QS-MAR",
    )

    values = np.arange(
        10.0,
        26.0,
        dtype=float,
    )

    frame = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": values,
        }
    )

    return BatteryDataset(frame)


@pytest.fixture
def evaluator() -> OptimizationEvaluator:
    """Return a small deterministic temporal evaluator."""

    splitter = ExpandingWindowSplitter(
        folds=2,
        validation_steps=2,
    )

    return OptimizationEvaluator(
        splitter,
        metric="rmse",
    )


def test_failed_candidate_does_not_abort_search(
    dataset,
    evaluator,
):
    """One candidate failure must not terminate the whole search."""

    optimizer = GenericOptimizer(
        evaluator,
        model="Synthetic",
    )

    result = optimizer.optimize(
        dataset=dataset,
        model_factory=FailingForecaster,
        candidates=(
            {"value": 0.0},
            {"value": 999},
            {"value": 1.0},
        ),
    )

    assert result.candidate_count == 3
    assert result.successful_count == 2
    assert result.failed_count == 1


def test_failed_candidate_contract(
    dataset,
    evaluator,
):
    """Failed candidates must use the explicit failed-state contract."""

    optimizer = GenericOptimizer(
        evaluator,
        model="Synthetic",
    )

    result = optimizer.optimize(
        dataset=dataset,
        model_factory=FailingForecaster,
        candidates=(
            {"value": 0.0},
            {"value": 999},
        ),
    )

    failed = next(
        candidate
        for candidate in result.candidates
        if candidate.status == "failed"
    )

    assert failed.params == {"value": 999}
    assert failed.score is None
    assert failed.metrics == {}


def test_failed_candidate_records_error_metadata(
    dataset,
    evaluator,
):
    """Failure type and message must remain available for audit."""

    optimizer = GenericOptimizer(
        evaluator,
        model="Synthetic",
    )

    result = optimizer.optimize(
        dataset=dataset,
        model_factory=FailingForecaster,
        candidates=(
            {"value": 0.0},
            {"value": 999},
        ),
    )

    failed = next(
        candidate
        for candidate in result.candidates
        if candidate.status == "failed"
    )

    assert failed.metadata["error_type"] == "LinAlgError"
    assert failed.metadata["error_message"] == "synthetic failure"


def test_failed_candidate_cannot_be_selected(
    dataset,
    evaluator,
):
    """Failed candidates must never participate in best-score selection."""

    optimizer = GenericOptimizer(
        evaluator,
        model="Synthetic",
    )

    result = optimizer.optimize(
        dataset=dataset,
        model_factory=FailingForecaster,
        candidates=(
            {"value": 999},
            {"value": 0.0},
            {"value": 1.0},
        ),
    )

    assert result.best_params != {"value": 999}

    selected = next(
        candidate
        for candidate in result.candidates
        if candidate.params == result.best_params
    )

    assert selected.status == "evaluated"
    assert selected.score is not None
    assert np.isfinite(selected.score)


def test_candidate_counts_include_failures(
    dataset,
    evaluator,
):
    """Attempted, successful, and failed counts must remain consistent."""

    optimizer = GenericOptimizer(
        evaluator,
        model="Synthetic",
    )

    result = optimizer.optimize(
        dataset=dataset,
        model_factory=FailingForecaster,
        candidates=(
            {"value": 0.0},
            {"value": 999},
            {"value": 1.0},
            {"value": 2.0},
        ),
    )

    assert result.candidate_count == 4
    assert result.successful_count == 3
    assert result.failed_count == 1

    assert (
        result.successful_count
        + result.failed_count
        == result.candidate_count
    )


def test_all_candidates_failing_raises_runtime_error(
    dataset,
    evaluator,
):
    """Optimization must fail clearly when no candidate is evaluable."""

    optimizer = GenericOptimizer(
        evaluator,
        model="Synthetic",
    )

    with pytest.raises(
        RuntimeError,
        match="no candidate completed evaluation",
    ):
        optimizer.optimize(
            dataset=dataset,
            model_factory=AlwaysFailingForecaster,
            candidates=(
                {"value": 1.0},
                {"value": 2.0},
                {"value": 3.0},
            ),
        )


def test_keyboard_interrupt_is_not_swallowed(
    dataset,
    evaluator,
):
    """User interrupts must propagate rather than become failures."""

    optimizer = GenericOptimizer(
        evaluator,
        model="Synthetic",
    )

    with pytest.raises(
        KeyboardInterrupt,
        match="synthetic interrupt",
    ):
        optimizer.optimize(
            dataset=dataset,
            model_factory=InterruptingForecaster,
            candidates=(
                {"value": 1.0},
            ),
        )