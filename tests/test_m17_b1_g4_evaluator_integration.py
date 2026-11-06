
"""M17-B1-G4: opt-in accuracy diagnostics integration contracts."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.optimization.evaluator import (
    ForecastAlignmentError,
    OptimizationEvaluator,
)
from zenerestimation.optimization.validation import ExpandingWindowSplitter


def make_dataset():
    dates = pd.date_range(
        "2020-03-01", periods=20, freq="QS-MAR"
    )
    return BatteryDataset(pd.DataFrame({
        "ds": dates,
        "microVolt": np.arange(1, 21, dtype=float),
    }))


class CountingForecaster:
    fit_calls = 0
    predict_calls = 0

    def __init__(self, offset=0.0, with_dates=True):
        self.offset = float(offset)
        self.with_dates = with_dates
        self.training = None

    def fit(self, dataset):
        type(self).fit_calls += 1
        self.training = dataset
        return self

    def predict(self, steps):
        type(self).predict_calls += 1
        value = float(self.training.target.iloc[-1]) + self.offset
        result = SimpleNamespace(
            forecast=np.full(steps, value, dtype=float)
        )
        if self.with_dates:
            last_date = pd.Timestamp(
                self.training.data["ds"].iloc[-1]
            )
            result.dates = pd.date_range(
                last_date + pd.DateOffset(months=3),
                periods=steps,
                freq="QS-MAR",
            )
        return result


@pytest.fixture(autouse=True)
def reset_counters():
    CountingForecaster.fit_calls = 0
    CountingForecaster.predict_calls = 0


def make_evaluator(
    *,
    record=False,
    strict=False,
    metric="rmse",
):
    return OptimizationEvaluator(
        ExpandingWindowSplitter(folds=3, validation_steps=2),
        metric=metric,
        require_verified_alignment=strict,
        record_accuracy_diagnostics=record,
    )


def evaluate(evaluator, *, offset=0.0, with_dates=True):
    return evaluator.evaluate(
        make_dataset(),
        CountingForecaster,
        {"offset": offset, "with_dates": with_dates},
    )


def test_g4_default_behavior_is_unchanged():
    result = evaluate(make_evaluator())
    assert "accuracy_diagnostics" not in result.metadata
    assert result.metadata["folds"] == 3
    assert len(result.metadata["fold_metrics"]) == 3


def test_g4_enabled_report_contains_measured_rows():
    result = evaluate(make_evaluator(record=True))
    report = result.metadata["accuracy_diagnostics"]

    assert len(report["rows"]) == 6
    assert len(report["fold_metrics"]) == 3
    assert len(report["horizon_metrics"]) == 2
    assert report["provenance"]["scope"] == (
        "internal_expanding_window"
    )
    assert report["provenance"]["units"] == "microVolt"


def test_g4_signed_errors_and_horizon_metrics():
    result = evaluate(make_evaluator(record=True))
    report = result.metadata["accuracy_diagnostics"]

    rows = report["rows"]
    assert [row["error_microVolt"] for row in rows] == [
        -1.0, -2.0, -1.0, -2.0, -1.0, -2.0
    ]

    assert report["summary"]["rmse_microVolt"] == pytest.approx(
        np.sqrt(2.5)
    )
    assert report["summary"]["mae_microVolt"] == pytest.approx(1.5)
    assert report["summary"]["bias_microVolt"] == pytest.approx(-1.5)
    assert report["summary"][
        "first_step_error_microVolt"
    ] == pytest.approx(-1.0)

    horizons = report["horizon_metrics"]
    assert [h["bias_microVolt"] for h in horizons] == [-1.0, -2.0]


@pytest.mark.parametrize("metric", ["rmse", "mae", "mape"])
def test_g4_scoring_and_existing_metadata_are_invariant(metric):
    baseline = evaluate(make_evaluator(metric=metric))
    enabled = evaluate(make_evaluator(record=True, metric=metric))

    assert enabled.score == pytest.approx(baseline.score)
    assert enabled.metrics == pytest.approx(baseline.metrics)

    for key, value in baseline.metadata.items():
        assert enabled.metadata[key] == value

    assert set(enabled.metadata) == (
        set(baseline.metadata) | {"accuracy_diagnostics"}
    )


def test_g4_no_additional_fit_or_predict():
    evaluate(make_evaluator(record=True))
    assert CountingForecaster.fit_calls == 3
    assert CountingForecaster.predict_calls == 3


def test_g4_alignment_status_is_preserved():
    result = evaluate(
        make_evaluator(record=True),
        with_dates=False,
    )
    report = result.metadata["accuracy_diagnostics"]

    assert all(
        row["alignment_status"] == "positional_only"
        for row in report["rows"]
    )
    assert all(
        evidence["status"] == "positional_only"
        for evidence in result.metadata["fold_alignment"]
    )


def test_g4_strict_alignment_still_rejects_missing_dates():
    evaluator = make_evaluator(record=True, strict=True)
    with pytest.raises(
        ForecastAlignmentError,
        match="positional_only",
    ):
        evaluate(evaluator, with_dates=False)


def test_g4_does_not_mutate_observed_dataset():
    dataset = make_dataset()
    original = dataset.data.copy(deep=True)

    evaluator = make_evaluator(record=True)
    evaluator.evaluate(
        dataset,
        CountingForecaster,
        {"offset": 0.0, "with_dates": True},
    )

    pd.testing.assert_frame_equal(dataset.data, original)


def test_g4_requires_boolean_opt_in():
    with pytest.raises(
        TypeError,
        match="record_accuracy_diagnostics",
    ):
        OptimizationEvaluator(
            ExpandingWindowSplitter(folds=3, validation_steps=2),
            record_accuracy_diagnostics="yes",
        )
