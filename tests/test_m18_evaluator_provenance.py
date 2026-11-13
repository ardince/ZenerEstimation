"""M18-B2b red-phase tests: opt-in provenance recorded at evaluation time.

These tests deliberately precede the implementation. No external holdout is
used, and the existing optimizer's candidate-selection policy is unchanged.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.optimization.evaluator import OptimizationEvaluator
from zenerestimation.optimization.internal_provenance import (
    build_internal_validation_context,
)
from zenerestimation.optimization.search import GenericOptimizer
from zenerestimation.optimization.validation import ExpandingWindowSplitter


DATASET_ID = "synthetic-m18-development"


def make_dataset(rows=16, *, observed=False):
    frame = pd.DataFrame({
        "ds": pd.date_range("2020-03-01", periods=rows, freq="QS-MAR"),
        "microVolt": np.arange(1, rows + 1, dtype=float),
    })
    if observed:
        frame["is_observed"] = True
    return BatteryDataset(frame)


def make_splitter():
    return ExpandingWindowSplitter(folds=2, validation_steps=2)


class PositionalForecast:
    def __init__(self, values):
        self.forecast = np.asarray(values, dtype=float)


class ConstantForecaster:
    fit_calls = 0

    def __init__(self, offset=0.0):
        self.offset = float(offset)
        self.last = None

    def fit(self, dataset):
        type(self).fit_calls += 1
        self.last = float(dataset.data["microVolt"].iloc[-1])
        return self

    def predict(self, steps):
        return PositionalForecast(np.full(steps, self.last + self.offset))


def evaluate(dataset=None, *, provenance=True, **options):
    evaluator = OptimizationEvaluator(
        make_splitter(),
        metric="rmse",
        record_internal_provenance=provenance,
        accuracy_dataset_id=DATASET_ID,
        **options,
    )
    return evaluator.evaluate(
        dataset=make_dataset() if dataset is None else dataset,
        model_factory=ConstantForecaster,
        params={"offset": 0.0},
    )


def test_provenance_is_absent_by_default():
    evaluator = OptimizationEvaluator(make_splitter(), metric="rmse")
    result = evaluator.evaluate(make_dataset(), ConstantForecaster, {"offset": 0.0})
    assert "internal_provenance" not in result.metadata


def test_opt_in_provenance_matches_independently_derived_context():
    data = make_dataset()
    expected = build_internal_validation_context(
        data, make_splitter(), dataset_id=DATASET_ID
    )
    result = evaluate(data)
    assert result.metadata["internal_provenance"] == expected
    assert expected["scope"] == "internal_expanding_window"
    assert expected["dataset_fingerprint"].startswith("sha256:")
    assert expected["folds"] == 2
    assert expected["validation_steps"] == 2


def test_default_dataset_id_reuses_existing_accuracy_identifier():
    evaluator = OptimizationEvaluator(
        make_splitter(), record_internal_provenance=True
    )
    result = evaluator.evaluate(make_dataset(), ConstantForecaster, {"offset": 0.0})
    assert result.metadata["internal_provenance"]["dataset_id"] == (
        "internal_validation_dataset"
    )


def test_historical_measurement_change_changes_recorded_fingerprint():
    original = make_dataset()
    changed = make_dataset()
    changed.data.loc[0, "microVolt"] += 0.25
    assert (
        evaluate(original).metadata["internal_provenance"]["dataset_fingerprint"]
        != evaluate(changed).metadata["internal_provenance"]["dataset_fingerprint"]
    )


@pytest.mark.parametrize("invalid", [None, 1, "yes", []])
def test_provenance_opt_in_requires_boolean(invalid):
    with pytest.raises(TypeError, match="record_internal_provenance"):
        OptimizationEvaluator(make_splitter(), record_internal_provenance=invalid)


def test_invalid_dataset_id_rejected_when_recording():
    with pytest.raises(ValueError, match="accuracy_dataset_id"):
        OptimizationEvaluator(
            make_splitter(),
            record_internal_provenance=True,
            accuracy_dataset_id=" ",
        )


@pytest.mark.parametrize("invalidity", ["unsorted", "missing_validation", "inserted_validation"])
def test_invalid_provenance_fails_before_model_fit(invalidity):
    data = make_dataset(observed=True)
    if invalidity == "unsorted":
        data.data = data.data.iloc[::-1].reset_index(drop=True)
    elif invalidity == "missing_validation":
        data.data.loc[len(data.data) - 1, "microVolt"] = np.nan
    else:
        data.data.loc[len(data.data) - 1, "microVolt"] = np.nan
        data.data.loc[len(data.data) - 1, "is_observed"] = False

    ConstantForecaster.fit_calls = 0
    with pytest.raises(ValueError):
        evaluate(data)
    assert ConstantForecaster.fit_calls == 0


def test_recording_does_not_change_metrics_or_selection_score():
    data = make_dataset()
    plain = evaluate(data, provenance=False)
    recorded = evaluate(data, provenance=True)
    assert recorded.score == pytest.approx(plain.score)
    assert recorded.metrics == pytest.approx(plain.metrics)
    assert recorded.metadata["fold_metrics"] == plain.metadata["fold_metrics"]
    assert recorded.metadata["fold_alignment"] == plain.metadata["fold_alignment"]


def test_recording_does_not_mutate_development_dataset():
    data = make_dataset(observed=True)
    before = data.data.copy(deep=True)
    evaluate(data)
    pd.testing.assert_frame_equal(data.data, before)


def test_provenance_coexists_with_accuracy_diagnostics_and_profile():
    result = evaluate(
        record_accuracy_diagnostics=True,
        record_accuracy_profile=True,
    )
    assert "internal_provenance" in result.metadata
    assert "accuracy_diagnostics" in result.metadata
    assert "accuracy_profile" in result.metadata
    assert result.metadata["accuracy_profile"]["provenance"]["scope"] == (
        "internal_expanding_window"
    )


def test_generic_optimizer_preserves_provenance_on_evaluated_candidates():
    evaluator = OptimizationEvaluator(
        make_splitter(),
        metric="rmse",
        record_internal_provenance=True,
        accuracy_dataset_id=DATASET_ID,
    )
    optimizer = GenericOptimizer(evaluator, model="ConstantForecaster")
    result = optimizer.optimize(
        make_dataset(), ConstantForecaster,
        candidates=[{"offset": 0.0}, {"offset": 1.0}],
    )
    assert len(result.candidates) == 2
    assert all(candidate.status == "evaluated" for candidate in result.candidates)
    fingerprints = {
        candidate.metadata["internal_provenance"]["dataset_fingerprint"]
        for candidate in result.candidates
    }
    assert len(fingerprints) == 1
    assert result.metadata["validation"] == "internal_temporal"


def test_failed_candidate_is_not_misrepresented_as_successful_provenance():
    class FailingForecaster(ConstantForecaster):
        def fit(self, dataset):
            raise RuntimeError("deliberate training failure")

    evaluator = OptimizationEvaluator(
        make_splitter(),
        record_internal_provenance=True,
        accuracy_dataset_id=DATASET_ID,
    )
    optimizer = GenericOptimizer(evaluator, model="ConstantForecaster")

    def factory(offset):
        return FailingForecaster(offset) if offset == -1.0 else ConstantForecaster(offset)

    result = optimizer.optimize(
        make_dataset(), factory,
        candidates=[{"offset": -1.0}, {"offset": 0.0}],
    )
    failed, successful = result.candidates
    assert failed.status != "evaluated"
    assert "internal_provenance" not in failed.metadata
    assert "internal_provenance" in successful.metadata
