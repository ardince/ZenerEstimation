"""Focused tests for opt-in fold-specific neural training evidence."""
from types import SimpleNamespace

import pytest

from zenerestimation.optimization import ExpandingWindowSplitter, OptimizationEvaluator
from tests.test_generic_optimizer import OffsetForecaster, make_dataset


class HistoryForecaster(OffsetForecaster):
    def fit(self, dataset):
        super().fit(dataset)
        self.history = SimpleNamespace(history={"loss": [1.0, 0.8, 0.7, 0.6]})
        return self


class HybridHistoryForecaster(OffsetForecaster):
    def fit(self, dataset):
        super().fit(dataset)
        self.residual_model = SimpleNamespace(
            history=SimpleNamespace(history={"loss": [1.0, 0.8, 0.7, 0.6]})
        )
        return self


def evaluator(*, enabled):
    return OptimizationEvaluator(
        ExpandingWindowSplitter(folds=2, validation_steps=2),
        record_neural_diagnostics=enabled,
    )


@pytest.mark.parametrize("factory", [HistoryForecaster, HybridHistoryForecaster])
def test_fold_training_records_all_folds(factory):
    result = evaluator(enabled=True).evaluate(make_dataset(), factory, {"offset": 1.0})
    records = result.metadata["fold_training"]
    assert len(records) == 2
    assert [row["fold_index"] for row in records] == [1, 2]
    assert all(row["status"] == "recorded" for row in records)
    assert all(row["diagnostics"]["best_training_loss_epoch"] == 4 for row in records)
    assert all(row["events"] is None for row in records)


def test_disabled_preserves_metadata_contract():
    result = evaluator(enabled=False).evaluate(make_dataset(), HistoryForecaster, {"offset": 1.0})
    assert "fold_training" not in result.metadata


def test_non_neural_candidate_is_explicitly_not_recorded():
    result = evaluator(enabled=True).evaluate(make_dataset(), OffsetForecaster, {"offset": 1.0})
    assert all(row["status"] == "not_recorded" for row in result.metadata["fold_training"])


def test_diagnostics_do_not_change_selection_metric():
    before = evaluator(enabled=False).evaluate(make_dataset(), HistoryForecaster, {"offset": 1.0})
    after = evaluator(enabled=True).evaluate(make_dataset(), HistoryForecaster, {"offset": 1.0})
    assert before.score == after.score
    assert before.metrics == after.metrics


def test_flag_must_be_boolean():
    with pytest.raises(TypeError, match="record_neural_diagnostics"):
        OptimizationEvaluator(ExpandingWindowSplitter(folds=2, validation_steps=2), record_neural_diagnostics="yes")
