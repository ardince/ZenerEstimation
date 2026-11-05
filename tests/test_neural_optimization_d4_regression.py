"""M16.3-D4: cross-component regression, with lightweight deterministic models.

No TensorFlow training is needed. These tests verify orchestration contracts,
not neural forecasting accuracy or generalization on external benchmarks.
"""
from types import SimpleNamespace

import pytest

from tests.test_generic_optimizer import OffsetForecaster, make_dataset
from zenerestimation.optimization import (
    ExpandingWindowSplitter,
    GenericOptimizer,
    OptimizationEvaluator,
)
from zenerestimation.optimization.neural_reporting import summarize_neural_candidates
from zenerestimation.optimization.spaces import (
    GRUParameterSpace,
    HybridNeuralParameterSpace,
    LSTMParameterSpace,
)


class InstrumentedNeuralForecaster(OffsetForecaster):
    """Accept neural knobs and expose a fold-specific Keras-shaped history."""

    created = []

    def __init__(self, offset=0.0, window=2, units=8, epochs=4,
                 batch_size=8):
        super().__init__(offset=offset)
        self.window = window
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.__class__.created.append(self)

    def fit(self, dataset):
        super().fit(dataset)
        # Distinct histories make accidental reuse across folds observable.
        length = len(dataset.target)
        self.history = SimpleNamespace(history={
            "loss": [1.0 + length / 1000, 0.8, 0.7, 0.6][:self.epochs]
        })
        return self


class InstrumentedHybridForecaster(InstrumentedNeuralForecaster):
    def fit(self, dataset):
        super().fit(dataset)
        self.residual_model = SimpleNamespace(history=self.history)
        del self.history
        return self


def run(candidates, *, model=InstrumentedNeuralForecaster,
        max_candidates=None, max_epoch_folds=None, diagnostics=True,
        strict=False):
    evaluator = OptimizationEvaluator(
        ExpandingWindowSplitter(folds=2, validation_steps=2),
        metric="rmse",
        record_neural_diagnostics=diagnostics,
        require_verified_alignment=strict,
    )
    optimizer = GenericOptimizer(
        evaluator, model="LSTM", max_candidates=max_candidates,
        max_epoch_folds=max_epoch_folds,
    )
    return optimizer.optimize(make_dataset(), model, candidates)


def test_end_to_end_space_budget_and_report():
    InstrumentedNeuralForecaster.created = []
    space = LSTMParameterSpace(window=[2], units=[8, 16],
                               epochs=[4], batch_size=[4, 8])
    assert space.candidate_count == 4
    result = run(space.candidates(), max_candidates=2, max_epoch_folds=16)
    assert len(result.candidates) == 2
    assert result.metadata["budget"]["planned_epoch_folds"] == 16
    assert result.metadata["budget"]["stop_reason"] == "max_candidates"
    assert len(InstrumentedNeuralForecaster.created) == 4  # 2 candidates x 2 folds
    assert {(m.epochs, m.batch_size) for m in InstrumentedNeuralForecaster.created} == {(4, 4), (4, 8)}
    report = summarize_neural_candidates(result)
    assert report["candidate_count"] == 2
    assert report["budget"]["planned_epoch_folds"] == 16
    assert all(row["recorded_history_folds"] == 2 for row in report["candidates"])
    assert all(row["observed_completed_epoch_folds"] == 8 for row in report["candidates"])
    assert all(row["mean_recorded_duration_seconds"] is None for row in report["candidates"])


def test_epoch_budget_boundary_does_not_evaluate_excluded_candidate():
    InstrumentedNeuralForecaster.created = []
    candidates = [{"offset": x, "epochs": 4} for x in (1.0, 2.0, 3.0)]
    result = run(candidates, max_epoch_folds=16)
    assert len(result.candidates) == 2
    assert len(InstrumentedNeuralForecaster.created) == 4
    assert result.metadata["budget"]["stop_reason"] == "max_epoch_folds"
    assert result.metadata["budget"]["planned_epoch_folds"] == 16


def test_diagnostics_do_not_change_scores_or_winner():
    candidates = [{"offset": 0.0, "epochs": 4}, {"offset": 1.0, "epochs": 4}]
    plain = run(candidates, diagnostics=False)
    monitored = run(candidates, diagnostics=True)
    assert plain.best_params == monitored.best_params
    assert plain.best_score == monitored.best_score
    assert [c.score for c in plain.candidates] == [c.score for c in monitored.candidates]
    assert all("fold_training" not in c.metadata for c in plain.candidates)
    assert all(len(c.metadata["fold_training"]) == 2 for c in monitored.candidates)


@pytest.mark.parametrize("factory", [InstrumentedNeuralForecaster, InstrumentedHybridForecaster])
def test_fold_evidence_is_independent_and_one_based(factory):
    result = run([{"offset": 1.0, "epochs": 4}], model=factory)
    folds = result.candidates[0].metadata["fold_training"]
    assert [f["fold_index"] for f in folds] == [1, 2]
    assert all(f["status"] == "recorded" for f in folds)
    assert all(f["diagnostics"]["best_training_loss_epoch"] == 4 for f in folds)
    assert folds[0]["history"] is not folds[1]["history"]
    assert folds[0]["history"]["epochs"][0]["metrics"]["loss"] != folds[1]["history"]["epochs"][0]["metrics"]["loss"]
    report = summarize_neural_candidates(result)
    assert [f["fold_index"] for f in report["candidates"][0]["folds"]] == [1, 2]


def test_candidate_report_preserves_search_order_and_best_score():
    result = run([{"offset": 2.0, "epochs": 4}, {"offset": 0.0, "epochs": 4}])
    report = summarize_neural_candidates(result)
    assert [r["params"] for r in report["candidates"]] == [dict(c.params) for c in result.candidates]
    assert report["best_params"] == dict(result.best_params)
    assert report["best_score"] == result.best_score
    assert report["selection_metric"] == "rmse"


def test_gru_and_hybrid_spaces_keep_explicit_training_dimensions():
    for cls in (GRUParameterSpace, HybridNeuralParameterSpace):
        space = cls(window=[2], units=[8], epochs=[4, 6], batch_size=[8])
        assert space.candidate_count == 2
        assert [row["epochs"] for row in space.candidates()] == [4, 6]
        assert all(row["batch_size"] == 8 for row in space.candidates())


def test_strict_alignment_failure_is_not_converted_to_success():
    # OffsetForecaster returns values without timestamp evidence.
    with pytest.raises(RuntimeError, match="no candidate completed"):
        run([{"offset": 1.0, "epochs": 4}], strict=True)


def test_report_is_read_only():
    result = run([{"offset": 1.0, "epochs": 4}])
    before = result.candidates[0].metadata["fold_training"][0]["diagnostics"]["completed_epochs"]
    report = summarize_neural_candidates(result)
    report["candidates"][0]["folds"][0]["completed_epochs"] = -100
    after = result.candidates[0].metadata["fold_training"][0]["diagnostics"]["completed_epochs"]
    assert before == after == 4
