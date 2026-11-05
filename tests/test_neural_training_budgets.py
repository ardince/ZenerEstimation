import pytest

from zenerestimation.optimization import GenericOptimizer
from tests.test_generic_optimizer import make_dataset, make_optimizer, OffsetForecaster

class BudgetOffsetForecaster(OffsetForecaster):
    def __init__(self, offset, epochs=1):
        super().__init__(offset=offset)
        self.epochs = epochs


def budget_optimizer(**kwargs):
    original = make_optimizer()
    return GenericOptimizer(original.evaluator, model="LSTM", **kwargs)


def test_candidate_limit_preserves_order_and_selection():
    result = budget_optimizer(max_candidates=2).optimize(
        make_dataset(), OffsetForecaster,
        [{"offset": 3.0}, {"offset": 1.0}, {"offset": 0.0}])
    assert len(result.candidates) == 2
    assert result.best_params == {"offset": 1.0}
    assert result.metadata["budget"]["stop_reason"] == "max_candidates"


def test_epoch_fold_budget_stops_at_boundary():
    result = budget_optimizer(max_epoch_folds=12).optimize(
        make_dataset(), BudgetOffsetForecaster,
        [{"offset": 1.0, "epochs": 2}, {"offset": 2.0, "epochs": 2},
         {"offset": 3.0, "epochs": 1}])
    assert len(result.candidates) == 2
    assert result.metadata["budget"]["planned_epoch_folds"] == 12
    assert result.metadata["budget"]["stop_reason"] == "max_epoch_folds"


def test_combined_limits():
    result = budget_optimizer(max_candidates=1, max_epoch_folds=20).optimize(
        make_dataset(), BudgetOffsetForecaster,
        [{"offset": 1.0, "epochs": 2}, {"offset": 2.0, "epochs": 2}])
    assert len(result.candidates) == 1
    assert result.metadata["budget"]["planned_epoch_folds"] == 6


@pytest.mark.parametrize("kwargs", [{"max_candidates": 0}, {"max_candidates": True},
                                    {"max_epoch_folds": -1}, {"max_epoch_folds": 1.2}])
def test_invalid_budget(kwargs):
    with pytest.raises(ValueError):
        budget_optimizer(**kwargs)


def test_epoch_budget_requires_explicit_epochs():
    with pytest.raises(ValueError, match="epochs"):
        budget_optimizer(max_epoch_folds=10).optimize(
            make_dataset(), OffsetForecaster, [{"offset": 1.0}])


def test_too_small_budget_rejected():
    with pytest.raises(ValueError, match="first candidate"):
        budget_optimizer(max_epoch_folds=2).optimize(
            make_dataset(), OffsetForecaster, [{"offset": 1.0, "epochs": 1}])


def test_legacy_metadata_unchanged():
    result = budget_optimizer().optimize(
        make_dataset(), OffsetForecaster, [{"offset": 1.0}])
    assert "budget" not in result.metadata
