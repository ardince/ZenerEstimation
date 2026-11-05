import copy
import pytest
from zenerestimation.optimization.neural_diagnostics import analyze_neural_training, analyze_neural_model


def trace(loss, val=None):
    return {"epochs": [{"epoch": i + 1, "metrics": {"loss": x, **({"val_loss": val[i]} if val is not None else {})}} for i, x in enumerate(loss)]}


def test_onset_best_and_plateau():
    result = analyze_neural_training(trace([1.0, .8, .7, .6, .6, .6, .6, .6, .6]), plateau_patience=5)
    assert result["estimated_loss_improvement_onset_epoch"] == 2
    assert result["best_training_loss_epoch"] == 4
    assert result["training_plateau_at_end"] is True


def test_validation_is_optional():
    result = analyze_neural_training(trace([1, .9, .8]))
    assert result["validation_status"] == "not_recorded"
    assert result["best_validation_loss_epoch"] is None


def test_validation_best_is_independent():
    result = analyze_neural_training(trace([1, .8, .6], [1, .5, .7]))
    assert result["best_training_loss_epoch"] == 3
    assert result["best_validation_loss_epoch"] == 2


def test_no_onset_for_flat_loss():
    assert analyze_neural_training(trace([1] * 5))["estimated_loss_improvement_onset_epoch"] is None


def test_short_trace_not_plateau():
    assert analyze_neural_training(trace([1, .9]), plateau_patience=5)["training_plateau_at_end"] is False


@pytest.mark.parametrize("values", [[1, float("nan")], [1, float("inf")]])
def test_reject_nonfinite(values):
    with pytest.raises(ValueError):
        analyze_neural_training(trace(values))


def test_reject_partial_validation():
    t = trace([1, .8, .7], [1, .9, .8]); del t["epochs"][1]["metrics"]["val_loss"]
    with pytest.raises(ValueError, match="every epoch"):
        analyze_neural_training(t)


def test_reject_nonconsecutive_epochs():
    t = trace([1, .9]); t["epochs"][1]["epoch"] = 3
    with pytest.raises(ValueError, match="consecutive"):
        analyze_neural_training(t)


def test_input_unchanged():
    t = trace([1, .8, .7]); old = copy.deepcopy(t)
    analyze_neural_training(t)
    assert t == old


def test_hybrid_residual_adapter():
    class History:
        history = {"loss": [1.0, .8, .7, .6]}
    class Neural:
        history = History()
    class Hybrid:
        residual_model = Neural()
    result = analyze_neural_model(Hybrid())
    assert result["best_training_loss_epoch"] == 4


def test_no_history_returns_none():
    assert analyze_neural_model(object()) is None


@pytest.mark.parametrize("kwargs", [{"plateau_patience": 0}, {"min_delta": -1}, {"sustained_epochs": 0}, {"min_relative_improvement": 1}])
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        analyze_neural_training(trace([1, .9]), **kwargs)
