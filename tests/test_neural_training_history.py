from types import SimpleNamespace

import pytest

from zenerestimation.optimization.neural_training import (
    estimate_loss_improvement_onset,
    extract_neural_training_history,
)


def fake_model(history):
    return SimpleNamespace(history=SimpleNamespace(history=history))


def test_missing_history_is_explicit():
    assert extract_neural_training_history(SimpleNamespace(history=None)) is None


def test_extract_epoch_metrics_without_inventing_timestamps():
    trace = extract_neural_training_history(fake_model({"loss": [1.0, 0.9, 0.8], "mae": [0.8, 0.7, 0.6]}))
    assert trace["completed_epochs"] == 3
    assert trace["epochs"][1] == {"epoch": 2, "metrics": {"loss": 0.9, "mae": 0.7}}
    assert trace["training_started_at"] is None
    assert trace["event_timing_status"] == "not_recorded"


def test_hybrid_uses_residual_history():
    residual = fake_model({"loss": [1.0, 0.7]})
    hybrid = SimpleNamespace(residual_model=residual)
    assert extract_neural_training_history(hybrid)["completed_epochs"] == 2


@pytest.mark.parametrize("history", [{"mae": [0.2]}, {"loss": [0.1, float("nan")]}, {"loss": [1.0], "mae": [1.0, 0.5]}, {}])
def test_invalid_history_rejected(history):
    with pytest.raises((ValueError, TypeError)):
        extract_neural_training_history(fake_model(history))


def test_learning_onset_heuristic_and_no_onset():
    trace = extract_neural_training_history(fake_model({"loss": [1.0, 0.98, 0.97, 0.96, 0.99]}))
    assert estimate_loss_improvement_onset(trace) == 2
    assert estimate_loss_improvement_onset(trace, sustained_epochs=5) is None


def test_learning_onset_does_not_assume_generalization():
    trace = extract_neural_training_history(fake_model({"loss": [0.0, 0.0, 0.0, 0.0]}))
    assert estimate_loss_improvement_onset(trace) is None
