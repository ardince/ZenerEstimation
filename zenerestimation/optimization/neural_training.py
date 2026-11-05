"""Read-only extraction of completed Keras neural training histories.

No training, callbacks, model state, or validation boundaries are changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any


def _neural_component(model: Any) -> Any:
    """Find the neural residual component of a hybrid, if present."""
    return getattr(model, "residual_model", model)


def extract_neural_training_history(model: Any) -> dict[str, Any] | None:
    """Extract a JSON-compatible trace from a completed Keras History.

    Returns None if the model has no recorded training history. An empty or
    malformed history is rejected rather than misreported as successful learning.
    """
    neural = _neural_component(model)
    history = getattr(neural, "history", None)
    if history is None:
        return None
    raw = getattr(history, "history", None)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("neural training history must contain metric sequences")

    metrics: dict[str, list[float]] = {}
    length: int | None = None
    for name, sequence in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError("training metric names must be non-empty strings")
        if isinstance(sequence, (str, bytes)):
            raise TypeError("training metrics must be numeric sequences")
        try:
            values = [float(value) for value in sequence]
        except (TypeError, ValueError) as exc:
            raise TypeError(f"training metric {name!r} must be numeric") from exc
        if not values or any(not isfinite(value) for value in values):
            raise ValueError(f"training metric {name!r} must contain finite values")
        if length is None:
            length = len(values)
        elif len(values) != length:
            raise ValueError("training metric sequences must have equal lengths")
        metrics[name] = values

    if "loss" not in metrics:
        raise ValueError("neural training history must contain 'loss'")
    assert length is not None
    return {
        "schema_version": "neural-training-history-v1",
        "component": type(neural).__name__,
        "completed_epochs": length,
        "epochs": [
            {"epoch": index + 1, "metrics": {name: values[index] for name, values in metrics.items()}}
            for index in range(length)
        ],
        "training_started_at": None,
        "training_ended_at": None,
        "duration_seconds": None,
        "event_timing_status": "not_recorded",
    }


def estimate_loss_improvement_onset(
    trace: Mapping[str, Any], *, min_relative_improvement: float = 0.01, sustained_epochs: int = 3
) -> int | None:
    """Return first 1-based epoch beginning sustained improvement vs epoch 1.

    This is a descriptive loss-curve heuristic, NOT evidence of generalization.
    """
    if not isfinite(min_relative_improvement) or not 0 < min_relative_improvement < 1:
        raise ValueError("min_relative_improvement must be finite and between 0 and 1")
    if isinstance(sustained_epochs, bool) or not isinstance(sustained_epochs, int) or sustained_epochs < 1:
        raise ValueError("sustained_epochs must be a positive integer")
    epochs = trace["epochs"]
    losses = [float(row["metrics"]["loss"]) for row in epochs]
    if not losses or not all(isfinite(v) for v in losses):
        raise ValueError("losses must be nonempty and finite")
    baseline = losses[0]
    if baseline <= 0:
        return None
    threshold = baseline * (1 - min_relative_improvement)
    for start in range(1, len(losses) - sustained_epochs + 1):
        if all(value <= threshold for value in losses[start:start + sustained_epochs]):
            return start + 1
    return None
