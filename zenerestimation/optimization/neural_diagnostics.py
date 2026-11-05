"""Descriptive neural loss-curve diagnostics, independent of model training.

Training and validation loss are in the model's training scale; they are not
forecast RMSE or evidence of out-of-sample generalization.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from zenerestimation.optimization.neural_training import (
    extract_neural_training_history,
    estimate_loss_improvement_onset,
)


def _losses(trace: Mapping[str, Any], key: str) -> list[float] | None:
    rows = trace.get("epochs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("trace must contain nonempty epoch rows")
    numbers: list[float] = []
    seen: set[int] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping) or row.get("epoch") != index:
            raise ValueError("epochs must be consecutive and 1-based")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("each epoch must contain metrics")
        if key not in metrics:
            if any(key in r.get("metrics", {}) for r in rows if isinstance(r, Mapping)):
                raise ValueError(f"metric {key!r} must be present at every epoch")
            return None
        value = metrics[key]
        if isinstance(value, bool):
            raise ValueError(f"metric {key!r} must be finite numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metric {key!r} must be finite numeric") from exc
        if not isfinite(numeric):
            raise ValueError(f"metric {key!r} must be finite numeric")
        numbers.append(numeric)
    return numbers


def analyze_neural_training(
    trace: Mapping[str, Any], *, min_relative_improvement: float = 0.01,
    sustained_epochs: int = 3, plateau_patience: int = 5,
    min_delta: float = 0.0,
) -> dict[str, Any]:
    """Analyze completed training history without modifying or retraining models.

    Plateau: last ``plateau_patience`` epochs contain no new best training
    loss improved by more than ``min_delta`` (absolute training-loss units).
    Best epochs use earliest-minimum tie resolution.
    """
    if not isinstance(trace, Mapping):
        raise TypeError("trace must be a mapping")
    if isinstance(plateau_patience, bool) or not isinstance(plateau_patience, int) or plateau_patience < 1:
        raise ValueError("plateau_patience must be a positive integer")
    if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)) or not isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be nonnegative and finite")
    # Validate onset parameters even if the trace is short.
    if isinstance(sustained_epochs, bool) or not isinstance(sustained_epochs, int) or sustained_epochs < 1:
        raise ValueError("sustained_epochs must be a positive integer")
    if isinstance(min_relative_improvement, bool) or not isinstance(min_relative_improvement, (int, float)) or not isfinite(min_relative_improvement) or not 0 < min_relative_improvement < 1:
        raise ValueError("min_relative_improvement must be between zero and one")
    train = _losses(trace, "loss")
    if train is None:
        raise ValueError("trace requires training loss")
    validation = _losses(trace, "val_loss")
    best_index = min(range(len(train)), key=train.__getitem__)
    best_validation = min(range(len(validation)), key=validation.__getitem__) if validation is not None else None
    best_so_far = train[0]
    last_significant_improvement_epoch = 1
    for i, value in enumerate(train[1:], 2):
        if value < best_so_far - min_delta:
            last_significant_improvement_epoch = i
            best_so_far = value
    return {
        "schema_version": "neural-training-diagnostics-v1",
        "completed_epochs": len(train),
        "estimated_loss_improvement_onset_epoch": estimate_loss_improvement_onset(
            trace, min_relative_improvement=min_relative_improvement,
            sustained_epochs=sustained_epochs,
        ),
        "best_training_loss_epoch": best_index + 1,
        "best_training_loss": train[best_index],
        "final_training_loss": train[-1],
        "best_validation_loss_epoch": None if best_validation is None else best_validation + 1,
        "best_validation_loss": None if best_validation is None else validation[best_validation],
        "validation_status": "recorded" if validation is not None else "not_recorded",
        "training_plateau_at_end": len(train) - last_significant_improvement_epoch >= plateau_patience,
        "last_significant_improvement_epoch": last_significant_improvement_epoch,
        "parameters": {
            "min_relative_improvement": float(min_relative_improvement),
            "sustained_epochs": sustained_epochs,
            "plateau_patience": plateau_patience,
            "min_delta": float(min_delta),
        },
        "interpretation": "descriptive_training_loss_only_not_generalization_evidence",
    }


def analyze_neural_model(model: Any, **kwargs: Any) -> dict[str, Any] | None:
    """Convenience adapter for LSTM, GRU, or hybrid neural residual model."""
    trace = extract_neural_training_history(model)
    return None if trace is None else analyze_neural_training(trace, **kwargs)
