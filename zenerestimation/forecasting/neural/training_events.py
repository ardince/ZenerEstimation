"""Opt-in, per-fit Keras training event recorder (no validation data added)."""
from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from time import perf_counter
from typing import Any

from keras.callbacks import Callback


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NeuralTrainingRecorder(Callback):
    """Record observed Keras lifecycle events and per-epoch logs.

    A recorder belongs to one fit call. Timing is observational and is not
    evidence of generalization or of a particular first weight update.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.started_at: str | None = None
        self.ended_at: str | None = None
        self.duration_seconds: float | None = None
        self.epochs: list[dict[str, Any]] = []
        self.status = "not_started"
        self._start: float | None = None
        self._epoch_start: float | None = None

    def on_train_begin(self, logs=None) -> None:
        self.reset()
        self.started_at = _utc_now()
        self._start = perf_counter()
        self.status = "running"

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self._epoch_start = perf_counter()

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        metrics: dict[str, float] = {}
        for key, value in (logs or {}).items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if isfinite(numeric):
                metrics[str(key)] = numeric
        self.epochs.append({
            "epoch": int(epoch) + 1,
            "ended_at": _utc_now(),
            "duration_seconds": None if self._epoch_start is None else max(0.0, perf_counter() - self._epoch_start),
            "metrics": metrics,
        })
        self._epoch_start = None

    def on_train_end(self, logs=None) -> None:
        self.ended_at = _utc_now()
        self.duration_seconds = None if self._start is None else max(0.0, perf_counter() - self._start)
        self.status = "completed"

    def mark_failed(self) -> None:
        """Mark an interrupted fit even if Keras skips on_train_end."""
        self.ended_at = _utc_now()
        self.duration_seconds = None if self._start is None else max(0.0, perf_counter() - self._start)
        self.status = "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "neural-training-events-v1",
            "status": self.status,
            "training_started_at": self.started_at,
            "training_ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "completed_epochs": len(self.epochs),
            "epochs": [{"epoch": row["epoch"], "ended_at": row["ended_at"],
                        "duration_seconds": row["duration_seconds"],
                        "metrics": dict(row["metrics"])} for row in self.epochs],
        }
