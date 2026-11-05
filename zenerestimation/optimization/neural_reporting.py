"""Read-only candidate comparison for Milestone 16.3-D3.

Does not select candidates, change scores, or consume benchmark holdouts.
Only reports metrics already present in an OptimizationResult.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if isfinite(number) else None


def summarize_neural_candidates(result: Any) -> dict[str, Any]:
    """Summarize candidate/fold evidence without changing optimization ranking.

    Missing training histories and event traces are explicitly represented.
    Epoch-fold counts are *observed completed epochs*, distinct from the
    optimizer's planned epoch-fold budget.
    """
    candidates = getattr(result, "candidates", None)
    if candidates is None:
        raise TypeError("result must expose candidates")
    rows = []
    for position, candidate in enumerate(candidates, 1):
        metadata = candidate.metadata or {}
        if not isinstance(metadata, Mapping):
            raise TypeError("candidate metadata must be a mapping")
        folds = metadata.get("fold_training")
        if folds is not None and not isinstance(folds, (list, tuple)):
            raise ValueError("fold_training must be a sequence or absent")
        onset, best, duration = [], [], []
        completed = 0
        recorded = 0
        plateau_count = 0
        fold_rows = []
        for fold in folds or ():
            if not isinstance(fold, Mapping):
                raise ValueError("fold_training entries must be mappings")
            diag = fold.get("diagnostics")
            events = fold.get("events")
            if diag is not None and not isinstance(diag, Mapping):
                raise ValueError("diagnostics must be a mapping or None")
            if events is not None and not isinstance(events, Mapping):
                raise ValueError("events must be a mapping or None")
            diag = diag or {}
            events = events or {}
            if fold.get("status") == "recorded" or fold.get("history") is not None:
                recorded += 1
            onset_value = _numeric(diag.get("estimated_loss_improvement_onset_epoch"))
            best_value = _numeric(diag.get("best_training_loss_epoch"))
            duration_value = _numeric(events.get("duration_seconds"))
            epochs_value = _numeric(diag.get("completed_epochs"))
            if onset_value is not None:
                onset.append(onset_value)
            if best_value is not None:
                best.append(best_value)
            if duration_value is not None:
                duration.append(duration_value)
            if epochs_value is not None:
                completed += int(epochs_value)
            if diag.get("training_plateau_at_end") is True:
                plateau_count += 1
            fold_rows.append({
                "fold_index": fold.get("fold_index"),
                "component": fold.get("component"),
                "history_status": fold.get("status", "unknown"),
                "estimated_onset_epoch": onset_value,
                "best_training_loss_epoch": best_value,
                "training_plateau_at_end": diag.get("training_plateau_at_end"),
                "completed_epochs": epochs_value,
                "duration_seconds": duration_value,
                "event_status": events.get("status", "not_recorded"),
            })
        rows.append({
            "candidate_index": position,
            "status": candidate.status,
            "params": dict(candidate.params),
            "score": candidate.score,
            "metrics": dict(candidate.metrics or {}),
            "fold_count": len(folds) if folds is not None else None,
            "recorded_history_folds": recorded,
            "observed_completed_epoch_folds": completed if folds is not None else None,
            "mean_estimated_onset_epoch": _mean(onset),
            "mean_best_training_loss_epoch": _mean(best),
            "mean_recorded_duration_seconds": _mean(duration),
            "recorded_duration_folds": len(duration),
            "plateau_folds": plateau_count if folds is not None else None,
            "folds": fold_rows,
        })
    optimization_metadata = getattr(result, "metadata", None) or {}
    return {
        "schema_version": "neural-candidate-report-v1",
        "model": getattr(result, "model", None),
        "selection_metric": getattr(result, "metric", None),
        "best_params": dict(getattr(result, "best_params", {}) or {}),
        "best_score": getattr(result, "best_score", None),
        "candidate_count": len(rows),
        "budget": dict(optimization_metadata.get("budget") or {}),
        "candidates": rows,
        "interpretation": "descriptive_training_evidence_only; selection_unchanged; not_generalization_evidence",
    }
