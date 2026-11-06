
"""Additive, measurement-only forecast accuracy diagnostics for M17-B1.

This module does not fit models, rank candidates, or modify evaluation results.
Input predictions must already have been aligned by the caller; alignment
status is preserved as evidence, never inferred from the supplied dates.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

_ALLOWED_SCOPES = frozenset({"internal_expanding_window", "external_holdout"})
_ALLOWED_ALIGNMENTS = frozenset({"verified", "positional_only"})


def _metrics(errors: np.ndarray) -> dict:
    return {
        "n": int(errors.size),
        "rmse_microVolt": float(np.sqrt(np.mean(np.square(errors)))),
        "mae_microVolt": float(np.mean(np.abs(errors))),
        "bias_microVolt": float(np.mean(errors)),
    }


def _validate_fold(raw: Mapping) -> tuple[int, pd.DatetimeIndex, np.ndarray, np.ndarray, str]:
    if not isinstance(raw, Mapping):
        raise ValueError("Each fold must be a mapping")
    try:
        index = raw["fold_index"]
        dates_raw = raw["dates"]
        actual_raw = raw["actual"]
        predicted_raw = raw["predicted"]
        status = raw["alignment_status"]
    except KeyError as exc:
        raise ValueError(f"Missing fold field: {exc.args[0]}") from exc

    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or index < 0:
        raise ValueError("fold_index must be a nonnegative integer")
    if status not in _ALLOWED_ALIGNMENTS:
        raise ValueError("Unknown alignment status")
    try:
        dates = pd.DatetimeIndex(dates_raw)
        actual = np.asarray(actual_raw, dtype=float)
        predicted = np.asarray(predicted_raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid fold dates or numeric values") from exc
    if actual.ndim != 1 or predicted.ndim != 1 or len(actual) == 0:
        raise ValueError("Actual and predicted must be nonempty 1D arrays")
    if len(actual) != len(predicted) or len(dates) != len(actual):
        raise ValueError("Dates, actual and predicted lengths must match")
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be nonmissing, unique and increasing")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted must contain only finite values")
    return int(index), dates, actual, predicted, status


def build_accuracy_diagnostics(
    folds: Iterable[Mapping], *, dataset: str, model: str, scope: str
) -> dict:
    """Summarize already-aligned, measured fold forecasts in microvolts.

    Supported scopes are ``internal_expanding_window`` and ``external_holdout``.
    Operational future forecasts are deliberately not scoreable here.
    For multiple folds, first-step summary errors are the mean signed/absolute
    first-step errors; horizon metrics pool observations at each step.
    """
    if scope not in _ALLOWED_SCOPES:
        raise ValueError("scope must identify a measured evaluation, not an operational forecast")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a nonempty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    try:
        items = list(folds)
    except TypeError as exc:
        raise ValueError("folds must be iterable") from exc
    if not items:
        raise ValueError("At least one fold is required")

    rows = []
    fold_metrics = []
    statuses = []
    seen = set()
    for raw in items:
        index, dates, actual, predicted, status = _validate_fold(raw)
        if index in seen:
            raise ValueError("Duplicate fold_index")
        seen.add(index)
        if status not in statuses:
            statuses.append(status)
        errors = predicted - actual
        if not np.isfinite(errors).all() or not np.isfinite(np.square(errors)).all():
            raise ValueError("Error magnitude exceeds finite numeric range")
        fold_metrics.append({"fold_index": index, **_metrics(errors)})
        for step, (date, a, p, error) in enumerate(zip(dates, actual, predicted, errors), 1):
            rows.append({
                "fold_index": index,
                "horizon_step": step,
                "date": date.strftime("%Y-%m-%d"),
                "actual_microVolt": float(a),
                "predicted_microVolt": float(p),
                "error_microVolt": float(error),
                "absolute_error_microVolt": float(abs(error)),
                "squared_error_microVolt2": float(error * error),
                "alignment_status": status,
            })

    all_errors = np.array([r["error_microVolt"] for r in rows], dtype=float)
    summary = _metrics(all_errors)
    first = [r["error_microVolt"] for r in rows if r["horizon_step"] == 1]
    summary["first_step_error_microVolt"] = float(np.mean(first))
    summary["first_step_absolute_error_microVolt"] = float(np.mean(np.abs(first)))

    rmses = np.array([f["rmse_microVolt"] for f in fold_metrics])
    worst = int(np.argmax(rmses))
    summary["fold_stability"] = {
        "fold_count": len(fold_metrics),
        "rmse_mean_microVolt": float(np.mean(rmses)),
        "rmse_std_population_microVolt": float(np.std(rmses, ddof=0)),
        "worst_fold_index": fold_metrics[worst]["fold_index"],
        "worst_fold_rmse_microVolt": float(rmses[worst]),
    }
    horizon_metrics = []
    for step in sorted({r["horizon_step"] for r in rows}):
        errors = np.array([r["error_microVolt"] for r in rows if r["horizon_step"] == step])
        horizon_metrics.append({"horizon_step": step, **_metrics(errors)})

    return {
        "rows": rows,
        "summary": summary,
        "horizon_metrics": horizon_metrics,
        "fold_metrics": fold_metrics,
        "provenance": {
            "dataset": dataset,
            "model": model,
            "scope": scope,
            "units": "microVolt",
            "alignment_statuses": statuses,
        },
    }
