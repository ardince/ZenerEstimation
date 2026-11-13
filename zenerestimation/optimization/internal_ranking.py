
"""M18 Policy A: read-only ranking of comparable internal validation results.

This first version accepts trusted validation context plus OptimizationResult
objects. It does not fit, select, or use external holdout observations.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np
import pandas as pd

from zenerestimation.evaluation.accuracy_diagnostics import build_accuracy_diagnostics
from zenerestimation.evaluation.accuracy_profile import build_accuracy_profile
from math import isclose, isfinite

from .result import OptimizationResult

from .internal_provenance import PROTOCOL_VERSION


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)



def _verify_internal_provenance(
    recorded: Mapping,
    trusted: Mapping,
) -> None:
    """Require matching development-dataset and validation evidence."""

    _require(
        isinstance(recorded, Mapping),
        "selected candidate requires internal provenance",
    )

    _require(
        trusted.get("protocol_version") == PROTOCOL_VERSION,
        "unsupported trusted provenance protocol",
    )

    _require(
        recorded.get("protocol_version") == PROTOCOL_VERSION,
        "unsupported candidate provenance protocol",
    )

    for field in (
        "dataset_id",
        "dataset_fingerprint",
        "scope",
        "folds",
        "validation_steps",
        "fold_validation_dates",
        "fold_training_end_dates",
    ):
        _require(
            field in trusted and field in recorded,
            f"missing provenance field: {field}",
        )

    fingerprint = trusted["dataset_fingerprint"]

    _require(
        isinstance(fingerprint, str)
        and fingerprint.startswith("sha256:")
        and len(fingerprint) == 71
        and all(
            character in "0123456789abcdef"
            for character in fingerprint[7:]
        ),
        "invalid trusted dataset fingerprint",
    )

    _require(
        trusted["scope"] == "internal_expanding_window",
        "trusted provenance requires internal scope",
    )

    for field in (
        "dataset_id",
        "dataset_fingerprint",
        "scope",
        "folds",
        "validation_steps",
    ):
        _require(
            type(recorded[field]) is type(trusted[field])
            and recorded[field] == trusted[field],
            f"candidate provenance mismatch: {field}",
        )

    for field in (
        "fold_validation_dates",
        "fold_training_end_dates",
    ):
        _require(
            isinstance(trusted[field], (list, tuple))
            and isinstance(recorded[field], (list, tuple))
            and recorded[field] == trusted[field],
            f"candidate provenance mismatch: {field}",
        )



def _compare_canonical(actual, expected, path="accuracy_profile") -> None:
    """Compare a supplied M17 profile with canonical, regenerated evidence."""
    if isinstance(expected, Mapping):
        _require(isinstance(actual, Mapping), f"{path} must be a mapping")
        _require(set(actual) == set(expected), f"{path} fields mismatch")
        for key, value in expected.items():
            _compare_canonical(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        _require(isinstance(actual, list), f"{path} must be a list")
        _require(len(actual) == len(expected), f"{path} length mismatch")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_canonical(left, right, f"{path}[{index}]")
    elif isinstance(expected, bool):
        _require(type(actual) is bool and actual == expected, f"{path} mismatch")
    elif isinstance(expected, int):
        _require(type(actual) is int and actual == expected, f"{path} mismatch")
    elif isinstance(expected, float):
        _require(
            isinstance(actual, Real) and not isinstance(actual, bool)
            and isfinite(float(actual))
            and isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{path} mismatch",
        )
    else:
        _require(actual == expected, f"{path} mismatch")


def _verify_accuracy_profile(profile, *, model, dataset_id, expected, fold_scores):
    """Validate measured rows and all M17 aggregates against internal folds."""
    _require(isinstance(profile, Mapping), "internal accuracy profile required")
    _require(profile.get("profile_version") == "m17_b2_v1", "unsupported accuracy profile")
    provenance = profile.get("provenance")
    _require(isinstance(provenance, Mapping), "accuracy provenance required")
    _require(provenance.get("dataset") == dataset_id, "accuracy dataset mismatch")
    _require(provenance.get("model") == model, "accuracy model mismatch")
    _require(provenance.get("scope") == "internal_expanding_window", "internal accuracy scope required")
    _require(provenance.get("units") == "microVolt", "accuracy units mismatch")
    _require(provenance.get("alignment_statuses") == ["verified"], "verified accuracy rows required")

    rows = profile.get("by_date")
    _require(isinstance(rows, list), "accuracy observation rows required")
    _require(len(rows) == sum(map(len, expected)), "accuracy observation count mismatch")
    folds = []
    for index, dates in enumerate(expected):
        block = rows[index * len(dates): (index + 1) * len(dates)]
        actual, predicted = [], []
        for step, (row, date) in enumerate(zip(block, dates), start=1):
            _require(isinstance(row, Mapping), "invalid accuracy observation")
            _require(row.get("fold_index") == index, "accuracy fold index mismatch")
            _require(row.get("horizon_step") == step, "accuracy horizon mismatch")
            _require(row.get("date") == date, "accuracy date mismatch")
            _require(row.get("alignment_status") == "verified", "unverified accuracy observation")
            for field in ("actual_microVolt", "predicted_microVolt"):
                value = row.get(field)
                _require(
                    isinstance(value, Real) and not isinstance(value, bool)
                    and isfinite(float(value)),
                    f"invalid {field}",
                )
            actual.append(float(row["actual_microVolt"]))
            predicted.append(float(row["predicted_microVolt"]))
        folds.append({
            "fold_index": index,
            "dates": pd.DatetimeIndex(dates),
            "actual": np.asarray(actual, dtype=float),
            "predicted": np.asarray(predicted, dtype=float),
            "alignment_status": "verified",
        })
    try:
        diagnostics = build_accuracy_diagnostics(
            folds, dataset=dataset_id, model=model, scope="internal_expanding_window"
        )
        canonical = build_accuracy_profile(diagnostics)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("invalid M17 accuracy profile observations") from exc
    _compare_canonical(profile, canonical)
    for index, (recorded, canonical_fold) in enumerate(zip(fold_scores, diagnostics["fold_metrics"])):
        _require(
            isclose(recorded, canonical_fold["rmse_microVolt"], rel_tol=1e-12, abs_tol=1e-12),
            f"accuracy fold RMSE mismatch: {index}",
        )


def rank_internal_models(results: Mapping, *, validation_context: Mapping) -> dict:
    """Rank models by mean fold RMSE, with shared ranks for exact ties.

    Context is caller-provided and must be derived from the trusted development
    dataset; matching strings alone do not authenticate dataset provenance.
    """
    _require(isinstance(validation_context, Mapping), "trusted validation_context required")
    _require(validation_context.get("scope") == "internal_expanding_window", "internal validation scope required")
    dataset_id = validation_context.get("dataset_id")
    _require(isinstance(dataset_id, str) and bool(dataset_id.strip()), "dataset_id required")
    expected_dates = validation_context.get("fold_validation_dates")
    _require(isinstance(expected_dates, (list, tuple)) and bool(expected_dates), "fold validation dates required")
    expected = tuple(tuple(dates) for dates in expected_dates)
    _require(all(dates and all(isinstance(d, str) and d for d in dates) for dates in expected), "invalid fold validation dates")
    _require(isinstance(results, Mapping) and bool(results), "nonempty model results mapping required")

    rows = []
    for label, result in results.items():
        _require(isinstance(label, str) and bool(label.strip()), "invalid model label")
        _require(isinstance(result, OptimizationResult), "OptimizationResult required")
        _require(label == result.model, "model identity mismatch")
        _require(result.metric == "rmse", "ranking requires RMSE optimization")
        _require(result.metadata.get("validation") == "internal_temporal", "internal optimization provenance required")
        selected = [c for c in result.candidates if c.status == "evaluated" and c.params == result.best_params and c.score is not None and isclose(float(c.score), float(result.best_score), rel_tol=1e-12, abs_tol=1e-12)]
        _require(len(selected) == 1, "selected candidate must resolve uniquely")
        candidate = selected[0]
        md = candidate.metadata
        _require(md.get("validation") == "expanding_window", "expanding-window evidence required")
        _require(md.get("selection_metric") == "rmse", "RMSE selection evidence required")
        profile = md.get("accuracy_profile")
        _require(isinstance(profile, Mapping), "internal accuracy profile required")
        provenance = profile.get("provenance")
        _require(isinstance(provenance, Mapping) and provenance.get("scope") == "internal_expanding_window", "external or missing accuracy scope")
        _require(provenance.get("dataset") == dataset_id, "development dataset identity mismatch")
        _require(profile.get("profile_version") == "m17_b2_v1", "unsupported accuracy profile")
        folds = md.get("fold_metrics")
        alignment = md.get("fold_alignment")
        _require(isinstance(folds, list) and isinstance(alignment, list), "fold evidence required")
        _require(len(folds) == len(alignment) == len(expected), "fold count mismatch")
        _require(md.get("folds") == len(expected), "fold metadata mismatch")
        _require(md.get("validation_steps") == len(expected[0]) and all(len(x) == len(expected[0]) for x in expected), "validation step mismatch")

        _verify_internal_provenance(
            md.get("internal_provenance"),
            validation_context,
        )

        fold_scores = []
        for index, (metric, evidence, dates) in enumerate(zip(folds, alignment, expected)):
            _require(isinstance(metric, Mapping) and isinstance(evidence, Mapping), "invalid fold evidence")
            score = metric.get("rmse")
            _require(isinstance(score, (int, float)) and not isinstance(score, bool) and isfinite(score) and score >= 0, "invalid fold RMSE")
            fold_scores.append(float(score))
            _require(evidence.get("fold_index") == index, "fold index mismatch")
            _require(evidence.get("status") == "verified", "verified alignment required")
            _require(evidence.get("forecast_dates_provided") is True and evidence.get("timestamp_match") is True, "timestamp verification required")
            _require(evidence.get("validation_steps") == len(dates), "fold horizon mismatch")
            _require(tuple(evidence.get("validation_dates", ())) == dates, "validation dates mismatch")
            _require(tuple(evidence.get("forecast_dates", ())) == dates, "forecast dates mismatch")
        score = sum(fold_scores) / len(fold_scores)
        _require(isclose(score, float(candidate.score), rel_tol=1e-12, abs_tol=1e-12), "candidate score inconsistent with fold RMSE")
        _require(isclose(score, float(candidate.metrics.get("rmse", float("nan"))), rel_tol=1e-12, abs_tol=1e-12), "aggregate RMSE inconsistent")
        _verify_accuracy_profile(
            profile, model=label, dataset_id=dataset_id,
            expected=expected, fold_scores=fold_scores,
        )
        rows.append({"model": label, "score": score})

    rows.sort(key=lambda row: (row["score"], row["model"]))
    prior_score = None
    current_rank = 0
    for position, row in enumerate(rows, start=1):
        if prior_score is None or row["score"] != prior_score:
            current_rank = position
        row["rank"] = current_rank
        prior_score = row["score"]
    return {"ranking": rows, "ineligible": [], "selected_model": None, "ranking_version": "m18_internal_ranking_v1"}
