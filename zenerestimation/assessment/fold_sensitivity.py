"""M19-D2-B2: read-only internal expanding-window fold sensitivity.

This is retrospective reaggregation of already verified fold RMSE, not
retraining, prospective validation, holdout access, or authorization.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import fsum, isclose, isfinite
from numbers import Real

from zenerestimation.optimization.internal_ranking import (
    extract_verified_internal_fold_scores,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def assess_internal_fold_sensitivity(
    results: Mapping, *, validation_context: Mapping
) -> dict:
    """Compute deterministic per-model LOFO influence after M18 verification.

    ``validation_context`` must originate independently from the trusted
    development dataset. Matching strings are not dataset authentication.
    An empty or one-fold calendar cannot produce LOFO sensitivity.
    """
    verified = extract_verified_internal_fold_scores(
        results, validation_context=validation_context
    )
    _require(
        verified.get("evidence_version") == "m19_d2_verified_internal_folds_v1"
        and verified.get("scope") == "internal_expanding_window",
        "unsupported internal fold evidence",
    )
    dates = verified["fold_validation_dates"]
    _require(isinstance(dates, list) and bool(dates), "fold calendar required")
    _require(
        all(isinstance(fold, list) and bool(fold) for fold in dates),
        "invalid fold calendar",
    )
    count = len(dates)
    model_rows = verified["models"]
    _require(isinstance(model_rows, list) and bool(model_rows), "model evidence required")
    output = []
    seen = set()
    for row in sorted(model_rows, key=lambda item: item["model"]):
        model = row["model"]
        _require(isinstance(model, str) and bool(model.strip()) and model not in seen,
                 "invalid or duplicate model identity")
        seen.add(model)
        scores = row["fold_rmse"]
        _require(isinstance(scores, list) and len(scores) == count,
                 "fold RMSE count mismatch")
        _require(all(isinstance(x, Real) and not isinstance(x, bool)
                     and isfinite(float(x)) and x >= 0 for x in scores),
                 "invalid fold RMSE")
        scores = [float(x) for x in scores]
        baseline = fsum(scores) / count
        _require(isfinite(baseline), "nonfinite baseline RMSE")
        recorded = row["score"]
        _require(isinstance(recorded, Real) and not isinstance(recorded, bool)
                 and isfinite(float(recorded))
                 and isclose(baseline, float(recorded), rel_tol=1e-12, abs_tol=1e-12),
                 "baseline RMSE inconsistent with verified ranking score")
        if count < 2:
            output.append({
                "model": model,
                "baseline_rmse": baseline,
                "fold_rmse": scores,
                "status": "unavailable_insufficient_folds",
                "leave_one_fold_out": [],
                "max_absolute_influence": None,
            })
            continue
        omitted = []
        for index in range(count):
            remaining = fsum(scores[:index] + scores[index + 1:]) / (count - 1)
            influence = remaining - baseline
            _require(isfinite(remaining) and isfinite(influence),
                     "nonfinite fold sensitivity")
            omitted.append({
                "fold_index": index,
                "validation_dates": list(dates[index]),
                "omitted_fold_rmse": scores[index],
                "lofo_rmse": remaining,
                "influence": influence,
                "absolute_influence": abs(influence),
            })
        output.append({
            "model": model,
            "baseline_rmse": baseline,
            "fold_rmse": scores,
            "status": "available",
            "leave_one_fold_out": omitted,
            "max_absolute_influence": max(item["absolute_influence"] for item in omitted),
        })
    return {
        "assessment_version": "m19_d2_b2_fold_sensitivity_v1",
        "scope": "internal_expanding_window",
        "evidence_version": verified["evidence_version"],
        "dataset_id": verified["dataset_id"],
        "dataset_fingerprint": verified["dataset_fingerprint"],
        "fold_validation_dates": [list(fold) for fold in dates],
        "fold_count": count,
        "method": "leave_one_fold_out_reaggregation",
        "models": output,
    }
