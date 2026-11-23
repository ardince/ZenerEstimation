"""M19-D1-B1: read-only verification of internal selection and accuracy evidence.

No holdout inputs, model execution, optimization, or authorization grants.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from math import isclose, isfinite, sqrt

from zenerestimation.optimization.internal_selection_artifact import (
    verify_internal_selection_artifact,
)

VERSION = "m19_d1_b1_internal_evidence_v1"


def _number(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _matches(actual: float, expected: float) -> bool:
    return isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)


def _require_match(actual, expected, name: str) -> None:
    if not _matches(_number(actual, name), _number(expected, name)):
        raise ValueError(f"{name} does not match reconstructed observations")


def verify_internal_generalization_evidence(
    *,
    selection_artifact: Mapping,
    results: Mapping,
    validation_context: Mapping,
) -> dict:
    """Verify selected candidate and reconstruct internal metrics only.

    `results` must map model labels to trusted OptimizationResult objects,
    as required by the M18 ranking/selection replay. The unique evaluated
    best candidate must carry an M17-B2 accuracy profile in its metadata.
    """
    # Reuse the scientific ranking and selection replay; never reselect here.
    verification = verify_internal_selection_artifact(
        selection_artifact,
        results=results,
        validation_context=validation_context,
    )
    if verification.get("status") != "verified":
        raise ValueError("selection artifact verification was not successful")

    record = selection_artifact["selection_decision_record"]
    decision = record["decision"]
    if decision.get("status") != "SELECTED" or decision.get("authorized") is not True:
        raise ValueError("internal selection is not explicitly SELECTED")
    model = decision.get("selected_model")
    if not isinstance(model, str) or not model or model not in results:
        raise ValueError("selected model is missing from trusted results")
    if decision.get("selection_metric") != "rmse":
        raise ValueError("D1-B1 requires the M18 RMSE selection contract")

    optimization_result = results[model]
    if optimization_result.model != model or optimization_result.metric != "rmse":
        raise ValueError("optimization result identity or metric mismatch")
    matching_candidates = [
        candidate for candidate in optimization_result.candidates
        if candidate.status == "evaluated"
        and candidate.params == optimization_result.best_params
        and candidate.score is not None
        and _matches(_number(candidate.score, "candidate.score"),
                     _number(optimization_result.best_score, "best_score"))
    ]
    if len(matching_candidates) != 1:
        raise ValueError("selected optimization candidate must resolve uniquely")
    candidate = matching_candidates[0]
    metadata = candidate.metadata or {}
    if metadata.get("selection_metric") != "rmse":
        raise ValueError("candidate selection metric mismatch")
    profile = metadata.get("accuracy_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("selected candidate requires an accuracy profile")
    if profile.get("profile_version") != "m17_b2_v1":
        raise ValueError("unsupported accuracy profile version")
    if profile.get("provenance", {}).get("scope") != "internal_expanding_window":
        raise ValueError("accuracy profile is not internal validation evidence")
    if profile.get("provenance", {}).get("model") != model:
        raise ValueError("accuracy profile model mismatch")
    alignment = profile.get("alignment_summary", {})
    if alignment.get("fully_verified") is not True or alignment.get("positional_only_rows") != 0:
        raise ValueError("internal observation alignment is not fully verified")

    rows = profile.get("by_date")
    if not isinstance(rows, list) or not rows:
        raise ValueError("accuracy profile has no observation rows")
    folds = defaultdict(list)
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("alignment_status") != "verified":
            raise ValueError("invalid or unverified accuracy observation")
        fold, step = row.get("fold_index"), row.get("horizon_step")
        if type(fold) is not int or fold < 0 or type(step) is not int or step < 1:
            raise ValueError("invalid fold or horizon step")
        identity = (fold, step)
        if identity in seen:
            raise ValueError("duplicate accuracy observation")
        seen.add(identity)
        actual = _number(row.get("actual_microVolt"), "actual_microVolt")
        predicted = _number(row.get("predicted_microVolt"), "predicted_microVolt")
        error = predicted - actual
        if not isfinite(error) or not isfinite(error * error):
            raise ValueError("nonfinite observation error")
        _require_match(row.get("error_microVolt"), error, "error_microVolt")
        _require_match(row.get("absolute_error_microVolt"), abs(error), "absolute_error_microVolt")
        _require_match(row.get("squared_error_microVolt2"), error * error, "squared_error_microVolt2")
        folds[fold].append((step, error))

    fold_metrics = metadata.get("fold_metrics")
    if not isinstance(fold_metrics, list) or len(fold_metrics) != len(folds):
        raise ValueError("candidate fold metrics are missing or inconsistent")
    if sorted(folds) != list(range(len(folds))):
        raise ValueError("fold indices must be consecutive starting at zero")

    per_fold = []
    all_errors = []
    for index, supplied in enumerate(fold_metrics):
        observations = folds[index]
        if [step for step, _ in observations] != list(range(1, len(observations) + 1)):
            raise ValueError("fold horizon steps are not consecutive and ordered")
        errors = [error for _, error in observations]
        all_errors.extend(errors)
        rmse = sqrt(sum(e * e for e in errors) / len(errors))
        mae = sum(abs(e) for e in errors) / len(errors)
        _require_match(supplied["rmse"], rmse, f"fold[{index}].rmse")
        if "mae" in supplied:
            _require_match(supplied["mae"], mae, f"fold[{index}].mae")
        per_fold.append({"fold_index": index, "n": len(errors), "rmse_microVolt": rmse, "mae_microVolt": mae})

    ranking_rmse = sum(row["rmse_microVolt"] for row in per_fold) / len(per_fold)
    ranking_mae = sum(row["mae_microVolt"] for row in per_fold) / len(per_fold)
    pooled_rmse = sqrt(sum(e * e for e in all_errors) / len(all_errors))
    pooled_mae = sum(abs(e) for e in all_errors) / len(all_errors)
    _require_match(optimization_result.best_score, ranking_rmse, "optimization_result.best_score")
    _require_match(candidate.score, ranking_rmse, "candidate.score")
    _require_match(candidate.metrics["rmse"], ranking_rmse, "candidate.metrics.rmse")
    if "mae" in candidate.metrics:
        _require_match(candidate.metrics["mae"], ranking_mae, "candidate.metrics.mae")
    _require_match(decision["selection_score"], ranking_rmse, "decision.selection_score")
    overall = profile.get("overall", {})
    _require_match(overall.get("rmse_microVolt"), pooled_rmse, "profile.overall.rmse")
    _require_match(overall.get("mae_microVolt"), pooled_mae, "profile.overall.mae")
    if overall.get("n") != len(all_errors):
        raise ValueError("profile observation count mismatch")
    if alignment.get("total_rows") != len(rows) or alignment.get("verified_rows") != len(rows):
        raise ValueError("profile alignment counts mismatch")

    return deepcopy({
        "assessment_version": VERSION,
        "status": "INTERNAL_EVIDENCE_VERIFIED",
        "selected_model": model,
        "selection_decision_digest": record["decision_digest"],
        "selection_replay_verified": True,
        "selection_metric": "rmse",
        "units": "microVolt",
        "fold_count": len(per_fold),
        "observation_count": len(all_errors),
        "ranking_rmse_microVolt": ranking_rmse,
        "ranking_mae_microVolt": ranking_mae,
        "pooled_rmse_microVolt": pooled_rmse,
        "pooled_mae_microVolt": pooled_mae,
        "fold_metrics": per_fold,
        "holdout_assessed": False,
        "generalization_gap": None,
        "protected_holdout_comparison_permitted": False,
        "protected_holdout_ranking_permitted": False,
    })
