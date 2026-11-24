"""M19-D2-C1: deterministic cross-model sensitivity to common fold omission.

Retrospective reaggregation of M18-verified *internal* folds only. This is not
retraining, protected-holdout evaluation, or model-selection authorization.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import isclose, isfinite
from numbers import Real

from zenerestimation.assessment.fold_sensitivity import assess_internal_fold_sensitivity


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _score(value: object) -> float:
    _require(isinstance(value, Real) and not isinstance(value, bool), "invalid ranking score")
    number = float(value)
    _require(isfinite(number) and number >= 0, "invalid ranking score")
    return number


def _rank(scores: Mapping[str, float]) -> list[dict]:
    """M18 competition ranks: exact ties share rank, with skipped positions."""
    ordered = sorted(scores.items(), key=lambda pair: (pair[1], pair[0]))
    rows: list[dict] = []
    previous = None
    current_rank = 0
    for position, (model, score) in enumerate(ordered, start=1):
        if previous is None or score != previous:
            current_rank = position
        rows.append({"model": model, "score": score, "rank": current_rank})
        previous = score
    return rows


def assess_internal_ranking_sensitivity(
    results: Mapping, *, validation_context: Mapping
) -> dict:
    """Compare verified baseline ranks with each *common* LOFO perturbation.

    The trusted development-dataset context is supplied independently by the
    caller. This function does not accept serialized assessment evidence as
    authoritative input and never selects a model.
    """
    report = assess_internal_fold_sensitivity(results, validation_context=validation_context)
    _require(report.get("assessment_version") == "m19_d2_b2_fold_sensitivity_v1"
             and report.get("scope") == "internal_expanding_window"
             and report.get("method") == "leave_one_fold_out_reaggregation",
             "unsupported internal sensitivity evidence")
    dates = report.get("fold_validation_dates")
    count = report.get("fold_count")
    _require(isinstance(count, int) and not isinstance(count, bool) and count >= 1
             and isinstance(dates, list) and len(dates) == count
             and all(isinstance(d, list) and bool(d) for d in dates),
             "invalid common fold calendar")
    models = report.get("models")
    _require(isinstance(models, list) and bool(models), "model evidence required")
    baseline_scores: dict[str, float] = {}
    omission_scores: list[dict[str, float]] = [dict() for _ in range(count)]
    for model_row in models:
        _require(isinstance(model_row, Mapping), "invalid model evidence")
        model = model_row.get("model")
        _require(isinstance(model, str) and bool(model.strip())
                 and model not in baseline_scores, "invalid or duplicate model")
        baseline = _score(model_row.get("baseline_rmse"))
        baseline_scores[model] = baseline
        if count < 2:
            _require(model_row.get("status") == "unavailable_insufficient_folds"
                     and model_row.get("leave_one_fold_out") == [],
                     "inconsistent unavailable sensitivity")
            continue
        _require(model_row.get("status") == "available", "unavailable fold sensitivity")
        omissions = model_row.get("leave_one_fold_out")
        _require(isinstance(omissions, list) and len(omissions) == count,
                 "incomplete model fold sensitivity")
        seen_folds: set[int] = set()
        for omission in omissions:
            _require(isinstance(omission, Mapping), "invalid fold omission")
            index = omission.get("fold_index")
            _require(isinstance(index, int) and not isinstance(index, bool)
                     and 0 <= index < count and index not in seen_folds,
                     "invalid or duplicate fold index")
            seen_folds.add(index)
            _require(omission.get("validation_dates") == dates[index],
                     "common fold calendar mismatch")
            score = _score(omission.get("lofo_rmse"))
            influence = omission.get("influence")
            _require(isinstance(influence, Real) and not isinstance(influence, bool)
                     and isfinite(float(influence))
                     and isclose(float(influence), score - baseline,
                                 rel_tol=1e-12, abs_tol=1e-12),
                     "inconsistent fold influence")
            omission_scores[index][model] = score
    baseline_ranking = _rank(baseline_scores)
    baseline_winners = [row["model"] for row in baseline_ranking if row["rank"] == 1]
    # Ranking sensitivity requires at least two models and two folds.
    if count < 2 or len(baseline_scores) < 2:
        return {
            "assessment_version": "m19_d2_c1_ranking_sensitivity_v1",
            "scope": "internal_expanding_window",
            "source_assessment_version": report["assessment_version"],
            "dataset_id": report["dataset_id"],
            "dataset_fingerprint": report["dataset_fingerprint"],
            "fold_validation_dates": [list(d) for d in dates],
            "fold_count": count,
            "model_count": len(baseline_scores),
            "status": "unavailable_insufficient_folds" if count < 2 else "unavailable_insufficient_models",
            "baseline_ranking": baseline_ranking,
            "baseline_top_models": baseline_winners,
            "fold_omissions": [],
            "any_rank_change": None,
            "any_top_model_set_change": None,
        }
    original_ranks = {row["model"]: row["rank"] for row in baseline_ranking}
    fold_omissions = []
    for index, scores in enumerate(omission_scores):
        _require(set(scores) == set(baseline_scores), "incomplete common-fold comparison")
        ranking = _rank(scores)
        changes = [
            {"model": row["model"], "baseline_rank": original_ranks[row["model"]],
             "lofo_rank": row["rank"], "rank_delta": row["rank"] - original_ranks[row["model"]]}
            for row in sorted(ranking, key=lambda item: item["model"])
            if row["rank"] != original_ranks[row["model"]]
        ]
        winners = [row["model"] for row in ranking if row["rank"] == 1]
        fold_omissions.append({
            "fold_index": index,
            "validation_dates": list(dates[index]),
            "ranking": ranking,
            "top_models": winners,
            "rank_changes": changes,
            "rank_changed": bool(changes),
            "top_model_set_changed": set(winners) != set(baseline_winners),
        })
    return {
        "assessment_version": "m19_d2_c1_ranking_sensitivity_v1",
        "scope": "internal_expanding_window",
        "source_assessment_version": report["assessment_version"],
        "dataset_id": report["dataset_id"],
        "dataset_fingerprint": report["dataset_fingerprint"],
        "fold_validation_dates": [list(d) for d in dates],
        "fold_count": count,
        "model_count": len(baseline_scores),
        "status": "available",
        "baseline_ranking": baseline_ranking,
        "baseline_top_models": baseline_winners,
        "fold_omissions": fold_omissions,
        "any_rank_change": any(fold["rank_changed"] for fold in fold_omissions),
        "any_top_model_set_change": any(fold["top_model_set_changed"] for fold in fold_omissions),
    }
