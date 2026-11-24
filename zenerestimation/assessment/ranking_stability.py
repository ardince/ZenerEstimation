"""M19-D2-C2: descriptive stability of M18-verified internal LOFO rankings.

No retraining, holdout access, prospective certification, or selection authority.
The public entry point accepts original results and independently trusted context,
not serialized ranking reports supplied by callers.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real

from zenerestimation.assessment.ranking_sensitivity import assess_internal_ranking_sensitivity


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validated_ranking(rows: object, *, models: set[str], label: str) -> dict[str, int]:
    _require(isinstance(rows, list) and len(rows) == len(models), f"invalid {label} ranking")
    seen: set[str] = set()
    ranks: dict[str, int] = {}
    scores: list[tuple[str, float, int]] = []
    for row in rows:
        _require(isinstance(row, Mapping), f"invalid {label} ranking row")
        model, score, rank = row.get("model"), row.get("score"), row.get("rank")
        _require(isinstance(model, str) and model in models and model not in seen,
                 f"invalid {label} model")
        _require(isinstance(score, Real) and not isinstance(score, bool)
                 and isfinite(float(score)) and score >= 0, f"invalid {label} score")
        _require(isinstance(rank, int) and not isinstance(rank, bool) and rank >= 1,
                 f"invalid {label} rank")
        seen.add(model)
        ranks[model] = rank
        scores.append((model, float(score), rank))
    ordered = sorted(scores, key=lambda item: (item[1], item[0]))
    previous_score = None
    expected_rank = 0
    for position, (_, score, rank) in enumerate(ordered, 1):
        if previous_score is None or score != previous_score:
            expected_rank = position
        _require(rank == expected_rank, f"inconsistent {label} competition rank")
        previous_score = score
    return ranks


def assess_internal_ranking_stability(results: Mapping, *, validation_context: Mapping) -> dict:
    """Summarize common-fold rank perturbations, retaining D2-C1 evidence boundaries."""
    report = assess_internal_ranking_sensitivity(results, validation_context=validation_context)
    _require(isinstance(report, Mapping)
             and report.get("assessment_version") == "m19_d2_c1_ranking_sensitivity_v1"
             and report.get("scope") == "internal_expanding_window"
             and report.get("source_assessment_version") == "m19_d2_b2_fold_sensitivity_v1",
             "unsupported internal ranking sensitivity evidence")
    count, model_count = report.get("fold_count"), report.get("model_count")
    dates = report.get("fold_validation_dates")
    _require(isinstance(count, int) and not isinstance(count, bool) and count >= 1
             and isinstance(model_count, int) and not isinstance(model_count, bool) and model_count >= 1
             and isinstance(dates, list) and len(dates) == count
             and all(isinstance(d, list) and bool(d) for d in dates), "invalid fold calendar")
    baseline = report.get("baseline_ranking")
    _require(isinstance(baseline, list), "missing baseline ranking")
    models = {row.get("model") for row in baseline if isinstance(row, Mapping)}
    _require(len(models) == model_count and all(isinstance(m, str) and m.strip() for m in models),
             "invalid baseline model set")
    baseline_ranks = _validated_ranking(baseline, models=models, label="baseline")
    winners = sorted(m for m, rank in baseline_ranks.items() if rank == 1)
    _require(report.get("baseline_top_models") == winners, "baseline winner set mismatch")
    status = report.get("status")
    common = {
        "assessment_version": "m19_d2_c2_ranking_stability_v1",
        "scope": "internal_expanding_window",
        "source_assessment_version": report["assessment_version"],
        "dataset_id": report.get("dataset_id"),
        "dataset_fingerprint": report.get("dataset_fingerprint"),
        "fold_validation_dates": [list(d) for d in dates],
        "fold_count": count,
        "model_count": model_count,
        "method": "common_fold_lofo_ranking_stability",
        "baseline_top_models": winners,
    }
    if status != "available":
        expected = "unavailable_insufficient_folds" if count < 2 else "unavailable_insufficient_models" if model_count < 2 else None
        _require(status == expected and report.get("fold_omissions") == []
                 and report.get("any_rank_change") is None
                 and report.get("any_top_model_set_change") is None,
                 "inconsistent unavailable ranking sensitivity")
        return {**common, "status": status, "rank_change_count": None,
                "winner_set_change_count": None, "rank_stability_fraction": None,
                "winner_set_stability_fraction": None,
                "max_absolute_rank_displacement": None,
                "per_model_rank_change_count": None}
    _require(count >= 2 and model_count >= 2, "invalid available ranking sensitivity")
    omissions = report.get("fold_omissions")
    _require(isinstance(omissions, list) and len(omissions) == count,
             "incomplete common fold omissions")
    seen_folds: set[int] = set()
    rank_changes = 0
    winner_changes = 0
    maximum = 0
    per_model = {model: 0 for model in sorted(models)}
    for omission in omissions:
        _require(isinstance(omission, Mapping), "invalid fold omission")
        index = omission.get("fold_index")
        _require(isinstance(index, int) and not isinstance(index, bool)
                 and 0 <= index < count and index not in seen_folds,
                 "invalid or duplicate fold index")
        seen_folds.add(index)
        _require(omission.get("validation_dates") == dates[index],
                 "fold calendar mismatch")
        ranks = _validated_ranking(omission.get("ranking"), models=models, label="LOFO")
        fold_winners = sorted(m for m, rank in ranks.items() if rank == 1)
        _require(omission.get("top_models") == fold_winners, "LOFO winner set mismatch")
        changed = {model: ranks[model] - baseline_ranks[model]
                   for model in models if ranks[model] != baseline_ranks[model]}
        expected_changes = [
            {"model": model, "baseline_rank": baseline_ranks[model],
             "lofo_rank": ranks[model], "rank_delta": changed[model]}
            for model in sorted(changed)
        ]
        _require(omission.get("rank_changes") == expected_changes
                 and omission.get("rank_changed") is bool(changed)
                 and omission.get("top_model_set_changed") is (fold_winners != winners),
                 "inconsistent LOFO change evidence")
        rank_changes += bool(changed)
        winner_changes += fold_winners != winners
        for model, delta in changed.items():
            per_model[model] += 1
            maximum = max(maximum, abs(delta))
    _require(seen_folds == set(range(count)), "missing fold omission")
    _require(report.get("any_rank_change") is bool(rank_changes)
             and report.get("any_top_model_set_change") is bool(winner_changes),
             "inconsistent aggregate change evidence")
    return {**common, "status": "available", "rank_change_count": rank_changes,
            "winner_set_change_count": winner_changes,
            "rank_stability_fraction": (count - rank_changes) / count,
            "winner_set_stability_fraction": (count - winner_changes) / count,
            "max_absolute_rank_displacement": maximum,
            "per_model_rank_change_count": per_model}
