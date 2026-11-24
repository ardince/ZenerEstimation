
"""M19-D3-B1: diagnostic-only internal model-selection robustness.

Retrospective interpretation of M18-verified internal rankings under
common-fold leave-one-out reaggregation.

This module does not select, authorize, prepare, execute, or deploy models.
It does not access protected-holdout observations or certify prospective
model performance.
"""

from __future__ import annotations

from collections.abc import Mapping

from math import isfinite
from numbers import Real

from zenerestimation.assessment.ranking_sensitivity import (
    assess_internal_ranking_sensitivity,
)


ASSESSMENT_VERSION = "m19_d3_b1_selection_robustness_v1"
SOURCE_VERSION = "m19_d2_c1_ranking_sensitivity_v1"
SCOPE = "internal_expanding_window"

UNAVAILABLE_STATUSES = {
    "unavailable_insufficient_folds",
    "unavailable_insufficient_models",
}


def _verify_competition_ranking(
    ranking: object,
    *,
    expected_models: set[str],
) -> list[str]:
    """Verify ascending RMSE ordering and exact competition ranks."""
    _require(
        isinstance(ranking, list)
        and len(ranking) == len(expected_models),
        "invalid ranking length",
    )

    models = []
    previous_score = None
    current_rank = 0
    winners = []

    for position, row in enumerate(ranking, start=1):
        _require(isinstance(row, Mapping), "invalid ranking row")

        model = row.get("model")
        score = row.get("score")
        rank = row.get("rank")

        _require(
            isinstance(model, str)
            and model in expected_models
            and model not in models,
            "invalid ranking model identity",
        )
        _require(
            isinstance(score, Real)
            and not isinstance(score, bool)
            and isfinite(score)
            and score >= 0,
            "invalid ranking RMSE",
        )
        _require(
            type(rank) is int,
            "invalid competition rank",
        )

        if previous_score is not None:
            _require(
                (score, model) >= (previous_score, models[-1]),
                "ranking not sorted by score and model",
            )

        if previous_score is None or score != previous_score:
            current_rank = position

        _require(
            rank == current_rank,
            "inconsistent competition rank",
        )

        if rank == 1:
            winners.append(model)

        models.append(model)
        previous_score = score

    _require(
        set(models) == expected_models,
        "incomplete ranking model set",
    )

    return winners


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _model_set(value: object, *, label: str) -> list[str]:
    _require(
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(model, str) and bool(model.strip())
            for model in value
        )
        and len(value) == len(set(value)),
        f"invalid {label}",
    )
    _require(value == sorted(value), f"unordered {label}")
    return list(value)


def _unique_recommendation(winners: list[str]) -> str | None:
    return winners[0] if len(winners) == 1 else None


def assess_internal_selection_robustness(
    results: Mapping,
    *,
    validation_context: Mapping,
) -> dict:
    """Describe robustness of a hypothetical unique internal recommendation.

    The original results and independently trusted validation context are
    reverified through M19-D2-C1. Serialized assessment reports are not
    accepted as authoritative input.

    Exact rank-1 ties never produce an implicit recommendation.

    A consistency fraction is defined only when the baseline has a unique
    winner and common-fold LOFO evidence is available.
    """
    report = assess_internal_ranking_sensitivity(
        results,
        validation_context=validation_context,
    )

    _require(
        isinstance(report, Mapping)
        and report.get("assessment_version") == SOURCE_VERSION
        and report.get("scope") == SCOPE
        and report.get("source_assessment_version")
        == "m19_d2_b2_fold_sensitivity_v1",
        "unsupported internal ranking sensitivity evidence",
    )

    count = report.get("fold_count")
    model_count = report.get("model_count")
    dates = report.get("fold_validation_dates")

    _require(
        type(count) is int
        and count >= 1
        and type(model_count) is int
        and model_count >= 1
        and isinstance(dates, list)
        and len(dates) == count
        and all(isinstance(item, list) and item for item in dates),
        "invalid internal fold calendar or model count",
    )

    baseline = report.get("baseline_ranking")
    _require(
        isinstance(baseline, list)
        and len(baseline) == model_count,
        "invalid baseline ranking",
    )

    baseline_models = [
        row.get("model")
        for row in baseline
        if isinstance(row, Mapping)
    ]
    _require(
        len(baseline_models) == model_count
        and all(
            isinstance(model, str) and bool(model.strip())
            for model in baseline_models
        )
        and len(set(baseline_models)) == model_count,
        "invalid baseline model identities",
    )

    winners = _model_set(
        report.get("baseline_top_models"),
        label="baseline winner set",
    )
    _require(
        set(winners).issubset(set(baseline_models)),
        "baseline winner outside verified model set",
    )


    verified_baseline_winners = _verify_competition_ranking(
        baseline,
        expected_models=set(baseline_models),
    )

    _require(
        winners == verified_baseline_winners,
        "inconsistent baseline winner set",
    )


    baseline_recommended = _unique_recommendation(winners)

    common = {
        "assessment_version": ASSESSMENT_VERSION,
        "source_assessment_version": SOURCE_VERSION,
        "scope": SCOPE,
        "method": "common_fold_lofo_selection_diagnostics",
        "dataset_id": report.get("dataset_id"),
        "dataset_fingerprint": report.get("dataset_fingerprint"),
        "fold_validation_dates": [list(item) for item in dates],
        "fold_count": count,
        "model_count": model_count,
        "baseline_top_models": winners,
        "baseline_recommended_model": baseline_recommended,
        "selection_authorized": False,
    }

    status = report.get("status")

    if status in UNAVAILABLE_STATUSES:
        expected = (
            "unavailable_insufficient_folds"
            if count < 2
            else "unavailable_insufficient_models"
            if model_count < 2
            else None
        )
        _require(
            status == expected
            and report.get("fold_omissions") == []
            and report.get("any_rank_change") is None
            and report.get("any_top_model_set_change") is None,
            "inconsistent unavailable sensitivity evidence",
        )

        return {
            **common,
            "status": status,
            "fold_omissions": [],
            "recommendation_consistency_count": None,
            "recommendation_consistency_fraction": None,
            "ambiguous_omission_count": None,
            "alternative_winner_models": None,
        }

    _require(
        status == "available"
        and count >= 2
        and model_count >= 2,
        "invalid available sensitivity status",
    )

    omissions = report.get("fold_omissions")
    _require(
        isinstance(omissions, list)
        and len(omissions) == count,
        "incomplete common-fold sensitivity evidence",
    )

    diagnostic_omissions: list[dict] = []
    seen_folds: set[int] = set()
    consistent_count = 0
    ambiguous_count = 0
    alternatives: set[str] = set()

    for omission in omissions:
        _require(
            isinstance(omission, Mapping),
            "invalid fold omission",
        )

        index = omission.get("fold_index")
        _require(
            type(index) is int
            and 0 <= index < count
            and index not in seen_folds,
            "invalid or duplicate fold index",
        )
        seen_folds.add(index)

        _require(
            omission.get("validation_dates") == dates[index],
            "fold calendar mismatch",
        )

        fold_winners = _model_set(
            omission.get("top_models"),
            label="omission winner set",
        )
        _require(
            set(fold_winners).issubset(set(baseline_models)),
            "omission winner outside verified model set",
        )



        ranking = omission.get("ranking")

        verified_omission_winners = _verify_competition_ranking(
            ranking,
            expected_models=set(baseline_models),
        )

        _require(
            fold_winners == verified_omission_winners,
            "inconsistent omission winner set",
        )

        recommended = _unique_recommendation(fold_winners)
        ambiguous = recommended is None

        if ambiguous:
            ambiguous_count += 1

        consistent = (
            baseline_recommended is not None
            and recommended == baseline_recommended
        )

        if consistent:
            consistent_count += 1

        if (
            recommended is not None
            and recommended != baseline_recommended
        ):
            alternatives.add(recommended)

        diagnostic_omissions.append({
            "fold_index": index,
            "validation_dates": list(dates[index]),
            "top_models": fold_winners,
            "recommended_model": recommended,
            "ambiguous": ambiguous,
            "recommendation_consistent": (
                consistent
                if baseline_recommended is not None
                else None
            ),
        })

    _require(
        seen_folds == set(range(count)),
        "missing common-fold omission",
    )

    diagnostic_omissions.sort(key=lambda row: row["fold_index"])

    has_unique_baseline = baseline_recommended is not None

    return {
        **common,
        "status": "available",
        "fold_omissions": diagnostic_omissions,
        "recommendation_consistency_count": (
            consistent_count if has_unique_baseline else None
        ),
        "recommendation_consistency_fraction": (
            consistent_count / count if has_unique_baseline else None
        ),
        "ambiguous_omission_count": ambiguous_count,
        "alternative_winner_models": sorted(alternatives),
    }



