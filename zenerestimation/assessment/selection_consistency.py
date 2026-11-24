"""M19-D3-B2: diagnostic-only cross-contract internal selection consistency.

Recomputes trusted internal evidence through B1 and D2-C2; never authorizes
selection, retraining, or protected-holdout use.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real

from zenerestimation.assessment.selection_robustness import assess_internal_selection_robustness
from zenerestimation.assessment.ranking_stability import assess_internal_ranking_stability

ASSESSMENT_VERSION = "m19_d3_b2_selection_consistency_v1"
SCOPE = "internal_expanding_window"
B1_VERSION = "m19_d3_b1_selection_robustness_v1"
C2_VERSION = "m19_d2_c2_ranking_stability_v1"
C1_VERSION = "m19_d2_c1_ranking_sensitivity_v1"
UNAVAILABLE = {"unavailable_insufficient_folds", "unavailable_insufficient_models"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"invalid {label}")
    return value


def _fraction(value: object, *, label: str) -> float:
    _require(isinstance(value, Real) and not isinstance(value, bool), f"invalid {label}")
    value = float(value)
    _require(isfinite(value) and 0 <= value <= 1, f"invalid {label}")
    return value


def _models(value: object, *, label: str) -> list[str]:
    _require(isinstance(value, list) and bool(value), f"invalid {label}")
    _require(all(isinstance(x, str) and bool(x.strip()) for x in value), f"invalid {label}")
    _require(value == sorted(set(value)), f"invalid {label} ordering or duplicates")
    return value


def assess_internal_selection_consistency(
    results: Mapping, *, validation_context: Mapping
) -> dict:
    """Verify consistency of independent internal recommendation/rank diagnostics."""
    robustness = assess_internal_selection_robustness(
        results, validation_context=validation_context
    )
    stability = assess_internal_ranking_stability(
        results, validation_context=validation_context
    )
    _require(isinstance(robustness, Mapping) and isinstance(stability, Mapping),
             "invalid upstream assessment")
    _require(robustness.get("assessment_version") == B1_VERSION
             and robustness.get("source_assessment_version") == C1_VERSION
             and stability.get("assessment_version") == C2_VERSION
             and stability.get("source_assessment_version") == C1_VERSION,
             "unsupported upstream assessment version")
    _require(robustness.get("scope") == stability.get("scope") == SCOPE,
             "inconsistent internal assessment scope")
    _require(robustness.get("selection_authorized") is False,
             "upstream selection authorization forbidden")

    for key in ("dataset_id", "dataset_fingerprint", "fold_count", "model_count",
                "fold_validation_dates", "baseline_top_models", "status"):
        _require(robustness.get(key) == stability.get(key), f"upstream {key} mismatch")
    dataset_id = robustness.get("dataset_id")
    fingerprint = robustness.get("dataset_fingerprint")
    _require(isinstance(dataset_id, str) and bool(dataset_id.strip()), "invalid dataset id")
    _require(isinstance(fingerprint, str) and bool(fingerprint.strip()),
             "invalid dataset fingerprint")
    count = _integer(robustness.get("fold_count"), label="fold count", minimum=1)
    model_count = _integer(robustness.get("model_count"), label="model count", minimum=1)
    dates = robustness.get("fold_validation_dates")
    _require(isinstance(dates, list) and len(dates) == count
             and all(isinstance(d, list) and bool(d)
                     and all(isinstance(x, str) and bool(x) for x in d) for d in dates),
             "invalid common fold calendar")
    winners = _models(robustness.get("baseline_top_models"), label="baseline winners")
    _require(len(winners) <= model_count, "invalid baseline winner count")
    recommended = robustness.get("baseline_recommended_model")
    _require(recommended == (winners[0] if len(winners) == 1 else None),
             "inconsistent baseline recommendation")
    status = robustness.get("status")
    expected_unavailable = ("unavailable_insufficient_folds" if count < 2 else
                            "unavailable_insufficient_models" if model_count < 2 else None)
    _require(status == (expected_unavailable or "available"),
             "inconsistent availability status")

    common = {
        "assessment_version": ASSESSMENT_VERSION,
        "scope": SCOPE,
        "method": "cross_contract_internal_selection_consistency",
        "source_assessment_versions": [B1_VERSION, C2_VERSION],
        "dataset_id": dataset_id,
        "dataset_fingerprint": fingerprint,
        "fold_validation_dates": [list(d) for d in dates],
        "fold_count": count,
        "model_count": model_count,
        "baseline_top_models": list(winners),
        "baseline_recommended_model": recommended,
        "status": status,
        "selection_authorized": False,
    }
    if status in UNAVAILABLE:
        _require(robustness.get("fold_omissions") == []
                 and all(robustness.get(k) is None for k in (
                     "recommendation_consistency_count", "recommendation_consistency_fraction",
                     "ambiguous_omission_count", "alternative_winner_models")),
                 "inconsistent unavailable robustness")
        _require(all(stability.get(k) is None for k in (
                     "rank_change_count", "winner_set_change_count",
                     "rank_stability_fraction", "winner_set_stability_fraction",
                     "max_absolute_rank_displacement", "per_model_rank_change_count")),
                 "inconsistent unavailable stability")
        return {**common, "recommendation_consistency_fraction": None,
                "winner_set_stability_fraction": None, "rank_stability_fraction": None,
                "recommendation_winner_gap": None, "winner_rank_gap": None,
                "consistency_relationship_verified": True}

    omissions = robustness.get("fold_omissions")
    _require(isinstance(omissions, list) and len(omissions) == count,
             "incomplete robustness fold omissions")
    seen: set[int] = set()
    consistent_count = 0
    ambiguous_count = 0
    alternative_winners: set[str] = set()
    for omission in omissions:
        _require(isinstance(omission, Mapping), "invalid robustness omission")
        index = _integer(omission.get("fold_index"), label="fold index")
        _require(index < count and index not in seen, "duplicate or invalid fold index")
        seen.add(index)
        _require(omission.get("validation_dates") == dates[index], "omission calendar mismatch")
        top = _models(omission.get("top_models"), label="omission winners")
        _require(len(top) <= model_count, "invalid omission winner count")
        unique = top[0] if len(top) == 1 else None
        _require(omission.get("recommended_model") == unique
                 and omission.get("ambiguous") is (unique is None),
                 "inconsistent omission recommendation")
        expected_consistent = (unique == recommended) if recommended is not None else None
        _require(omission.get("recommendation_consistent") is expected_consistent,
                 "inconsistent omission consistency flag")
        consistent_count += expected_consistent is True
        ambiguous_count += unique is None
        if unique is not None and unique != recommended:
            alternative_winners.add(unique)
    _require(seen == set(range(count)), "missing fold omission")
    _require(robustness.get("ambiguous_omission_count") == ambiguous_count
             and robustness.get("alternative_winner_models") == sorted(alternative_winners),
             "inconsistent robustness aggregates")
    rc = robustness.get("recommendation_consistency_count")
    rf = robustness.get("recommendation_consistency_fraction")
    if recommended is None:
        _require(rc is None and rf is None, "baseline tie cannot imply consistency")
    else:
        _require(_integer(rc, label="recommendation count") == consistent_count
                 and _fraction(rf, label="recommendation fraction") == consistent_count / count,
                 "inconsistent recommendation aggregates")

    rank_changes = _integer(stability.get("rank_change_count"), label="rank changes")
    winner_changes = _integer(stability.get("winner_set_change_count"), label="winner changes")
    _require(rank_changes <= count and winner_changes <= rank_changes,
             "inconsistent rank/winner change counts")
    rank_fraction = _fraction(stability.get("rank_stability_fraction"), label="rank fraction")
    winner_fraction = _fraction(stability.get("winner_set_stability_fraction"), label="winner fraction")
    _require(rank_fraction == (count - rank_changes) / count
             and winner_fraction == (count - winner_changes) / count,
             "inconsistent stability fractions")
    displacement = _integer(stability.get("max_absolute_rank_displacement"),
                            label="maximum rank displacement")
    per_model = stability.get("per_model_rank_change_count")
    _require(isinstance(per_model, Mapping) and len(per_model) == model_count
             and all(isinstance(k, str) and bool(k.strip())
                     and type(v) is int and 0 <= v <= count for k, v in per_model.items()),
             "invalid per-model rank change counts")
    _require((rank_changes == 0) == (displacement == 0)
             and (rank_changes == 0) == all(v == 0 for v in per_model.values()),
             "inconsistent rank displacement aggregates")
    if recommended is not None:
        _require(consistent_count == count - winner_changes,
                 "recommendation/winner-set mismatch")
        _require(rf == winner_fraction, "recommendation/winner stability mismatch")
    _require(winner_fraction >= rank_fraction, "winner-set stability below rank stability")

    return {**common, "recommendation_consistency_fraction": rf,
            "winner_set_stability_fraction": winner_fraction,
            "rank_stability_fraction": rank_fraction,
            "recommendation_winner_gap": (rf - winner_fraction) if rf is not None else None,
            "winner_rank_gap": winner_fraction - rank_fraction,
            "consistency_relationship_verified": True}
