
"""M18-C1: controlled selection from verified internal-validation rankings.

Selection is explicit opt-in. Scientific ranking, provenance checks, and
accuracy-profile verification remain delegated to the M18-B4 ranking engine.
External holdout observations are never used.
"""
from __future__ import annotations

from collections.abc import Mapping

from zenerestimation.optimization.internal_ranking import rank_internal_models


SELECTION_VERSION = "m18_internal_selection_v1"
RANKING_VERSION = "m18_internal_ranking_v1"


def select_internal_model(
    results: Mapping,
    *,
    validation_context: Mapping,
    authorized: bool = False,
) -> dict:
    """Return an auditable selection decision without modifying the inputs.

    A unique rank-1 model can be selected only with explicit authorization.
    Exact rank-1 ties remain unresolved, with no manual override. All ranking
    and scientific-evidence validation is performed by ``rank_internal_models``.
    """
    if type(authorized) is not bool:
        raise TypeError("authorized must be a bool")

    report = rank_internal_models(
        results,
        validation_context=validation_context,
    )
    if report.get("ranking_version") != RANKING_VERSION:
        raise ValueError("unsupported internal ranking version")
    if report.get("selected_model") is not None:
        raise ValueError("ranking report must not contain a selected model")

    rows = report["ranking"]
    if not rows:
        raise ValueError("verified ranking must contain at least one model")

    winners = [row for row in rows if row["rank"] == 1]
    if not winners:
        raise ValueError("verified ranking must contain a rank-1 model")

    if len(winners) > 1:
        status = "TIE_UNRESOLVED"
        recommended_model = None
        selection_score = None
        selected_model = None
    else:
        winner = winners[0]
        recommended_model = winner["model"]
        selection_score = winner["score"]
        selected_model = recommended_model if authorized else None
        status = "SELECTED" if authorized else "PENDING_AUTHORIZATION"

    return {
        "selection_version": SELECTION_VERSION,
        "status": status,
        "selected_model": selected_model,
        "recommended_model": recommended_model,
        "selection_metric": "rmse",
        "selection_score": selection_score,
        "authorized": authorized,
        "ranking_version": RANKING_VERSION,
        "ranking": [dict(row) for row in rows],
    }
