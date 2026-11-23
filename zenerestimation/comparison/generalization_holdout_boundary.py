
"""M19-D1-B2-B1: internal-to-protected-holdout authorization boundary.

Verifies internal selection evidence without granting protected-holdout
assessment, comparison, ranking, or disclosure permissions.

This module does not inspect protected observations, execute models,
or replace the independent M19-C2 authorization contracts.
"""

from __future__ import annotations

from collections.abc import Mapping

from zenerestimation.comparison.generalization_internal_evidence import (
    verify_internal_generalization_evidence,
)

VERSION = "m19_d1_b2_b1_holdout_boundary_v1"


def assess_generalization_holdout_boundary(
    *,
    selection_artifact: Mapping,
    results: Mapping,
    validation_context: Mapping,
) -> dict:
    """Reverify internal evidence and return a fail-closed boundary decision.

    Internal selection authorization is not protected-holdout authorization.
    Invalid internal evidence raises through the existing D1-B1 verifier.
    """

    evidence = verify_internal_generalization_evidence(
        selection_artifact=selection_artifact,
        results=results,
        validation_context=validation_context,
    )

    if (
        type(evidence) is not dict
        or evidence.get("status") != "INTERNAL_EVIDENCE_VERIFIED"
        or evidence.get("selection_replay_verified") is not True
        or evidence.get("holdout_assessed") is not False
        or evidence.get("generalization_gap") is not None
        or evidence.get("protected_holdout_comparison_permitted") is not False
        or evidence.get("protected_holdout_ranking_permitted") is not False
    ):
        raise ValueError("invalid internal evidence boundary contract")

    model = evidence.get("selected_model")
    digest = evidence.get("selection_decision_digest")

    if type(model) is not str or not model:
        raise ValueError("invalid verified selected model")

    if type(digest) is not str or not digest:
        raise ValueError("missing verified selection decision digest")

    return {
        "assessment_version": VERSION,
        "status": "AUTHORIZATION_REQUIRED",
        "reason": "PROTECTED_HOLDOUT_AUTHORIZATION_NOT_ESTABLISHED",
        "internal_evidence_verified": True,
        "selected_model": model,
        "selection_decision_digest": digest,
        "holdout_assessed": False,
        "generalization_gap": None,
        "protected_holdout_assessment_permitted": False,
        "protected_holdout_comparison_permitted": False,
        "protected_holdout_ranking_permitted": False,
        "protected_holdout_disclosure_permitted": False,
    }
