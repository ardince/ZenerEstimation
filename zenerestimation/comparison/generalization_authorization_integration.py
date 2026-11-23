
"""M19-D1-B2-B2: read-only reconciliation of authorization boundaries.

Recomputes internal evidence and independent benchmark authorization.
Never grants protected-holdout access, assessment, comparison, ranking,
or disclosure permissions.

A decision digest establishes consistency, not independent authenticity.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from zenerestimation.comparison.generalization_holdout_boundary import (
    assess_generalization_holdout_boundary,
)
from zenerestimation.comparison.benchmark_authorization_integration import (
    ASSESSMENT_VERSION as C2_VERSION,
    assess_independent_benchmark_authorization,
)
from zenerestimation.assessment.holdout_target_disclosure import _digest


VERSION = "m19_d1_b2_b2_authorization_reconciliation_v1"

_C2_ARGUMENTS = frozenset({
    "inputs",
    "scope_declaration",
    "authorization_evidence",
    "prospective_commitment",
    "authorization_registration",
    "protected_holdout_events",
    "trust_registry",
    "checkpoint",
    "signature_trust_registry",
    "chronology_trust_registry",
})

_C2_REQUIRED = frozenset({
    "inputs",
    "scope_declaration",
    "authorization_evidence",
    "prospective_commitment",
    "authorization_registration",
    "protected_holdout_events",
})


def reconcile_generalization_authorization(
    *,
    selection_artifact: Mapping,
    results: Mapping,
    validation_context: Mapping,
    benchmark_authorization_inputs: Mapping,
    supplied_authorization_decision: Mapping | None = None,
) -> dict:
    """Reverify both boundaries without promoting either into permission.

    Invalid internal evidence raises through D1-B1/B2-B1.
    Invalid or contradictory C2 decisions are rejected.
    """

    internal = assess_generalization_holdout_boundary(
        selection_artifact=selection_artifact,
        results=results,
        validation_context=validation_context,
    )

    if (
        type(internal) is not dict
        or internal.get("status") != "AUTHORIZATION_REQUIRED"
        or internal.get("internal_evidence_verified") is not True
        or internal.get("holdout_assessed") is not False
        or internal.get("generalization_gap") is not None
        or any(
            internal.get(field) is not False
            for field in (
                "protected_holdout_assessment_permitted",
                "protected_holdout_comparison_permitted",
                "protected_holdout_ranking_permitted",
                "protected_holdout_disclosure_permitted",
            )
        )
    ):
        raise ValueError("invalid internal holdout boundary decision")

    if type(benchmark_authorization_inputs) is not dict:
        raise TypeError("benchmark authorization inputs must be a dict")

    keys = set(benchmark_authorization_inputs)
    if not _C2_REQUIRED <= keys or not keys <= _C2_ARGUMENTS:
        raise ValueError("invalid benchmark authorization arguments")

    # The existing C2 verifier owns scope, cryptographic, chronology,
    # and upstream benchmark verification. Never duplicate those checks.
    c2 = assess_independent_benchmark_authorization(
        **benchmark_authorization_inputs
    )

    if type(c2) is not dict:
        raise ValueError("invalid C2 authorization decision")

    if c2.get("assessment_version") != C2_VERSION:
        raise ValueError("unsupported C2 authorization decision version")

    if c2.get("status") not in (
        "AUTHORIZATION_REQUIRED",
        "AUTHORIZATION_REJECTED",
    ):
        raise ValueError("unexpected C2 authorization status")

    digest = c2.get("decision_digest")
    payload = {k: v for k, v in c2.items() if k != "decision_digest"}

    if type(digest) is not str or digest != _digest(payload):
        raise ValueError("C2 decision digest mismatch")

    if (
        c2.get("authorization") != "not_established"
        or c2.get("prospective_authorization_verified") is not False
        or c2.get("independent_time_provenance_verified") is not False
        or c2.get("access_completeness_verified") is not False
        or c2.get("comparison_permitted") is not False
        or c2.get("ranking_permitted") is not False
    ):
        raise ValueError("C2 decision contradicts fail-closed contract")

    if supplied_authorization_decision is not None:
        if (
            type(supplied_authorization_decision) is not dict
            or supplied_authorization_decision != c2
        ):
            return {
                "assessment_version": VERSION,
                "status": "AUTHORIZATION_REJECTED",
                "reasons": ["SUPPLIED_AUTHORIZATION_DECISION_MISMATCH"],
                "internal_evidence_verified": True,
                "selected_model": internal["selected_model"],
                "selection_decision_digest": internal[
                    "selection_decision_digest"
                ],
                "holdout_assessed": False,
                "generalization_gap": None,
                "protected_holdout_assessment_permitted": False,
                "protected_holdout_comparison_permitted": False,
                "protected_holdout_ranking_permitted": False,
                "protected_holdout_disclosure_permitted": False,
            }

    rejected = c2["status"] == "AUTHORIZATION_REJECTED"

    return deepcopy({
        "assessment_version": VERSION,
        "status": (
            "AUTHORIZATION_REJECTED"
            if rejected else "AUTHORIZATION_REQUIRED"
        ),
        "reasons": (
            ["INDEPENDENT_BENCHMARK_AUTHORIZATION_REJECTED"]
            if rejected
            else ["PROTECTED_HOLDOUT_AUTHORIZATION_NOT_ESTABLISHED"]
        ),
        "internal_evidence_verified": True,
        "selected_model": internal["selected_model"],
        "selection_decision_digest": internal[
            "selection_decision_digest"
        ],
        "benchmark_scope_digest": c2.get("scope_digest"),
        "benchmark_authorization_decision_digest": digest,
        "benchmark_authorization_status": c2["status"],
        "holdout_assessed": False,
        "generalization_gap": None,
        "protected_holdout_assessment_permitted": False,
        "protected_holdout_comparison_permitted": False,
        "protected_holdout_ranking_permitted": False,
        "protected_holdout_disclosure_permitted": False,
    })
