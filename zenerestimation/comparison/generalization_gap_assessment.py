"""M19-D1-C1: read-only, fail-closed generalization-gap availability assessment.

This facade does not score, read, or disclose protected holdout observations.
It cannot establish scientific independence or authorize a numerical gap.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from zenerestimation.comparison.generalization_authorization_integration import (
    reconcile_generalization_authorization,
)

VERSION = "m19_d1_c1_gap_assessment_v1"
_DENIED_FIELDS = (
    "protected_holdout_assessment_permitted",
    "protected_holdout_comparison_permitted",
    "protected_holdout_ranking_permitted",
    "protected_holdout_disclosure_permitted",
)


def assess_generalization_gap(
    *,
    selection_artifact: Mapping,
    results: Mapping,
    validation_context: Mapping,
    benchmark_authorization_inputs: Mapping,
    supplied_authorization_decision: Mapping | None = None,
) -> dict:
    """Reconcile existing evidence without granting access or computing a gap.

    Raises for invalid internal evidence or unexpected upstream contracts.
    An upstream rejected authorization remains rejected. Neither a benchmark
    participant's name nor a C2 scope digest proves selected-model binding.
    """
    decision = reconcile_generalization_authorization(
        selection_artifact=selection_artifact,
        results=results,
        validation_context=validation_context,
        benchmark_authorization_inputs=benchmark_authorization_inputs,
        supplied_authorization_decision=supplied_authorization_decision,
    )
    if type(decision) is not dict:
        raise ValueError("invalid D1-B2-B2 reconciliation result")
    status = decision.get("status")
    if status not in ("AUTHORIZATION_REQUIRED", "AUTHORIZATION_REJECTED"):
        raise ValueError("unexpected D1-B2-B2 authorization status")
    if (
        decision.get("internal_evidence_verified") is not True
        or decision.get("holdout_assessed") is not False
        or decision.get("generalization_gap") is not None
        or any(decision.get(field) is not False for field in _DENIED_FIELDS)
    ):
        raise ValueError("D1-B2-B2 decision contradicts fail-closed contract")
    if (
        type(decision.get("selected_model")) is not str
        or not decision["selected_model"]
        or type(decision.get("selection_decision_digest")) is not str
        or not decision["selection_decision_digest"]
    ):
        raise ValueError("missing verified internal selection identity")
    reasons = decision.get("reasons")
    if (
        type(reasons) is not list
        or not reasons
        or any(type(reason) is not str or not reason for reason in reasons)
    ):
        raise ValueError("invalid reconciliation reasons")

    # The upstream mismatch-rejection record intentionally omits C2 digests.
    # Preserve that shape; never invent benchmark linkage or comparability.
    return deepcopy({
        "assessment_version": VERSION,
        "status": status,
        "reasons": reasons,
        "internal_evidence_verified": True,
        "selected_model": decision["selected_model"],
        "selection_decision_digest": decision["selection_decision_digest"],
        "benchmark_scope_digest": decision.get("benchmark_scope_digest"),
        "benchmark_authorization_decision_digest": decision.get(
            "benchmark_authorization_decision_digest"
        ),
        "benchmark_authorization_status": decision.get(
            "benchmark_authorization_status"
        ),
        "comparability_verified": False,
        "holdout_assessed": False,
        "generalization_gap": None,
        "generalization_gap_metric": None,
        **{field: False for field in _DENIED_FIELDS},
    })
