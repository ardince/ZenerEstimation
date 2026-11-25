"""M19-D4-B2: read-only consolidation of internal selection and eligibility evidence.

This is a diagnostic report, not an evidence authenticator, certification,
model-selection authorization, or protected-holdout evaluation.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

from .decision import evaluate_scientific_eligibility
from .eligibility import AssessmentDeclaration, TrustedVerificationEvidence
from .selection_consistency import assess_internal_selection_consistency

ASSESSMENT_VERSION = "m19_d4_b2_consolidated_assessment_v1"
SOURCE_VERSION = "m19_d3_b2_selection_consistency_v1"
SCOPE = "internal_selection_assessment"
_ALLOWED_STATUSES = {
    "available", "unavailable_insufficient_folds", "unavailable_insufficient_models"
}


def _detached_json(value: object) -> object:
    """Detach a report, reject non-JSON values and nonfinite numeric values."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def build_consolidated_assessment(
    results: Mapping,
    *,
    validation_context: Mapping,
    declaration: AssessmentDeclaration | None = None,
    verification_evidence: TrustedVerificationEvidence | None = None,
) -> dict:
    """Recompute D3-B2 from original inputs; keep eligibility unbound and conditional.

    The caller must supply trusted validation context independently of ``results``.
    Eligibility evidence, when supplied, is structurally checked, not authenticated.
    This function does not read protected holdout targets or authorize selection.
    """
    if not isinstance(results, Mapping) or not isinstance(validation_context, Mapping):
        raise TypeError("results and validation_context must be mappings")
    if (declaration is None) != (verification_evidence is None):
        raise ValueError("declaration and verification_evidence must be supplied together")
    if declaration is not None and not isinstance(declaration, AssessmentDeclaration):
        raise TypeError("declaration must be AssessmentDeclaration")
    if verification_evidence is not None and not isinstance(verification_evidence, TrustedVerificationEvidence):
        raise TypeError("verification_evidence must be TrustedVerificationEvidence")

    internal = assess_internal_selection_consistency(
        results, validation_context=validation_context
    )
    if not isinstance(internal, Mapping):
        raise ValueError("invalid D3-B2 assessment")
    if internal.get("assessment_version") != SOURCE_VERSION:
        raise ValueError("unsupported D3-B2 assessment version")
    if internal.get("scope") != "internal_expanding_window":
        raise ValueError("invalid D3-B2 scope")
    if internal.get("status") not in _ALLOWED_STATUSES:
        raise ValueError("invalid D3-B2 status")
    if internal.get("selection_authorized") is not False:
        raise ValueError("D3-B2 must not authorize selection")
    if internal.get("consistency_relationship_verified") is not True:
        raise ValueError("D3-B2 consistency relationship not verified")

    internal_snapshot = _detached_json(dict(internal))
    eligibility_snapshot = None
    eligibility_dataset_id = None
    if declaration is not None:
        # This policy checks structural evidence binding, not authenticity.
        decision = evaluate_scientific_eligibility(declaration, verification_evidence)
        if decision.prospective_independence_certified is not False:
            raise ValueError("eligibility policy must not issue certification")
        eligibility_snapshot = _detached_json(decision.to_dict())
        eligibility_dataset_id = declaration.dataset_id

    internal_dataset_id = internal_snapshot.get("dataset_id")
    return {
        "assessment_version": ASSESSMENT_VERSION,
        "scope": SCOPE,
        "status": internal_snapshot["status"],
        "internal_assessment": internal_snapshot,
        "scientific_eligibility": eligibility_snapshot,
        "evidence_relationship": {
            "internal_dataset_id": internal_dataset_id,
            "eligibility_dataset_id": eligibility_dataset_id,
            "dataset_id_match": (
                internal_dataset_id == eligibility_dataset_id
                if eligibility_dataset_id is not None else None
            ),
            "cross_track_identity_verified": False,
        },
        "scientific_boundaries": {
            "selection_authorized": False,
            "prospective_certification_issued": False,
            "protected_holdout_accessed": False,
            "optimization_executed": False,
        },
    }
