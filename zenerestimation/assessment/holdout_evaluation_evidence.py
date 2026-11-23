"""M19-B3-B4: deterministic, independently reconstructed holdout evidence.

Consumes only approved B1/B2/B3 snapshots; never executes models or accesses
holdout partitions. Digests establish internal consistency, not authentication.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import fsum, isclose, isfinite, sqrt
import json

from zenerestimation.assessment.eligibility import AssessmentDeclaration, validate_assessment_declaration
from zenerestimation.assessment.holdout_target_disclosure import _canonical, _digest, _check_eligibility
from zenerestimation.assessment.holdout_error_scoring import score_disclosed_holdout, SCORING_VERSION

EVIDENCE_VERSION = "m19_b3_b4_deterministic_evaluation_evidence_v1"
_SCORING_KEYS = frozenset({
    "scoring_version", "status", "assessment_id", "declaration_digest", "dataset_id",
    "eligibility_digest", "disclosure_digest", "execution_artifact_digest",
    "forecast_dates", "forecast_horizon", "scored_count", "error_convention",
    "metrics", "denominators", "mape_zero_policy", "mape_zero_actual_excluded_dates",
    "by_date", "scientific_independence", "scoring_digest",
})


def _detach(snapshot: Mapping, name: str) -> dict:
    if type(snapshot) is not dict:
        raise TypeError(f"{name} must be a dict")
    try:
        detached = json.loads(_canonical(snapshot))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} is not canonical JSON") from exc
    if detached != snapshot:
        raise ValueError(f"{name} contains noncanonical values")
    return detached


def _check_scoring_digest(snapshot: Mapping) -> dict:
    scored = _detach(snapshot, "B3 scoring")
    if set(scored) != _SCORING_KEYS:
        raise ValueError("B3 scoring schema mismatch")
    digest = scored.pop("scoring_digest")
    if type(digest) is not str or digest != _digest(scored):
        raise ValueError("B3 scoring digest mismatch")
    scored["scoring_digest"] = digest
    return scored


def build_holdout_evaluation_evidence(
    *, declaration: AssessmentDeclaration, eligibility_snapshot: Mapping,
    disclosure_snapshot: Mapping, scoring_snapshot: Mapping,
) -> dict:
    """Recompute B3 from B1/B2, compare every field, and seal a detached record.

    Caller must enforce access controls and trusted origins for B1/B2 inputs.
    This function does not establish historical independence or authenticity.
    """
    validate_assessment_declaration(declaration)
    _check_eligibility(eligibility_snapshot, declaration)
    scored = _check_scoring_digest(scoring_snapshot)
    expected = score_disclosed_holdout(
        declaration=declaration, eligibility_snapshot=eligibility_snapshot,
        disclosure_snapshot=disclosure_snapshot,
    )
    # Exact canonical equality is intentional: includes all per-date values,
    # MAPE exclusions, denominators, provenance digests and aggregate metrics.
    if _canonical(scored) != _canonical(expected):
        raise ValueError("B3 scoring differs from independently reconstructed metrics or evidence")
    eligibility = _detach(eligibility_snapshot, "B1 eligibility")
    disclosure = _detach(disclosure_snapshot, "B2 disclosure")
    payload = {
        "evidence_version": EVIDENCE_VERSION,
        "status": "HOLDOUT_EVALUATION_EVIDENCE_VERIFIED",
        "assessment_id": declaration.assessment_id,
        "declaration_digest": declaration.digest,
        "dataset_id": declaration.dataset_id,
        "development_cutoff": declaration.development_cutoff,
        "selected_model": eligibility["selected_model"],
        "decision_digest": eligibility["decision_digest"],
        "plan_digest": eligibility["plan_digest"],
        "execution_artifact_digest": eligibility["execution_artifact_digest"],
        "training_fingerprint": eligibility["training_fingerprint"],
        "eligibility_digest": eligibility["eligibility_digest"],
        "disclosure_digest": disclosure["disclosure_digest"],
        "scoring_digest": scored["scoring_digest"],
        "forecast_dates": list(declaration.forecast_dates),
        "forecast_horizon": len(declaration.forecast_dates),
        "error_convention": "forecast_minus_observed",
        "metrics": scored["metrics"],
        "denominators": scored["denominators"],
        "mape_zero_policy": scored["mape_zero_policy"],
        "mape_zero_actual_excluded_dates": scored["mape_zero_actual_excluded_dates"],
        "by_date": scored["by_date"],
        "scientific_independence": "not_established",
        "authentication": "not_established",
    }
    result = json.loads(_canonical(payload))
    result["evaluation_evidence_digest"] = _digest(result)
    return result


def verify_holdout_evaluation_evidence(
    evidence: Mapping, *, declaration: AssessmentDeclaration,
    eligibility_snapshot: Mapping, disclosure_snapshot: Mapping,
    scoring_snapshot: Mapping,
) -> bool:
    """Fail closed: verify evidence against independently rebuilt full chain."""
    if type(evidence) is not dict:
        return False
    try:
        expected = build_holdout_evaluation_evidence(
            declaration=declaration, eligibility_snapshot=eligibility_snapshot,
            disclosure_snapshot=disclosure_snapshot, scoring_snapshot=scoring_snapshot,
        )
        return _canonical(evidence) == _canonical(expected)
    except (TypeError, ValueError, OverflowError, KeyError):
        return False
