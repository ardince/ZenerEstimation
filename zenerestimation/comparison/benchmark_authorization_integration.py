"""M19-C2-B2-C1: deterministic fail-closed integration of accepted evidence checks.

This module does not grant protected-holdout comparison or ranking rights.
It does not prove external chronology or completeness of holdout-access logs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from zenerestimation.assessment.holdout_target_disclosure import _digest
from zenerestimation.comparison.benchmark_authorization import assess_benchmark_scope
from zenerestimation.comparison.benchmark_authorization_evidence import (
    assess_authorization_evidence_structure,
)
from zenerestimation.comparison.benchmark_authorization_crypto import (
    verify_authorization_signature,
)
from zenerestimation.comparison.benchmark_authorization_prospective import (
    assess_prospective_commitment_binding,
)
from zenerestimation.comparison.benchmark_authorization_chronology import (
    verify_trusted_authorization_chronology,
)
from zenerestimation.comparison.benchmark_comparability import VerifiedBenchmarkInput

ASSESSMENT_VERSION = "m19_c2b2c1_integrated_authorization_v1"
_STAGE_NAMES = (
    "benchmark_scope", "evidence_structure", "signature_verification",
    "prospective_binding", "signed_chronology",
)


def _decision(*, status: str, reasons: list[str], stages: dict,
              scope_digest: str | None, commitment_digest: str | None) -> dict:
    """Create a stable, serializable decision without echoing untrusted records."""
    result = {
        "assessment_version": ASSESSMENT_VERSION,
        "status": status,
        "reasons": reasons,
        "scope_digest": scope_digest,
        "commitment_digest": commitment_digest,
        "stages": stages,
        "scientific_alignment_verified": stages.get("benchmark_scope", {}).get(
            "scientific_alignment_status") == "SCIENTIFICALLY_ALIGNED" and (
                stages.get("benchmark_scope", {}).get("status") == "AUTHORIZATION_REQUIRED"
            ),
        "scope_binding_verified": stages.get("benchmark_scope", {}).get("status") == "AUTHORIZATION_REQUIRED",
        "signature_verified": stages.get("signature_verification", {}).get("signature_verified") is True,
        "issuer_trusted": stages.get("signature_verification", {}).get("issuer_trusted") is True,
        "commitment_binding_verified": stages.get("prospective_binding", {}).get("commitment_binding_verified") is True,
        "signed_events_verified": stages.get("signed_chronology", {}).get("signed_events_verified") is True,
        "trusted_checkpoint_verified": stages.get("signed_chronology", {}).get("trusted_checkpoint_verified") is True,
        "independent_time_provenance_verified": False,
        "access_completeness_verified": False,
        "prospective_authorization_verified": False,
        "authorization": "not_established",
        "authentication": (
            "signature_verified" if stages.get("signature_verification", {}).get("signature_verified") is True
            else "not_established"
        ),
        "comparison_permitted": False,
        "ranking_permitted": False,
    }
    result["decision_digest"] = _digest(result)
    return result


def assess_independent_benchmark_authorization(
    *,
    inputs: Sequence[VerifiedBenchmarkInput],
    scope_declaration: Mapping,
    authorization_evidence: Mapping | None,
    prospective_commitment: Mapping,
    authorization_registration: Mapping | None,
    protected_holdout_events: Sequence[Mapping] | None,
    trust_registry: Mapping | None = None,
    checkpoint: Mapping | None = None,
    signature_trust_registry: Mapping | None = None,
    chronology_trust_registry: Mapping | None = None,
) -> dict:
    """Assess all B2 contracts without ever granting authorization.

    `trust_registry` is an optional backwards-compatible alias for the B2-B1
    signature registry. B2-B2-B uses a DIFFERENT registry schema and must be
    supplied through `chronology_trust_registry`. Passing two competing B2-B1
    registries is rejected, not silently resolved.

    C1 upstream integrity failures and malformed B1 declarations are rejected.
    Missing external authorization evidence remains required, while malformed
    or contradictory supplied evidence is rejected. All stages are evaluated
    where their prerequisites allow it; no stage's success overrides a failure.
    """
    stages: dict[str, dict] = {}
    reasons: list[str] = []
    scope_digest = None
    commitment_digest = None

    if trust_registry is not None and signature_trust_registry is not None:
        return _decision(status="AUTHORIZATION_REJECTED",
                         reasons=["CONFLICTING_SIGNATURE_REGISTRY_ARGUMENTS"],
                         stages=stages, scope_digest=None, commitment_digest=None)
    signature_registry = signature_trust_registry if signature_trust_registry is not None else trust_registry

    # Each stage is invoked through the existing accepted verifier. Do not
    # reconstruct their cryptographic claims, timestamps or C1 chain.
    try:
        scope = assess_benchmark_scope(inputs=inputs, declaration=scope_declaration)
    except (ValueError, TypeError, KeyError, OverflowError):
        return _decision(status="AUTHORIZATION_REJECTED", reasons=["INVALID_C1_OR_B1_SCOPE"],
                         stages=stages, scope_digest=None, commitment_digest=None)
    stages["benchmark_scope"] = scope
    # A stage may reject before it can provide a digest. Do not let an
    # incomplete or unexpected response bypass the final fail-closed decision.
    if type(scope) is not dict:
        return _decision(status="AUTHORIZATION_REJECTED",
                         reasons=["BENCHMARK_SCOPE_INVALID_RESULT"],
                         stages={}, scope_digest=None, commitment_digest=None)
    scope_digest = scope.get("scope_digest") if type(scope.get("scope_digest")) is str else None

    try:
        structure = assess_authorization_evidence_structure(
            scope_declaration=scope_declaration,
            authorization_evidence=authorization_evidence,
        )
        stages["evidence_structure"] = structure
    except (ValueError, TypeError, KeyError, OverflowError):
        reasons.append("EVIDENCE_STRUCTURE_EXCEPTION")

    # B2-B1 requires its own independently provisioned registry. If no evidence
    # is supplied, there is no need to invent a registry to explain its absence.
    if authorization_evidence is None:
        reasons.append("EXTERNAL_EVIDENCE_MISSING")
    elif signature_registry is None:
        reasons.append("SIGNATURE_TRUST_REGISTRY_MISSING")
    else:
        try:
            stages["signature_verification"] = verify_authorization_signature(
                authorization_evidence=authorization_evidence,
                trust_registry=signature_registry,
                scope_declaration=scope_declaration,
            )
        except (ValueError, TypeError, KeyError, OverflowError):
            reasons.append("SIGNATURE_VERIFICATION_EXCEPTION")

    try:
        prospective = assess_prospective_commitment_binding(
            commitment=prospective_commitment,
            scope_declaration=scope_declaration,
        )
        stages["prospective_binding"] = prospective
        if type(prospective) is not dict:
            raise ValueError("invalid prospective stage result")
        commitment_digest = (prospective.get("commitment_digest")
                             if type(prospective.get("commitment_digest")) is str else None)
    except (ValueError, TypeError, KeyError, OverflowError):
        reasons.append("INVALID_PROSPECTIVE_COMMITMENT")

    if "prospective_binding" in stages:
        if chronology_trust_registry is None and (
            authorization_registration is not None or protected_holdout_events is not None or checkpoint is not None
        ):
            reasons.append("CHRONOLOGY_TRUST_REGISTRY_MISSING")
        else:
            try:
                stages["signed_chronology"] = verify_trusted_authorization_chronology(
                    prospective_commitment=prospective_commitment,
                    authorization_registration=authorization_registration,
                    protected_holdout_events=protected_holdout_events,
                    trust_registry=chronology_trust_registry,
                    checkpoint=checkpoint,
                )
            except (ValueError, TypeError, KeyError, OverflowError):
                reasons.append("CHRONOLOGY_VERIFICATION_EXCEPTION")

    # Reject contradictory/invalid supplied evidence, but do not confuse
    # genuinely absent proof with an invalid signature or false chronology.
    for name in _STAGE_NAMES:
        stage = stages.get(name)
        if stage is None:
            continue
        if type(stage) is not dict:
            reasons.append(f"{name.upper()}_INVALID_RESULT")
            stages[name] = {"status": "INVALID_STAGE_RESULT"}
        elif stage.get("status") == "AUTHORIZATION_REJECTED":
            reasons.append(f"{name.upper()}_REJECTED")
        elif stage.get("status") != "AUTHORIZATION_REQUIRED":
            reasons.append(f"{name.upper()}_UNEXPECTED_STATUS")

    rejection_reasons = {
        "CONFLICTING_SIGNATURE_REGISTRY_ARGUMENTS", "INVALID_C1_OR_B1_SCOPE",
        "EVIDENCE_STRUCTURE_EXCEPTION", "SIGNATURE_VERIFICATION_EXCEPTION",
        "INVALID_PROSPECTIVE_COMMITMENT", "CHRONOLOGY_VERIFICATION_EXCEPTION",
    }
    rejected = any(r in rejection_reasons or r.endswith(("_REJECTED", "_UNEXPECTED_STATUS", "_INVALID_RESULT")) for r in reasons)
    if not rejected:
        if "signature_verification" not in stages and authorization_evidence is not None:
            reasons.append("SIGNATURE_NOT_VERIFIED")
        if "signed_chronology" not in stages:
            reasons.append("CHRONOLOGY_NOT_VERIFIED")
        if not reasons:
            reasons.append("INDEPENDENT_TIME_AND_ACCESS_COMPLETENESS_NOT_ESTABLISHED")
    return _decision(
        status="AUTHORIZATION_REJECTED" if rejected else "AUTHORIZATION_REQUIRED",
        reasons=list(dict.fromkeys(reasons)), stages=stages,
        scope_digest=scope_digest, commitment_digest=commitment_digest,
    )
