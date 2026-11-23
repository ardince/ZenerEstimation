"""M19-C2-B2-C2: A1 structural contract and A2 timestamp-bound facade.

A1 assessment stays structural; A2 verifies timestamped signed-event bytes.
A timestamp is not event-signature authentication or benchmark authorization.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

ATTESTATION_VERSION = "m19_c2b2c2a_temporal_attestation_v1"
TRUST_REGISTRY_VERSION = "m19_c2b2c2a_temporal_trust_registry_v1"
SUPPORTED_MECHANISMS = frozenset({"rfc3161"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _json_object(value: Any) -> dict[str, Any] | None:
    """Reject non-JSON, non-string keys, non-finite numbers, and cycles."""
    if not isinstance(value, Mapping):
        return None
    try:
        raw = _canonical(value)
        result = json.loads(raw)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return None
    return result if isinstance(result, dict) else None


def assess_temporal_attestation_contract(
    *,
    temporal_attestation: Mapping[str, Any] | None,
    temporal_trust_registry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the A1 envelope; never authenticate time or authorize use.

    Envelope schema:
      attestation_version, attestation_id, mechanism, authority_id,
      subject_digest, scope_digest, commitment_digest, proof.
    `proof` is a nonempty opaque string (base64 token for future RFC 3161).

    Registry schema:
      registry_version, authorities: {authority_id: {mechanisms: [str],
      trust_material: object}}.

    The trust_material field is intentionally opaque in A1; its contents MUST
    be validated by a mechanism-specific verifier before being trusted.
    """
    problems: list[str] = []
    att = _json_object(temporal_attestation)
    registry = _json_object(temporal_trust_registry)

    if temporal_attestation is None:
        problems.append("TEMPORAL_ATTESTATION_MISSING")
    elif att is None:
        problems.append("TEMPORAL_ATTESTATION_INVALID")
    else:
        required = {
            "attestation_version", "attestation_id", "mechanism",
            "authority_id", "subject_digest", "scope_digest",
            "commitment_digest", "proof",
        }
        if set(att) != required:
            problems.append("TEMPORAL_ATTESTATION_SCHEMA_INVALID")
        if att.get("attestation_version") != ATTESTATION_VERSION:
            problems.append("TEMPORAL_ATTESTATION_VERSION_UNSUPPORTED")
        for key in ("attestation_id", "authority_id", "proof"):
            if not _nonempty(att.get(key)):
                problems.append(f"{key.upper()}_INVALID")
        for key in ("subject_digest", "scope_digest", "commitment_digest"):
            if not _is_digest(att.get(key)):
                problems.append(f"{key.upper()}_INVALID")
        if not _nonempty(att.get("mechanism")) or att.get("mechanism") not in SUPPORTED_MECHANISMS:
            problems.append("TEMPORAL_MECHANISM_UNSUPPORTED")

    if temporal_trust_registry is None:
        problems.append("TEMPORAL_TRUST_REGISTRY_MISSING")
    elif registry is None:
        problems.append("TEMPORAL_TRUST_REGISTRY_INVALID")
    else:
        if set(registry) != {"registry_version", "authorities"}:
            problems.append("TEMPORAL_TRUST_REGISTRY_SCHEMA_INVALID")
        if registry.get("registry_version") != TRUST_REGISTRY_VERSION:
            problems.append("TEMPORAL_TRUST_REGISTRY_VERSION_UNSUPPORTED")
        authorities = registry.get("authorities")
        if not isinstance(authorities, dict):
            problems.append("TEMPORAL_AUTHORITIES_INVALID")
        elif att is not None and _nonempty(att.get("authority_id")):
            authority = authorities.get(att["authority_id"])
            if not isinstance(authority, dict):
                problems.append("TEMPORAL_AUTHORITY_UNTRUSTED")
            elif (set(authority) != {"mechanisms", "trust_material"}
                  or not isinstance(authority.get("mechanisms"), list)
                  or not all(_nonempty(m) for m in authority["mechanisms"])
                  or not isinstance(authority.get("trust_material"), dict)
                  or not authority["trust_material"]):
                problems.append("TEMPORAL_AUTHORITY_POLICY_INVALID")
            elif att.get("mechanism") not in authority["mechanisms"]:
                problems.append("TEMPORAL_AUTHORITY_MECHANISM_NOT_ALLOWED")

    # A1 cannot establish provenance even if the envelope is structurally valid.
    problems = sorted(set(problems))
    outcome = {
        "contract_version": ATTESTATION_VERSION,
        "status": "AUTHORIZATION_REJECTED" if problems else "AUTHORIZATION_REQUIRED",
        "contract_valid": not problems,
        "mechanism": att.get("mechanism") if att else None,
        "independent_temporal_attestation_verified": False,
        "prospective_authorization_verified": False,
        "comparison_permitted": False,
        "ranking_permitted": False,
        "reason_codes": problems if problems else ["TEMPORAL_VERIFIER_NOT_IMPLEMENTED"],
    }
    outcome["assessment_digest"] = _sha256(outcome)
    return outcome


# Timestamp the COMPLETE signed registration event, including its signature.
# The domain prefix prevents reuse of a timestamp from another application.
_REGISTRATION_TIMESTAMP_DOMAIN = b"ZenerEstimation/M19-C2-B2-C2/registration/v1\x00"


def registration_timestamp_subject_bytes(registration: Mapping[str, Any]) -> bytes:
    """Deterministic A2 subject; never sign only the unsigned event payload."""
    from zenerestimation.comparison.benchmark_authorization_chronology import (
        _validate_event,
    )

    if type(registration) is not dict:
        raise ValueError("invalid registration")
    _validate_event(registration)
    if registration["event_type"] != "authorization_registered":
        raise ValueError("not an authorization registration")
    return _REGISTRATION_TIMESTAMP_DOMAIN + _canonical(registration)


def _a2_result(base: dict[str, Any], reasons: list[str], *,
               token: dict[str, Any] | None = None,
               scope_bound: bool = False,
               commitment_bound: bool = False) -> dict[str, Any]:
    outcome = {k: v for k, v in base.items() if k != "assessment_digest"}
    verified = bool(token and token.get("independent_temporal_attestation_verified") is True
                    and not reasons)
    outcome.update({
        "status": "AUTHORIZATION_REJECTED" if reasons else "AUTHORIZATION_REQUIRED",
        "reason_codes": sorted(set(reasons)) if reasons else ["FURTHER_AUTHORIZATION_EVIDENCE_REQUIRED"],
        "independent_temporal_attestation_verified": verified,
        "timestamped_registration_bound": verified,
        "scope_binding_verified": scope_bound,
        "commitment_binding_verified": commitment_bound,
        "registration_signature_verified": False,
        "independent_access_chronology_verified": False,
        "attested_at": token.get("attested_at") if verified else None,
        "prospective_authorization_verified": False,
        "comparison_permitted": False,
        "ranking_permitted": False,
    })
    outcome["assessment_digest"] = _sha256(outcome)
    return outcome


def verify_independent_temporal_attestation(
    *,
    temporal_attestation: Mapping[str, Any] | None,
    authorization_registration: Mapping[str, Any] | None,
    prospective_commitment: Mapping[str, Any] | None,
    scope_declaration: Mapping[str, Any] | None,
    temporal_trust_registry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind a pinned-TSA timestamp to a complete signed registration record.

    Does not authenticate the registration Ed25519 signature: that requires an
    independently supplied audit-event trust registry, which is not an input
    to this stable facade. Does not establish prospective holdout chronology.
    """
    from zenerestimation.comparison.benchmark_authorization import (
        verify_benchmark_scope_declaration,
    )
    from zenerestimation.comparison.benchmark_authorization_prospective import (
        verify_prospective_benchmark_commitment,
        assess_prospective_commitment_binding,
    )
    from zenerestimation.comparison.benchmark_authorization_rfc3161 import (
        verify_rfc3161_timestamp_token,
    )

    base = assess_temporal_attestation_contract(
        temporal_attestation=temporal_attestation,
        temporal_trust_registry=temporal_trust_registry,
    )
    if not base["contract_valid"]:
        return _a2_result(base, base["reason_codes"])

    att = _json_object(temporal_attestation)
    registry = _json_object(temporal_trust_registry)
    registration = _json_object(authorization_registration)
    commitment = _json_object(prospective_commitment)
    scope = _json_object(scope_declaration)
    reasons: list[str] = []
    for name, value in (("AUTHORIZATION_REGISTRATION", registration),
                        ("PROSPECTIVE_COMMITMENT", commitment),
                        ("SCOPE_DECLARATION", scope)):
        if value is None:
            reasons.append(f"{name}_INVALID")
    if reasons:
        return _a2_result(base, reasons)

    try:
        subject = registration_timestamp_subject_bytes(registration)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return _a2_result(base, ["AUTHORIZATION_REGISTRATION_INVALID"])

    if not verify_prospective_benchmark_commitment(commitment):
        reasons.append("PROSPECTIVE_COMMITMENT_INVALID")
    if not verify_benchmark_scope_declaration(scope):
        reasons.append("SCOPE_DECLARATION_INVALID")
    if reasons:
        return _a2_result(base, reasons)

    binding = assess_prospective_commitment_binding(
        commitment=commitment, scope_declaration=scope,
    )
    scope_bound = binding["commitment_binding_verified"] is True
    if not scope_bound:
        reasons.append("COMMITMENT_SCOPE_BINDING_MISMATCH")

    # Existing chronology events carry sha256:-prefixed commitment digests.
    if (registration["benchmark_id"] != commitment["benchmark_id"]
            or registration["commitment_digest"] != commitment["commitment_digest"]):
        reasons.append("REGISTRATION_COMMITMENT_BINDING_MISMATCH")
    if att["commitment_digest"] != commitment["commitment_digest"].removeprefix("sha256:"):
        reasons.append("ATTESTATION_COMMITMENT_DIGEST_MISMATCH")
    if att["scope_digest"] != scope["scope_digest"].removeprefix("sha256:"):
        reasons.append("ATTESTATION_SCOPE_DIGEST_MISMATCH")
    expected_subject_digest = hashlib.sha256(subject).hexdigest()
    if att["subject_digest"] != expected_subject_digest:
        reasons.append("ATTESTATION_SUBJECT_DIGEST_MISMATCH")
    if reasons:
        return _a2_result(base, reasons, scope_bound=scope_bound,
                          commitment_bound=False)

    authority = registry["authorities"][att["authority_id"]]
    token = verify_rfc3161_timestamp_token(
        proof=att["proof"], subject_bytes=subject,
        subject_digest=expected_subject_digest,
        trust_material=authority["trust_material"],
    )
    if token["independent_temporal_attestation_verified"] is not True:
        reasons.append("RFC3161_TOKEN_INVALID_OR_UNTRUSTED")
    return _a2_result(base, reasons, token=token,
                      scope_bound=scope_bound, commitment_bound=True)
