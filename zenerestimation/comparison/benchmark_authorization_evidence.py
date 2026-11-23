"""M19-C2-B2-A: external-authorization evidence *structure*, not approval.

No signature, trust anchor, timestamp, or protected-holdout event is verified
in B2-A. In particular, this module MUST NOT return AUTHORIZED.
B2-B will add independently configured trust and prospective-time verification.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone

from zenerestimation.assessment.holdout_target_disclosure import _digest
from zenerestimation.comparison.benchmark_authorization import (
    verify_benchmark_scope_declaration,
)

EVIDENCE_VERSION = "m19_c2b2_external_authorization_evidence_v1"
ASSESSMENT_VERSION = "m19_c2b2a_evidence_structure_assessment_v1"
ALGORITHM = "Ed25519"
PURPOSE = "cross_model_benchmark"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z", re.ASCII)
_SIG = re.compile(r"[A-Za-z0-9_-]{86}\Z", re.ASCII)  # unpadded base64url, 64 bytes
_CLAIMS = frozenset({
    "evidence_version", "authorization_id", "issuer_id", "key_id",
    "algorithm", "purpose", "scope_digest", "issued_at", "not_before",
    "expires_at", "authorized_operation",
})
_FIELDS = _CLAIMS | {"signature"}


def _identifier(value: object, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"invalid {field}")
    return value


def _timestamp(value: object, field: str) -> datetime:
    """Require canonical UTC timestamps; no ambiguous naive or local clocks."""
    if type(value) is not str or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ) is None:
        raise ValueError(f"{field} must be UTC YYYY-MM-DDTHH:MM:SSZ")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid {field}") from exc
    if parsed.tzinfo != timezone.utc or parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError(f"noncanonical {field}")
    return parsed


def validate_external_authorization_evidence(evidence: Mapping) -> dict:
    """Validate and detach an untrusted signed-claim envelope.

    This is *not* cryptographic verification. The signature is opaque here.
    No embedded public key, issuer-provided trust registry, or audit timestamp
    is permitted in the envelope.
    """
    if type(evidence) is not dict or set(evidence) != _FIELDS:
        raise ValueError("external evidence must contain exactly the required fields")
    if evidence["evidence_version"] != EVIDENCE_VERSION:
        raise ValueError("unsupported evidence version")
    result = dict(evidence)
    for field in ("authorization_id", "issuer_id", "key_id"):
        _identifier(result[field], field)
    if result["algorithm"] != ALGORITHM or type(result["algorithm"]) is not str:
        raise ValueError("unsupported signature algorithm")
    if result["purpose"] != PURPOSE or type(result["purpose"]) is not str:
        raise ValueError("unsupported authorization purpose")
    if result["authorized_operation"] != "compare_and_rank" or type(result["authorized_operation"]) is not str:
        raise ValueError("unsupported authorized operation")
    if type(result["scope_digest"]) is not str or _SHA.fullmatch(result["scope_digest"]) is None:
        raise ValueError("invalid scope digest format")
    issued = _timestamp(result["issued_at"], "issued_at")
    start = _timestamp(result["not_before"], "not_before")
    end = _timestamp(result["expires_at"], "expires_at")
    if not (issued <= start < end):
        raise ValueError("invalid claimed authorization time window")
    signature = result["signature"]
    if type(signature) is not str or _SIG.fullmatch(signature) is None:
        raise ValueError("signature must be 64-byte unpadded base64url")
    # 86 characters are not by themselves proof of a 64-byte signature.
    import base64
    try:
        raw = base64.b64decode(signature + "==", altchars=b"-_", validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("invalid signature encoding") from exc
    if len(raw) != 64 or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != signature:
        raise ValueError("noncanonical signature encoding")
    return result


def authorization_signed_claims(evidence: Mapping) -> dict:
    """Return the exact claims intended for signing; never include signature."""
    validated = validate_external_authorization_evidence(evidence)
    return {field: validated[field] for field in sorted(_CLAIMS)}


def assess_authorization_evidence_structure(
    *, scope_declaration: Mapping, authorization_evidence: Mapping | None,
) -> dict:
    """Assess B2-A syntax and B1 binding without conferring approval.

    Missing evidence -> AUTHORIZATION_REQUIRED; malformed or wrong scope ->
    AUTHORIZATION_REJECTED. A structurally matching envelope still remains
    AUTHORIZATION_REQUIRED until independent B2-B verification.
    """
    if not verify_benchmark_scope_declaration(scope_declaration):
        raise ValueError("invalid B1 scope declaration")
    reasons: list[str] = []
    evidence_digest = None
    if authorization_evidence is None:
        reasons.append("EXTERNAL_EVIDENCE_MISSING")
    else:
        try:
            validated = validate_external_authorization_evidence(authorization_evidence)
        except (ValueError, TypeError, KeyError) as exc:
            reasons.append("INVALID_EXTERNAL_EVIDENCE")
        else:
            evidence_digest = _digest(validated)
            if validated["scope_digest"] != scope_declaration["scope_digest"]:
                reasons.append("SCOPE_BINDING_MISMATCH")
    rejected = "INVALID_EXTERNAL_EVIDENCE" in reasons or "SCOPE_BINDING_MISMATCH" in reasons
    result = {
        "assessment_version": ASSESSMENT_VERSION,
        "benchmark_id": scope_declaration["benchmark_id"],
        "scope_digest": scope_declaration["scope_digest"],
        "evidence_digest": evidence_digest,
        "status": "AUTHORIZATION_REJECTED" if rejected else "AUTHORIZATION_REQUIRED",
        "reasons": reasons,
        "signature_verified": False,
        "issuer_trusted": False,
        "prospective_authorization_verified": False,
        "authorization": "not_established",
        "authentication": "not_established",
        "comparison_permitted": False,
        "ranking_permitted": False,
    }
    result["decision_digest"] = _digest(result)
    return result
