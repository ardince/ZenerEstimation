"""M19-C2-B2-B1: independently anchored Ed25519 authentication, NOT authorization.

Trust registry must be provisioned independently of authorization evidence. This
module does not verify trusted time, holdout chronology, or permit comparison.
"""
from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from zenerestimation.assessment.holdout_target_disclosure import _digest
from zenerestimation.comparison.benchmark_authorization import verify_benchmark_scope_declaration
from zenerestimation.comparison.benchmark_authorization_evidence import (
    assess_authorization_evidence_structure,
    authorization_signed_claims,
    validate_external_authorization_evidence,
)

CRYPTO_ASSESSMENT_VERSION = "m19_c2b2b1_signature_verification_v1"
TRUST_REGISTRY_VERSION = "m19_c2b2b1_trust_registry_v1"
SIGNING_CONTEXT = b"ZenerEstimation/M19-C2-B2/Ed25519/authorization/v1\n"
_KEY_ENCODING = re.compile(r"[A-Za-z0-9_-]{43}\Z", re.ASCII)


def authorization_signing_bytes(evidence: Mapping) -> bytes:
    """Domain-separated, deterministic UTF-8 bytes of validated signed claims."""
    claims = authorization_signed_claims(evidence)
    return SIGNING_CONTEXT + json.dumps(
        claims, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _decode_canonical(value: object, length: int, field: str) -> bytes:
    if type(value) is not str:
        raise ValueError(f"invalid {field}")
    expected_length = {32: 43, 64: 86}[length]
    pattern = _KEY_ENCODING if length == 32 else re.compile(r"[A-Za-z0-9_-]{86}\Z", re.ASCII)
    if len(value) != expected_length or pattern.fullmatch(value) is None:
        raise ValueError(f"invalid {field}")
    try:
        raw = base64.b64decode(value + ("=" if length == 32 else "=="), altchars=b"-_", validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError(f"invalid {field}") from exc
    if len(raw) != length or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != value:
        raise ValueError(f"noncanonical {field}")
    return raw


def _trusted_key(registry: Mapping, issuer_id: str, key_id: str) -> bytes:
    """Strict, immutable-by-convention external registry shape; no envelope keys."""
    if type(registry) is not dict or set(registry) != {"registry_version", "issuers"}:
        raise ValueError("invalid independent trust registry")
    if registry["registry_version"] != TRUST_REGISTRY_VERSION:
        raise ValueError("unsupported trust registry version")
    issuers = registry["issuers"]
    if type(issuers) is not dict or any(type(k) is not str for k in issuers):
        raise ValueError("invalid issuers registry")
    issuer = issuers.get(issuer_id)
    if issuer is None:
        raise LookupError("UNTRUSTED_ISSUER")
    if type(issuer) is not dict or set(issuer) != {"status", "keys"}:
        raise ValueError("invalid issuer registry record")
    if issuer["status"] not in ("active", "revoked") or type(issuer["status"]) is not str:
        raise ValueError("invalid issuer status")
    if issuer["status"] != "active":
        raise LookupError("REVOKED_ISSUER")
    keys = issuer["keys"]
    if type(keys) is not dict or any(type(k) is not str for k in keys):
        raise ValueError("invalid keys registry")
    key = keys.get(key_id)
    if key is None:
        raise LookupError("UNTRUSTED_KEY")
    if type(key) is not dict or set(key) != {"algorithm", "status", "public_key"}:
        raise ValueError("invalid trusted key record")
    if key["algorithm"] != "Ed25519" or type(key["algorithm"]) is not str:
        raise ValueError("unsupported trusted key algorithm")
    if key["status"] not in ("active", "revoked") or type(key["status"]) is not str:
        raise ValueError("invalid trusted key status")
    if key["status"] != "active":
        raise LookupError("REVOKED_KEY")
    return _decode_canonical(key["public_key"], 32, "trusted public key")


def verify_authorization_signature(
    *, authorization_evidence: Mapping | None, trust_registry: Mapping,
    scope_declaration: Mapping | None = None,
) -> dict:
    """Authenticate signature and optional B1 binding; never authorize use.

    If scope_declaration is omitted, only signed-claim authentication is
    established; scope_binding_verified remains False. Invalid supplied
    evidence fails closed. Missing evidence remains AUTHORIZATION_REQUIRED.
    """
    reasons: list[str] = []
    signature_verified = False
    issuer_trusted = False
    scope_binding_verified = False
    evidence_digest = None
    benchmark_id = None
    scope_digest = None

    if scope_declaration is not None:
        if not verify_benchmark_scope_declaration(scope_declaration):
            raise ValueError("invalid B1 scope declaration")
        benchmark_id = scope_declaration["benchmark_id"]
        scope_digest = scope_declaration["scope_digest"]

    if authorization_evidence is None:
        reasons.append("EXTERNAL_EVIDENCE_MISSING")
    else:
        try:
            validated = validate_external_authorization_evidence(authorization_evidence)
        except (ValueError, TypeError, KeyError):
            reasons.append("INVALID_EXTERNAL_EVIDENCE")
        else:
            evidence_digest = _digest(validated)
            if scope_declaration is not None:
                scope_binding_verified = validated["scope_digest"] == scope_digest
                if not scope_binding_verified:
                    reasons.append("SCOPE_BINDING_MISMATCH")
            try:
                trusted_key = _trusted_key(
                    trust_registry, validated["issuer_id"], validated["key_id"]
                )
                issuer_trusted = True
            except LookupError as exc:
                reasons.append(str(exc))
            except (ValueError, TypeError, KeyError):
                reasons.append("INVALID_TRUST_REGISTRY")
            else:
                try:
                    signature = _decode_canonical(validated["signature"], 64, "signature")
                    Ed25519PublicKey.from_public_bytes(trusted_key).verify(
                        signature, authorization_signing_bytes(validated)
                    )
                except (InvalidSignature, ValueError, TypeError):
                    reasons.append("INVALID_SIGNATURE")
                else:
                    signature_verified = True

    rejected = any(reason != "EXTERNAL_EVIDENCE_MISSING" for reason in reasons)
    result = {
        "assessment_version": CRYPTO_ASSESSMENT_VERSION,
        "benchmark_id": benchmark_id,
        "scope_digest": scope_digest,
        "evidence_digest": evidence_digest,
        "status": "AUTHORIZATION_REJECTED" if rejected else "AUTHORIZATION_REQUIRED",
        "reasons": reasons,
        "signature_verified": signature_verified,
        "issuer_trusted": issuer_trusted,
        "scope_binding_verified": scope_binding_verified,
        "prospective_authorization_verified": False,
        "authorization": "not_established",
        "authentication": "signature_verified" if signature_verified else "not_established",
        "comparison_permitted": False,
        "ranking_permitted": False,
    }
    result["decision_digest"] = _digest(result)
    return result
