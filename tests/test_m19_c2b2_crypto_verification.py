"""M19-C2-B2-B1 cryptographic trust, adversarial and fail-closed tests."""
from __future__ import annotations

import base64
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from zenerestimation.comparison.benchmark_authorization import build_benchmark_scope_declaration
from zenerestimation.comparison.benchmark_authorization_evidence import EVIDENCE_VERSION
from zenerestimation.comparison.benchmark_authorization_crypto import (
    TRUST_REGISTRY_VERSION, authorization_signing_bytes, verify_authorization_signature,
)


def b64(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def setup():
    scope = build_benchmark_scope_declaration(
        benchmark_id="benchmark.410", dataset_id="410",
        forecast_dates=["2025-03-01", "2025-06-01"],
        participants=[
            {"run_id": "run.gru", "assessment_id": "assess.gru", "model_family": "GRU",
             "benchmark_entry_digest": "sha256:" + "a" * 64},
            {"run_id": "run.lstm", "assessment_id": "assess.lstm", "model_family": "LSTM",
             "benchmark_entry_digest": "sha256:" + "b" * 64},
        ],
    )
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    registry = {"registry_version": TRUST_REGISTRY_VERSION, "issuers": {
        "trusted.example": {"status": "active", "keys": {
            "key.001": {"algorithm": "Ed25519", "status": "active", "public_key": b64(public)}
        }}
    }}
    evidence = {
        "evidence_version": EVIDENCE_VERSION, "authorization_id": "auth.001",
        "issuer_id": "trusted.example", "key_id": "key.001", "algorithm": "Ed25519",
        "purpose": "cross_model_benchmark", "authorized_operation": "compare_and_rank",
        "scope_digest": scope["scope_digest"],
        "issued_at": "2026-01-01T00:00:00Z", "not_before": "2026-01-01T00:00:01Z",
        "expires_at": "2027-01-01T00:00:00Z", "signature": b64(bytes(64)),
    }
    evidence["signature"] = b64(private.sign(authorization_signing_bytes(evidence)))
    return scope, registry, evidence, private


def check(scope, registry, evidence):
    return verify_authorization_signature(
        scope_declaration=scope, trust_registry=registry, authorization_evidence=evidence
    )


def test_valid_signature_is_authenticated_but_never_authorized():
    scope, registry, evidence, _ = setup()
    result = check(scope, registry, evidence)
    assert result["signature_verified"] is True
    assert result["issuer_trusted"] is True
    assert result["scope_binding_verified"] is True
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert result["prospective_authorization_verified"] is False
    assert result["comparison_permitted"] is False and result["ranking_permitted"] is False


def test_signing_bytes_are_stable_and_domain_separated():
    _, _, evidence, _ = setup()
    assert authorization_signing_bytes(evidence) == authorization_signing_bytes(deepcopy(evidence))
    assert authorization_signing_bytes(evidence).startswith(b"ZenerEstimation/M19-C2-B2/")
    assert b'"signature"' not in authorization_signing_bytes(evidence)


@pytest.mark.parametrize("field,value", [
    ("authorization_id", "auth.002"), ("issuer_id", "attacker.example"),
    ("key_id", "key.002"), ("scope_digest", "sha256:" + "f" * 64),
    ("issued_at", "2026-01-02T00:00:00Z"),
    ("not_before", "2026-01-02T00:00:01Z"),
    ("expires_at", "2027-01-02T00:00:00Z"),
])
def test_tampered_claims_fail_closed(field, value):
    scope, registry, evidence, _ = setup()
    evidence[field] = value
    result = check(scope, registry, evidence)
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert result["comparison_permitted"] is False


def test_invalid_signature_bytes_rejected():
    scope, registry, evidence, _ = setup()
    evidence["signature"] = b64(bytes(64))
    result = check(scope, registry, evidence)
    assert result["reasons"] == ["INVALID_SIGNATURE"]
    assert result["signature_verified"] is False


def test_wrong_trusted_key_rejected():
    scope, registry, evidence, _ = setup()
    other = Ed25519PrivateKey.generate()
    registry["issuers"]["trusted.example"]["keys"]["key.001"]["public_key"] = b64(
        other.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )
    assert check(scope, registry, evidence)["reasons"] == ["INVALID_SIGNATURE"]


@pytest.mark.parametrize("change,reason", [
    ("issuer_revoked", "REVOKED_ISSUER"),
    ("key_revoked", "REVOKED_KEY"),
    ("issuer_missing", "UNTRUSTED_ISSUER"),
    ("key_missing", "UNTRUSTED_KEY"),
    ("malformed_registry", "INVALID_TRUST_REGISTRY"),
    ("wrong_algorithm", "INVALID_TRUST_REGISTRY"),
    ("invalid_public_key", "INVALID_TRUST_REGISTRY"),
])
def test_untrusted_registry_conditions(change, reason):
    scope, registry, evidence, _ = setup()
    issuer = registry["issuers"]["trusted.example"]
    key = issuer["keys"]["key.001"]
    if change == "issuer_revoked": issuer["status"] = "revoked"
    elif change == "key_revoked": key["status"] = "revoked"
    elif change == "issuer_missing": del registry["issuers"]["trusted.example"]
    elif change == "key_missing": del issuer["keys"]["key.001"]
    elif change == "malformed_registry": registry["extra"] = "attacker"
    elif change == "wrong_algorithm": key["algorithm"] = "RSA"
    elif change == "invalid_public_key": key["public_key"] = "attacker"
    result = check(scope, registry, evidence)
    assert reason in result["reasons"]
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert result["signature_verified"] is False


def test_missing_evidence_never_authorizes():
    scope, registry, _, _ = setup()
    result = check(scope, registry, None)
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert result["reasons"] == ["EXTERNAL_EVIDENCE_MISSING"]


def test_scope_substitution_rejected_even_with_valid_signature():
    scope, registry, evidence, private = setup()
    evidence["scope_digest"] = "sha256:" + "f" * 64
    evidence["signature"] = b64(private.sign(authorization_signing_bytes(evidence)))
    result = check(scope, registry, evidence)
    assert result["signature_verified"] is True
    assert result["scope_binding_verified"] is False
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert "SCOPE_BINDING_MISMATCH" in result["reasons"]


def test_deterministic_and_read_only():
    scope, registry, evidence, _ = setup()
    before = deepcopy((scope, registry, evidence))
    assert check(scope, registry, evidence) == check(scope, registry, evidence)
    assert (scope, registry, evidence) == before


def test_embedded_public_key_rejected():
    scope, registry, evidence, _ = setup()
    evidence["public_key"] = registry["issuers"]["trusted.example"]["keys"]["key.001"]["public_key"]
    assert check(scope, registry, evidence)["reasons"] == ["INVALID_EXTERNAL_EVIDENCE"]


def test_unbound_signature_does_not_claim_scope_verification():
    _, registry, evidence, _ = setup()
    result = verify_authorization_signature(authorization_evidence=evidence, trust_registry=registry)
    assert result["signature_verified"] is True
    assert result["scope_binding_verified"] is False
    assert result["comparison_permitted"] is False
