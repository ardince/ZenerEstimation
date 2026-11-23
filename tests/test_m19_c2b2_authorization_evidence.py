"""M19-C2-B2-A: strict external evidence envelope; NO approval in this phase."""
from __future__ import annotations

from copy import deepcopy
import base64

import pytest

from zenerestimation.comparison.benchmark_authorization import build_benchmark_scope_declaration
from zenerestimation.comparison.benchmark_authorization_evidence import (
    EVIDENCE_VERSION, assess_authorization_evidence_structure,
    authorization_signed_claims, validate_external_authorization_evidence,
)


def _scope():
    return build_benchmark_scope_declaration(
        benchmark_id="benchmark.410", dataset_id="410",
        forecast_dates=["2025-03-01", "2025-06-01"],
        participants=[
            {"run_id": "run.gru", "assessment_id": "assess.gru", "model_family": "GRU",
             "benchmark_entry_digest": "sha256:" + "a" * 64},
            {"run_id": "run.lstm", "assessment_id": "assess.lstm", "model_family": "LSTM",
             "benchmark_entry_digest": "sha256:" + "b" * 64},
        ],
    )


def _evidence(scope=None):
    scope = scope or _scope()
    # Test-only placeholder signature, not a cryptographically valid approval.
    sig = base64.urlsafe_b64encode(bytes(range(64))).rstrip(b"=").decode("ascii")
    return {
        "evidence_version": EVIDENCE_VERSION,
        "authorization_id": "auth.001", "issuer_id": "trusted.example",
        "key_id": "key.001", "algorithm": "Ed25519",
        "purpose": "cross_model_benchmark", "authorized_operation": "compare_and_rank",
        "scope_digest": scope["scope_digest"],
        "issued_at": "2026-01-01T00:00:00Z",
        "not_before": "2026-01-01T00:00:01Z",
        "expires_at": "2027-01-01T00:00:00Z",
        "signature": sig,
    }


def test_matching_structure_never_authorizes():
    scope = _scope()
    result = assess_authorization_evidence_structure(
        scope_declaration=scope, authorization_evidence=_evidence(scope))
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert result["reasons"] == []
    assert result["signature_verified"] is False
    assert result["issuer_trusted"] is False
    assert result["prospective_authorization_verified"] is False
    assert result["comparison_permitted"] is False
    assert result["ranking_permitted"] is False


def test_missing_evidence_is_required_not_approved():
    result = assess_authorization_evidence_structure(
        scope_declaration=_scope(), authorization_evidence=None)
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert result["reasons"] == ["EXTERNAL_EVIDENCE_MISSING"]


def test_signed_claims_are_detached_and_exclude_signature():
    evidence = _evidence()
    claims = authorization_signed_claims(evidence)
    assert "signature" not in claims
    assert claims["scope_digest"] == evidence["scope_digest"]
    claims["scope_digest"] = "sha256:" + "f" * 64
    assert evidence["scope_digest"] != claims["scope_digest"]


def test_determinism_and_read_only():
    scope, evidence = _scope(), _evidence()
    before_scope, before_evidence = deepcopy(scope), deepcopy(evidence)
    a = assess_authorization_evidence_structure(scope_declaration=scope, authorization_evidence=evidence)
    b = assess_authorization_evidence_structure(scope_declaration=scope, authorization_evidence=evidence)
    assert a == b
    assert scope == before_scope and evidence == before_evidence
    assert a["decision_digest"].startswith("sha256:")


def test_other_scope_is_rejected():
    scope = _scope()
    evidence = _evidence(scope)
    evidence["scope_digest"] = "sha256:" + "f" * 64
    result = assess_authorization_evidence_structure(
        scope_declaration=scope, authorization_evidence=evidence)
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert result["reasons"] == ["SCOPE_BINDING_MISMATCH"]


@pytest.mark.parametrize("field,value", [
    ("algorithm", "RSA"), ("purpose", "ranking"),
    ("authorized_operation", "compare"),
    ("evidence_version", "future"), ("issuer_id", ""),
    ("key_id", "bad key"), ("authorization_id", ""),
    ("scope_digest", "f" * 64), ("signature", "x" * 86),
    ("issued_at", "2026-01-01T00:00:00+00:00"),
    ("not_before", "2025-12-31T00:00:00Z"),
    ("expires_at", "2026-01-01T00:00:01Z"),
])
def test_invalid_claim_rejected(field, value):
    evidence = _evidence()
    evidence[field] = value
    with pytest.raises(ValueError):
        validate_external_authorization_evidence(evidence)
    decision = assess_authorization_evidence_structure(
        scope_declaration=_scope(), authorization_evidence=evidence)
    assert decision["status"] == "AUTHORIZATION_REJECTED"
    assert decision["reasons"] == ["INVALID_EXTERNAL_EVIDENCE"]


@pytest.mark.parametrize("extra", ["public_key", "trust_registry", "approved", "audit_timestamp", "approval_digest"])
def test_untrusted_extra_fields_rejected(extra):
    evidence = _evidence()
    evidence[extra] = "attacker-controlled"
    with pytest.raises(ValueError):
        validate_external_authorization_evidence(evidence)


def test_missing_claim_rejected():
    evidence = _evidence()
    del evidence["key_id"]
    with pytest.raises(ValueError):
        validate_external_authorization_evidence(evidence)


def test_bad_scope_fails_closed():
    scope = _scope()
    scope["scope_digest"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="invalid B1 scope"):
        assess_authorization_evidence_structure(
            scope_declaration=scope, authorization_evidence=_evidence())


def test_signature_is_only_syntactically_valid_not_verified():
    evidence = _evidence()
    assert validate_external_authorization_evidence(evidence) == evidence
    decision = assess_authorization_evidence_structure(
        scope_declaration=_scope(), authorization_evidence=evidence)
    assert decision["status"] != "AUTHORIZED"
    assert decision["signature_verified"] is False
