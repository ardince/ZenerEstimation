"""A2 facade binding tests; crypto itself is covered by the real-token A2 suite."""
import base64
import copy
import hashlib

import pytest

from zenerestimation.comparison import benchmark_authorization_temporal as temporal
from zenerestimation.comparison import benchmark_authorization_rfc3161 as rfc
from zenerestimation.comparison.benchmark_authorization import build_benchmark_scope_declaration
from zenerestimation.comparison.benchmark_authorization_prospective import build_prospective_benchmark_commitment
from zenerestimation.comparison.benchmark_authorization_chronology import EVENT_VERSION


def evidence():
    participants = [
        {"run_id": "r1", "assessment_id": "a1", "model_family": "GRU"},
        {"run_id": "r2", "assessment_id": "a2", "model_family": "LSTM"},
    ]
    commitment = build_prospective_benchmark_commitment(
        benchmark_id="b1", dataset_id="d1", forecast_dates=["2025-03-01"],
        participants=participants,
    )
    scope = build_benchmark_scope_declaration(
        benchmark_id="b1", dataset_id="d1", forecast_dates=["2025-03-01"],
        participants=[{**p, "benchmark_entry_digest": "sha256:" + "1" * 64} for p in participants],
    )
    registration = {
        "event_version": EVENT_VERSION, "event_id": "ev1", "event_type": "authorization_registered",
        "issuer_id": "issuer1", "key_id": "key1", "benchmark_id": "b1",
        "commitment_digest": commitment["commitment_digest"],
        "assessment_id": None, "event_time": "2025-01-01T00:00:00Z",
        "sequence": 1, "signature": base64.urlsafe_b64encode(b"s" * 64).decode().rstrip("="),
    }
    subject = temporal.registration_timestamp_subject_bytes(registration)
    att = {
        "attestation_version": temporal.ATTESTATION_VERSION, "attestation_id": "tsa1",
        "mechanism": "rfc3161", "authority_id": "tsa1",
        "subject_digest": hashlib.sha256(subject).hexdigest(),
        "scope_digest": scope["scope_digest"].removeprefix("sha256:"),
        "commitment_digest": commitment["commitment_digest"].removeprefix("sha256:"),
        "proof": "placeholder",
    }
    registry = {
        "registry_version": temporal.TRUST_REGISTRY_VERSION,
        "authorities": {"tsa1": {"mechanisms": ["rfc3161"],
                                  "trust_material": {"tsa_certificate_der_b64": "placeholder"}}},
    }
    return dict(temporal_attestation=att, authorization_registration=registration,
                prospective_commitment=commitment, scope_declaration=scope,
                temporal_trust_registry=registry)


def _fake_valid(*, proof, subject_bytes, subject_digest, trust_material):
    assert hashlib.sha256(subject_bytes).hexdigest() == subject_digest
    return {"independent_temporal_attestation_verified": True,
            "attested_at": "2025-01-01T00:00:01Z"}


def test_bound_timestamp_is_not_benchmark_authorization(monkeypatch):
    monkeypatch.setattr(rfc, "verify_rfc3161_timestamp_token", _fake_valid)
    inputs = evidence()
    result = temporal.verify_independent_temporal_attestation(**inputs)
    assert result["independent_temporal_attestation_verified"] is True
    assert result["timestamped_registration_bound"] is True
    assert result["registration_signature_verified"] is False
    assert result["independent_access_chronology_verified"] is False
    assert result["prospective_authorization_verified"] is False
    assert result["comparison_permitted"] is False
    assert result["ranking_permitted"] is False
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert result == temporal.verify_independent_temporal_attestation(**inputs)


@pytest.mark.parametrize("mutation,reason", [
    ("registration_signature", "ATTESTATION_SUBJECT_DIGEST_MISMATCH"),
    ("registration_commitment", "REGISTRATION_COMMITMENT_BINDING_MISMATCH"),
    ("attestation_scope", "ATTESTATION_SCOPE_DIGEST_MISMATCH"),
    ("attestation_commitment", "ATTESTATION_COMMITMENT_DIGEST_MISMATCH"),
    ("attestation_subject", "ATTESTATION_SUBJECT_DIGEST_MISMATCH"),
    ("commitment", "PROSPECTIVE_COMMITMENT_INVALID"),
    ("scope", "SCOPE_DECLARATION_INVALID"),
])
def test_fail_closed_before_crypto(monkeypatch, mutation, reason):
    def forbidden(**kwargs):
        pytest.fail("invalid binding reached cryptographic verifier")
    monkeypatch.setattr(rfc, "verify_rfc3161_timestamp_token", forbidden)
    inputs = evidence()
    if mutation == "registration_signature":
        inputs["authorization_registration"]["signature"] = base64.urlsafe_b64encode(b"x" * 64).decode().rstrip("=")
    elif mutation == "registration_commitment":
        inputs["authorization_registration"]["commitment_digest"] = "sha256:" + "a" * 64
    elif mutation == "attestation_scope":
        inputs["temporal_attestation"]["scope_digest"] = "a" * 64
    elif mutation == "attestation_commitment":
        inputs["temporal_attestation"]["commitment_digest"] = "a" * 64
    elif mutation == "attestation_subject":
        inputs["temporal_attestation"]["subject_digest"] = "a" * 64
    elif mutation == "commitment":
        inputs["prospective_commitment"]["dataset_id"] = "changed"
    elif mutation == "scope":
        inputs["scope_declaration"]["dataset_id"] = "changed"
    before = copy.deepcopy(inputs)
    result = temporal.verify_independent_temporal_attestation(**inputs)
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert reason in result["reason_codes"]
    assert result["independent_temporal_attestation_verified"] is False
    assert result["comparison_permitted"] is False
    assert inputs == before


def test_invalid_token_is_rejected(monkeypatch):
    monkeypatch.setattr(rfc, "verify_rfc3161_timestamp_token",
                        lambda **kwargs: {"independent_temporal_attestation_verified": False})
    result = temporal.verify_independent_temporal_attestation(**evidence())
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert "RFC3161_TOKEN_INVALID_OR_UNTRUSTED" in result["reason_codes"]


def test_structural_assessment_remains_noncryptographic():
    inputs = evidence()
    result = temporal.assess_temporal_attestation_contract(
        temporal_attestation=inputs["temporal_attestation"],
        temporal_trust_registry=inputs["temporal_trust_registry"],
    )
    assert result["independent_temporal_attestation_verified"] is False
    assert result["reason_codes"] == ["TEMPORAL_VERIFIER_NOT_IMPLEMENTED"]
