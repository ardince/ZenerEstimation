"""Standalone A1 contract tests; no cryptographic verification is claimed."""

import copy
import pytest

from zenerestimation.comparison.benchmark_authorization_temporal import (
    ATTESTATION_VERSION, TRUST_REGISTRY_VERSION,
    assess_temporal_attestation_contract, verify_independent_temporal_attestation,
)


def fixture():
    att = {
        "attestation_version": ATTESTATION_VERSION,
        "attestation_id": "tsa-receipt-1",
        "mechanism": "rfc3161",
        "authority_id": "external-tsa-1",
        "subject_digest": "a" * 64,
        "scope_digest": "b" * 64,
        "commitment_digest": "c" * 64,
        "proof": "opaque-token-placeholder",
    }
    registry = {
        "registry_version": TRUST_REGISTRY_VERSION,
        "authorities": {"external-tsa-1": {
            "mechanisms": ["rfc3161"],
            "trust_material": {"certificates": ["opaque-placeholder"]},
        }},
    }
    return att, registry


def assess(att, registry):
    return assess_temporal_attestation_contract(
        temporal_attestation=att, temporal_trust_registry=registry)


def test_well_formed_envelope_is_not_proof():
    att, registry = fixture()
    result = assess(att, registry)
    assert result["contract_valid"] is True
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert result["reason_codes"] == ["TEMPORAL_VERIFIER_NOT_IMPLEMENTED"]
    assert result["independent_temporal_attestation_verified"] is False
    assert result["prospective_authorization_verified"] is False
    assert result["comparison_permitted"] is False
    assert result["ranking_permitted"] is False


@pytest.mark.parametrize("field,value,reason", [
    ("mechanism", "self_signed", "TEMPORAL_MECHANISM_UNSUPPORTED"),
    ("subject_digest", "not-a-digest", "SUBJECT_DIGEST_INVALID"),
    ("scope_digest", "A" * 64, "SCOPE_DIGEST_INVALID"),
    ("commitment_digest", "", "COMMITMENT_DIGEST_INVALID"),
    ("attestation_version", "unknown", "TEMPORAL_ATTESTATION_VERSION_UNSUPPORTED"),
    ("proof", "", "PROOF_INVALID"),
])
def test_invalid_envelopes_rejected(field, value, reason):
    att, registry = fixture()
    att[field] = value
    result = assess(att, registry)
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert reason in result["reason_codes"]
    assert not result["independent_temporal_attestation_verified"]


def test_missing_attestation_and_registry_rejected():
    assert assess(None, None)["status"] == "AUTHORIZATION_REJECTED"


def test_unknown_authority_rejected():
    att, registry = fixture()
    att["authority_id"] = "attacker"
    assert "TEMPORAL_AUTHORITY_UNTRUSTED" in assess(att, registry)["reason_codes"]


def test_authority_mechanism_policy_enforced():
    att, registry = fixture()
    registry["authorities"]["external-tsa-1"]["mechanisms"] = ["other"]
    assert "TEMPORAL_AUTHORITY_MECHANISM_NOT_ALLOWED" in assess(att, registry)["reason_codes"]


def test_deterministic_read_only():
    att, registry = fixture()
    before = copy.deepcopy((att, registry))
    assert assess(att, registry) == assess(att, registry)
    assert (att, registry) == before


def test_public_facade_cannot_grant_authorization():
    att, registry = fixture()

    result = verify_independent_temporal_attestation(
        temporal_attestation=att,
        temporal_trust_registry=registry,
        authorization_registration={"example": "record"},
        prospective_commitment={"example": "record"},
        scope_declaration={"example": "record"},
    )

    # A2 rejects records that are structurally valid JSON
    # but invalid under the actual authorization contracts.
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert "AUTHORIZATION_REGISTRATION_INVALID" in result["reason_codes"]

    # Neither an A1 envelope nor invalid source records
    # can establish independent temporal verification.
    assert result["independent_temporal_attestation_verified"] is False

    # Authorization must never be inferred from this facade.
    assert result["prospective_authorization_verified"] is False
    assert result["comparison_permitted"] is False
    assert result["ranking_permitted"] is False

def test_public_facade_rejects_missing_source_record():
    att, registry = fixture()
    result = verify_independent_temporal_attestation(
        temporal_attestation=att, temporal_trust_registry=registry,
        authorization_registration=None,
        prospective_commitment={}, scope_declaration={},
    )
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert "AUTHORIZATION_REGISTRATION_INVALID" in result["reason_codes"]
