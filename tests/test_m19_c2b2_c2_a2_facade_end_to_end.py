"""Real RFC 3161 timestamp through the A2 public facade (no crypto mocks).

Requires the OpenSSL CLI plus asn1crypto and cryptography. Uses an ephemeral
self-signed TSA certificate pinned explicitly by this test's trust registry;
this is not a production PKI validation or authorization approval.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import shutil
import subprocess

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from zenerestimation.comparison.benchmark_authorization import (
    build_benchmark_scope_declaration,
)
from zenerestimation.comparison.benchmark_authorization_chronology import EVENT_VERSION
from zenerestimation.comparison.benchmark_authorization_prospective import (
    build_prospective_benchmark_commitment,
)
from zenerestimation.comparison.benchmark_authorization_temporal import (
    ATTESTATION_VERSION,
    TRUST_REGISTRY_VERSION,
    registration_timestamp_subject_bytes,
    verify_independent_temporal_attestation,
)


def _openssl(cwd, *args):
    result = subprocess.run(
        ["openssl", *args], cwd=cwd, capture_output=True, text=True,
        check=False,
    )
    if result.returncode:
        pytest.fail(f"OpenSSL {' '.join(args[:2])} failed: {result.stderr}")


@pytest.fixture(scope="module")
def genuine_facade_evidence(tmp_path_factory):
    if not shutil.which("openssl"):
        pytest.skip("OpenSSL CLI unavailable")

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
        participants=[
            {**p, "benchmark_entry_digest": "sha256:" + "1" * 64}
            for p in participants
        ],
    )
    # This is a syntactically valid registration, NOT a verified Ed25519
    # signature. A2 does not claim to authenticate the audit-event signer.
    registration = {
        "event_version": EVENT_VERSION,
        "event_id": "ev1",
        "event_type": "authorization_registered",
        "issuer_id": "issuer1",
        "key_id": "key1",
        "benchmark_id": "b1",
        "commitment_digest": commitment["commitment_digest"],
        "assessment_id": None,
        "event_time": "2025-01-01T00:00:00Z",
        "sequence": 1,
        "signature": base64.urlsafe_b64encode(b"s" * 64).decode().rstrip("="),
    }
    subject = registration_timestamp_subject_bytes(registration)

    path = tmp_path_factory.mktemp("a2-facade-rfc3161")
    _openssl(path, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
             "-keyout", "key.pem", "-out", "cert.pem", "-days", "2",
             "-subj", "/CN=A2 Facade Test TSA",
             "-addext", "extendedKeyUsage=critical,timeStamping",
             "-addext", "keyUsage=critical,digitalSignature")
    (path / "registration.bin").write_bytes(subject)
    _openssl(path, "ts", "-query", "-data", "registration.bin", "-sha256",
             "-cert", "-out", "request.tsq")
    (path / "serial").write_text("01\n", encoding="ascii")
    # OpenSSL config must use POSIX-style paths, including on Windows.
    (path / "ts.cnf").write_text(
        "[ tsa ]\ndefault_tsa = tsa_config\n[ tsa_config ]\n"
        f"dir = {path.as_posix()}\nserial = $dir/serial\ncrypto_device = builtin\n"
        "signer_cert = $dir/cert.pem\nsigner_key = $dir/key.pem\n"
        "signer_digest = sha256\ndefault_policy = 1.2.3.4\n"
        "digests = sha256\naccuracy = secs:1\nordering = yes\n"
        "tsa_name = yes\ness_cert_id_chain = no\n",
        encoding="ascii",
    )
    _openssl(path, "ts", "-reply", "-config", "ts.cnf", "-queryfile",
             "request.tsq", "-out", "response.tsr")
    _openssl(path, "ts", "-reply", "-in", "response.tsr", "-token_out",
             "-out", "token.der")

    cert = x509.load_pem_x509_certificate((path / "cert.pem").read_bytes())
    cert_der = cert.public_bytes(serialization.Encoding.DER)
    trust_material = {
        "tsa_certificate_der_b64": base64.b64encode(cert_der).decode("ascii"),
        "tsa_certificate_sha256": hashlib.sha256(cert_der).hexdigest(),
        "allowed_policy_oids": ["1.2.3.4"],
        "revocation_policy": "not_established",
    }
    attestation = {
        "attestation_version": ATTESTATION_VERSION,
        "attestation_id": "tsa1",
        "mechanism": "rfc3161",
        "authority_id": "tsa1",
        "subject_digest": hashlib.sha256(subject).hexdigest(),
        "scope_digest": scope["scope_digest"].removeprefix("sha256:"),
        "commitment_digest": commitment["commitment_digest"].removeprefix("sha256:"),
        "proof": base64.b64encode((path / "token.der").read_bytes()).decode("ascii"),
    }
    registry = {
        "registry_version": TRUST_REGISTRY_VERSION,
        "authorities": {
            "tsa1": {"mechanisms": ["rfc3161"], "trust_material": trust_material},
        },
    }
    return {
        "temporal_attestation": attestation,
        "authorization_registration": registration,
        "prospective_commitment": commitment,
        "scope_declaration": scope,
        "temporal_trust_registry": registry,
    }


def _verify(evidence):
    return verify_independent_temporal_attestation(**evidence)


def test_genuine_rfc3161_token_passes_public_facade_without_authorization(genuine_facade_evidence):
    evidence = genuine_facade_evidence
    original = copy.deepcopy(evidence)
    result = _verify(evidence)
    assert result["status"] == "AUTHORIZATION_REQUIRED", result
    assert result["independent_temporal_attestation_verified"] is True
    assert result["timestamped_registration_bound"] is True
    assert result["scope_binding_verified"] is True
    assert result["commitment_binding_verified"] is True
    assert result["attested_at"] is not None
    assert result["registration_signature_verified"] is False
    assert result["independent_access_chronology_verified"] is False
    assert result["prospective_authorization_verified"] is False
    assert result["comparison_permitted"] is False
    assert result["ranking_permitted"] is False
    assert result == _verify(evidence)
    assert evidence == original


def test_one_byte_registration_signature_change_rejected(genuine_facade_evidence):
    evidence = copy.deepcopy(genuine_facade_evidence)
    # Alter the signed registration's signature bytes, preserving valid encoding.
    signature = base64.urlsafe_b64decode(
        evidence["authorization_registration"]["signature"] + "=="
    )
    changed = bytes([signature[0] ^ 1]) + signature[1:]
    evidence["authorization_registration"]["signature"] = (
        base64.urlsafe_b64encode(changed).decode("ascii").rstrip("=")
    )
    result = _verify(evidence)
    assert result["status"] == "AUTHORIZATION_REJECTED", result
    assert "ATTESTATION_SUBJECT_DIGEST_MISMATCH" in result["reason_codes"]
    assert result["independent_temporal_attestation_verified"] is False
    assert result["comparison_permitted"] is False
    assert result["ranking_permitted"] is False


def test_recomputed_digest_cannot_reuse_original_token(genuine_facade_evidence):
    evidence = copy.deepcopy(genuine_facade_evidence)
    evidence["authorization_registration"]["event_id"] = "ev2"
    # Recompute the public envelope digest: binding checks will pass, but the
    # original RFC3161 token's signed message imprint must still fail.
    subject = registration_timestamp_subject_bytes(evidence["authorization_registration"])
    evidence["temporal_attestation"]["subject_digest"] = hashlib.sha256(subject).hexdigest()
    result = _verify(evidence)
    assert result["status"] == "AUTHORIZATION_REJECTED", result
    assert "RFC3161_TOKEN_INVALID_OR_UNTRUSTED" in result["reason_codes"]
    assert result["independent_temporal_attestation_verified"] is False
    assert result["comparison_permitted"] is False
    assert result["ranking_permitted"] is False
