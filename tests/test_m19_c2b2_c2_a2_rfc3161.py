"""A2 cryptographic tests using actual OpenSSL-generated RFC3161 tokens.

Requires openssl executable, asn1crypto, cryptography. No mocked CMS signatures.
"""
import base64
import hashlib
import shutil
import subprocess
from copy import deepcopy

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from zenerestimation.comparison.benchmark_authorization_rfc3161 import (
    verify_rfc3161_timestamp_token,
)


def _openssl(*args, cwd):
    result = subprocess.run(["openssl", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"OpenSSL failed: {result.stderr}")


@pytest.fixture(scope="module")
def timestamp(tmp_path_factory):
    if not shutil.which("openssl"):
        pytest.skip("OpenSSL CLI unavailable")
    path = tmp_path_factory.mktemp("rfc3161")
    _openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "key.pem",
             "-out", "cert.pem", "-days", "2", "-subj", "/CN=A2 Local Test TSA",
             "-addext", "extendedKeyUsage=critical,timeStamping",
             "-addext", "keyUsage=critical,digitalSignature", cwd=path)
    subject = b"canonical signed registration fixture\n"
    (path / "subject.bin").write_bytes(subject)
    _openssl("ts", "-query", "-data", "subject.bin", "-sha256", "-cert", "-out", "request.tsq", cwd=path)
    # Provide a serial file to prevent ts -reply warning/failure on some OpenSSL versions.
    (path / "serial").write_text("01\n")


    # OpenSSL configuration files require forward-slash paths on Windows.
    # pathlib.Path.as_posix() provides a portable representation.
    openssl_dir = path.as_posix()

    (path / "ts.cnf").write_text(
        "[ tsa ]\ndefault_tsa = tsa_config\n[ tsa_config ]\n"
        f"dir = {openssl_dir}\nserial = $dir/serial\ncrypto_device = builtin\n"
        "signer_cert = $dir/cert.pem\nsigner_key = $dir/key.pem\n"
        "signer_digest = sha256\ndefault_policy = 1.2.3.4\n"
        "digests = sha256\naccuracy = secs:1\nordering = yes\n"
        "tsa_name = yes\ness_cert_id_chain = no\n"
    )

    _openssl("ts", "-reply", "-config", "ts.cnf", "-queryfile", "request.tsq", "-out", "response.tsr", cwd=path)
    _openssl("ts", "-reply", "-in", "response.tsr", "-token_out", "-out", "token.der", cwd=path)
    cert = x509.load_pem_x509_certificate((path / "cert.pem").read_bytes())
    der = cert.public_bytes(serialization.Encoding.DER)
    return dict(
        proof=base64.b64encode((path / "token.der").read_bytes()).decode(),
        subject_bytes=subject,
        subject_digest=hashlib.sha256(subject).hexdigest(),
        trust_material={
            "tsa_certificate_der_b64": base64.b64encode(der).decode(),
            "tsa_certificate_sha256": hashlib.sha256(der).hexdigest(),
            "allowed_policy_oids": ["1.2.3.4"],
            "revocation_policy": "not_established",
        },
    )


def test_real_rfc3161_token_verified_without_authorization(timestamp):
    result = verify_rfc3161_timestamp_token(**timestamp)
    assert result["independent_temporal_attestation_verified"] is True, result
    assert result["timestamp_signature_verified"] is True
    assert result["message_imprint_verified"] is True
    assert result["certificate_chain_verified"] is False
    assert result["revocation_verified"] is False
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert result["prospective_authorization_verified"] is False
    assert result["comparison_permitted"] is False
    assert result["ranking_permitted"] is False


@pytest.mark.parametrize("change", [
    "subject", "digest", "token", "pin", "policy", "revocation", "cert",
])
def test_adversarial_fail_closed(timestamp, change):
    data = deepcopy(timestamp)
    if change == "subject":
        data["subject_bytes"] += b"tampered"
    elif change == "digest":
        data["subject_digest"] = "0" * 64
    elif change == "token":
        raw = bytearray(base64.b64decode(data["proof"]))
        raw[-1] ^= 1
        data["proof"] = base64.b64encode(raw).decode()
    elif change == "pin":
        data["trust_material"]["tsa_certificate_sha256"] = "0" * 64
    elif change == "policy":
        data["trust_material"]["allowed_policy_oids"] = ["1.2.3.5"]
    elif change == "revocation":
        data["trust_material"]["revocation_policy"] = "verified"
    elif change == "cert":
        data["trust_material"]["tsa_certificate_der_b64"] = base64.b64encode(b"forged").decode()
    result = verify_rfc3161_timestamp_token(**data)
    assert result["status"] == "AUTHORIZATION_REJECTED", (change, result)
    assert result["independent_temporal_attestation_verified"] is False
    assert result["comparison_permitted"] is False
    assert result["ranking_permitted"] is False


def test_deterministic_read_only(timestamp):
    original = deepcopy(timestamp)
    first = verify_rfc3161_timestamp_token(**timestamp)
    second = verify_rfc3161_timestamp_token(**timestamp)
    assert first == second
    assert timestamp == original
