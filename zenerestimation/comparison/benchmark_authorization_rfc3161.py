"""M19-C2-B2-C2-A2: restricted RFC 3161 TimeStampToken verifier.

Trust profile: independently provisioned *exact TSA signer certificate* pin.
No PKI path building, revocation determination, or independent access-log
completeness is claimed. Never authorizes model comparison or ranking.

Requires: asn1crypto, cryptography. Pure, offline, read-only verification.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone

from asn1crypto import cms, tsp
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID

VERSION = "m19_c2b2c2a2_rfc3161_verification_v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_B64 = re.compile(r"[A-Za-z0-9+/]+={0,2}\Z", re.ASCII)


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _result(reasons, *, gen_time=None, imprint_verified=False,
            signature_verified=False, tsa_trusted=False):
    result = {
        "verification_version": VERSION,
        "status": "AUTHORIZATION_REJECTED" if reasons else "AUTHORIZATION_REQUIRED",
        "reason_codes": sorted(set(reasons)) if reasons else ["FURTHER_AUTHORIZATION_EVIDENCE_REQUIRED"],
        "independent_temporal_attestation_verified": not reasons,
        "message_imprint_verified": imprint_verified,
        "timestamp_signature_verified": signature_verified,
        "tsa_certificate_pinned": tsa_trusted,
        "attested_at": gen_time if not reasons else None,
        "certificate_chain_verified": False,  # pinned leaf, not a validated PKI chain
        "revocation_verified": False,
        "prospective_authorization_verified": False,
        "comparison_permitted": False,
        "ranking_permitted": False,
    }
    result["assessment_digest"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _decode_proof(value):
    if type(value) is not str or not value or len(value) > 2_000_000 or _B64.fullmatch(value) is None:
        raise ValueError("invalid proof encoding")
    raw = base64.b64decode(value, validate=True)
    if base64.b64encode(raw).decode("ascii") != value:
        raise ValueError("noncanonical proof encoding")
    return raw


def _cert_policy(cert, at, policy):
    if cert.not_valid_before_utc > at or cert.not_valid_after_utc < at:
        raise ValueError("TSA certificate not valid at genTime")
    eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
    if not eku.critical or list(eku.value) != [ExtendedKeyUsageOID.TIME_STAMPING]:
        raise ValueError("TSA EKU must be critical and exclusively timeStamping")
    ku = cert.extensions.get_extension_for_class(x509.KeyUsage).value
    if not ku.digital_signature:
        raise ValueError("TSA key must allow digital signature")
    if policy.get("require_current_certificate_validity", False):
        # This is a *current-time policy check*, not proof of historical time.
        now = datetime.now(timezone.utc)
        if not (cert.not_valid_before_utc <= now <= cert.not_valid_after_utc):
            raise ValueError("TSA certificate not currently valid")


def _signer_matches(sid, cert):
    if sid.name == "issuer_and_serial_number":
        from asn1crypto import x509 as asn1_x509
        parsed = asn1_x509.Certificate.load(cert.public_bytes(serialization.Encoding.DER))
        info = sid.chosen
        return (info["serial_number"].native == parsed.serial_number
                and info["issuer"].dump() == parsed.issuer.dump())
    if sid.name == "subject_key_identifier":
        try:
            ski = cert.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value.digest
        except x509.ExtensionNotFound:
            return False
        return sid.chosen.native == ski
    return False


def _verify_signature(public_key, signature, signed_attrs, algorithm):
    if isinstance(public_key, rsa.RSAPublicKey) and algorithm == "rsassa_pkcs1v15":
        public_key.verify(signature, signed_attrs, padding.PKCS1v15(), hashes.SHA256())
    elif isinstance(public_key, ec.EllipticCurvePublicKey) and algorithm == "sha256_ecdsa":
        public_key.verify(signature, signed_attrs, ec.ECDSA(hashes.SHA256()))
    else:
        raise ValueError("unsupported TSA signature algorithm")


def verify_rfc3161_timestamp_token(*, proof: str, subject_bytes: bytes,
                                   subject_digest: str, trust_material: Mapping) -> dict:
    """Verify a DER CMS TimeStampToken using a pinned TSA signer certificate.

    `trust_material` must be independently supplied (never from token/envelope):
      {"tsa_certificate_der_b64": canonical-base64 DER,
       "tsa_certificate_sha256": lowercase 64-hex,
       "allowed_policy_oids": [nonempty OID, ...],
       "revocation_policy": "not_established"}

    Only SHA-256 message imprint and SHA-256 RSA PKCS#1 v1.5 / ECDSA signatures
    are supported. A valid token proves authenticated existence no later than
    `genTime` under the pinned TSA trust assumption, not certificate revocation.
    """
    imprint_ok = signature_ok = trusted = False
    try:
        if type(subject_bytes) is not bytes or not subject_bytes:
            raise ValueError("subject bytes invalid")
        if type(subject_digest) is not str or _DIGEST.fullmatch(subject_digest) is None:
            raise ValueError("subject digest invalid")
        if hashlib.sha256(subject_bytes).hexdigest() != subject_digest:
            raise ValueError("subject digest mismatch")
        if type(trust_material) is not dict or set(trust_material) != {
            "tsa_certificate_der_b64", "tsa_certificate_sha256", "allowed_policy_oids", "revocation_policy"
        }:
            raise ValueError("invalid trust material")
        if trust_material["revocation_policy"] != "not_established":
            raise ValueError("unsupported revocation policy")
        oids = trust_material["allowed_policy_oids"]
        if type(oids) is not list or not oids or not all(type(v) is str and re.fullmatch(r"\d+(?:\.\d+)+", v) for v in oids):
            raise ValueError("invalid timestamp policy allowlist")
        cert_der = _decode_proof(trust_material["tsa_certificate_der_b64"])
        pin = trust_material["tsa_certificate_sha256"]
        if type(pin) is not str or _DIGEST.fullmatch(pin) is None or hashlib.sha256(cert_der).hexdigest() != pin:
            raise ValueError("TSA certificate pin mismatch")
        cert = x509.load_der_x509_certificate(cert_der)
        trusted = True
        token_der = _decode_proof(proof)
        token = cms.ContentInfo.load(token_der, strict=True)
        if token["content_type"].native != "signed_data":
            raise ValueError("not CMS SignedData")
        signed = token["content"]
        if signed["encap_content_info"]["content_type"].native != "tst_info":
            raise ValueError("not RFC3161 TSTInfo")
        content = signed["encap_content_info"]["content"]
        if content.native is None:
            raise ValueError("detached timestamp token")
        tst_bytes = content.parsed.dump() if hasattr(content, "parsed") and content.parsed is not None else content.native
        if not isinstance(tst_bytes, bytes):
            raise ValueError("invalid encapsulated content")
        info = tsp.TSTInfo.load(tst_bytes, strict=True)
        if info["message_imprint"]["hash_algorithm"]["algorithm"].native != "sha256":
            raise ValueError("unsupported message imprint hash")
        if info["message_imprint"]["hashed_message"].native != hashlib.sha256(subject_bytes).digest():
            raise ValueError("message imprint mismatch")
        imprint_ok = True
        if info["policy"].dotted not in oids:
            raise ValueError("timestamp policy not allowed")
        if len(signed["signer_infos"]) != 1:
            raise ValueError("ambiguous CMS signer")
        signer = signed["signer_infos"][0]
        if not _signer_matches(signer["sid"], cert):
            raise ValueError("CMS signer not pinned TSA")
        if signer["digest_algorithm"]["algorithm"].native != "sha256":
            raise ValueError("unsupported signer digest")
        attrs = signer["signed_attrs"]
        if attrs.native is None:
            raise ValueError("missing CMS signed attributes")
        by_type = {}
        for attr in attrs:
            key = attr["type"].native
            if key in by_type or len(attr["values"]) != 1:
                raise ValueError("ambiguous signed attribute")
            by_type[key] = attr["values"][0]
        if by_type.get("content_type") is None or by_type["content_type"].native != "tst_info":
            raise ValueError("CMS content type not authenticated")
        if by_type.get("message_digest") is None or by_type["message_digest"].native != hashlib.sha256(tst_bytes).digest():
            raise ValueError("CMS content digest mismatch")
        # RFC 5816 ESSCertIDv2 / RFC 3161 ESSCertID signer binding.
        ess_v2 = by_type.get("signing_certificate_v2")
        ess_v1 = by_type.get("signing_certificate")
        if ess_v2 is None and ess_v1 is None:
            raise ValueError("missing ESS signer certificate binding")
        if ess_v2 is not None:
            parsed_ess = tsp.SigningCertificateV2.load(ess_v2.dump())
            cert_ids = parsed_ess["certs"]
            if len(cert_ids) != 1 or cert_ids[0]["hash_algorithm"]["algorithm"].native != "sha256" or cert_ids[0]["cert_hash"].native != hashlib.sha256(cert_der).digest():
                raise ValueError("ESSCertIDv2 signer mismatch")
        else:
            parsed_ess = tsp.SigningCertificate.load(ess_v1.dump())
            cert_ids = parsed_ess["certs"]
            if len(cert_ids) != 1 or cert_ids[0]["cert_hash"].native != hashlib.sha1(cert_der).digest():
                raise ValueError("ESSCertID signer mismatch")
        signed_der = attrs.untag().dump()
        _verify_signature(cert.public_key(), signer["signature"].native,
                          signed_der, signer["signature_algorithm"]["algorithm"].native)
        signature_ok = True
        at = info["gen_time"].native
        if at.tzinfo is None:
            raise ValueError("genTime missing timezone")
        at = at.astimezone(timezone.utc)
        _cert_policy(cert, at, trust_material)
        # Preserve fractional seconds, if present; no claim of sub-second accuracy.
        timestamp = at.isoformat().replace("+00:00", "Z")
        return _result([], gen_time=timestamp, imprint_verified=True,
                       signature_verified=True, tsa_trusted=True)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError,
            InvalidSignature, x509.ExtensionNotFound, OverflowError) as exc:
        return _result(["RFC3161_TOKEN_INVALID_OR_UNTRUSTED"],
                       imprint_verified=imprint_ok, signature_verified=signature_ok,
                       tsa_trusted=trusted)
