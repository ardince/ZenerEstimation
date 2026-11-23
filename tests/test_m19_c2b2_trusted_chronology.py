"""M19-C2-B2-B2-B adversarial verification of signed chronology claims.

Test-only keys and preprovisioned checkpoint anchors do NOT prove real-world time.
"""
import base64
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from zenerestimation.comparison.benchmark_authorization_prospective import (
    build_prospective_benchmark_commitment,
)
from zenerestimation.comparison.benchmark_authorization_chronology import (
    EVENT_VERSION, CHECKPOINT_VERSION, REGISTRY_VERSION,
    event_signing_bytes, checkpoint_signing_bytes, event_chain_digest,
    verify_trusted_authorization_chronology, _canonical, _hash,
)


def _b64(data):
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _fixture():
    commitment = build_prospective_benchmark_commitment(
        benchmark_id="benchmark.410", dataset_id="410",
        forecast_dates=["2025-03-01", "2025-06-01"],
        participants=[
            {"run_id": "run.gru", "assessment_id": "assess.gru", "model_family": "GRU"},
            {"run_id": "run.lstm", "assessment_id": "assess.lstm", "model_family": "LSTM"},
        ],
    )
    audit_key = Ed25519PrivateKey.generate()
    witness_key = Ed25519PrivateKey.generate()
    def public(key):
        return _b64(key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ))
    registry = {
        "registry_version": REGISTRY_VERSION,
        "issuers": {
            "audit.example": {
                "status": "active", "role": "audit_event",
                "keys": {"audit.key": {
                    "algorithm": "Ed25519", "status": "active", "public_key": public(audit_key)
                }},
            },
            "witness.example": {
                "status": "active", "role": "checkpoint_witness",
                "keys": {"witness.key": {
                    "algorithm": "Ed25519", "status": "active", "public_key": public(witness_key)
                }},
            },
        },
        "checkpoint_anchors": {},
    }
    def make_event(eid, etype, seq, time, assessment):
        e = {
            "event_version": EVENT_VERSION, "event_id": eid, "event_type": etype,
            "issuer_id": "audit.example", "key_id": "audit.key",
            "benchmark_id": commitment["benchmark_id"],
            "commitment_digest": commitment["commitment_digest"],
            "assessment_id": assessment, "event_time": time,
            "sequence": seq, "signature": _b64(b"\x00" * 64),
        }
        e["signature"] = _b64(audit_key.sign(event_signing_bytes(e)))
        return e
    approval = make_event("event.approval", "authorization_registered", 1,
                          "2024-01-01T00:00:00Z", None)
    access = [
        make_event("event.access.gru", "protected_holdout_access", 2,
                   "2024-01-02T00:00:00Z", "assess.gru"),
        make_event("event.access.lstm", "protected_holdout_access", 3,
                   "2024-01-02T00:00:01Z", "assess.lstm"),
    ]
    def make_checkpoint(events):
        cp = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_id": "checkpoint.001", "issuer_id": "witness.example",
            "key_id": "witness.key", "log_id": "log.001",
            "last_sequence": events[-1]["sequence"],
            "event_chain_digest": event_chain_digest(events),
            "signature": _b64(b"\x00" * 64),
        }
        cp["signature"] = _b64(witness_key.sign(checkpoint_signing_bytes(cp)))
        registry["checkpoint_anchors"][cp["checkpoint_id"]] = _hash(_canonical(cp))
        return cp
    checkpoint = make_checkpoint([approval, *access])
    return {
        "prospective_commitment": commitment,
        "authorization_registration": approval,
        "protected_holdout_events": access,
        "trust_registry": registry,
        "checkpoint": checkpoint,
    }, make_event, make_checkpoint


def _verify(args):
    return verify_trusted_authorization_chronology(**args)


def test_valid_signed_chain_still_does_not_authorize():
    args, _, _ = _fixture()
    r = _verify(args)
    assert r["status"] == "AUTHORIZATION_REQUIRED"
    assert r["signed_events_verified"] is True
    assert r["trusted_checkpoint_verified"] is True
    assert r["independent_time_provenance_verified"] is False
    assert r["prospective_authorization_verified"] is False
    assert not r["comparison_permitted"] and not r["ranking_permitted"]


@pytest.mark.parametrize("missing", [
    "authorization_registration", "protected_holdout_events", "checkpoint",
])
def test_missing_evidence_fails_closed(missing):
    args, _, _ = _fixture()
    args[missing] = None
    assert _verify(args)["status"] == "AUTHORIZATION_REQUIRED"


@pytest.mark.parametrize("mutate", [
    lambda a: a["authorization_registration"].update(event_time="2023-01-01T00:00:00Z"),
    lambda a: a["authorization_registration"].update(commitment_digest="sha256:" + "f"*64),
    lambda a: a["protected_holdout_events"][0].update(assessment_id="assess.other"),
    lambda a: a["protected_holdout_events"][0].update(signature=_b64(b"\x01"*64)),
    lambda a: a["checkpoint"].update(event_chain_digest="sha256:" + "f"*64),
    lambda a: a["checkpoint"].update(last_sequence=4),
    lambda a: a["checkpoint"].update(issuer_id="audit.example"),
    lambda a: a["trust_registry"]["issuers"]["audit.example"].update(status="revoked"),
    lambda a: a["trust_registry"]["issuers"]["witness.example"].update(status="revoked"),
    lambda a: a["trust_registry"]["checkpoint_anchors"].clear(),
    lambda a: a["protected_holdout_events"].pop(),
    lambda a: a["protected_holdout_events"][0].update(sequence=3),
    lambda a: a["authorization_registration"].update(public_key="attacker"),
    lambda a: a["trust_registry"].update(registry_version="future"),
])
def test_tampering_and_untrusted_sources_rejected(mutate):
    args, _, _ = _fixture()
    mutate(args)
    r = _verify(args)
    assert r["status"] == "AUTHORIZATION_REJECTED"
    assert not r["comparison_permitted"]


@pytest.mark.parametrize("approval_time", [
    "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z",
])
def test_equal_or_late_approval_rejected_even_if_resigned_and_reanchored(approval_time):
    args, make_event, make_checkpoint = _fixture()
    args["authorization_registration"] = make_event(
        "event.approval", "authorization_registered", 1, approval_time, None
    )
    args["checkpoint"] = make_checkpoint(
        [args["authorization_registration"], *args["protected_holdout_events"]]
    )
    r = _verify(args)
    assert r["status"] == "AUTHORIZATION_REJECTED"
    assert r["reasons"] == ["NON_PROSPECTIVE_APPROVAL"]


def test_untrusted_key_substitution_rejected():
    args, _, _ = _fixture()
    args["trust_registry"]["issuers"]["audit.example"]["keys"]["audit.key"]["public_key"] = (
        _b64(b"\x01" * 32)
    )
    assert _verify(args)["status"] == "AUTHORIZATION_REJECTED"


def test_missing_registry_rejected():
    args, _, _ = _fixture()
    args["trust_registry"] = None
    assert _verify(args)["status"] == "AUTHORIZATION_REJECTED"


def test_replay_id_rejected():
    args, _, _ = _fixture()
    args["protected_holdout_events"][0]["event_id"] = "event.approval"
    assert _verify(args)["status"] == "AUTHORIZATION_REJECTED"


def test_deterministic_read_only():
    args, _, _ = _fixture()
    original = deepcopy(args)
    r1, r2 = _verify(args), _verify(args)
    assert r1 == r2
    assert args == original


def test_prospective_commitment_tampering_rejected():
    args, _, _ = _fixture()
    args["prospective_commitment"]["dataset_id"] = "110"
    assert _verify(args)["status"] == "AUTHORIZATION_REJECTED"


def test_domain_separation():
    args, _, _ = _fixture()
    assert event_signing_bytes(args["authorization_registration"]) != checkpoint_signing_bytes(args["checkpoint"])


def test_checkpoint_is_not_independent_time_proof():
    args, _, _ = _fixture()
    result = _verify(args)
    assert result["trusted_checkpoint_verified"]
    assert not result["independent_time_provenance_verified"]
    assert result["status"] != "AUTHORIZED"
