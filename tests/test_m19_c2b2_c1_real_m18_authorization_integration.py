"""M19-C2-B2-C1 real-verifier integration using one genuine M18 GRU chain.

The second LSTM chain is synthetic, not an independently executed M18 model.
Test-issued keys and checkpoint anchors do not establish independently trusted time.
"""
from __future__ import annotations

from copy import deepcopy
import re

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from tests.test_m19_c2b1_real_m18_integration import mixed_chain, _scope, _no_execution
from tests.test_m19_c2b2_crypto_verification import b64
from zenerestimation.comparison.benchmark_authorization_crypto import (
    TRUST_REGISTRY_VERSION, authorization_signing_bytes,
)

from tests.test_m19_c1_real_m18_integration import genuine_chain  # noqa: F401

from zenerestimation.comparison.benchmark_authorization_evidence import EVIDENCE_VERSION
from zenerestimation.comparison.benchmark_authorization_prospective import (
    build_prospective_benchmark_commitment,
)
from zenerestimation.comparison.benchmark_authorization_chronology import (
    EVENT_VERSION, CHECKPOINT_VERSION, REGISTRY_VERSION,
    event_signing_bytes, checkpoint_signing_bytes, event_chain_digest,
    _canonical, _hash,
)
from zenerestimation.comparison.benchmark_authorization_integration import (
    assess_independent_benchmark_authorization,
)

from tests.test_m18_forecast_plan_contract import (
    training as processed_training,
)

from tests.test_m19_b3_b1_m18_real_integration import (
    declaration,
    full_frame,
)

from tests.test_m19_c1_real_m18_integration import genuine_chain
from tests.test_m19_c2b1_real_m18_integration import mixed_chain


def _real_inputs(mixed_chain):
    pair, events = mixed_chain
    scope = _scope(pair)
    participants = [
        {k: p[k] for k in ("run_id", "assessment_id", "model_family")}
        for p in scope["participants"]
    ]
    commitment = build_prospective_benchmark_commitment(
        benchmark_id=scope["benchmark_id"], dataset_id=scope["dataset_id"],
        forecast_dates=scope["forecast_dates"], participants=participants,
    )
    return pair, events, scope, commitment


def _signed_evidence(scope):
    key = Ed25519PrivateKey.generate()
    pub = b64(key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw,
    ))
    registry = {"registry_version": TRUST_REGISTRY_VERSION, "issuers": {
        "trusted.example": {"status": "active", "keys": {
            "key.001": {"algorithm": "Ed25519", "status": "active", "public_key": pub}
        }}
    }}
    evidence = {
        "evidence_version": EVIDENCE_VERSION, "authorization_id": "auth.001",
        "issuer_id": "trusted.example", "key_id": "key.001", "algorithm": "Ed25519",
        "purpose": "cross_model_benchmark", "authorized_operation": "compare_and_rank",
        "scope_digest": scope["scope_digest"],
        "issued_at": "2023-01-01T00:00:00Z",
        "not_before": "2023-01-01T00:00:01Z",
        "expires_at": "2027-01-01T00:00:00Z", "signature": b64(bytes(64)),
    }
    evidence["signature"] = b64(key.sign(authorization_signing_bytes(evidence)))
    return registry, evidence


def _signed_chronology(commitment, *, approval_time="2024-01-01T00:00:00Z"):
    audit = Ed25519PrivateKey.generate()
    witness = Ed25519PrivateKey.generate()

    def pub(key):
        return b64(key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw,
        ))

    registry = {"registry_version": REGISTRY_VERSION, "issuers": {
        "audit.example": {"status": "active", "role": "audit_event", "keys": {
            "audit.key": {"algorithm": "Ed25519", "status": "active", "public_key": pub(audit)}
        }},
        "witness.example": {"status": "active", "role": "checkpoint_witness", "keys": {
            "witness.key": {"algorithm": "Ed25519", "status": "active", "public_key": pub(witness)}
        }},
    }, "checkpoint_anchors": {}}

    def event(event_id, event_type, seq, timestamp, assessment_id):
        record = {
            "event_version": EVENT_VERSION, "event_id": event_id,
            "event_type": event_type, "issuer_id": "audit.example", "key_id": "audit.key",
            "benchmark_id": commitment["benchmark_id"],
            "commitment_digest": commitment["commitment_digest"],
            "assessment_id": assessment_id, "event_time": timestamp,
            "sequence": seq, "signature": b64(bytes(64)),
        }
        record["signature"] = b64(audit.sign(event_signing_bytes(record)))
        return record

    approval = event("event.approval", "authorization_registered", 1, approval_time, None)
    accesses = [event(f"event.access.{i}", "protected_holdout_access", i + 2,
                      "2024-01-02T00:00:00Z", p["assessment_id"])
                for i, p in enumerate(commitment["participants"])]
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION, "checkpoint_id": "checkpoint.001",
        "issuer_id": "witness.example", "key_id": "witness.key", "log_id": "log.001",
        "last_sequence": accesses[-1]["sequence"],
        "event_chain_digest": event_chain_digest([approval, *accesses]),
        "signature": b64(bytes(64)),
    }
    checkpoint["signature"] = b64(witness.sign(checkpoint_signing_bytes(checkpoint)))
    registry["checkpoint_anchors"][checkpoint["checkpoint_id"]] = _hash(_canonical(checkpoint))
    return registry, approval, accesses, checkpoint


@pytest.fixture
def integrated_real_chain(mixed_chain):
    pair, events, scope, commitment = _real_inputs(mixed_chain)
    signature_registry, evidence = _signed_evidence(scope)
    chronology_registry, approval, accesses, checkpoint = _signed_chronology(commitment)
    args = dict(
        inputs=pair, scope_declaration=scope, authorization_evidence=evidence,
        prospective_commitment=commitment, authorization_registration=approval,
        protected_holdout_events=accesses, signature_trust_registry=signature_registry,
        chronology_trust_registry=chronology_registry, checkpoint=checkpoint,
    )
    return args, events


def _assert_no_authorization(result):
    assert result["status"] != "AUTHORIZED"
    assert result["authorization"] == "not_established"
    assert result["comparison_permitted"] is False
    assert result["ranking_permitted"] is False
    assert result["independent_time_provenance_verified"] is False
    assert result["access_completeness_verified"] is False
    assert result["prospective_authorization_verified"] is False


def test_real_m18_signed_chain_is_valid_but_not_authorized(integrated_real_chain):
    args, events = integrated_real_chain
    before = deepcopy({k: v for k, v in args.items() if k != "inputs"})
    before_entries = [deepcopy(p.entry) for p in args["inputs"]]
    result = assess_independent_benchmark_authorization(**args)
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert result["scientific_alignment_verified"] is True
    assert result["scope_binding_verified"] is True
    assert result["signature_verified"] is True
    assert result["issuer_trusted"] is True
    assert result["commitment_binding_verified"] is True
    assert result["signed_events_verified"] is True
    assert result["trusted_checkpoint_verified"] is True
    assert all(s["status"] == "AUTHORIZATION_REQUIRED" for s in result["stages"].values())
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", result["decision_digest"])
    assert result == assess_independent_benchmark_authorization(**args)
    assert before == {k: v for k, v in args.items() if k != "inputs"}
    assert before_entries == [p.entry for p in args["inputs"]]
    _assert_no_authorization(result)
    _no_execution(events)


def test_real_m18_invalid_signature_rejected(integrated_real_chain):
    args, events = integrated_real_chain
    args["authorization_evidence"]["signature"] = b64(bytes(64))
    result = assess_independent_benchmark_authorization(**args)
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert result["signature_verified"] is False
    _assert_no_authorization(result)
    _no_execution(events)


def test_real_m18_signed_but_retrospective_chronology_rejected(integrated_real_chain):
    args, events = integrated_real_chain
    registry, approval, accesses, checkpoint = _signed_chronology(
        args["prospective_commitment"], approval_time="2024-01-02T00:00:00Z",
    )
    args.update(chronology_trust_registry=registry, authorization_registration=approval,
                protected_holdout_events=accesses, checkpoint=checkpoint)
    result = assess_independent_benchmark_authorization(**args)
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert "NON_PROSPECTIVE_APPROVAL" in result["stages"]["signed_chronology"]["reasons"]
    _assert_no_authorization(result)
    _no_execution(events)


def test_real_m18_missing_checkpoint_remains_required(integrated_real_chain):
    args, events = integrated_real_chain
    args["checkpoint"] = None
    result = assess_independent_benchmark_authorization(**args)
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert result["trusted_checkpoint_verified"] is False
    _assert_no_authorization(result)
    _no_execution(events)


def test_real_m18_cross_entry_tampering_rejected(integrated_real_chain):
    args, events = integrated_real_chain
    entry = deepcopy(args["inputs"][1].entry)
    entry["metrics"]["rmse_microVolt"] = 999.0
    from zenerestimation.comparison.benchmark_comparability import VerifiedBenchmarkInput
    original = args["inputs"][1]
    args["inputs"] = [args["inputs"][0], VerifiedBenchmarkInput(
        entry=entry, declaration=original.declaration,
        eligibility_snapshot=original.eligibility_snapshot,
        disclosure_snapshot=original.disclosure_snapshot,
        scoring_snapshot=original.scoring_snapshot,
        evaluation_evidence=original.evaluation_evidence,
    )]
    result = assess_independent_benchmark_authorization(**args)
    assert result["status"] == "AUTHORIZATION_REJECTED"
    _assert_no_authorization(result)
    _no_execution(events)
