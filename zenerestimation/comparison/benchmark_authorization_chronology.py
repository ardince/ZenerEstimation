"""M19-C2-B2-B2-B: signed audit-event chronology, fail-closed.

A signed timestamp is an assertion, NOT independently proven event time.
This module requires a separately provisioned, authenticated checkpoint
covering the event sequence, and STILL never grants benchmark authorization.
Production use must establish checkpoint provenance and its temporal anchoring
outside the event producer before relying on prospective claims.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from zenerestimation.comparison.benchmark_authorization_prospective import (
    verify_prospective_benchmark_commitment,
)

EVENT_VERSION = "m19_c2b2b2b_audit_event_v1"
CHECKPOINT_VERSION = "m19_c2b2b2b_checkpoint_v1"
REGISTRY_VERSION = "m19_c2b2b2b_audit_trust_v1"
DECISION_VERSION = "m19_c2b2b2b_chronology_decision_v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z", re.ASCII)
_B64 = re.compile(r"[A-Za-z0-9_-]+\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_TIME = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\Z", re.ASCII)
_EVENT_FIELDS = frozenset({
    "event_version", "event_id", "event_type", "issuer_id", "key_id",
    "benchmark_id", "commitment_digest", "assessment_id", "event_time",
    "sequence", "signature",
})
_CHECKPOINT_FIELDS = frozenset({
    "checkpoint_version", "checkpoint_id", "issuer_id", "key_id",
    "log_id", "last_sequence", "event_chain_digest", "signature",
})
_REGISTRY_FIELDS = frozenset({"registry_version", "issuers", "checkpoint_anchors"})
_EVENT_TYPES = frozenset({"authorization_registered", "protected_holdout_access"})


def _canonical(obj: Mapping) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _hash(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _signed_bytes(domain: str, record: Mapping) -> bytes:
    return domain.encode("ascii") + b"\x00" + _canonical({
        k: v for k, v in record.items() if k != "signature"
    })


def event_signing_bytes(event: Mapping) -> bytes:
    """Stable, domain-separated bytes for independently signed audit events."""
    _validate_event(event, verify_signature=False)
    return _signed_bytes("ZenerEstimation/M19-C2-B2-B2-B/event/v1", event)


def checkpoint_signing_bytes(checkpoint: Mapping) -> bytes:
    _validate_checkpoint(checkpoint, verify_signature=False)
    return _signed_bytes("ZenerEstimation/M19-C2-B2-B2-B/checkpoint/v1", checkpoint)


def _id(v: object) -> bool:
    return type(v) is str and _ID.fullmatch(v) is not None


def _digest(v: object) -> bool:
    return type(v) is str and _DIGEST.fullmatch(v) is not None


def _decode(v: object, size: int) -> bytes:
    if type(v) is not str or _B64.fullmatch(v) is None or "=" in v:
        raise ValueError("noncanonical base64url")
    try:
        raw = base64.urlsafe_b64decode(v + "=" * (-len(v) % 4))
    except (ValueError, Exception) as exc:
        raise ValueError("invalid base64url") from exc
    if len(raw) != size or base64.urlsafe_b64encode(raw).decode().rstrip("=") != v:
        raise ValueError("invalid encoded length or canonical form")
    return raw


def _time(v: object) -> datetime:
    if type(v) is not str or _TIME.fullmatch(v) is None:
        raise ValueError("timestamp must be canonical UTC")
    try:
        dt = datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("invalid timestamp") from exc
    if dt.strftime("%Y-%m-%dT%H:%M:%SZ") != v:
        raise ValueError("noncanonical timestamp")
    return dt


def _validate_event(v: Mapping, *, verify_signature: bool = True) -> None:
    if type(v) is not dict or set(v) != _EVENT_FIELDS:
        raise ValueError("invalid event fields")
    if v["event_version"] != EVENT_VERSION or v["event_type"] not in _EVENT_TYPES:
        raise ValueError("invalid event version/type")
    if not all(_id(v[k]) for k in ("event_id", "issuer_id", "key_id", "benchmark_id")):
        raise ValueError("invalid event identifier")
    if not _digest(v["commitment_digest"]):
        raise ValueError("invalid commitment digest")
    if v["event_type"] == "authorization_registered":
        if v["assessment_id"] is not None:
            raise ValueError("authorization event must not name assessment")
    elif not _id(v["assessment_id"]):
        raise ValueError("access event must name assessment")
    _time(v["event_time"])
    if type(v["sequence"]) is not int or v["sequence"] < 1:
        raise ValueError("invalid event sequence")
    if verify_signature:
        _decode(v["signature"], 64)


def _validate_checkpoint(v: Mapping, *, verify_signature: bool = True) -> None:
    if type(v) is not dict or set(v) != _CHECKPOINT_FIELDS:
        raise ValueError("invalid checkpoint fields")
    if v["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError("invalid checkpoint version")
    if not all(_id(v[k]) for k in ("checkpoint_id", "issuer_id", "key_id", "log_id")):
        raise ValueError("invalid checkpoint identifier")
    if type(v["last_sequence"]) is not int or v["last_sequence"] < 1:
        raise ValueError("invalid checkpoint sequence")
    if not _digest(v["event_chain_digest"]):
        raise ValueError("invalid checkpoint digest")
    if verify_signature:
        _decode(v["signature"], 64)


def event_chain_digest(events: Sequence[Mapping]) -> str:
    """Digest of the complete ordered event records, including signatures."""
    if type(events) not in (list, tuple) or not events:
        raise ValueError("empty event chain")
    for e in events:
        _validate_event(e)
    return _hash(_canonical({"events": list(events)}))


def _trusted_key(registry: Mapping, issuer: str, key: str, *, role: str) -> Ed25519PublicKey:
    issuers = registry["issuers"]
    record = issuers.get(issuer)
    if type(record) is not dict or set(record) != {"status", "role", "keys"}:
        raise ValueError("untrusted issuer")
    if record["status"] != "active" or record["role"] != role:
        raise ValueError("untrusted issuer role/status")
    keys = record["keys"]
    if type(keys) is not dict:
        raise ValueError("invalid key registry")
    k = keys.get(key)
    if type(k) is not dict or set(k) != {"algorithm", "status", "public_key"}:
        raise ValueError("untrusted key")
    if k["algorithm"] != "Ed25519" or k["status"] != "active":
        raise ValueError("untrusted key status/algorithm")
    return Ed25519PublicKey.from_public_bytes(_decode(k["public_key"], 32))


def _verify_record(record: Mapping, registry: Mapping, *, role: str, domain: str) -> None:
    key = _trusted_key(registry, record["issuer_id"], record["key_id"], role=role)
    key.verify(_decode(record["signature"], 64), _signed_bytes(domain, record))


def _result(status: str, reasons: list[str], *, commitment_digest: str | None = None,
            verified: bool = False) -> dict:
    result = {
        "decision_version": DECISION_VERSION,
        "status": status, "reasons": reasons,
        "commitment_digest": commitment_digest,
        "signed_events_verified": verified,
        "trusted_checkpoint_verified": verified,
        "independent_time_provenance_verified": False,
        "prospective_authorization_verified": False,
        "authorization": "not_established",
        "authentication": "not_established",
        "comparison_permitted": False, "ranking_permitted": False,
    }
    result["decision_digest"] = _hash(_canonical(result))
    return result


def verify_trusted_authorization_chronology(
    *, prospective_commitment: Mapping, authorization_registration: Mapping | None,
    protected_holdout_events: Sequence[Mapping] | None,
    trust_registry: Mapping | None,
    checkpoint: Mapping | None = None,
) -> dict:
    """Verify cryptographic chronology claims; never grant authorization.

    The checkpoint digest MUST be independently provisioned in trust_registry
    (checkpoint_anchors), not copied from the untrusted checkpoint argument.
    Even then, the checkpoint's real-world publication time is not established
    by this function, so prospective authorization remains unestablished.
    """
    cd = prospective_commitment.get("commitment_digest") if isinstance(prospective_commitment, Mapping) else None
    if not verify_prospective_benchmark_commitment(prospective_commitment):
        return _result("AUTHORIZATION_REJECTED", ["INVALID_COMMITMENT"])
    if authorization_registration is None or protected_holdout_events is None or checkpoint is None:
        return _result("AUTHORIZATION_REQUIRED", ["MISSING_CHRONOLOGY_EVIDENCE"], commitment_digest=cd)
    try:
        if type(trust_registry) is not dict or set(trust_registry) != _REGISTRY_FIELDS:
            raise ValueError("invalid trust registry")
        if trust_registry["registry_version"] != REGISTRY_VERSION:
            raise ValueError("invalid registry version")
        if type(trust_registry["checkpoint_anchors"]) is not dict:
            raise ValueError("invalid checkpoint anchors")
        _validate_event(authorization_registration)
        if authorization_registration["event_type"] != "authorization_registered":
            raise ValueError("incorrect authorization event type")
        if type(protected_holdout_events) not in (list, tuple) or not protected_holdout_events:
            raise ValueError("missing holdout events")
        for event in protected_holdout_events:
            _validate_event(event)
            if event["event_type"] != "protected_holdout_access":
                raise ValueError("incorrect access event type")
        events = [authorization_registration, *protected_holdout_events]
        if len({e["event_id"] for e in events}) != len(events):
            raise ValueError("replayed event ID")
        seq = [e["sequence"] for e in events]
        if seq != sorted(seq) or len(set(seq)) != len(seq):
            raise ValueError("inconsistent event order")
        benchmark = prospective_commitment["benchmark_id"]
        if any(e["benchmark_id"] != benchmark or e["commitment_digest"] != cd for e in events):
            raise ValueError("commitment/benchmark substitution")
        assessments = {p["assessment_id"] for p in prospective_commitment["participants"]}
        if {e["assessment_id"] for e in protected_holdout_events} != assessments:
            raise ValueError("incomplete assessment access coverage")
        for e in events:
            _verify_record(e, trust_registry, role="audit_event",
                           domain="ZenerEstimation/M19-C2-B2-B2-B/event/v1")
        _validate_checkpoint(checkpoint)
        if checkpoint["last_sequence"] != seq[-1]:
            raise ValueError("checkpoint sequence mismatch")
        if checkpoint["event_chain_digest"] != event_chain_digest(events):
            raise ValueError("checkpoint chain mismatch")
        _verify_record(checkpoint, trust_registry, role="checkpoint_witness",
                       domain="ZenerEstimation/M19-C2-B2-B2-B/checkpoint/v1")
        anchored = trust_registry["checkpoint_anchors"].get(checkpoint["checkpoint_id"])
        if type(anchored) is not str or anchored != _hash(_canonical(checkpoint)):
            raise ValueError("checkpoint not independently anchored")
        approval = _time(authorization_registration["event_time"])
        earliest = min(_time(e["event_time"]) for e in protected_holdout_events)
        if approval >= earliest:
            return _result("AUTHORIZATION_REJECTED", ["NON_PROSPECTIVE_APPROVAL"], commitment_digest=cd)
        return _result("AUTHORIZATION_REQUIRED", ["INDEPENDENT_TIME_PROVENANCE_NOT_ESTABLISHED"],
                       commitment_digest=cd, verified=True)
    except (ValueError, TypeError, KeyError, InvalidSignature, OverflowError):
        return _result("AUTHORIZATION_REJECTED", ["INVALID_OR_UNTRUSTED_CHRONOLOGY"],
                       commitment_digest=cd)
