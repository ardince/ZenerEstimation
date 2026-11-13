"""M18-C2: deterministic audit records for controlled internal model selection.

This layer preserves M18-C1's public interface. Verification replays C1 using
independently supplied optimization results and trusted validation context.
A digest detects changes relative to an independently trusted digest; it is
not a signature or proof of human authorization.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math

from zenerestimation.optimization.internal_selection import select_internal_model

DECISION_RECORD_VERSION = "m18_c2_decision_v1"
_CONTEXT_FIELDS = frozenset({
    "protocol_version", "dataset_id", "dataset_fingerprint", "scope",
    "fold_validation_dates", "fold_training_end_dates", "folds",
    "validation_steps",
})
_RECORD_FIELDS = frozenset({
    "decision_record_version", "decision", "validation_provenance",
    "decision_digest",
})
_DECISION_FIELDS = frozenset({
    "selection_version", "status", "selected_model", "recommended_model",
    "selection_metric", "selection_score", "authorized", "ranking_version",
    "ranking",
})
_RANKING_FIELDS = frozenset({"model", "score", "rank"})


def _canonical_bytes(payload: Mapping) -> bytes:
    try:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("decision record is not canonical JSON") from exc


def _digest(payload: Mapping) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _strict_equal(actual, expected) -> bool:
    """Compare JSON values with exact types (not bool/int coercion)."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (
            actual.keys() == expected.keys()
            and all(_strict_equal(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(a, b) for a, b in zip(actual, expected)
        )
    if isinstance(expected, float):
        return math.isfinite(actual) and actual == expected
    return actual == expected


def _require_shape(record: Mapping) -> None:
    if not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS:
        raise ValueError("invalid decision record fields")
    if record["decision_record_version"] != DECISION_RECORD_VERSION:
        raise ValueError("unsupported decision record version")
    decision = record["decision"]
    provenance = record["validation_provenance"]
    if not isinstance(decision, Mapping) or set(decision) != _DECISION_FIELDS:
        raise ValueError("invalid selection decision fields")
    if not isinstance(provenance, Mapping) or set(provenance) != _CONTEXT_FIELDS:
        raise ValueError("invalid validation provenance fields")
    ranking = decision["ranking"]
    if not isinstance(ranking, list) or not ranking or any(
        not isinstance(row, Mapping) or set(row) != _RANKING_FIELDS
        for row in ranking
    ):
        raise ValueError("invalid ranking rows")
    if type(decision["authorized"]) is not bool:
        raise ValueError("authorization must be a bool")
    digest = record["decision_digest"]
    if not isinstance(digest, str) or len(digest) != 71 or not digest.startswith("sha256:") or any(
        char not in "0123456789abcdef" for char in digest[7:]
    ):
        raise ValueError("invalid decision digest")


def build_selection_decision_record(
    results: Mapping,
    *,
    validation_context: Mapping,
    authorized: bool = False,
) -> dict:
    """Build a canonical audit record from a verified M18-C1 decision."""
    decision = select_internal_model(
        results, validation_context=validation_context, authorized=authorized,
    )
    provenance = deepcopy(dict(validation_context))
    if set(provenance) != _CONTEXT_FIELDS:
        raise ValueError("trusted context has missing or unexpected fields")
    payload = {
        "decision_record_version": DECISION_RECORD_VERSION,
        "decision": deepcopy(decision),
        "validation_provenance": provenance,
    }
    record = {**payload, "decision_digest": _digest(payload)}
    _require_shape(record)
    return record


def verify_selection_decision_record(
    record: Mapping,
    *,
    results: Mapping,
    validation_context: Mapping,
) -> dict:
    """Verify digest, exact schema and independently replayed C1 decision.

    Caller must independently trust the development dataset/context and results;
    a self-consistent digest does not authenticate their origin or approval.
    """
    _require_shape(record)
    payload = {key: value for key, value in record.items() if key != "decision_digest"}
    if record["decision_digest"] != _digest(payload):
        raise ValueError("decision digest mismatch")

    expected = build_selection_decision_record(
        results,
        validation_context=validation_context,
        authorized=record["decision"]["authorized"],
    )
    if not _strict_equal(record["validation_provenance"], expected["validation_provenance"]):
        raise ValueError("trusted validation provenance mismatch")
    if not _strict_equal(record["decision"], expected["decision"]):
        raise ValueError("recomputed selection decision mismatch")
    if not _strict_equal(record, expected):
        raise ValueError("canonical decision record mismatch")
    return {
        "status": "verified",
        "decision_record_version": DECISION_RECORD_VERSION,
        "decision_digest_match": True,
        "ranking_match": True,
        "provenance_match": True,
        "selection_match": True,
    }
