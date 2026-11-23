"""M19-C2-B2-B2-A: prospective benchmark commitment, not authorization.

The commitment deliberately excludes post-execution C1 entry digests.
Its digest is an integrity identifier, not an independently witnessed approval.
"""
from __future__ import annotations

import re
from datetime import date
from collections.abc import Mapping, Sequence

from zenerestimation.assessment.holdout_target_disclosure import _digest
from zenerestimation.comparison.benchmark_authorization import (
    verify_benchmark_scope_declaration,
)

COMMITMENT_VERSION = "m19_c2b2b2a_prospective_commitment_v1"
ASSESSMENT_VERSION = "m19_c2b2b2a_commitment_binding_v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_PARTICIPANT_FIELDS = {"run_id", "assessment_id", "model_family"}
_PAYLOAD_FIELDS = {
    "commitment_version", "benchmark_id", "purpose", "dataset_id",
    "forecast_horizon", "forecast_dates", "authorized_operation",
    "participants",
}
_FAMILIES = frozenset({"LSTM", "GRU", "ARIMA", "Kalman", "Hybrid"})


def _id(value: object, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"invalid {field}")
    return value


def _payload(value: object) -> dict:
    if type(value) is not dict or set(value) != _PAYLOAD_FIELDS:
        raise ValueError("invalid prospective commitment fields")
    if value["commitment_version"] != COMMITMENT_VERSION:
        raise ValueError("unsupported commitment version")
    if value["purpose"] != "cross_model_benchmark":
        raise ValueError("unsupported commitment purpose")
    if value["authorized_operation"] != "compare_and_rank":
        raise ValueError("unsupported operation")
    benchmark_id = _id(value["benchmark_id"], "benchmark_id")
    dataset_id = _id(value["dataset_id"], "dataset_id")
    horizon = value["forecast_horizon"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("invalid forecast horizon")
    dates = value["forecast_dates"]
    if type(dates) is not list or len(dates) != horizon:
        raise ValueError("invalid forecast dates")
    try:
        if any(type(d) is not str or date.fromisoformat(d).isoformat() != d for d in dates):
            raise ValueError("invalid forecast date")
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid forecast date") from exc
    if dates != sorted(set(dates)):
        raise ValueError("forecast dates must be strictly increasing")
    raw = value["participants"]
    if type(raw) is not list or len(raw) < 2:
        raise ValueError("at least two committed participants required")
    participants = []
    for p in raw:
        if type(p) is not dict or set(p) != _PARTICIPANT_FIELDS:
            raise ValueError("invalid committed participant fields")
        family = p["model_family"]
        if type(family) is not str or family not in _FAMILIES:
            raise ValueError("invalid model family")
        participants.append({
            "run_id": _id(p["run_id"], "run_id"),
            "assessment_id": _id(p["assessment_id"], "assessment_id"),
            "model_family": family,
        })
    if participants != sorted(participants, key=lambda p: p["run_id"]):
        raise ValueError("participants must be sorted by run_id")
    if len({p["run_id"] for p in participants}) != len(participants):
        raise ValueError("duplicate run IDs")
    if len({p["assessment_id"] for p in participants}) != len(participants):
        raise ValueError("duplicate assessment IDs")
    if len({p["model_family"] for p in participants}) < 2:
        raise ValueError("two model families required")
    return {
        "commitment_version": COMMITMENT_VERSION,
        "benchmark_id": benchmark_id,
        "purpose": "cross_model_benchmark",
        "dataset_id": dataset_id,
        "forecast_horizon": horizon,
        "forecast_dates": list(dates),
        "authorized_operation": "compare_and_rank",
        "participants": participants,
    }


def build_prospective_benchmark_commitment(
    *, benchmark_id: str, dataset_id: str, forecast_dates: Sequence[str],
    participants: Sequence[Mapping],
) -> dict:
    """Build an unsigned, pre-execution commitment; never an approval."""
    if type(forecast_dates) not in (list, tuple) or type(participants) not in (list, tuple):
        raise TypeError("forecast dates and participants must be sequences")
    payload = {
        "commitment_version": COMMITMENT_VERSION,
        "benchmark_id": benchmark_id, "purpose": "cross_model_benchmark",
        "dataset_id": dataset_id, "forecast_horizon": len(forecast_dates),
        "forecast_dates": list(forecast_dates),
        "authorized_operation": "compare_and_rank",
        "participants": sorted((dict(p) for p in participants), key=lambda p: str(p.get("run_id", ""))),
    }
    validated = _payload(payload)
    return {**validated, "commitment_digest": _digest(validated)}


def verify_prospective_benchmark_commitment(commitment: Mapping) -> bool:
    """Check canonical structure and digest; NOT approval or prior existence."""
    if type(commitment) is not dict or set(commitment) != _PAYLOAD_FIELDS | {"commitment_digest"}:
        return False
    try:
        payload = {k: v for k, v in commitment.items() if k != "commitment_digest"}
        validated = _payload(payload)
        return validated == payload and type(commitment["commitment_digest"]) is str and (
            commitment["commitment_digest"] == _digest(validated)
        )
    except (ValueError, TypeError, KeyError, OverflowError):
        return False


def assess_prospective_commitment_binding(
    *, commitment: Mapping, scope_declaration: Mapping,
) -> dict:
    """Bind retrospective B1 scope to prospective fields, never authorize.

    This does not establish that the commitment existed before holdout access.
    The retrospective entry digests remain verified by the C1/B1 pipeline.
    """
    if not verify_prospective_benchmark_commitment(commitment):
        raise ValueError("invalid prospective commitment")
    if not verify_benchmark_scope_declaration(scope_declaration):
        raise ValueError("invalid retrospective B1 declaration")
    reasons = []
    for field, reason in (
        ("benchmark_id", "BENCHMARK_MISMATCH"),
        ("dataset_id", "DATASET_MISMATCH"),
        ("forecast_horizon", "HORIZON_MISMATCH"),
        ("forecast_dates", "TIMESTAMP_MISMATCH"),
        ("purpose", "PURPOSE_MISMATCH"),
    ):
        if commitment[field] != scope_declaration[field]:
            reasons.append(reason)
    actual = [
        {field: p[field] for field in ("run_id", "assessment_id", "model_family")}
        for p in scope_declaration["participants"]
    ]
    if actual != commitment["participants"]:
        reasons.append("PARTICIPANT_MISMATCH")
    result = {
        "assessment_version": ASSESSMENT_VERSION,
        "benchmark_id": commitment["benchmark_id"],
        "commitment_digest": commitment["commitment_digest"],
        "scope_digest": scope_declaration["scope_digest"],
        "status": "AUTHORIZATION_REJECTED" if reasons else "AUTHORIZATION_REQUIRED",
        "reasons": reasons,
        "commitment_binding_verified": not reasons,
        "signature_verified": False,
        "prospective_authorization_verified": False,
        "authorization": "not_established",
        "authentication": "not_established",
        "comparison_permitted": False,
        "ranking_permitted": False,
    }
    result["decision_digest"] = _digest(result)
    return result
