"""M19-C2-B1: deterministic, self-declared benchmark scope validation.

A declaration is NOT approval, authentication, evidence of prior issuance,
scientific independence, or permission to compare/rank protected holdouts.
No models execute, and the accepted C1/C2-A contracts are unchanged.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date

from zenerestimation.assessment.holdout_target_disclosure import _digest
from zenerestimation.comparison.benchmark_comparability import (
    VerifiedBenchmarkInput, assess_scientific_alignment,
)

SCOPE_VERSION = "m19_c2b1_benchmark_scope_v1"
DECISION_VERSION = "m19_c2b1_scope_assessment_v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_FAMILIES = frozenset({"LSTM", "GRU", "ARIMA", "Kalman", "Hybrid"})
_PARTICIPANT_FIELDS = frozenset({
    "run_id", "assessment_id", "model_family", "benchmark_entry_digest",
})
_SCOPE_FIELDS = frozenset({
    "scope_version", "benchmark_id", "purpose", "scope_status",
    "dataset_id", "forecast_horizon", "forecast_dates", "participants",
})


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical nonempty identifier (max 128 characters)")
    return value


def _dates(value: object, horizon: int) -> list[str]:
    if type(value) is not list or len(value) != horizon:
        raise ValueError("forecast_dates must match forecast_horizon")
    try:
        if any(type(d) is not str or date.fromisoformat(d).isoformat() != d for d in value):
            raise ValueError("noncanonical forecast date")
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid forecast date") from exc
    if value != sorted(set(value)):
        raise ValueError("forecast_dates must be unique and chronological")
    return list(value)


def _participant(value: object) -> dict:
    if type(value) is not dict or set(value) != _PARTICIPANT_FIELDS:
        raise ValueError("participant must contain exactly the required fields")
    result = {
        "run_id": _identifier(value["run_id"], "run_id"),
        "assessment_id": _identifier(value["assessment_id"], "assessment_id"),
        "model_family": value["model_family"],
        "benchmark_entry_digest": value["benchmark_entry_digest"],
    }
    if type(result["model_family"]) is not str or result["model_family"] not in _FAMILIES:
        raise ValueError("unsupported model family")
    digest = result["benchmark_entry_digest"]
    if type(digest) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ValueError("benchmark_entry_digest must be sha256:-prefixed lowercase SHA-256 hex")
    return result


def _validate_scope_payload(value: object) -> dict:
    if type(value) is not dict or set(value) != _SCOPE_FIELDS:
        raise ValueError("scope declaration must contain exactly the required fields")
    if value["scope_version"] != SCOPE_VERSION:
        raise ValueError("unsupported scope version")
    if value["purpose"] != "cross_model_benchmark":
        raise ValueError("unsupported purpose")
    if value["scope_status"] != "SELF_DECLARED_SCOPE_ONLY":
        raise ValueError("scope cannot claim authorization")
    benchmark_id = _identifier(value["benchmark_id"], "benchmark_id")
    dataset_id = _identifier(value["dataset_id"], "dataset_id")
    horizon = value["forecast_horizon"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("forecast_horizon must be a positive integer")
    dates = _dates(value["forecast_dates"], horizon)
    raw = value["participants"]
    if type(raw) is not list or len(raw) < 2:
        raise ValueError("at least two participants are required")
    participants = [_participant(p) for p in raw]
    ids = [p["run_id"] for p in participants]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate participant run_id")
    if len({p["model_family"] for p in participants}) < 2:
        raise ValueError("cross-model scope requires at least two distinct model families")
    if participants != sorted(participants, key=lambda p: p["run_id"]):
        raise ValueError("participants must be ordered by run_id")
    return {
        "scope_version": SCOPE_VERSION, "benchmark_id": benchmark_id,
        "purpose": "cross_model_benchmark", "scope_status": "SELF_DECLARED_SCOPE_ONLY",
        "dataset_id": dataset_id, "forecast_horizon": horizon,
        "forecast_dates": dates, "participants": participants,
    }


def build_benchmark_scope_declaration(
    *, benchmark_id: str, dataset_id: str, forecast_dates: Sequence[str],
    participants: Sequence[Mapping],
) -> dict:
    """Build an integrity-checkable *scope*, never an approval artifact."""
    if type(forecast_dates) not in (list, tuple) or type(participants) not in (list, tuple):
        raise TypeError("forecast_dates and participants must be lists or tuples")
    payload = {
        "scope_version": SCOPE_VERSION, "benchmark_id": benchmark_id,
        "purpose": "cross_model_benchmark", "scope_status": "SELF_DECLARED_SCOPE_ONLY",
        "dataset_id": dataset_id, "forecast_horizon": len(forecast_dates),
        "forecast_dates": list(forecast_dates),
        "participants": sorted([dict(p) for p in participants], key=lambda p: str(p.get("run_id", ""))),
    }
    validated = _validate_scope_payload(payload)
    validated["scope_digest"] = _digest(validated)
    return validated


def verify_benchmark_scope_declaration(declaration: Mapping) -> bool:
    """Internal consistency only; never authentication or prior authorization."""
    if type(declaration) is not dict or set(declaration) != _SCOPE_FIELDS | {"scope_digest"}:
        return False
    try:
        payload = {k: v for k, v in declaration.items() if k != "scope_digest"}
        validated = _validate_scope_payload(payload)
        return validated == payload and type(declaration["scope_digest"]) is str and (
            declaration["scope_digest"] == _digest(validated)
        )
    except (TypeError, ValueError, OverflowError, KeyError):
        return False


def assess_benchmark_scope(
    *, inputs: Sequence[VerifiedBenchmarkInput], declaration: Mapping,
) -> dict:
    """Reverify C1/C2-A; match scope; never grant comparison authorization.

    Malformed or unverified C1 inputs fail closed with an exception. A valid
    self-declaration with a mismatched population or participants is rejected.
    """
    alignment = assess_scientific_alignment(inputs)
    if not verify_benchmark_scope_declaration(declaration):
        raise ValueError("invalid or internally inconsistent scope declaration")
    reasons = []
    if alignment["status"] != "SCIENTIFICALLY_ALIGNED":
        reasons.append("SCIENTIFIC_ALIGNMENT_FAILED")
    if declaration["dataset_id"] != alignment["shared_dataset_id"]:
        reasons.append("DATASET_SCOPE_MISMATCH")
    if declaration["forecast_horizon"] != alignment["shared_horizon"]:
        reasons.append("HORIZON_SCOPE_MISMATCH")
    if declaration["forecast_dates"] != alignment["shared_forecast_dates"]:
        reasons.append("TIMESTAMP_SCOPE_MISMATCH")
    actual = sorted(({
        "run_id": item.entry["run_id"],
        "assessment_id": item.entry["assessment_id"],
        "model_family": item.entry["model_family"],
        "benchmark_entry_digest": item.entry["benchmark_entry_digest"],
    } for item in inputs), key=lambda p: p["run_id"])
    if declaration["participants"] != actual:
        reasons.append("PARTICIPANT_SCOPE_MISMATCH")
    result = {
        "decision_version": DECISION_VERSION,
        "benchmark_id": declaration["benchmark_id"],
        "scope_digest": declaration["scope_digest"],
        "status": "AUTHORIZATION_REJECTED" if reasons else "AUTHORIZATION_REQUIRED",
        "reasons": reasons,
        "scientific_alignment_status": alignment["status"],
        "observation_digest": alignment["observation_digest"],
        "authorization": "not_established",
        "authentication": "not_established",
        "comparison_permitted": False,
        "ranking_permitted": False,
    }
    result["decision_digest"] = _digest(result)
    return result
