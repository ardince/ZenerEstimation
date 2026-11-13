"""M18-C7-B2: read-only consumption of verified forecast execution evidence.

Verification checks consistency with independently supplied trusted inputs;
SHA-256 is not authentication or proof that the recorded model ran.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path

from zenerestimation.optimization.internal_forecast_execution_artifact import (
    verify_internal_forecast_execution_artifact,
)

VIEW_VERSION = "m18_c7_forecast_view_v1"
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024


def consume_internal_forecast_execution_artifact(
    execution_artifact: Mapping, *, plan: Mapping, artifact: Mapping,
    results: Mapping, validation_context: Mapping, registry, training_dataset,
    training_context: Mapping, forecast_horizon: int, forecast_dates,
) -> dict:
    """Return a detached JSON-safe forecast view only after C6-D verification.

    This function performs no filesystem I/O and never constructs or runs models.
    """
    verified = verify_internal_forecast_execution_artifact(
        execution_artifact, plan=plan, artifact=artifact, results=results,
        validation_context=validation_context, registry=registry,
        training_dataset=training_dataset, training_context=training_context,
        forecast_horizon=forecast_horizon, forecast_dates=forecast_dates,
    )
    if verified is not True:
        raise ValueError("execution artifact failed independent verification")

    record = execution_artifact["execution_record"]
    forecast = record["forecast"]
    return {
        "view_version": VIEW_VERSION,
        "status": "VERIFIED",
        "selected_model": record["selected_model"],
        "forecast": {
            "horizon": forecast["horizon"],
            "dates": deepcopy(forecast["dates"]),
            "values_microVolt": deepcopy(forecast["values_microVolt"]),
        },
        "provenance": {
            "plan_digest": record["plan_digest"],
            "decision_digest": record["decision_digest"],
            "execution_artifact_digest": execution_artifact["artifact_digest"],
            "training_fingerprint": deepcopy(
                record["execution_evidence"]["training_fingerprint"]
            ),
        },
    }


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    """Reject duplicate JSON members rather than silently choosing one."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_internal_forecast_execution_artifact(
    path: str | Path, *, plan: Mapping, artifact: Mapping,
    results: Mapping, validation_context: Mapping, registry, training_dataset,
    training_context: Mapping, forecast_horizon: int, forecast_dates,
) -> dict:
    """Read a bounded UTF-8 JSON artifact and return its verified forecast view.

    Does not modify the file, create temporary files, or execute models.
    """
    target = Path(path)
    # Reading at most one byte beyond the limit prevents unbounded allocations
    # even if the file grows after a preliminary size check.
    with target.open("rb") as handle:
        raw = handle.read(MAX_ARTIFACT_BYTES + 1)
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("execution artifact exceeds maximum JSON file size")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("execution artifact must be UTF-8 JSON") from exc
    try:
        execution_artifact = json.loads(
            decoded, object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("invalid execution artifact JSON") from exc
    return consume_internal_forecast_execution_artifact(
        execution_artifact, plan=plan, artifact=artifact, results=results,
        validation_context=validation_context, registry=registry,
        training_dataset=training_dataset, training_context=training_context,
        forecast_horizon=forecast_horizon, forecast_dates=forecast_dates,
    )
