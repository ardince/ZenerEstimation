"""M19-B3-B1: target-blind eligibility for evaluating an approved M18 forecast.

Replays trusted M18/M19 verification; does not access holdout targets, score
predictions, authenticate trusted sources, or establish scientific independence.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from math import isfinite
from typing import Any

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, validate_assessment_declaration,
)
from zenerestimation.assessment.holdout_binding import verify_holdout_binding
from zenerestimation.optimization.internal_forecast_execution_consumer import (
    consume_internal_forecast_execution_artifact,
)

ELIGIBILITY_VERSION = "m19_b3_b1_evaluation_eligibility_v1"
_BINDING_FIELDS = (
    "assessment_id", "declaration_digest", "dataset_id", "selected_model",
    "decision_digest", "plan_digest", "execution_artifact_digest",
    "training_fingerprint", "development_cutoff", "forecast_horizon",
    "forecast_dates", "scientific_independence", "binding_version", "status",
)


def _equal(a: Any, b: Any, name: str) -> None:
    if type(a) is not type(b) or a != b:
        raise ValueError(f"{name} mismatch")


def establish_holdout_evaluation_eligibility(
    *, declaration: AssessmentDeclaration, binding_report: Mapping,
    selection_artifact: Mapping, results: Mapping, validation_context: Mapping,
    registry: Any, training_dataset: Any, training_context: Mapping,
    plan: Mapping, execution_artifact: Mapping, trusted_dataset_id: str,
) -> dict[str, Any]:
    """Return a detached prediction-only eligibility snapshot; never read targets.

    `binding_report` is rechecked against freshly replayed trusted verification.
    This is a consistency gate, not evidence-origin authentication or freeze proof.
    """
    validate_assessment_declaration(declaration)
    if not isinstance(binding_report, Mapping):
        raise TypeError("binding_report must be a mapping")
    if set(binding_report) != set(_BINDING_FIELDS):
        raise ValueError("binding_report schema mismatch")

    kwargs = dict(
        declaration=declaration, selection_artifact=selection_artifact,
        results=results, validation_context=validation_context, registry=registry,
        training_dataset=training_dataset, training_context=training_context,
        plan=plan, execution_artifact=execution_artifact,
        trusted_dataset_id=trusted_dataset_id,
    )
    expected = verify_holdout_binding(**kwargs)
    for field in _BINDING_FIELDS:
        _equal(binding_report[field], expected[field], f"binding {field}")

    horizon = len(declaration.forecast_dates)
    view = consume_internal_forecast_execution_artifact(
        execution_artifact, plan=plan, artifact=selection_artifact,
        results=results, validation_context=validation_context, registry=registry,
        training_dataset=training_dataset, training_context=training_context,
        forecast_horizon=horizon, forecast_dates=tuple(declaration.forecast_dates),
    )
    if not isinstance(view, Mapping) or view.get("status") != "VERIFIED":
        raise ValueError("execution view must be independently verified")
    forecast = view.get("forecast")
    provenance = view.get("provenance")
    if not isinstance(forecast, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("execution view lacks forecast or provenance")
    _equal(view.get("selected_model"), expected["selected_model"], "model")
    _equal(forecast.get("horizon"), horizon, "horizon")
    _equal(forecast.get("dates"), list(declaration.forecast_dates), "calendar")
    for field in ("decision_digest", "plan_digest", "execution_artifact_digest", "training_fingerprint"):
        _equal(provenance.get(field), expected[field], field)

    values = forecast.get("values_microVolt")
    if type(values) is not list or len(values) != horizon:
        raise ValueError("forecast values must match declared horizon")
    if any(type(v) not in (int, float) or not isfinite(v) for v in values):
        raise ValueError("forecast values must be finite numeric scalars")

    # JSON round-trip rejects noncanonical/nonserializable provenance and
    # ensures no caller-owned nested mutable references survive.
    payload = {
        "eligibility_version": ELIGIBILITY_VERSION,
        "status": "ELIGIBLE_FOR_HOLDOUT_DISCLOSURE",
        "assessment_id": declaration.assessment_id,
        "declaration_digest": declaration.digest,
        "dataset_id": declaration.dataset_id,
        "development_cutoff": declaration.development_cutoff,
        "selected_model": expected["selected_model"],
        "decision_digest": expected["decision_digest"],
        "plan_digest": expected["plan_digest"],
        "execution_artifact_digest": expected["execution_artifact_digest"],
        "training_fingerprint": expected["training_fingerprint"],
        "forecast_horizon": horizon,
        "forecast_dates": list(declaration.forecast_dates),
        "predicted_microVolt": [float(v) for v in values],
        "scientific_independence": "not_established",
        "target_values_included": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False, ensure_ascii=False)
    detached = json.loads(encoded)
    detached["eligibility_digest"] = "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return detached
