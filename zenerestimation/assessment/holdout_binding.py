"""M19-B3-A1: fail-closed, read-only binding of M18 execution to M19 assessment.

A successful binding establishes consistency against caller-supplied trusted
inputs, not provenance authentication or scientific holdout independence.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date
from typing import Any

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration,
    validate_assessment_declaration,
)
from zenerestimation.optimization.internal_selection_consumer import (
    consume_internal_selection_artifact,
)
from zenerestimation.optimization.internal_forecast_plan import (
    verify_internal_forecast_plan,
)
from zenerestimation.optimization.internal_forecast_execution_consumer import (
    consume_internal_forecast_execution_artifact,
)

BINDING_VERSION = "m19_b3_a1_holdout_binding_v1"


def _require_mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{name} mismatch")


def verify_holdout_binding(
    *,
    declaration: AssessmentDeclaration,
    selection_artifact: Mapping,
    results: Mapping,
    validation_context: Mapping,
    registry: Any,
    training_dataset: Any,
    training_context: Mapping,
    plan: Mapping,
    execution_artifact: Mapping,
    trusted_dataset_id: str,
) -> dict[str, Any]:
    """Reverify the M18 chain and bind it to an M19 assessment declaration.

    All trusted inputs must originate outside the artifact being verified.
    No models are instantiated or run. Does not authenticate the provenance
    of trusted inputs, establish forecast freeze, or prove target non-exposure.
    """
    validate_assessment_declaration(declaration)
    for name, value in (
        ("selection_artifact", selection_artifact),
        ("results", results),
        ("validation_context", validation_context),
        ("training_context", training_context),
        ("plan", plan),
        ("execution_artifact", execution_artifact),
    ):
        _require_mapping(value, name)
    if type(trusted_dataset_id) is not str or not trusted_dataset_id:
        raise ValueError("trusted_dataset_id must be a nonempty string")
    _require_equal(declaration.dataset_id, trusted_dataset_id, "dataset identity")

    # Replays C1/C2/C3, not merely a caller-provided READY status.
    ready = consume_internal_selection_artifact(
        selection_artifact, results=results, validation_context=validation_context,
    )
    if ready.get("status") != "READY":
        raise ValueError("selection is not independently verified and ready")

    dates = tuple(declaration.forecast_dates)
    horizon = len(dates)
    if verify_internal_forecast_plan(
        plan,
        artifact=selection_artifact,
        results=results,
        validation_context=validation_context,
        registry=registry,
        training_dataset=training_dataset,
        training_context=training_context,
        forecast_horizon=horizon,
        forecast_dates=dates,
    ) is not True:
        raise ValueError("forecast plan failed independent verification")

    _require_equal(plan["selected_model"], ready["selected_model"], "selected model")
    _require_equal(plan["decision_digest"], ready["decision_digest"], "selection digest")
    _require_equal(plan["forecast_horizon"], horizon, "forecast horizon")
    _require_equal(plan["forecast_dates"], list(dates), "forecast dates")
    _require_equal(
        plan["training_provenance"]["train_end"],
        declaration.development_cutoff,
        "development cutoff",
    )
    # Explicit chronology, even if an upstream verifier changes its policy.
    if date.fromisoformat(declaration.development_cutoff) >= date.fromisoformat(dates[0]):
        raise ValueError("forecast begins on or before development cutoff")

    # Replays C6-D validation, not merely a caller-provided VERIFIED status.
    view = consume_internal_forecast_execution_artifact(
        execution_artifact,
        plan=plan,
        artifact=selection_artifact,
        results=results,
        validation_context=validation_context,
        registry=registry,
        training_dataset=training_dataset,
        training_context=training_context,
        forecast_horizon=horizon,
        forecast_dates=dates,
    )
    if view.get("status") != "VERIFIED":
        raise ValueError("execution artifact is not independently verified")
    _require_equal(view["selected_model"], ready["selected_model"], "execution model")
    _require_equal(view["forecast"]["horizon"], horizon, "execution horizon")
    _require_equal(view["forecast"]["dates"], list(dates), "execution dates")
    provenance = view["provenance"]
    _require_equal(provenance["decision_digest"], ready["decision_digest"], "execution decision digest")
    _require_equal(provenance["plan_digest"], plan["plan_digest"], "execution plan digest")
    _require_equal(
        provenance["training_fingerprint"],
        {k: plan["training_provenance"][k] for k in (
            "algorithm", "digest", "canonicalization_version", "row_count"
        )},
        "execution training fingerprint",
    )
    _require_equal(
        provenance["execution_artifact_digest"],
        execution_artifact["artifact_digest"],
        "execution artifact digest",
    )

    # Detached report: intentionally no claim about non-exposure or independence.
    return deepcopy({
        "binding_version": BINDING_VERSION,
        "status": "BOUND",
        "assessment_id": declaration.assessment_id,
        "declaration_digest": declaration.digest,
        "dataset_id": trusted_dataset_id,
        "selected_model": ready["selected_model"],
        "decision_digest": ready["decision_digest"],
        "plan_digest": plan["plan_digest"],
        "execution_artifact_digest": provenance["execution_artifact_digest"],
        "training_fingerprint": provenance["training_fingerprint"],
        "development_cutoff": declaration.development_cutoff,
        "forecast_horizon": horizon,
        "forecast_dates": list(dates),
        "scientific_independence": "not_established",
    })
