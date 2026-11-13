"""M18-C5: read-only preparation of an independently verified model selection.

PREPARED records describe point-in-time registry resolution, not execution,
configuration reproducibility, deployment approval, or verified human identity.
"""
from __future__ import annotations

from collections.abc import Mapping

from zenerestimation.forecasting.base import BaseForecastModel
from zenerestimation.forecasting.models import ModelRegistry
from zenerestimation.optimization.internal_selection_consumer import (
    CONSUMPTION_VERSION,
    consume_internal_selection_artifact,
)

PREPARATION_VERSION = "m18_c5_preparation_v1"


def prepare_selected_forecast(
    artifact: Mapping,
    *,
    results: Mapping,
    validation_context: Mapping,
    registry: ModelRegistry,
) -> dict:
    """Verify a C3 selection through C4 and resolve its model without executing it.

    Requires an explicit ModelRegistry; does not instantiate, fit, predict,
    modify the registry, or persist anything. Registry identity is not pinned.
    """
    if not isinstance(registry, ModelRegistry):
        raise TypeError("registry must be an explicit ModelRegistry instance")

    ready = consume_internal_selection_artifact(
        artifact,
        results=results,
        validation_context=validation_context,
    )
    if ready.get("status") != "READY":
        raise ValueError("selection is not ready for preparation")

    selected = ready["selected_model"]
    if type(selected) is not str or not selected:
        raise ValueError("verified selection has no valid model name")

    model_cls = registry.get(selected)
    if not isinstance(model_cls, type) or not issubclass(model_cls, BaseForecastModel):
        raise TypeError("registered model must be a BaseForecastModel subclass")

    return {
        "preparation_version": PREPARATION_VERSION,
        "status": "PREPARED",
        "selected_model": selected,
        "registry_model": selected.lower(),
        "decision_digest": ready["decision_digest"],
        "artifact_version": ready["artifact_version"],
        "consumption_version": CONSUMPTION_VERSION,
    }
