"""M18-C4: fail-closed, read-only consumption of verified C3 selection artifacts.

A READY record indicates that a recorded selection was independently replayed
against caller-supplied trusted evidence and has the C1 SELECTED state. It is
not proof of human identity, a deployment approval, or model activation.
"""
from __future__ import annotations

from collections.abc import Mapping

from zenerestimation.optimization.internal_selection_artifact import (
    ARTIFACT_VERSION,
    verify_internal_selection_artifact,
)

CONSUMPTION_VERSION = "m18_c4_consumption_v1"


def consume_internal_selection_artifact(
    artifact: Mapping,
    *,
    results: Mapping,
    validation_context: Mapping,
) -> dict:
    """Return a detached READY record only for an independently verified selection.

    Verification replays C1/C2 via C3 using independently trusted results and
    validation context. All invalid or non-selected decisions fail closed.
    No forecasting, model construction, persistence, or activation occurs.
    """
    if not isinstance(results, Mapping) or not isinstance(validation_context, Mapping):
        raise TypeError("trusted results and validation_context must be mappings")

    # Do not inspect the selection as an authority until the entire C3 artifact
    # has passed independent verification against the caller's trusted inputs.
    verification = verify_internal_selection_artifact(
        artifact,
        results=results,
        validation_context=validation_context,
    )
    if verification.get("status") != "verified" or any(
        verification.get(field) is not True
        for field in (
            "decision_digest_match", "ranking_match",
            "provenance_match", "selection_match",
        )
    ):
        raise ValueError("selection artifact was not independently verified")

    record = artifact["selection_decision_record"]
    decision = record["decision"]
    selected = decision["selected_model"]
    if (
        decision["status"] != "SELECTED"
        or decision["authorized"] is not True
        or type(selected) is not str
        or not selected
        or selected != decision["recommended_model"]
    ):
        raise ValueError("selection decision is not authorized and uniquely selected")

    return {
        "consumption_version": CONSUMPTION_VERSION,
        "status": "READY",
        "selected_model": selected,
        "decision_digest": record["decision_digest"],
        "artifact_version": ARTIFACT_VERSION,
    }
