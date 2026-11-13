"""M18-C3: standalone persistence for audited internal model selection.

This is a serialization boundary, not a ranking or authorization engine.
A locally consistent digest does not authenticate the decision's origin;
independent verification requires trusted results and validation context.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from zenerestimation.optimization.internal_selection_audit import (
    _require_shape as _require_c2_shape,
    build_selection_decision_record,
    verify_selection_decision_record,
)

ARTIFACT_VERSION = "m18_c3_selection_artifact_v1"
ARTIFACT_TYPE = "internal_model_selection"
_ARTIFACT_FIELDS = frozenset({
    "artifact_version", "artifact_type", "selection_decision_record",
})


def _validated_artifact(artifact: Mapping) -> dict:
    """Check wrapper, C2 schema, and local digest without replaying selection."""
    if not isinstance(artifact, Mapping) or set(artifact) != _ARTIFACT_FIELDS:
        raise ValueError("invalid internal selection artifact fields")
    if type(artifact["artifact_version"]) is not str or artifact["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError("unsupported internal selection artifact version")
    if type(artifact["artifact_type"]) is not str or artifact["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError("invalid internal selection artifact type")

    record = artifact["selection_decision_record"]
    _require_c2_shape(record)
    payload = {key: value for key, value in record.items() if key != "decision_digest"}
    try:
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid C2 decision record JSON") from exc
    digest = "sha256:" + hashlib.sha256(canonical).hexdigest()
    if record["decision_digest"] != digest:
        raise ValueError("C2 decision digest mismatch")

    # Verify JSON compatibility before touching any destination. The returned
    # snapshot cannot be mutated through the caller's original containers.
    try:
        snapshot = deepcopy(dict(artifact))
        json.dumps(snapshot, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid internal selection artifact JSON") from exc
    return snapshot


def build_internal_selection_artifact(
    results: Mapping,
    *,
    validation_context: Mapping,
    authorized: bool = False,
) -> dict:
    """Wrap the exact C2 audit record without changing selection semantics."""
    record = build_selection_decision_record(
        results, validation_context=validation_context, authorized=authorized,
    )
    return _validated_artifact({
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "selection_decision_record": record,
    })


def save_internal_selection_artifact(artifact: Mapping, destination: str | Path) -> Path:
    """Atomically persist deterministic UTF-8 JSON after local validation.

    This does NOT independently verify model selection or human authorization.
    """
    snapshot = _validated_artifact(artifact)
    path = Path(destination)
    if path.suffix.lower() != ".json":
        raise ValueError("artifact destination must end in .json")
    contents = (
        json.dumps(snapshot, sort_keys=True, indent=2,
                   ensure_ascii=False, allow_nan=False) + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def verify_internal_selection_artifact(
    artifact: Mapping,
    *,
    results: Mapping,
    validation_context: Mapping,
) -> dict:
    """Verify wrapper and independently replay C2 using trusted inputs."""
    snapshot = _validated_artifact(artifact)
    return verify_selection_decision_record(
        snapshot["selection_decision_record"],
        results=results, validation_context=validation_context,
    )
