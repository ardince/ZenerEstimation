"""Versioned, selection-only neural training evidence artifact (16.3-E1)."""
from __future__ import annotations

from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
from typing import Any

from .result import OptimizationResult

SCHEMA_VERSION = "neural-training-artifact-v1"


def _json_safe(value: Any, path: str = "root") -> Any:
    """Normalize standard Python numeric containers; reject non-finite/opaque data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"non-finite number at {path}")
        return value
    if isinstance(value, (tuple, list)):
        return [_json_safe(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError(f"non-string mapping key at {path}")
        return {k: _json_safe(v, f"{path}.{k}") for k, v in value.items()}
    # NumPy scalar support without a hard dependency on NumPy here.
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        return _json_safe(value.item(), path)
    raise TypeError(f"unsupported value at {path}: {type(value).__name__}")


def build_neural_training_artifact(result: OptimizationResult) -> dict[str, Any]:
    """Snapshot internal neural training evidence; never include benchmark outcomes.

    Does not change candidate ranking or modify OptimizationResult.
    """
    if not isinstance(result, OptimizationResult):
        raise TypeError("result must be an OptimizationResult")
    candidates = []
    for i, candidate in enumerate(result.candidates, 1):
        metadata = candidate.metadata or {}
        folds = metadata.get("fold_training")
        if folds is not None and not isinstance(folds, (list, tuple)):
            raise ValueError("fold_training must be a sequence or None")
        candidates.append({
            "candidate_index": i,
            "status": candidate.status,
            "params": _json_safe(candidate.params),
            "score": _json_safe(candidate.score),
            "metrics": _json_safe(candidate.metrics),
            "fold_training": _json_safe(folds) if folds is not None else None,
        })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": "internal_temporal_neural_training_only",
        "model": result.model,
        "selection_metric": result.metric,
        "best_params": _json_safe(result.best_params),
        "best_score": _json_safe(result.best_score),
        "budget": _json_safe((result.metadata or {}).get("budget")),
        "candidates": candidates,
        "interpretation": "descriptive_training_evidence_only; not_generalization_evidence",
    }
    return deepcopy(payload)


def write_neural_training_artifact(result: OptimizationResult, path: str | Path) -> Path:
    """Write validated JSON without NaN/Infinity; no implicit provenance claims."""
    payload = build_neural_training_artifact(result)
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise ValueError("artifact path must end in .json")
    # Serialize before writing to avoid partial output for invalid values.
    contents = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination.write_text(contents, encoding="utf-8")
    return destination
