"""Versioned, standalone experiment manifest contract (schema 1.0).

This module deliberately does not modify legacy Experiment or optimization artifacts.
"""
from __future__ import annotations

import copy
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


class ManifestValidationError(ValueError):
    """Raised when an experiment manifest violates the versioned contract."""


def _fail(path: str, reason: str) -> None:
    raise ManifestValidationError(f"{path}: {reason}")


def _object(value: Any, path: str, required: set[str], allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    missing = required - value.keys()
    unknown = value.keys() - allowed
    if missing:
        _fail(path, f"missing fields: {', '.join(sorted(missing))}")
    if unknown:
        _fail(path, f"unknown fields: {', '.join(sorted(unknown))}")
    return value


def _name(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a nonempty string")


def _json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "nonfinite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, "object keys must be strings")
            _json_value(item, f"{path}.{key}")
        return
    _fail(path, "value is not JSON-compatible")


def _extension(value: Any, path: str) -> None:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    _json_value(value, path)


def _validate(payload: Any) -> None:
    root = _object(payload, "manifest", {"schema_version", "experiment_id", "created_at", "specification", "execution"}, {"schema_version", "experiment_id", "created_at", "specification", "execution"})
    if root["schema_version"] != "1.0":
        _fail("schema_version", "unsupported schema version (expected '1.0')")
    _name(root["experiment_id"], "experiment_id")
    timestamp = root["created_at"]
    if not isinstance(timestamp, str):
        _fail("created_at", "must be a timezone-aware ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        _fail("created_at", "invalid ISO 8601 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None or "T" not in timestamp:
        _fail("created_at", "must include date, time and timezone")

    spec = _object(root["specification"], "specification", {"dataset", "model", "evaluation", "preprocessing"}, {"dataset", "model", "evaluation", "preprocessing"})
    dataset = _object(spec["dataset"], "specification.dataset", {"identifier", "fingerprint"}, {"identifier", "fingerprint"})
    _name(dataset["identifier"], "specification.dataset.identifier")
    fp = dataset["fingerprint"]
    if fp is not None:
        fp = _object(fp, "specification.dataset.fingerprint", {"algorithm", "digest"}, {"algorithm", "digest"})
        if fp["algorithm"] != "sha256" or not isinstance(fp["digest"], str) or re.fullmatch(r"[0-9a-fA-F]{64}", fp["digest"]) is None:
            _fail("specification.dataset.fingerprint", "expected sha256 with a 64-character hex digest")

    model = _object(spec["model"], "specification.model", {"name", "parameters", "seed"}, {"name", "parameters", "seed"})
    _name(model["name"], "specification.model.name")
    _extension(model["parameters"], "specification.model.parameters")
    seed = model["seed"]
    if seed is not None and (type(seed) is not int or seed < 0):
        _fail("specification.model.seed", "must be a nonnegative integer or null")

    evaluation = _object(spec["evaluation"], "specification.evaluation", {"metric", "alignment_policy", "splitter"}, {"metric", "alignment_policy", "splitter"})
    _name(evaluation["metric"], "specification.evaluation.metric")
    if evaluation["alignment_policy"] not in ("verified", "positional_only"):
        _fail("specification.evaluation.alignment_policy", "unsupported alignment policy")
    splitter = _object(evaluation["splitter"], "specification.evaluation.splitter", {"name", "parameters"}, {"name", "parameters"})
    _name(splitter["name"], "specification.evaluation.splitter.name")
    _extension(splitter["parameters"], "specification.evaluation.splitter.parameters")
    _extension(spec["preprocessing"], "specification.preprocessing")

    execution = _object(root["execution"], "execution", {"status", "environment", "metrics", "artifacts", "error"}, {"status", "environment", "metrics", "artifacts", "error"})
    if execution["status"] not in ("pending", "running", "completed", "failed"):
        _fail("execution.status", "unsupported execution status")
    _extension(execution["environment"], "execution.environment")
    metrics = execution["metrics"]
    if not isinstance(metrics, dict):
        _fail("execution.metrics", "must be an object")
    for key, value in metrics.items():
        _name(key, "execution.metrics key")
        if type(value) not in (int, float) or not math.isfinite(value):
            _fail(f"execution.metrics.{key}", "must be a finite number")
    _extension(execution["artifacts"], "execution.artifacts")
    error = execution["error"]
    if error is not None:
        _extension(error, "execution.error")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects, including duplicates in nested objects."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("manifest", f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


class ExperimentManifest:
    """Validated schema-1.0 manifest with defensively copied payload data."""

    def __init__(self, payload: Mapping[str, Any]):
        _validate(payload)
        self._payload = copy.deepcopy(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentManifest":
        return cls(payload)

    def validate(self) -> None:
        _validate(self._payload)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)

    def save_json(self, path: str | Path) -> None:
        self.validate()
        text = json.dumps(self._payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        Path(path).write_text(text, encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "ExperimentManifest":
        try:
            with Path(path).open("r", encoding="utf-8") as stream:
                payload = json.load(
                    stream,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=lambda constant: _fail("manifest", f"invalid JSON constant: {constant}"),
                )
        except (ValueError, UnicodeError) as exc:
            raise ManifestValidationError(f"Invalid manifest JSON: {exc}") from exc
        return cls.from_dict(payload)
