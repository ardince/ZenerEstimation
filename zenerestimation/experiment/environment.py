"""Side-effect-minimal, versioned runtime provenance for experiments.

Only allowlisted public runtime metadata is collected. Installed distribution
versions are queried without importing the corresponding packages.
"""
from __future__ import annotations

import json
import platform
import re
from importlib import metadata
from typing import Any, Mapping

SCHEMA_VERSION = "zener-environment-v1"
DEPENDENCIES = (
    "zenerestimation", "numpy", "pandas", "scipy", "scikit-learn",
    "statsmodels", "tensorflow", "keras", "matplotlib",
)


class EnvironmentProvenanceError(ValueError):
    """Raised for an invalid or unsupported environment provenance record."""


def _reject(path: str, message: str) -> None:
    raise EnvironmentProvenanceError(f"{path}: {message}")


def _object(value: Any, path: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _reject(path, "must be an object")
    if set(value) != keys:
        _reject(path, f"expected exactly: {', '.join(sorted(keys))}")
    return value


def _text(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _reject(path, "must be a nonempty string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        _reject(path, "must not contain control characters")


def _version(value: Any, path: str) -> None:
    _text(value, path)
    # Versions may be PEP 440 or vendor-specific; reject paths, whitespace,
    # shell metacharacters and control characters without imposing PEP 440.
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_~-]*", value) is None:
        _reject(path, "invalid version format")


def validate_environment_provenance(record: Mapping[str, Any]) -> None:
    """Validate a complete environment provenance record (schema v1)."""
    root = _object(record, "environment", {"schema_version", "python", "platform", "dependencies"})
    if root["schema_version"] != SCHEMA_VERSION:
        _reject("environment.schema_version", "unsupported schema version")

    py = _object(root["python"], "environment.python", {"implementation", "version"})
    _text(py["implementation"], "environment.python.implementation")
    _version(py["version"], "environment.python.version")

    os_info = _object(root["platform"], "environment.platform", {"system", "release", "machine"})
    for key in ("system", "release", "machine"):
        _text(os_info[key], f"environment.platform.{key}")

    deps = _object(root["dependencies"], "environment.dependencies", set(DEPENDENCIES))
    for package in DEPENDENCIES:
        value = deps[package]
        if value is not None:
            _version(value, f"environment.dependencies.{package}")

    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError) as exc:
        _reject("environment", f"not JSON-compatible: {exc}")


def collect_environment_provenance() -> dict[str, Any]:
    """Collect a stable, allowlisted snapshot without importing dependencies."""
    dependencies: dict[str, str | None] = {}
    for package in DEPENDENCIES:
        try:
            dependencies[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            dependencies[package] = None

    record = {
        "schema_version": SCHEMA_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": dependencies,
    }
    validate_environment_provenance(record)
    return record
