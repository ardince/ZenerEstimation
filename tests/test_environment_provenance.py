"""Contract tests for Milestone 16.2-E2-B environment provenance."""
from __future__ import annotations

import copy
import json
import sys
from importlib import metadata

import pytest

from zenerestimation.experiment import environment as env


@pytest.fixture
def record(monkeypatch):
    monkeypatch.setattr(env.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(env.platform, "python_version", lambda: "3.11.15")
    monkeypatch.setattr(env.platform, "system", lambda: "Windows")
    monkeypatch.setattr(env.platform, "release", lambda: "10")
    monkeypatch.setattr(env.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(env.metadata, "version", lambda name: "1.2.3")
    return env.collect_environment_provenance()


def test_schema_and_fixed_fields(record):
    assert record["schema_version"] == "zener-environment-v1"
    assert record["python"] == {"implementation": "CPython", "version": "3.11.15"}
    assert record["platform"] == {"system": "Windows", "release": "10", "machine": "AMD64"}
    assert set(record["dependencies"]) == set(env.DEPENDENCIES)


def test_repeated_collection_is_equal(record):
    assert env.collect_environment_provenance() == record


def test_all_dependencies_are_queried_in_allowlist_order(record, monkeypatch):
    seen = []
    def lookup(name):
        seen.append(name)
        return "1.2.3"
    monkeypatch.setattr(env.metadata, "version", lookup)
    env.collect_environment_provenance()
    assert seen == list(env.DEPENDENCIES)


def test_missing_optional_dependency_is_null(record, monkeypatch):
    def lookup(name):
        if name == "keras":
            raise metadata.PackageNotFoundError(name)
        return "1.2.3"
    monkeypatch.setattr(env.metadata, "version", lookup)
    result = env.collect_environment_provenance()
    assert result["dependencies"]["keras"] is None
    assert result["dependencies"]["tensorflow"] == "1.2.3"


def test_no_framework_imports(record, monkeypatch):
    before = {key for key in sys.modules if key == "tensorflow" or key.startswith("tensorflow.")}
    env.collect_environment_provenance()
    after = {key for key in sys.modules if key == "tensorflow" or key.startswith("tensorflow.")}
    assert after == before


def test_json_roundtrip(record):
    assert json.loads(json.dumps(record, sort_keys=True, allow_nan=False)) == record


def test_no_sensitive_fields(record):
    text = json.dumps(record)
    for token in ("username", "hostname", "home_directory", "environment_variables", "cwd", "executable"):
        assert f'"{token}"' not in text


@pytest.mark.parametrize("path,value", [
    (("schema_version",), "zener-environment-v2"),
    (("python", "implementation"), ""),
    (("python", "version"), "3.11 bad"),
    (("python", "version"), 3.11),
    (("platform", "system"), ""),
    (("platform", "machine"), "x\nsecret"),
    (("dependencies", "numpy"), ""),
    (("dependencies", "numpy"), "/private/path"),
    (("dependencies", "numpy"), 1.2),
    (("dependencies", "numpy"), False),
])
def test_reject_invalid_values(record, path, value):
    bad = copy.deepcopy(record)
    target = bad
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(env.EnvironmentProvenanceError):
        env.validate_environment_provenance(bad)


@pytest.mark.parametrize("path,key,value", [
    ((), "extra", True),
    (("python",), "executable", "C:/private/python.exe"),
    (("platform",), "node", "hostname"),
    (("dependencies",), "unknown-package", "1.0"),
])
def test_reject_unknown_fields(record, path, key, value):
    bad = copy.deepcopy(record)
    target = bad
    for part in path:
        target = target[part]
    target[key] = value
    with pytest.raises(env.EnvironmentProvenanceError):
        env.validate_environment_provenance(bad)


@pytest.mark.parametrize("path", [(), ("python",), ("platform",), ("dependencies",)])
def test_reject_missing_fields(record, path):
    bad = copy.deepcopy(record)
    target = bad
    for part in path:
        target = target[part]
    del target[next(iter(target))]
    with pytest.raises(env.EnvironmentProvenanceError):
        env.validate_environment_provenance(bad)


def test_reject_non_object(record):
    with pytest.raises(env.EnvironmentProvenanceError):
        env.validate_environment_provenance([])


def test_reject_non_object_dependencies(record):
    record["dependencies"] = []
    with pytest.raises(env.EnvironmentProvenanceError):
        env.validate_environment_provenance(record)


def test_reject_non_object_python(record):
    record["python"] = None
    with pytest.raises(env.EnvironmentProvenanceError):
        env.validate_environment_provenance(record)


def test_validator_does_not_mutate(record):
    original = copy.deepcopy(record)
    env.validate_environment_provenance(record)
    assert record == original


def test_manifest_schema_compatibility(record):
    from zenerestimation.experiment.manifest import ExperimentManifest
    payload = {
        "schema_version": "1.0",
        "experiment_id": "environment-contract-test",
        "created_at": "2026-10-26T12:00:00+00:00",
        "specification": {
            "dataset": {"identifier": "test", "fingerprint": None},
            "model": {"name": "kalman", "parameters": {}, "seed": None},
            "evaluation": {"metric": "rmse", "alignment_policy": "verified", "splitter": {"name": "expanding", "parameters": {}}},
            "preprocessing": {},
        },
        "execution": {"status": "pending", "environment": record, "metrics": {}, "artifacts": {}, "error": None},
    }
    manifest = ExperimentManifest.from_dict(payload)
    assert manifest.to_dict()["execution"]["environment"] == record
