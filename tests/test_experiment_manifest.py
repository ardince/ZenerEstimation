"""Milestone 16.2-E1-C1: contract-first tests for the experiment manifest.

Expected initial state: RED until zenerestimation.experiment.manifest is implemented.
No TensorFlow training, repository writes, or external resources are required.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from zenerestimation.experiment.manifest import (
    ExperimentManifest,
    ManifestValidationError,
)


@pytest.fixture
def valid_payload() -> dict:
    return {
        "schema_version": "1.0",
        "experiment_id": "gru-110-001",
        "created_at": "2026-10-20T09:00:00Z",
        "specification": {
            "dataset": {"identifier": "732B-5610110", "fingerprint": None},
            "model": {
                "name": "GRUForecaster",
                "parameters": {"units": 24, "window": 4},
                "seed": 42,
            },
            "evaluation": {
                "metric": "rmse",
                "alignment_policy": "verified",
                "splitter": {
                    "name": "ExpandingWindowSplitter",
                    "parameters": {"folds": 3},
                },
            },
            "preprocessing": {},
        },
        "execution": {
            "status": "pending",
            "environment": {},
            "metrics": {},
            "artifacts": {},
            "error": None,
        },
    }


def test_valid_manifest_roundtrip_dict(valid_payload):
    manifest = ExperimentManifest.from_dict(valid_payload)
    assert manifest.validate() is None
    assert manifest.to_dict() == valid_payload


def test_input_payload_is_defensively_copied(valid_payload):
    manifest = ExperimentManifest.from_dict(valid_payload)
    valid_payload["specification"]["model"]["parameters"]["units"] = 999
    assert manifest.to_dict()["specification"]["model"]["parameters"]["units"] == 24


def test_returned_payload_is_defensively_copied(valid_payload):
    manifest = ExperimentManifest.from_dict(valid_payload)
    returned = manifest.to_dict()
    returned["specification"]["model"]["parameters"]["units"] = 999
    assert manifest.to_dict()["specification"]["model"]["parameters"]["units"] == 24


@pytest.mark.parametrize("field", ["schema_version", "experiment_id", "created_at", "specification", "execution"])
def test_missing_top_level_required_field_rejected(valid_payload, field):
    del valid_payload[field]
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


@pytest.mark.parametrize("value", ["", "2.0", 1, None])
def test_unsupported_schema_version_rejected(valid_payload, value):
    valid_payload["schema_version"] = value
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


@pytest.mark.parametrize("value", ["", "  ", None, 123])
def test_invalid_experiment_id_rejected(valid_payload, value):
    valid_payload["experiment_id"] = value
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


@pytest.mark.parametrize("value", ["2026-10-20T09:00:00", "not-a-date", "2026-13-20T09:00:00Z", None])
def test_timestamp_requires_valid_timezone_aware_iso8601(valid_payload, value):
    valid_payload["created_at"] = value
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


def test_timezone_offset_timestamp_accepted(valid_payload):
    valid_payload["created_at"] = "2026-10-20T12:00:00+03:00"
    assert ExperimentManifest.from_dict(valid_payload).to_dict() == valid_payload


@pytest.mark.parametrize("value", ["", "  ", None, 7])
def test_invalid_dataset_identifier_rejected(valid_payload, value):
    valid_payload["specification"]["dataset"]["identifier"] = value
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


def test_valid_sha256_fingerprint_accepted(valid_payload):
    valid_payload["specification"]["dataset"]["fingerprint"] = {
        "algorithm": "sha256", "digest": "a" * 64
    }
    assert ExperimentManifest.from_dict(valid_payload).to_dict() == valid_payload


@pytest.mark.parametrize("fingerprint", ["bad", {"algorithm": "md5", "digest": "a" * 64}, {"algorithm": "sha256", "digest": "xyz"}, {"algorithm": "sha256"}])
def test_invalid_fingerprint_rejected(valid_payload, fingerprint):
    valid_payload["specification"]["dataset"]["fingerprint"] = fingerprint
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


@pytest.mark.parametrize("value", ["", "  ", None, 2])
def test_invalid_model_name_rejected(valid_payload, value):
    valid_payload["specification"]["model"]["name"] = value
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


@pytest.mark.parametrize("value", [-1, True, 1.5, "42"])
def test_invalid_model_seed_rejected(valid_payload, value):
    valid_payload["specification"]["model"]["seed"] = value
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


def test_null_seed_accepted(valid_payload):
    valid_payload["specification"]["model"]["seed"] = None
    assert ExperimentManifest.from_dict(valid_payload).to_dict() == valid_payload


@pytest.mark.parametrize("value", ["none", "", None, 1])
def test_invalid_alignment_policy_rejected(valid_payload, value):
    valid_payload["specification"]["evaluation"]["alignment_policy"] = value
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


def test_positional_alignment_policy_accepted(valid_payload):
    valid_payload["specification"]["evaluation"]["alignment_policy"] = "positional_only"
    assert ExperimentManifest.from_dict(valid_payload).to_dict() == valid_payload


@pytest.mark.parametrize("value", ["pending", "running", "completed", "failed"])
def test_valid_execution_statuses(valid_payload, value):
    valid_payload["execution"]["status"] = value
    assert ExperimentManifest.from_dict(valid_payload).to_dict() == valid_payload


@pytest.mark.parametrize("value", ["cancelled", "", None, 0])
def test_invalid_execution_status_rejected(valid_payload, value):
    valid_payload["execution"]["status"] = value
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0.1"])
def test_nonfinite_or_nonnumeric_metric_rejected(valid_payload, value):
    valid_payload["execution"]["metrics"] = {"rmse": value}
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


def test_finite_metrics_accepted(valid_payload):
    valid_payload["execution"]["metrics"] = {"rmse": 0.123, "mae": 0}
    assert ExperimentManifest.from_dict(valid_payload).to_dict() == valid_payload


def test_non_json_model_parameter_rejected(valid_payload):
    valid_payload["specification"]["model"]["parameters"]["bad"] = {1, 2}
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


def test_nonfinite_model_parameter_rejected(valid_payload):
    valid_payload["specification"]["model"]["parameters"]["bad"] = float("nan")
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


@pytest.mark.parametrize("section,extra", [
    ((), "unexpected"),
    (("specification",), "unexpected"),
    (("specification", "model"), "unitz"),
    (("execution",), "unexpected"),
])
def test_unknown_contract_fields_rejected(valid_payload, section, extra):
    target = valid_payload
    for key in section:
        target = target[key]
    target[extra] = 1
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


def test_deterministic_json_save_and_load(valid_payload, tmp_path: Path):
    manifest = ExperimentManifest.from_dict(valid_payload)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    manifest.save_json(first)
    manifest.save_json(second)
    assert first.read_bytes() == second.read_bytes()
    assert ExperimentManifest.load_json(first).to_dict() == valid_payload
    assert json.loads(first.read_text(encoding="utf-8")) == valid_payload


def test_load_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "invalid.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.load_json(path)


def test_roundtrip_does_not_mutate_source(valid_payload):
    original = copy.deepcopy(valid_payload)
    ExperimentManifest.from_dict(valid_payload).to_dict()
    assert valid_payload == original


@pytest.mark.parametrize("document", [
    '{"schema_version": "1.0", "schema_version": "2.0"}',
    '{"outer": {"seed": 42, "seed": 99}}',
])
def test_load_rejects_duplicate_json_keys(tmp_path: Path, document):
    path = tmp_path / "duplicate.json"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="duplicate JSON key"):
        ExperimentManifest.load_json(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {"nested": [float("-inf")]}, {"bad": {1, 2}}])
def test_nested_invalid_environment_metadata_rejected(valid_payload, value):
    valid_payload["execution"]["environment"]["dependency"] = value
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.from_dict(valid_payload)


def test_nested_returned_dict_does_not_modify_manifest(valid_payload):
    valid_payload["execution"]["environment"] = {"packages": {"tensorflow": "2.x"}}
    manifest = ExperimentManifest.from_dict(valid_payload)
    returned = manifest.to_dict()
    returned["execution"]["environment"]["packages"]["tensorflow"] = "changed"
    assert manifest.to_dict()["execution"]["environment"]["packages"]["tensorflow"] == "2.x"
