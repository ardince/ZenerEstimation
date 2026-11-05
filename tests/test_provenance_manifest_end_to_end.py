"""E2-C3: persistence lifecycle and supplied-input provenance verification.

These tests intentionally do not run forecasting or mutate production artifacts.
"""
from copy import deepcopy
import json

import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.experiment.assembly import (
    ManifestProvenanceError,
    build_provenance_manifest,
    verify_provenance_manifest,
)
from zenerestimation.experiment.environment import collect_environment_provenance
from zenerestimation.experiment.manifest import ExperimentManifest, ManifestValidationError
from zenerestimation.optimization.validation import ExpandingWindowSplitter


def _frame():
    return pd.DataFrame({
        "ds": pd.date_range("2022-03-01", periods=12, freq="QS-MAR"),
        "microVolt": [float(100 + i) for i in range(12)],
        "is_observed": [True] * 12,
    })


def _template():
    return {
        "schema_version": "1.0",
        "experiment_id": "e2-c3-roundtrip",
        "created_at": "2026-10-26T09:00:00+00:00",
        "specification": {
            "dataset": {"identifier": "410", "fingerprint": None},
            "model": {"name": "LSTM", "parameters": {"window": 4}, "seed": 42},
            "evaluation": {
                "metric": "rmse", "alignment_policy": "verified",
                "splitter": {"name": "ExpandingWindowSplitter", "parameters": {"folds": 2, "validation_steps": 2}},
            },
            "preprocessing": {"enabled": False},
        },
        "execution": {"status": "pending", "environment": {}, "metrics": {}, "artifacts": {}, "error": None},
    }


def _folds(data):
    return ExpandingWindowSplitter(folds=2, validation_steps=2).split(BatteryDataset(data))


@pytest.fixture
def scenario(tmp_path):
    data = _frame()
    folds = _folds(data)
    environment = collect_environment_provenance()
    manifest = build_provenance_manifest(_template(), data, folds, environment=environment)
    path = tmp_path / "experiment_manifest.json"
    manifest.save_json(path)
    loaded = ExperimentManifest.load_json(path)
    return loaded, data, folds, environment, path


def _alter(manifest, mutate):
    payload = manifest.to_dict()
    mutate(payload)
    return ExperimentManifest.from_dict(payload)


def test_complete_build_save_reload_verify(scenario):
    manifest, data, folds, _, path = scenario
    assert path.exists()
    assert manifest.to_dict()["schema_version"] == "1.0"
    assert verify_provenance_manifest(manifest, data, folds) is True


def test_reload_preserves_manifest_payload(scenario):
    manifest, _, _, _, path = scenario
    assert ExperimentManifest.load_json(path).to_dict() == manifest.to_dict()


def test_serialization_is_deterministic(scenario, tmp_path):
    manifest, _, _, _, first = scenario
    second = tmp_path / "second.json"
    manifest.save_json(second)
    assert first.read_bytes() == second.read_bytes()


def test_rebuilding_with_same_explicit_environment_is_deterministic(scenario):
    manifest, data, folds, environment, _ = scenario
    rebuilt = build_provenance_manifest(_template(), data, folds, environment=environment)
    assert rebuilt.to_dict() == manifest.to_dict()


def test_dataset_measurement_tampering_returns_false(scenario):
    manifest, data, folds, _, _ = scenario
    changed = data.copy(deep=True)
    changed.loc[0, "microVolt"] += 0.5
    assert verify_provenance_manifest(manifest, changed, folds) is False


def test_training_measurement_tampering_returns_false(scenario):
    manifest, data, _, _, _ = scenario
    changed = data.copy(deep=True)
    changed.loc[0, "microVolt"] += 0.5
    assert verify_provenance_manifest(manifest, data, _folds(changed)) is False


def test_fold_boundary_change_returns_false(scenario):
    manifest, data, _, _, _ = scenario
    changed_folds = ExpandingWindowSplitter(folds=2, validation_steps=1).split(BatteryDataset(data))
    # A different validation length conflicts with the declared splitter configuration.
    with pytest.raises(ManifestProvenanceError, match="validation_steps"):
        verify_provenance_manifest(manifest, data, changed_folds)


def test_missing_fold_returns_false(scenario):
    manifest, data, folds, _, _ = scenario
    assert verify_provenance_manifest(manifest, data, folds[:1]) is False


def test_missing_provenance_rejected(scenario):
    manifest, data, folds, _, _ = scenario
    altered = _alter(manifest, lambda p: p["specification"]["preprocessing"].pop("provenance"))
    with pytest.raises(ManifestProvenanceError, match="missing"):
        verify_provenance_manifest(altered, data, folds)


def test_missing_partition_evidence_rejected(scenario):
    manifest, data, folds, _, _ = scenario
    altered = _alter(manifest, lambda p: p["specification"]["preprocessing"]["provenance"].update(training_partitions=[]))
    with pytest.raises(ManifestProvenanceError, match="training_partitions"):
        verify_provenance_manifest(altered, data, folds)


def test_mismatched_dataset_fingerprint_fields_rejected(scenario):
    manifest, data, folds, _, _ = scenario
    altered = _alter(manifest, lambda p: p["specification"]["dataset"]["fingerprint"].update(digest="a" * 64))
    with pytest.raises(ManifestProvenanceError, match="dataset digest"):
        verify_provenance_manifest(altered, data, folds)


def test_environment_check_matching(scenario):
    manifest, data, folds, environment, _ = scenario
    assert verify_provenance_manifest(manifest, data, folds, environment=environment, check_environment=True)


def test_environment_check_mismatch_returns_false(scenario):
    manifest, data, folds, environment, _ = scenario
    changed = deepcopy(environment)
    changed["python"]["version"] = "9.9.9"
    assert verify_provenance_manifest(manifest, data, folds, environment=changed, check_environment=True) is False


def test_environment_mismatch_ignored_by_default(scenario):
    manifest, data, folds, _, _ = scenario
    assert verify_provenance_manifest(manifest, data, folds) is True


def test_environment_supplied_without_opt_in_rejected(scenario):
    manifest, data, folds, environment, _ = scenario
    with pytest.raises(ManifestProvenanceError, match="check_environment"):
        verify_provenance_manifest(manifest, data, folds, environment=environment)


def test_malformed_environment_rejected(scenario):
    manifest, data, folds, environment, _ = scenario
    changed = deepcopy(environment)
    changed["dependencies"]["numpy"] = ["not-a-version"]
    with pytest.raises(ValueError):
        verify_provenance_manifest(manifest, data, folds, environment=changed, check_environment=True)


def test_invalid_json_constant_rejected(scenario):
    _, _, _, _, path = scenario
    content = path.read_text(encoding="utf-8")
    content = content.replace('"seed": 42', '"seed": NaN')
    assert content != path.read_text(encoding="utf-8")
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.load_json(path)


def test_duplicate_json_key_rejected(scenario):
    _, _, _, _, path = scenario
    content = path.read_text(encoding="utf-8")
    content = content.replace('"schema_version": "1.0",', '"schema_version": "1.0", "schema_version": "1.0",', 1)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ManifestValidationError):
        ExperimentManifest.load_json(path)


def test_build_and_verify_do_not_mutate_inputs(scenario):
    manifest, data, folds, environment, _ = scenario
    original_data = data.copy(deep=True)
    original_train = [fold.train.data.copy(deep=True) for fold in folds]
    original_validation = [fold.validation.copy(deep=True) for fold in folds]
    original_environment = deepcopy(environment)
    original_manifest = manifest.to_dict()
    template = _template()
    original_template = deepcopy(template)
    build_provenance_manifest(template, data, folds, environment=environment)
    verify_provenance_manifest(manifest, data, folds, environment=environment, check_environment=True)
    assert template == original_template
    assert environment == original_environment
    assert manifest.to_dict() == original_manifest
    pd.testing.assert_frame_equal(data, original_data)
    for fold, train, validation in zip(folds, original_train, original_validation):
        pd.testing.assert_frame_equal(fold.train.data, train)
        pd.testing.assert_frame_equal(fold.validation, validation)


def test_manifest_contains_no_validation_targets(scenario):
    manifest, _, folds, _, _ = scenario
    provenance = manifest.to_dict()["specification"]["preprocessing"]["provenance"]
    assert len(provenance["training_partitions"]) == len(folds)
    for record in provenance["training_partitions"]:
        assert set(record) == {
            "algorithm", "digest", "canonicalization_version", "row_count",
            "train_start", "train_end", "validation_start", "validation_end", "fold_index",
        }
        assert "microVolt" not in record
