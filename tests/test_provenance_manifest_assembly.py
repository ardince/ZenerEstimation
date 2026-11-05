"""E2-C2: opt-in assembly and supplied-input verification of E1 manifests."""
from copy import deepcopy
import json

import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.experiment.assembly import (
    ManifestProvenanceError,
    PROVENANCE_VERSION,
    build_provenance_manifest,
    verify_provenance_manifest,
)
from zenerestimation.experiment.manifest import ExperimentManifest
from zenerestimation.optimization.validation import ExpandingWindowSplitter


def frame():
    return pd.DataFrame({
        "ds": pd.date_range("2022-03-01", periods=12, freq="QS-MAR"),
        "microVolt": [float(100 + i) for i in range(12)],
        "is_observed": [True] * 12,
    })


def inputs():
    data = frame()
    folds = ExpandingWindowSplitter(folds=2, validation_steps=2).split(BatteryDataset(data))
    return data, folds


def template():
    return {
        "schema_version": "1.0", "experiment_id": "e2-c2-test",
        "created_at": "2026-10-26T09:00:00+00:00",
        "specification": {
            "dataset": {"identifier": "410", "fingerprint": None},
            "model": {"name": "LSTM", "parameters": {"window": 4}, "seed": 42},
            "evaluation": {"metric": "rmse", "alignment_policy": "verified",
                           "splitter": {"name": "ExpandingWindowSplitter",
                                        "parameters": {"folds": 2, "validation_steps": 2}}},
            "preprocessing": {"enabled": False},
        },
        "execution": {"status": "pending", "environment": {}, "metrics": {},
                      "artifacts": {}, "error": None},
    }


def environment():
    from zenerestimation.experiment.environment import collect_environment_provenance
    return collect_environment_provenance()


def built():
    data, folds = inputs()
    return build_provenance_manifest(template(), data, folds, environment=environment()), data, folds


def test_build_returns_manifest():
    manifest, _, _ = built()
    assert isinstance(manifest, ExperimentManifest)
    assert manifest.to_dict()["schema_version"] == "1.0"


def test_dataset_fingerprint_is_e1_subset():
    manifest, _, _ = built()
    record = manifest.to_dict()["specification"]["dataset"]["fingerprint"]
    assert set(record) == {"algorithm", "digest"}
    assert len(record["digest"]) == 64


def test_versioned_dataset_metadata_preserved():
    manifest, _, _ = built()
    prov = manifest.to_dict()["specification"]["preprocessing"]["provenance"]
    assert prov["schema_version"] == PROVENANCE_VERSION
    assert prov["dataset"]["row_count"] == 12
    assert prov["dataset"]["canonicalization_version"] == "zener-canonical-dataset-v1"


def test_folds_recorded_in_order():
    manifest, _, _ = built()
    records = manifest.to_dict()["specification"]["preprocessing"]["provenance"]["training_partitions"]
    assert [r["fold_index"] for r in records] == [0, 1]
    assert records[0]["train_end"] < records[0]["validation_start"]


def test_environment_stored():
    manifest, _, _ = built()
    assert manifest.to_dict()["execution"]["environment"] == environment()


def test_original_specification_unchanged():
    data, folds = inputs()
    source = template()
    baseline = deepcopy(source)
    build_provenance_manifest(source, data, folds, environment=environment())
    assert source == baseline


def test_original_dataset_unchanged():
    data, folds = inputs()
    before = data.copy(deep=True)
    build_provenance_manifest(template(), data, folds, environment=environment())
    pd.testing.assert_frame_equal(data, before)


def test_successful_verification():
    manifest, data, folds = built()
    assert verify_provenance_manifest(manifest, data, folds)


def test_environment_comparison_success():
    manifest, data, folds = built()
    assert verify_provenance_manifest(manifest, data, folds, environment=environment(), check_environment=True)


def test_environment_mismatch_returns_false():
    manifest, data, folds = built()
    env = environment()
    env["python"]["version"] = "3.12.0"
    assert not verify_provenance_manifest(manifest, data, folds, environment=env, check_environment=True)


def test_environment_ignored_by_default():
    manifest, data, folds = built()
    assert verify_provenance_manifest(manifest, data, folds)


def test_environment_argument_requires_opt_in():
    manifest, data, folds = built()
    with pytest.raises(ManifestProvenanceError, match="check_environment"):
        verify_provenance_manifest(manifest, data, folds, environment=environment())


def test_dataset_tampering_detected():
    manifest, data, folds = built()
    changed = data.copy()
    changed.loc[0, "microVolt"] += 0.1
    assert not verify_provenance_manifest(manifest, changed, folds)


def test_training_partition_tampering_detected():
    manifest, data, folds = built()
    changed = frame()
    changed.loc[0, "microVolt"] += 0.1
    altered = ExpandingWindowSplitter(folds=2, validation_steps=2).split(BatteryDataset(changed))
    assert not verify_provenance_manifest(manifest, data, altered)


def test_missing_fold_detected():
    manifest, data, folds = built()
    assert not verify_provenance_manifest(manifest, data, folds[:1])


def test_reordered_folds_rejected():
    data, folds = inputs()
    with pytest.raises(ManifestProvenanceError, match="indices"):
        build_provenance_manifest(template(), data, folds[::-1], environment=environment())


def test_empty_folds_rejected():
    data, _ = inputs()
    with pytest.raises(ManifestProvenanceError, match="nonempty"):
        build_provenance_manifest(template(), data, [], environment=environment())


def test_declared_fold_count_rejected():
    data, folds = inputs()
    source = template()
    source["specification"]["evaluation"]["splitter"]["parameters"]["folds"] = 3
    with pytest.raises(ManifestProvenanceError, match="folds"):
        build_provenance_manifest(source, data, folds, environment=environment())


def test_validation_steps_mismatch_rejected():
    data, folds = inputs()
    source = template()
    source["specification"]["evaluation"]["splitter"]["parameters"]["validation_steps"] = 3
    with pytest.raises(ManifestProvenanceError, match="validation_steps"):
        build_provenance_manifest(source, data, folds, environment=environment())


def test_existing_fingerprint_rejected():
    data, folds = inputs()
    source = template()
    source["specification"]["dataset"]["fingerprint"] = {"algorithm": "sha256", "digest": "0" * 64}
    with pytest.raises(ManifestProvenanceError, match="fingerprint"):
        build_provenance_manifest(source, data, folds, environment=environment())


def test_existing_provenance_rejected():
    data, folds = inputs()
    source = template()
    source["specification"]["preprocessing"]["provenance"] = {}
    with pytest.raises(ManifestProvenanceError, match="namespace"):
        build_provenance_manifest(source, data, folds, environment=environment())


def test_existing_environment_rejected():
    data, folds = inputs()
    source = template()
    source["execution"]["environment"] = {"old": True}
    with pytest.raises(ManifestProvenanceError, match="environment"):
        build_provenance_manifest(source, data, folds, environment=environment())


def test_invalid_environment_rejected():
    data, folds = inputs()
    with pytest.raises(ValueError):
        build_provenance_manifest(template(), data, folds, environment={})


def test_missing_provenance_rejected():
    manifest, data, folds = built()
    payload = manifest.to_dict()
    del payload["specification"]["preprocessing"]["provenance"]
    with pytest.raises(ManifestProvenanceError, match="missing"):
        verify_provenance_manifest(ExperimentManifest.from_dict(payload), data, folds)


def test_unknown_provenance_version_rejected():
    manifest, data, folds = built()
    payload = manifest.to_dict()
    payload["specification"]["preprocessing"]["provenance"]["schema_version"] = "v999"
    with pytest.raises(ManifestProvenanceError, match="unsupported"):
        verify_provenance_manifest(ExperimentManifest.from_dict(payload), data, folds)


def test_cross_field_digest_tampering_rejected():
    manifest, data, folds = built()
    payload = manifest.to_dict()
    payload["specification"]["dataset"]["fingerprint"]["digest"] = "0" * 64
    with pytest.raises(ManifestProvenanceError, match="differs"):
        verify_provenance_manifest(ExperimentManifest.from_dict(payload), data, folds)


def test_json_roundtrip(tmp_path):
    manifest, data, folds = built()
    target = tmp_path / "manifest.json"
    manifest.save_json(target)
    restored = ExperimentManifest.load_json(target)
    assert restored.to_dict() == manifest.to_dict()
    assert verify_provenance_manifest(restored, data, folds)


def test_deterministic_serialization(tmp_path):
    manifest, _, _ = built()
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    manifest.save_json(first)
    manifest.save_json(second)
    assert first.read_bytes() == second.read_bytes()


def test_json_compatible():
    manifest, _, _ = built()
    json.dumps(manifest.to_dict(), allow_nan=False)


def test_preserves_metrics_and_artifacts():
    data, folds = inputs()
    source = template()
    source["execution"]["metrics"] = {"rmse": 0.25}
    source["execution"]["artifacts"] = {"optimization": "relative.json"}
    manifest = build_provenance_manifest(source, data, folds, environment=environment())
    assert manifest.to_dict()["execution"]["metrics"] == {"rmse": 0.25}
    assert manifest.to_dict()["execution"]["artifacts"] == {"optimization": "relative.json"}
