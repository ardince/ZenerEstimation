"""Opt-in assembly and supplied-input verification of E1 experiment manifests.

This module does not execute models, alter optimization artifacts, or persist files.
It attests to the *supplied* dataset and folds, not to historical execution.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from zenerestimation.data.fingerprint import (
    fingerprint_dataset,
    fingerprint_training_partition,
    verify_dataset_fingerprint,
    verify_training_partition_fingerprint,
)
from zenerestimation.experiment.environment import (
    collect_environment_provenance,
    validate_environment_provenance,
)
from zenerestimation.experiment.manifest import ExperimentManifest

PROVENANCE_KEY = "provenance"
PROVENANCE_VERSION = "zener-manifest-provenance-v1"


class ManifestProvenanceError(ValueError):
    """Missing, malformed or inconsistent provenance assembly inputs."""


def _fold_records(folds: Sequence[Any]) -> list[dict[str, Any]]:
    if isinstance(folds, (str, bytes)) or not isinstance(folds, Sequence) or not folds:
        raise ManifestProvenanceError("folds must be a nonempty sequence of TemporalFold objects")
    records = [fingerprint_training_partition(fold).to_dict() for fold in folds]
    indices = [record["fold_index"] for record in records]
    if indices != list(range(len(indices))):
        raise ManifestProvenanceError("fold indices must be unique, ordered and contiguous from zero")
    return records


def _check_splitter(payload: dict[str, Any], records: list[dict[str, Any]]) -> None:
    splitter = payload["specification"]["evaluation"]["splitter"]
    parameters = splitter["parameters"]
    if "folds" in parameters:
        declared = parameters["folds"]
        if type(declared) is not int or declared != len(records):
            raise ManifestProvenanceError("splitter.parameters.folds disagrees with supplied folds")
    if "validation_steps" in parameters:
        steps = parameters["validation_steps"]
        if type(steps) is not int or steps <= 0:
            raise ManifestProvenanceError("splitter.parameters.validation_steps must be positive")
    # Boundary dates are available, but observation counts of validation frames
    # are not represented by the fingerprint record. Validate lengths from folds
    # separately in the public functions, not by inferring date intervals.


def _check_validation_lengths(payload: dict[str, Any], folds: Sequence[Any]) -> None:
    parameters = payload["specification"]["evaluation"]["splitter"]["parameters"]
    if "validation_steps" in parameters:
        steps = parameters["validation_steps"]
        if any(len(fold.validation) != steps for fold in folds):
            raise ManifestProvenanceError("validation_steps disagrees with a supplied fold")


def _provenance(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        provenance = payload["specification"]["preprocessing"][PROVENANCE_KEY]
    except (KeyError, TypeError) as exc:
        raise ManifestProvenanceError("missing specification.preprocessing.provenance") from exc
    if not isinstance(provenance, dict) or set(provenance) != {
        "schema_version", "dataset", "training_partitions"
    }:
        raise ManifestProvenanceError("invalid provenance fields")
    if provenance["schema_version"] != PROVENANCE_VERSION:
        raise ManifestProvenanceError("unsupported provenance schema version")
    dataset = provenance["dataset"]
    if not isinstance(dataset, dict) or set(dataset) != {
        "algorithm", "digest", "canonicalization_version", "row_count"
    }:
        raise ManifestProvenanceError("invalid versioned dataset provenance")
    records = provenance["training_partitions"]
    if not isinstance(records, list) or not records:
        raise ManifestProvenanceError("training_partitions must be a nonempty list")
    expected_fields = {
        "algorithm", "digest", "canonicalization_version", "row_count",
        "train_start", "train_end", "validation_start", "validation_end", "fold_index"
    }
    if any(not isinstance(record, dict) or set(record) != expected_fields for record in records):
        raise ManifestProvenanceError("invalid training-partition record fields")
    indices = [record["fold_index"] for record in records]
    if indices != list(range(len(records))):
        raise ManifestProvenanceError("invalid training-partition indices")
    return provenance


def build_provenance_manifest(
    specification: Mapping[str, Any],
    dataset: Any,
    folds: Sequence[Any],
    *,
    environment: Mapping[str, Any] | None = None,
) -> ExperimentManifest:
    """Build an E1 manifest from a complete E1 payload and supplied evidence.

    ``specification`` is a complete schema-1.0 manifest payload template,
    including experiment_id, created_at, specification and execution fields.
    Its dataset fingerprint must be null and its provenance namespace absent.
    Other specification and execution fields are preserved unchanged.
    """
    if not isinstance(specification, Mapping):
        raise TypeError("specification must be a complete manifest payload mapping")
    payload = deepcopy(dict(specification))
    # Validate the caller's template before writing into nested fields.
    ExperimentManifest.from_dict(payload)
    dataset_spec = payload["specification"]["dataset"]
    if dataset_spec["fingerprint"] is not None:
        raise ManifestProvenanceError("template dataset fingerprint must be null")
    preprocessing = payload["specification"]["preprocessing"]
    if PROVENANCE_KEY in preprocessing:
        raise ManifestProvenanceError("template provenance namespace must be absent")
    if payload["execution"]["environment"]:
        raise ManifestProvenanceError("template execution.environment must be empty")

    dataset_record = fingerprint_dataset(dataset)
    fold_records = _fold_records(folds)
    _check_splitter(payload, fold_records)
    _check_validation_lengths(payload, folds)
    env = collect_environment_provenance() if environment is None else deepcopy(dict(environment))
    validate_environment_provenance(env)

    dataset_spec["fingerprint"] = dataset_record.to_manifest_fingerprint()
    preprocessing[PROVENANCE_KEY] = {
        "schema_version": PROVENANCE_VERSION,
        "dataset": dataset_record.to_dict(),
        "training_partitions": fold_records,
    }
    payload["execution"]["environment"] = deepcopy(env)
    return ExperimentManifest.from_dict(payload)


def verify_provenance_manifest(
    manifest: ExperimentManifest,
    dataset: Any,
    folds: Sequence[Any],
    *,
    environment: Mapping[str, Any] | None = None,
    check_environment: bool = False,
) -> bool:
    """Verify provenance against *supplied* inputs; return False on mismatch.

    Malformed or missing evidence raises ManifestProvenanceError. By default,
    environment is validated but not compared to the current runtime.
    """
    if not isinstance(manifest, ExperimentManifest):
        raise TypeError("manifest must be an ExperimentManifest")
    if type(check_environment) is not bool:
        raise TypeError("check_environment must be a bool")
    manifest.validate()
    payload = manifest.to_dict()
    provenance = _provenance(payload)
    env_record = payload["execution"]["environment"]
    validate_environment_provenance(env_record)
    if check_environment:
        current = collect_environment_provenance() if environment is None else deepcopy(dict(environment))
        validate_environment_provenance(current)
        if current != env_record:
            return False
    elif environment is not None:
        raise ManifestProvenanceError("environment supplied without check_environment=True")

    records = provenance["training_partitions"]
    if len(records) != len(folds):
        return False
    actual_records = _fold_records(folds)
    _check_splitter(payload, records)
    _check_validation_lengths(payload, folds)
    if payload["specification"]["dataset"]["fingerprint"] != {
        "algorithm": provenance["dataset"]["algorithm"],
        "digest": provenance["dataset"]["digest"],
    }:
        raise ManifestProvenanceError("dataset digest differs between manifest fields")
    if not verify_dataset_fingerprint(dataset, provenance["dataset"]):
        return False
    return all(
        verify_training_partition_fingerprint(fold, record)
        for fold, record in zip(folds, records)
    )
