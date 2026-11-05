"""16.3-E2: detached, content-addressed linkage of neural training evidence.

This is an assertion about supplied artifacts, not proof of historical execution.
Neither the existing ExperimentManifest nor the E1 artifact is modified.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from zenerestimation.experiment.manifest import ExperimentManifest
from zenerestimation.experiment.assembly import verify_provenance_manifest
from .neural_artifact import SCHEMA_VERSION as NEURAL_SCHEMA

LINK_SCHEMA = "neural-training-provenance-link-v1"


class NeuralProvenanceLinkError(ValueError):
    """Invalid or inconsistent neural artifact provenance linkage."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NeuralProvenanceLinkError("payload is not finite JSON data") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _check_artifact(artifact: Mapping[str, Any], manifest_payload: dict) -> None:
    if not isinstance(artifact, Mapping) or artifact.get("schema_version") != NEURAL_SCHEMA:
        raise NeuralProvenanceLinkError("unsupported neural training artifact")
    if artifact.get("scope") != "internal_temporal_neural_training_only":
        raise NeuralProvenanceLinkError("artifact scope must be internal temporal training")
    candidates = artifact.get("candidates")
    if not isinstance(candidates, list):
        raise NeuralProvenanceLinkError("artifact candidates must be a list")
    provenance = manifest_payload["specification"]["preprocessing"].get("provenance")
    if not isinstance(provenance, dict):
        raise NeuralProvenanceLinkError("manifest has no provenance namespace")
    partitions = provenance.get("training_partitions")
    if not isinstance(partitions, list) or not partitions:
        raise NeuralProvenanceLinkError("manifest has no training partitions")
    expected = set(range(1, len(partitions) + 1))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise NeuralProvenanceLinkError("invalid candidate")
        records = candidate.get("fold_training")
        if records is None:
            continue  # Evidence may be absent for failed/non-neural candidates.
        if not isinstance(records, list):
            raise NeuralProvenanceLinkError("fold_training must be a list or null")
        seen = []
        for record in records:
            if not isinstance(record, dict) or type(record.get("fold_index")) is not int:
                raise NeuralProvenanceLinkError("invalid fold training index")
            seen.append(record["fold_index"])
        if len(seen) != len(set(seen)) or not set(seen).issubset(expected):
            raise NeuralProvenanceLinkError("fold training indices do not match manifest folds")


def build_neural_provenance_link(artifact: Mapping[str, Any],
                                 manifest: ExperimentManifest, dataset: Any,
                                 folds: Any, *, environment: Mapping[str, Any] | None = None,
                                 check_environment: bool = False) -> dict[str, Any]:
    """Link supplied artifact and verified supplied dataset/folds to manifest.

    A successful check does not prove the artifact was produced by this execution.
    """
    if not isinstance(manifest, ExperimentManifest):
        raise TypeError("manifest must be an ExperimentManifest")
    if not verify_provenance_manifest(manifest, dataset, folds,
                                      environment=environment,
                                      check_environment=check_environment):
        raise NeuralProvenanceLinkError("manifest does not match supplied provenance inputs")
    payload = manifest.to_dict()
    _check_artifact(artifact, payload)
    return {
        "schema_version": LINK_SCHEMA,
        "claim": "supplied_artifact_and_verified_manifest_binding_only",
        "experiment_id": payload["experiment_id"],
        "artifact_sha256": _digest(artifact),
        "manifest_sha256": _digest(payload),
        "dataset_digest": payload["specification"]["preprocessing"]["provenance"]["dataset"]["digest"],
        "training_partition_digests": [p["digest"] for p in payload["specification"]["preprocessing"]["provenance"]["training_partitions"]],
        "fold_index_convention": "artifact_1_based_manifest_0_based",
    }


def verify_neural_provenance_link(link: Mapping[str, Any], artifact: Mapping[str, Any],
                                  manifest: ExperimentManifest, dataset: Any, folds: Any,
                                  *, environment: Mapping[str, Any] | None = None,
                                  check_environment: bool = False) -> bool:
    """Return False on mismatched evidence; malformed supplied inputs raise."""
    expected = build_neural_provenance_link(
        artifact, manifest, dataset, folds,
        environment=environment, check_environment=check_environment)
    return isinstance(link, Mapping) and dict(link) == expected


def write_neural_provenance_link(link: Mapping[str, Any], path: str | Path) -> Path:
    """Write detached linkage as strict JSON; does not overwrite the manifest."""
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise ValueError("link path must end in .json")
    data = _canonical(link)
    destination.write_text(json.dumps(json.loads(data), indent=2, sort_keys=True,
                                      ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return destination
