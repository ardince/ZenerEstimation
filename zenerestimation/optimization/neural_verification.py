"""16.3-E3: strict on-disk integrity and scope checks for E1/E2 evidence.

Verification binds *supplied* evidence; it cannot establish historical execution.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .neural_artifact import SCHEMA_VERSION
from .neural_provenance import (
    LINK_SCHEMA, NeuralProvenanceLinkError, verify_neural_provenance_link,
)


class NeuralEvidenceVerificationError(ValueError):
    """Malformed, out-of-scope or inconsistent neural evidence."""


# Keys denoting external holdout observations or outcomes, not training metrics.
_FORBIDDEN_KEYS = frozenset({
    'future_true', 'future_actual', 'holdout_actual', 'benchmark_actual',
    'benchmark_predictions', 'benchmark_rmse', 'external_test_rmse',
    'external_benchmark', 'holdout_predictions', 'test_targets',
})


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NeuralEvidenceVerificationError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise NeuralEvidenceVerificationError(f'non-finite JSON number: {value}')


def load_neural_evidence_json(path: str | Path) -> dict[str, Any]:
    """Load JSON strictly: reject duplicate keys, NaN, Infinity and non-objects."""
    source = Path(path)
    if source.suffix.lower() != '.json':
        raise NeuralEvidenceVerificationError('evidence file must end in .json')
    try:
        payload = json.loads(source.read_text(encoding='utf-8'),
                             object_pairs_hook=_unique_pairs,
                             parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NeuralEvidenceVerificationError('cannot read strict JSON evidence') from exc
    if not isinstance(payload, dict):
        raise NeuralEvidenceVerificationError('JSON evidence must be an object')
    return payload


def _check_no_external_targets(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() in _FORBIDDEN_KEYS:
                raise NeuralEvidenceVerificationError(f'external benchmark field forbidden: {key}')
            _check_no_external_targets(nested)
    elif isinstance(value, list):
        for nested in value:
            _check_no_external_targets(nested)


def validate_neural_training_evidence(artifact: Mapping[str, Any]) -> None:
    """Check E1 schema, candidate identity and internal-training-only scope."""
    if not isinstance(artifact, Mapping) or artifact.get('schema_version') != SCHEMA_VERSION:
        raise NeuralEvidenceVerificationError('unsupported neural training artifact schema')
    if artifact.get('scope') != 'internal_temporal_neural_training_only':
        raise NeuralEvidenceVerificationError('invalid neural training scope')
    if artifact.get('interpretation') != 'descriptive_training_evidence_only; not_generalization_evidence':
        raise NeuralEvidenceVerificationError('invalid training evidence interpretation')
    if not isinstance(artifact.get('candidates'), list):
        raise NeuralEvidenceVerificationError('missing candidate records')
    _check_no_external_targets(artifact)
    for index, candidate in enumerate(artifact['candidates'], 1):
        if not isinstance(candidate, dict) or type(candidate.get('candidate_index')) is not int or candidate['candidate_index'] != index:
            raise NeuralEvidenceVerificationError('candidate indices must be sequential and 1-based')
        if candidate.get('status') not in ('evaluated', 'failed'):
            raise NeuralEvidenceVerificationError('invalid candidate status')
        if not isinstance(candidate.get('params'), dict):
            raise NeuralEvidenceVerificationError('invalid candidate parameters')
        records = candidate.get('fold_training')
        if records is not None:
            if not isinstance(records, list):
                raise NeuralEvidenceVerificationError('fold training must be list or null')
            for record in records:
                if not isinstance(record, dict):
                    raise NeuralEvidenceVerificationError('invalid fold record')
    # No requirement for history: older and failed candidates may lack diagnostics.


def verify_neural_evidence_files(
    artifact_path: str | Path, link_path: str | Path,
    manifest: Any, dataset: Any, folds: Any,
    *, environment: Mapping[str, Any] | None = None,
    check_environment: bool = False,
) -> bool:
    """Validate strict files and verify E2 content binding and manifest inputs.

    Returns False for content mismatches; malformed evidence raises.
    """
    artifact = load_neural_evidence_json(artifact_path)
    link = load_neural_evidence_json(link_path)
    validate_neural_training_evidence(artifact)
    if link.get('schema_version') != LINK_SCHEMA:
        raise NeuralEvidenceVerificationError('unsupported neural provenance link schema')
    try:
        return verify_neural_provenance_link(
            link, artifact, manifest, dataset, folds,
            environment=environment, check_environment=check_environment,
        )
    except NeuralProvenanceLinkError as exc:
        raise NeuralEvidenceVerificationError(str(exc)) from exc
