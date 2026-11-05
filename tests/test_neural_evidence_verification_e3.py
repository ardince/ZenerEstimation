"""E3 on-disk integrity and benchmark-isolation tests (no TensorFlow)."""
from copy import deepcopy
import json
import pytest

from zenerestimation.optimization import neural_provenance as provenance
from zenerestimation.optimization import neural_verification as v
from tests.test_neural_provenance_link_e2 import _manifest


def artifact():
    return {
        'schema_version': 'neural-training-artifact-v1',
        'scope': 'internal_temporal_neural_training_only',
        'interpretation': 'descriptive_training_evidence_only; not_generalization_evidence',
        'model': 'LSTM', 'selection_metric': 'rmse',
        'best_params': {'epochs': 2}, 'best_score': 0.25, 'budget': None,
        'candidates': [
            {'candidate_index': 1, 'status': 'evaluated', 'params': {'epochs': 2},
             'score': 0.25, 'metrics': {'rmse': 0.25},
             'fold_training': [
                 {'fold_index': 1, 'history': {'loss': [1.0, 0.5]}, 'diagnostics': {'completed_epochs': 2}},
                 {'fold_index': 2, 'history': {'loss': [0.9, 0.4]}}]},
        ],
    }


@pytest.fixture
def evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, 'verify_provenance_manifest', lambda *a, **k: True)
    a, m = artifact(), _manifest()
    link = provenance.build_neural_provenance_link(a, m, None, [])
    ap, lp = tmp_path / 'training.json', tmp_path / 'link.json'
    ap.write_text(json.dumps(a), encoding='utf-8')
    lp.write_text(json.dumps(link), encoding='utf-8')
    return ap, lp, m


def test_roundtrip_verifies(evidence):
    ap, lp, manifest = evidence
    assert v.verify_neural_evidence_files(ap, lp, manifest, None, [])


def test_tampered_history_rejected(evidence):
    ap, lp, manifest = evidence
    data = json.loads(ap.read_text())
    data['candidates'][0]['fold_training'][0]['history']['loss'][0] = 999
    ap.write_text(json.dumps(data), encoding='utf-8')
    assert not v.verify_neural_evidence_files(ap, lp, manifest, None, [])


def test_tampered_link_rejected(evidence):
    ap, lp, manifest = evidence
    data = json.loads(lp.read_text())
    data['artifact_sha256'] = '0' * 64
    lp.write_text(json.dumps(data), encoding='utf-8')
    assert not v.verify_neural_evidence_files(ap, lp, manifest, None, [])


def test_mismatched_manifest_rejected(evidence):
    ap, lp, _ = evidence
    other = _manifest()
    payload = other.to_dict()
    payload['experiment_id'] = 'another-experiment'
    from zenerestimation.experiment.manifest import ExperimentManifest
    assert not v.verify_neural_evidence_files(ap, lp, ExperimentManifest.from_dict(payload), None, [])


@pytest.mark.parametrize('field', ['future_true', 'benchmark_rmse', 'external_benchmark'])
def test_external_benchmark_fields_rejected(field):
    data = artifact()
    data['candidates'][0]['fold_training'][0][field] = [123.4]
    with pytest.raises(v.NeuralEvidenceVerificationError, match='benchmark'):
        v.validate_neural_training_evidence(data)


def test_missing_fold_history_allowed():
    data = artifact()
    data['candidates'][0]['fold_training'] = None
    v.validate_neural_training_evidence(data)


def test_bad_candidate_index_rejected():
    data = artifact()
    data['candidates'][0]['candidate_index'] = 0
    with pytest.raises(v.NeuralEvidenceVerificationError, match='indices'):
        v.validate_neural_training_evidence(data)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_text('{"schema_version":"a","schema_version":"b"}', encoding='utf-8')
    with pytest.raises(v.NeuralEvidenceVerificationError, match='duplicate'):
        v.load_neural_evidence_json(path)


@pytest.mark.parametrize('constant', ['NaN', 'Infinity', '-Infinity'])
def test_nonfinite_json_rejected(tmp_path, constant):
    path = tmp_path / 'nonfinite.json'
    path.write_text('{"value":' + constant + '}', encoding='utf-8')
    with pytest.raises(v.NeuralEvidenceVerificationError, match='non-finite'):
        v.load_neural_evidence_json(path)


def test_wrong_link_schema_rejected(evidence):
    ap, lp, manifest = evidence
    data = json.loads(lp.read_text())
    data['schema_version'] = 'unexpected'
    lp.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(v.NeuralEvidenceVerificationError, match='schema'):
        v.verify_neural_evidence_files(ap, lp, manifest, None, [])


def test_manifest_input_verification_failure(evidence, monkeypatch):
    ap, lp, manifest = evidence
    monkeypatch.setattr(provenance, 'verify_provenance_manifest', lambda *a, **k: False)
    with pytest.raises(v.NeuralEvidenceVerificationError, match='manifest'):
        v.verify_neural_evidence_files(ap, lp, manifest, None, [])
