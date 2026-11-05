"""Fast isolated E2 linkage contract tests; full provenance tests run separately."""
from copy import deepcopy
import json
import pytest

from zenerestimation.experiment.manifest import ExperimentManifest
from zenerestimation.optimization import neural_provenance as p


def _manifest():
    payload = {
        'schema_version':'1.0', 'experiment_id':'neural-001',
        'created_at':'2026-10-20T09:00:00Z',
        'specification': {
            'dataset': {'identifier':'battery', 'fingerprint':None},
            'model': {'name':'LSTM', 'parameters':{}, 'seed':42},
            'evaluation': {'metric':'rmse', 'alignment_policy':'verified',
                           'splitter': {'name':'ExpandingWindowSplitter','parameters':{'folds':2}}},
            'preprocessing': {'provenance': {'dataset':{'digest':'abc'},
                'training_partitions':[{'digest':'one'},{'digest':'two'}]}}},
        'execution': {'status':'pending','environment':{},'metrics':{},'artifacts':{},'error':None}}
    return ExperimentManifest.from_dict(payload)


def _artifact():
    return {'schema_version':'neural-training-artifact-v1',
            'scope':'internal_temporal_neural_training_only',
            'candidates':[{'fold_training':[{'fold_index':1},{'fold_index':2}]}]}


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setattr(p, 'verify_provenance_manifest', lambda *a, **kw: True)
    return _artifact(), _manifest()


def test_build_and_verify(inputs):
    artifact, manifest = inputs
    link = p.build_neural_provenance_link(artifact, manifest, None, [])
    assert link['training_partition_digests'] == ['one','two']
    assert p.verify_neural_provenance_link(link, artifact, manifest, None, [])


def test_tampered_artifact_is_detected(inputs):
    artifact, manifest = inputs
    link = p.build_neural_provenance_link(artifact, manifest, None, [])
    modified = deepcopy(artifact)
    modified['candidates'][0]['fold_training'][0]['loss'] = 0.1
    assert not p.verify_neural_provenance_link(link, modified, manifest, None, [])


def test_tampered_link_is_detected(inputs):
    artifact, manifest = inputs
    link = p.build_neural_provenance_link(artifact, manifest, None, [])
    link['manifest_sha256'] = '0' * 64
    assert not p.verify_neural_provenance_link(link, artifact, manifest, None, [])


@pytest.mark.parametrize('indices', [[0,1], [1,1], [1,3], ['1',2]])
def test_bad_fold_index_rejected(inputs, indices):
    artifact, manifest = inputs
    artifact['candidates'][0]['fold_training'] = [{'fold_index':i} for i in indices]
    with pytest.raises(p.NeuralProvenanceLinkError):
        p.build_neural_provenance_link(artifact, manifest, None, [])


def test_mismatched_manifest_evidence_rejected(inputs, monkeypatch):
    artifact, manifest = inputs
    monkeypatch.setattr(p, 'verify_provenance_manifest', lambda *a, **kw: False)
    with pytest.raises(p.NeuralProvenanceLinkError):
        p.build_neural_provenance_link(artifact, manifest, None, [])


def test_write_link_roundtrip(inputs, tmp_path):
    artifact, manifest = inputs
    link = p.build_neural_provenance_link(artifact, manifest, None, [])
    path = p.write_neural_provenance_link(link, tmp_path / 'link.json')
    assert json.loads(path.read_text(encoding='utf-8')) == link
