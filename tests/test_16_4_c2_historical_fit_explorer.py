import hashlib
import json

import numpy as np
import pytest

from zenerestimation.visualization.historical_fit_explorer import (
    MODELS, RELATIVE, historical_frame, load_historical_fit,
)


def _payload(model):
    rows = [dict(ds=f'2023-{m:02d}-01', observed_microVolt=10.,
                 fitted_microVolt=None if model == 'Kalman' else 11.,
                 residual_microVolt=None if model == 'Kalman' else 1.)
            for m in (3, 6, 9, 12)]
    value = {'schema_version': 'historical-fit-evidence-v1', 'model': model,
             'scope': 'development_only_historical_reconstruction',
             'development_cutoff': '2023-12-01', 'rows': rows,
             'coverage': {'valid_fitted_observations': 0 if model == 'Kalman' else 4},
             'prediction_kind': 'synthetic_test', 'configuration': {}}
    canonical = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    value['sha256_payload'] = hashlib.sha256(canonical.encode()).hexdigest()
    return value


def _make(root):
    folder = root / RELATIVE
    folder.mkdir(parents=True)
    for model in MODELS:
        (folder / f'{model}_historical_fit.json').write_text(json.dumps(_payload(model)))
    included = [m for m in MODELS if m != 'Kalman']
    common = {'dates': ['2023-03-01', '2023-06-01', '2023-09-01', '2023-12-01'],
              'included_models': included, 'excluded_models': {'Kalman': 'no_fit'},
              'models': {m: {'n': 4, 'rmse': 1., 'mae': 1., 'bias': 1.} for m in included}}
    (folder / 'common_interval_metrics.json').write_text(json.dumps(common))
    return folder


def test_valid_evidence_and_unavailable_kalman(tmp_path):
    _make(tmp_path)
    artifacts, common = load_historical_fit(tmp_path)
    assert len(common['dates']) == 4
    assert len(historical_frame(artifacts, ['LSTM'])) == 4
    assert artifacts['Kalman']['rows'][0]['fitted_microVolt'] is None


def test_digest_tamper_rejected(tmp_path):
    folder = _make(tmp_path)
    path = folder / 'LSTM_historical_fit.json'
    payload = json.loads(path.read_text())
    payload['rows'][0]['fitted_microVolt'] = 123
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='SHA256'):
        load_historical_fit(tmp_path)


def test_common_metric_mismatch_rejected(tmp_path):
    folder = _make(tmp_path)
    path = folder / 'common_interval_metrics.json'
    payload = json.loads(path.read_text())
    payload['models']['LSTM']['rmse'] = 0.
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='common rmse'):
        load_historical_fit(tmp_path)
