"""C3 regression tests use synthetic saved evidence, never model training."""
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from zenerestimation.visualization.external_accuracy_evidence import (
    BATTERY, DATES, MODELS, SOURCE_KIND, build_external_accuracy_evidence,
    load_external_accuracy_evidence, write_external_accuracy_evidence,
)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root):
    base = root / 'results' / BATTERY
    bench = base / 'optimized_benchmark'
    diag = base / 'historical_fit_diagnostics'
    actual = [29.9, 30.506, 30.707, 31.319, 31.932, 32.0]
    # Six distinguishable vectors; metrics computed independently for fixture.
    forecasts = {m: [a + 0.1 * (i + 1) * (j + 1) for j, a in enumerate(actual)] for i, m in enumerate(MODELS)}
    metrics = {}
    for m in MODELS:
        e = [p - a for p, a in zip(forecasts[m], actual)]
        metrics[m] = {'rmse': math.sqrt(sum(v * v for v in e) / 6),
                      'mae': sum(abs(v) for v in e) / 6,
                      'mape': sum(abs(v / a) for v, a in zip(e, actual)) / 6 * 100}
    hashes = {}
    hashes['benchmark.json'] = _write(bench / 'benchmark.json', json.dumps({'status': 'benchmarked', 'battery': BATTERY, 'models': MODELS, 'metrics': metrics}))
    hashes['benchmark.csv'] = _write(bench / 'benchmark.csv', pd.DataFrame([{'model': m, **metrics[m]} for m in MODELS]).to_csv(index=False))
    hashes['provenance.json'] = _write(bench / 'provenance.json', json.dumps({'battery': BATTERY, 'metadata': {'development_end': '2023-12-01', 'benchmark_start': DATES[0], 'benchmark_end': DATES[-1], 'benchmark_steps': 6}}))
    frame = pd.DataFrame({'date': DATES, 'measured_microVolt': actual, **forecasts})
    _write(diag / 'regenerated_holdout_predictions.csv', frame.to_csv(index=False))
    _write(diag / 'diagnostic_provenance.json', json.dumps({
        'status': 'regenerated_410_diagnostic_not_frozen_benchmark', 'battery': BATTERY,
        'frozen_benchmark_modified': False, 'development_end': '2023-12-01',
        'development_quarters': 104, 'holdout_dates': DATES, 'models': MODELS,
        'frozen_benchmark_sha256': hashes, 'reconciliation_tolerance_abs': 1e-6,
    }))
    return bench, diag


def test_reconciles_all_six_and_roundtrips(tmp_path):
    _fixture(tmp_path)
    evidence = build_external_accuracy_evidence(tmp_path)
    assert evidence['prediction_source_kind'] == SOURCE_KIND
    assert evidence['max_abs_metric_delta'] < 1e-12
    assert len(evidence['rows']) == 36 and len(evidence['metrics']) == 6
    assert evidence['metrics'][0]['first_quarter_error'] == pytest.approx(0.1)
    output = tmp_path / 'results' / BATTERY / 'external_accuracy_16_4_c3'
    json_path, csv_path = write_external_accuracy_evidence(evidence, output)
    assert csv_path.is_file()
    assert load_external_accuracy_evidence(json_path)['metrics'] == evidence['metrics']


def test_rejects_modified_frozen_benchmark(tmp_path):
    bench, _ = _fixture(tmp_path)
    (bench / 'benchmark.json').write_text(json.dumps({'battery': BATTERY, 'status': 'benchmarked', 'models': MODELS}), encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        build_external_accuracy_evidence(tmp_path)


def test_rejects_shifted_dates(tmp_path):
    _, diag = _fixture(tmp_path)
    csv_path = diag / 'regenerated_holdout_predictions.csv'
    frame = pd.read_csv(csv_path)
    frame.loc[0, 'date'] = '2024-04-01'
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match='date alignment'):
        build_external_accuracy_evidence(tmp_path)


def test_rejects_metric_mismatch(tmp_path):
    _, diag = _fixture(tmp_path)
    csv_path = diag / 'regenerated_holdout_predictions.csv'
    frame = pd.read_csv(csv_path)
    frame.loc[0, 'LSTM'] += 5.0
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match='do not reconcile'):
        build_external_accuracy_evidence(tmp_path)


def test_rejects_corrupted_written_evidence(tmp_path):
    _fixture(tmp_path)
    evidence = build_external_accuracy_evidence(tmp_path)
    path, _ = write_external_accuracy_evidence(evidence, tmp_path / 'external_accuracy_16_4_c3')
    obj = json.loads(path.read_text(encoding='utf-8'))
    obj['metrics'][0]['rmse'] = 999
    path.write_text(json.dumps(obj), encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        load_external_accuracy_evidence(path)
