"""Isolated contract tests; no repository benchmark data is modified."""
import hashlib
import json
from pathlib import Path

import pytest

from zenerestimation.visualization.cross_battery_artifacts import (
    ArtifactContractError, CrossBatteryArtifactLoader, MODELS, METRICS,
)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def _fixture(tmp_path, battery):
    is_110 = battery.endswith('0110')
    base = tmp_path / 'results' / battery / ('transferred_benchmark' if is_110 else 'optimized_benchmark')
    base.mkdir(parents=True)
    metrics = {model: {'rmse': float(i + 1), 'mae': float(i + 1), 'mape': float(i + 1)} for i, model in enumerate(MODELS)}
    rankings = {metric: [{'rank': i + 1, 'model': model, 'value': float(i + 1)} for i, model in enumerate(MODELS)] for metric in METRICS}
    _write_json(base / 'benchmark.json', {'battery': battery, 'metrics': metrics, 'rankings': rankings, 'metadata': {'lower_is_better': True, 'comparison_metrics': list(METRICS)}})
    (base / 'benchmark.csv').write_text('model,rmse,mae,mape\n' + ''.join(f'{m},{i+1},{i+1},{i+1}\n' for i, m in enumerate(MODELS)), encoding='utf-8')
    if is_110:
        provenance = {'battery': battery, 'experiment_type': 'frozen_cross_battery_parameter_transfer', 'training_end': '2022-12-01', 'holdout_dates': ['2023-03-01'], 'holdout_steps': 1}
        _write_json(base / 'transfer_provenance_validated.json', provenance)
        (base / 'transfer_provenance.json').write_text(json.dumps(provenance) + r'\n', encoding='utf-8')
        prediction = base / 'holdout_predictions_full_precision.csv'
        prediction.write_text('date,measured_microVolt,' + ','.join(MODELS) + '\n2023-03-01,167.17,' + ','.join('167.2' for _ in MODELS) + '\n', encoding='utf-8')
        diag = tmp_path / 'results' / battery / 'historical_fit_diagnostics'
        diag.mkdir(parents=True)
        (diag / 'historical_fitted_values.csv').write_text('date,' + ','.join(MODELS) + '\n2022-12-01,' + ','.join('167.0' for _ in MODELS) + '\n', encoding='utf-8')
        _write_json(diag / 'diagnostic_provenance.json', {'status': 'regenerated_historical_fit_diagnostic_not_frozen_benchmark', 'benchmark_predictions_sha256': hashlib.sha256(prediction.read_bytes()).hexdigest()})
        plot_base = diag
    else:
        _write_json(base / 'provenance.json', {'battery': battery, 'metadata': {'development_end': '2023-12-01', 'benchmark_start': '2024-03-01', 'benchmark_end': '2025-06-01', 'benchmark_steps': 6}})
        plot_base = base
    for name in ('benchmark_history.png', 'benchmark_best_worst_history.png'):
        (plot_base / name).write_bytes(b'fake png: presence only')
    (base / 'benchmark_forecast.png').write_bytes(b'fake png: presence only')
    return base


def test_both_batteries_load_read_only(tmp_path):
    _fixture(tmp_path, '732B-5610410')
    _fixture(tmp_path, '732B-5610110')
    before = {str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file()}
    records = CrossBatteryArtifactLoader(tmp_path).load_both()
    after = {str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after
    a, b = records['732B-5610410'], records['732B-5610110']
    assert a.holdout_predictions is None and a.historical_fits is None
    assert b.holdout_predictions is not None and b.historical_fits is not None
    assert a.evidence['historical_fits'].visual and not a.evidence['historical_fits'].structured
    assert b.evidence['historical_fits'].origin == 'regenerated_diagnostic_not_frozen'
    assert b.holdout_steps == 1 and a.holdout_steps == 6


def test_rejects_wrong_battery(tmp_path):
    with pytest.raises(ArtifactContractError, match='Unsupported battery'):
        CrossBatteryArtifactLoader(tmp_path).load('unknown')


def test_rejects_tampered_prediction(tmp_path):
    base = _fixture(tmp_path, '732B-5610110')
    pred = base / 'holdout_predictions_full_precision.csv'
    pred.write_text(pred.read_text().replace('167.2', '169.2'), encoding='utf-8')
    with pytest.raises(ArtifactContractError, match='hash'):
        CrossBatteryArtifactLoader(tmp_path).load('732B-5610110')


def test_rejects_ranking_mismatch(tmp_path):
    base = _fixture(tmp_path, '732B-5610410')
    path = base / 'benchmark.json'
    data = json.loads(path.read_text())
    data['rankings']['rmse'][0]['value'] = 1000
    _write_json(path, data)
    with pytest.raises(ArtifactContractError, match='Ranking value'):
        CrossBatteryArtifactLoader(tmp_path).load('732B-5610410')


def test_rejects_nonfinite_json(tmp_path):
    base = _fixture(tmp_path, '732B-5610410')
    path = base / 'benchmark.json'
    path.write_text(path.read_text().replace('1.0', 'NaN', 1), encoding='utf-8')
    with pytest.raises(ArtifactContractError, match='Non-finite'):
        CrossBatteryArtifactLoader(tmp_path).load('732B-5610410')


def test_rejects_missing_plot(tmp_path):
    base = _fixture(tmp_path, '732B-5610410')
    (base / 'benchmark_history.png').unlink()
    with pytest.raises(ArtifactContractError, match='Missing expected plot'):
        CrossBatteryArtifactLoader(tmp_path).load('732B-5610410')
