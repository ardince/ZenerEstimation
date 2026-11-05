"""Sprint 15.3.6.7: diagnostic provenance remains separate from frozen evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_artifacts import (
    ArtifactContractError, CrossBatteryArtifactLoader, MODELS, METRICS,
)
from zenerestimation.visualization.cross_battery_comparison import build_comparison
from zenerestimation.visualization.cross_battery_dashboard_numerical import numerical_table
# Load the sibling test helpers by file path, independently of pytest import mode.
_HELPERS_PATH = Path(__file__).with_name('test_cross_battery_artifacts.py')
_HELPERS_SPEC = importlib.util.spec_from_file_location('_cross_battery_test_helpers', _HELPERS_PATH)
if _HELPERS_SPEC is None or _HELPERS_SPEC.loader is None:
    raise RuntimeError(f'Cannot load test helpers from {_HELPERS_PATH}')
_HELPERS = importlib.util.module_from_spec(_HELPERS_SPEC)
_HELPERS_SPEC.loader.exec_module(_HELPERS)
_fixture = _HELPERS._fixture
_write_json = _HELPERS._write_json

BATTERY = '732B-5610410'
OTHER = '732B-5610110'


def _diagnostic_fixture(tmp_path):
    base = _fixture(tmp_path, BATTERY)
    _fixture(tmp_path, OTHER)
    diag = tmp_path / 'results' / BATTERY / 'historical_fit_diagnostics'
    diag.mkdir(parents=True)
    dates = pd.date_range('2024-03-01', periods=6, freq='QS-MAR')
    actual = np.array([29.9, 30.506, 30.707, 31.319, 31.932, 32.0])
    predictions = pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'measured_microVolt': actual})
    metrics = {}
    for i, model in enumerate(MODELS):
        pred = actual + 0.1 * (i + 1)
        predictions[model] = pred
        error = pred - actual
        metrics[model] = {'rmse': float(np.sqrt(np.mean(error**2))),
                          'mae': float(np.mean(np.abs(error))),
                          'mape': float(np.mean(np.abs(error / actual)) * 100)}
    predictions.to_csv(diag / 'regenerated_holdout_predictions.csv', index=False, float_format='%.17g')
    history_dates = pd.date_range(end='2023-12-01', periods=104, freq='QS-MAR')
    fits = pd.DataFrame({'date': history_dates.strftime('%Y-%m-%d')})
    for model in MODELS:
        fits[model] = 10.0
    fits.loc[:3, 'LSTM'] = np.nan
    fits.to_csv(diag / 'historical_fitted_values.csv', index=False)
    benchmark = json.loads((base / 'benchmark.json').read_text())
    benchmark['metrics'] = metrics
    benchmark['rankings'] = {
        metric: [{'rank': rank, 'model': model, 'value': metrics[model][metric]}
                 for rank, model in enumerate(sorted(MODELS, key=lambda m: (metrics[m][metric], m)), 1)]
        for metric in METRICS
    }
    _write_json(base / 'benchmark.json', benchmark)
    pd.DataFrame([{'model': model, **metrics[model]} for model in MODELS]).to_csv(base / 'benchmark.csv', index=False)
    hashes = {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
              for name in ('benchmark.csv', 'benchmark.json', 'provenance.json')}
    provenance = {
        'status': 'regenerated_410_diagnostic_not_frozen_benchmark',
        'battery': BATTERY, 'frozen_benchmark_modified': False,
        'development_end': '2023-12-01', 'development_quarters': 104,
        'holdout_dates': dates.strftime('%Y-%m-%d').tolist(),
        'models': list(MODELS), 'frozen_benchmark_sha256': hashes,
        'all_metrics_within_tolerance': True,
        'artifacts': {'historical_fitted_values': 'historical_fitted_values.csv',
                      'regenerated_holdout_predictions': 'regenerated_holdout_predictions.csv',
                      'metric_reconciliation': 'metric_reconciliation.csv'},
    }
    _write_json(diag / 'diagnostic_provenance.json', provenance)
    return base, diag


def test_410_diagnostic_loaded_with_distinct_origin(tmp_path):
    _diagnostic_fixture(tmp_path)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in tmp_path.rglob('*') if p.is_file()}
    records = CrossBatteryArtifactLoader(tmp_path).load_both()
    comparison = build_comparison(records)
    holdout = numerical_table(comparison, battery_id=BATTERY, kind='holdout')
    fits = numerical_table(comparison, battery_id=BATTERY, kind='historical_fits')
    assert holdout.available and len(holdout.data) == 36
    assert fits.available and len(fits.data) == 624
    assert holdout.data.origin.eq('regenerated_diagnostic_not_frozen').all()
    assert 'NOT original frozen' in holdout.status
    assert pd.isna(fits.data.loc[(fits.data.model == 'LSTM') & (fits.data.date == '1998-03-01'), 'fitted_microvolt']).all()
    assert comparison.holdout_predictions.loc[comparison.holdout_predictions.battery_id.eq(OTHER), 'origin'].eq('frozen_transferred_benchmark').all()
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after


def test_410_diagnostic_metric_tampering_rejected(tmp_path):
    _, diag = _diagnostic_fixture(tmp_path)
    path = diag / 'regenerated_holdout_predictions.csv'
    df = pd.read_csv(path)
    df.loc[0, 'ARIMA'] += 0.5
    df.to_csv(path, index=False)
    with pytest.raises(ArtifactContractError, match='metric mismatch'):
        CrossBatteryArtifactLoader(tmp_path).load(BATTERY)


def test_410_diagnostic_frozen_hash_mismatch_rejected(tmp_path):
    base, _ = _diagnostic_fixture(tmp_path)
    (base / 'provenance.json').write_text((base / 'provenance.json').read_text() + '\n')
    with pytest.raises(ArtifactContractError, match='hash mismatch'):
        CrossBatteryArtifactLoader(tmp_path).load(BATTERY)


def test_410_missing_diagnostic_preserves_plot_only_contract(tmp_path):
    _fixture(tmp_path, BATTERY)
    record = CrossBatteryArtifactLoader(tmp_path).load(BATTERY)
    assert record.holdout_predictions is None
    assert not record.evidence['holdout_predictions'].structured
