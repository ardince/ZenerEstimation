"""Read-only, strict six-quarter external-accuracy evidence for battery 410.

Diagnostic regenerated forecasts are NEVER represented as original frozen vectors.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

BATTERY = '732B-5610410'
MODELS = ('ARIMA', 'Kalman', 'LSTM', 'GRU', 'LinearTrendLSTM', 'KalmanLSTM')
DATES = ('2024-03-01', '2024-06-01', '2024-09-01', '2024-12-01', '2025-03-01', '2025-06-01')
SOURCE_KIND = 'diagnostic_regeneration_reconciled_to_frozen_aggregate_metrics'
METRICS = ('rmse', 'mae', 'mape')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value, description: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{description}: expected numeric value') from exc
    if not math.isfinite(result):
        raise ValueError(f'{description}: non-finite value')
    return result


def _read_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return obj


def build_external_accuracy_evidence(root: Path, tolerance: float | None = None) -> dict:
    """Validate frozen sources and diagnostic forecast vectors without fitting models."""
    root = Path(root)
    benchmark_dir = root / 'results' / BATTERY / 'optimized_benchmark'
    diagnostic_dir = root / 'results' / BATTERY / 'historical_fit_diagnostics'
    benchmark = _read_json(benchmark_dir / 'benchmark.json')
    provenance = _read_json(benchmark_dir / 'provenance.json')
    diagnostic = _read_json(diagnostic_dir / 'diagnostic_provenance.json')
    if benchmark.get('battery') != BATTERY or provenance.get('battery') != BATTERY or diagnostic.get('battery') != BATTERY:
        raise ValueError('Battery identity mismatch')
    if benchmark.get('status') != 'benchmarked' or diagnostic.get('status') != 'regenerated_410_diagnostic_not_frozen_benchmark':
        raise ValueError('Unexpected evidence status; refusing to attribute prediction provenance')
    if diagnostic.get('frozen_benchmark_modified') is not False:
        raise ValueError('Diagnostic provenance does not certify unchanged benchmark')
    if tuple(benchmark.get('models', ())) != MODELS or tuple(diagnostic.get('models', ())) != MODELS:
        raise ValueError('Model set/order mismatch')
    if tuple(diagnostic.get('holdout_dates', ())) != DATES:
        raise ValueError('Diagnostic holdout dates mismatch')
    if diagnostic.get('development_end') != '2023-12-01' or diagnostic.get('development_quarters') != 104:
        raise ValueError('Unexpected development boundary')
    meta = provenance.get('metadata', {})
    if (meta.get('development_end'), meta.get('benchmark_start'), meta.get('benchmark_end'), meta.get('benchmark_steps')) != ('2023-12-01', DATES[0], DATES[-1], 6):
        raise ValueError('Frozen benchmark boundary mismatch')
    hashes = diagnostic.get('frozen_benchmark_sha256', {})
    for name in ('benchmark.json', 'benchmark.csv', 'provenance.json'):
        expected = hashes.get(name)
        if not isinstance(expected, str) or _sha256(benchmark_dir / name) != expected:
            raise ValueError(f'Frozen source hash mismatch: {name}')
    tol = _finite(diagnostic.get('reconciliation_tolerance_abs') if tolerance is None else tolerance, 'tolerance')
    if tol < 0:
        raise ValueError('Tolerance must be nonnegative')
    csv_path = diagnostic_dir / 'regenerated_holdout_predictions.csv'
    frame = pd.read_csv(csv_path, dtype={'date': str})
    expected_columns = ('date', 'measured_microVolt', *MODELS)
    if tuple(frame.columns) != expected_columns:
        raise ValueError('Unexpected diagnostic prediction columns')
    if tuple(frame['date'].tolist()) != DATES or frame['date'].duplicated().any():
        raise ValueError('Holdout date alignment failed')
    if frame.isna().any().any():
        raise ValueError('Missing holdout measurements or predictions')
    actual = [_finite(v, 'measured_microVolt') for v in frame['measured_microVolt']]
    if any(v == 0 for v in actual):
        raise ValueError('MAPE undefined for zero measured voltage')
    stored = benchmark.get('metrics', {})
    frozen_csv = pd.read_csv(benchmark_dir / 'benchmark.csv')
    if tuple(frozen_csv.columns) != ('model', *METRICS) or tuple(frozen_csv['model'].tolist()) != MODELS:
        raise ValueError('Frozen CSV schema or model order mismatch')
    for _, item in frozen_csv.iterrows():
        for metric in METRICS:
            if not math.isclose(_finite(item[metric], 'frozen CSV metric'),
                                _finite(stored[item['model']][metric], 'frozen JSON metric'),
                                rel_tol=0, abs_tol=1e-12):
                raise ValueError('Frozen CSV/JSON metrics disagree')
    if set(stored) != set(MODELS):
        raise ValueError('Frozen metrics model mismatch')
    rows = []
    summary = []
    max_delta = 0.0
    for model in MODELS:
        predictions = [_finite(v, f'{model} prediction') for v in frame[model]]
        errors = [p - a for p, a in zip(predictions, actual, strict=True)]
        metrics = {
            'rmse': math.sqrt(sum(e * e for e in errors) / len(errors)),
            'mae': sum(abs(e) for e in errors) / len(errors),
            'mape': 100 * sum(abs(e / a) for e, a in zip(errors, actual, strict=True)) / len(errors),
        }
        deltas = {metric: abs(metrics[metric] - _finite(stored[model][metric], f'{model} frozen {metric}')) for metric in METRICS}
        max_delta = max(max_delta, *deltas.values())
        if any(d > tol for d in deltas.values()):
            raise ValueError(f'{model}: regenerated metrics do not reconcile to frozen benchmark: {deltas}')
        summary.append({
            'model': model, **metrics, 'bias_predicted_minus_measured': sum(errors) / len(errors),
            'first_quarter_error': errors[0], 'first_quarter_absolute_error': abs(errors[0]),
            'n': len(errors), 'max_frozen_metric_delta': max(deltas.values()),
        })
        for date, a, p, e in zip(DATES, actual, predictions, errors, strict=True):
            rows.append({'model': model, 'date': date, 'measured_microVolt': a,
                         'predicted_microVolt': p, 'error_predicted_minus_measured': e,
                         'absolute_error_microVolt': abs(e)})
    return {
        'schema_version': 1, 'status': 'verified_aggregate_reconciliation', 'battery': BATTERY,
        'scope': 'external_six_quarter_holdout', 'development_end': '2023-12-01',
        'holdout_dates': list(DATES), 'prediction_source_kind': SOURCE_KIND,
        'original_frozen_forecast_vectors_available': False,
        'limitations': [
            'Regenerated diagnostic forecasts are not the original frozen benchmark prediction vectors.',
            'Aggregate metric agreement does not establish bitwise identity of individual forecasts.',
            'Holdout measurements must never be used for model tuning or selection.',
        ],
        'source_files_sha256': {
            'optimized_benchmark/' + name: _sha256(benchmark_dir / name)
            for name in ('benchmark.json', 'benchmark.csv', 'provenance.json')
        } | {
            'historical_fit_diagnostics/' + name: _sha256(diagnostic_dir / name)
            for name in ('diagnostic_provenance.json', 'regenerated_holdout_predictions.csv')
        },
        'tolerance_abs': tol, 'max_abs_metric_delta': max_delta,
        'metrics': summary, 'rows': rows,
    }


def write_external_accuracy_evidence(evidence: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write additive evidence only; never write into the frozen benchmark folder."""
    output_dir = Path(output_dir)
    if output_dir.name != 'external_accuracy_16_4_c3':
        raise ValueError('Output directory must be named external_accuracy_16_4_c3')
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(evidence)
    payload['sha256_payload'] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    json_path = output_dir / 'external_accuracy_evidence.json'
    csv_path = output_dir / 'external_accuracy_rows.csv'
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pd.DataFrame(payload['rows']).to_csv(csv_path, index=False, float_format='%.17g')
    return json_path, csv_path


def load_external_accuracy_evidence(path: Path) -> dict:
    payload = _read_json(Path(path))
    digest = payload.pop('sha256_payload', None)
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if digest != actual:
        raise ValueError('External accuracy evidence checksum mismatch')
    if payload.get('prediction_source_kind') != SOURCE_KIND or payload.get('status') != 'verified_aggregate_reconciliation':
        raise ValueError('Unexpected external accuracy evidence provenance')
    if tuple(payload.get('holdout_dates', ())) != DATES:
        raise ValueError('External accuracy evidence dates mismatch')
    if len(payload.get('rows', ())) != 36 or len(payload.get('metrics', ())) != 6:
        raise ValueError('Incomplete external accuracy evidence')
    return payload
