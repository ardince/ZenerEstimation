"""Offline, append-only 410 neural holdout evaluation and operational forecasting.

Run from repository root. No dashboard imports, no optimization, no benchmark writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BATTERY = '732B-5610410'
MODELS = ('LSTM', 'GRU', 'LinearTrendLSTM', 'KalmanLSTM')
DEV_END = pd.Timestamp('2023-12-01')
OP_END = pd.Timestamp('2025-06-01')
HORIZON = 6
SOURCE_DEFAULT = Path('results') / BATTERY / 'neural_expanded_16_4'
OUTPUT_BASE = Path('results') / BATTERY / 'optimized_neural_forecasts_16_4'
PARAM_KEYS = ('window', 'units', 'epochs', 'batch_size')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    def reject(value):
        raise ValueError(f'Invalid JSON constant: {value}')
    payload = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)
    if not isinstance(payload, dict):
        raise ValueError(f'{path}: expected JSON object')
    return payload


def load_selected(source: Path) -> dict:
    """Require a complete, internally consistent four-model artifact set."""
    selected = {}
    for model in MODELS:
        training = source / f'{model}_neural_training.json'
        report = source / f'{model}_learning_report.json'
        a, b = read_json(training), read_json(report)
        if a.get('model') != model or a.get('schema_version') != 'neural-training-artifact-v1':
            raise ValueError(f'{model}: incorrect training artifact model/schema')
        if a.get('scope') != 'internal_temporal_neural_training_only' or a.get('selection_metric') != 'rmse':
            raise ValueError(f'{model}: incorrect optimization scope/metric')
        if (b.get('schema_version') != 'neural-candidate-report-v1' or b.get('model') != model
                or b.get('selection_metric') != 'rmse'):
            raise ValueError(f'{model}: incorrect report model/schema')
        params = a.get('best_params')
        if not isinstance(params, dict) or set(params) != set(PARAM_KEYS):
            raise ValueError(f'{model}: invalid best_params')
        if any(type(params[k]) is not int or params[k] <= 0 for k in PARAM_KEYS):
            raise ValueError(f'{model}: parameters must be positive integers')
        score = a.get('best_score')
        if type(score) not in (float, int) or not math.isfinite(score) or score < 0:
            raise ValueError(f'{model}: invalid internal score')
        # Report metadata structures can evolve: enforce agreement wherever available.
        if b.get('best_params') != params:
            raise ValueError(f'{model}: report parameters disagree')
        if type(b.get('best_score')) not in (int, float) or not math.isclose(b['best_score'], score, abs_tol=1e-10, rel_tol=0):
            raise ValueError(f'{model}: report score disagrees')
        selected[model] = {'params': params, 'internal_rmse': float(score),
                           'training_sha256': sha256(training), 'report_sha256': sha256(report)}
    return selected


def load_canonical(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={'is_observed': 'string'})
    if not {'ds', 'microVolt', 'is_observed'}.issubset(frame.columns):
        raise ValueError('Canonical dataset lacks ds/microVolt/is_observed')
    frame['ds'] = pd.to_datetime(frame['ds'], errors='raise')
    if frame['ds'].isna().any() or frame['ds'].duplicated().any() or not frame['ds'].is_monotonic_increasing:
        raise ValueError('Invalid date sequence')
    if not frame['is_observed'].isin(['True', 'False']).all():
        raise ValueError('Invalid observation flags')
    frame['microVolt'] = pd.to_numeric(frame['microVolt'], errors='raise')
    if not np.isfinite(frame.loc[frame['is_observed'].eq('True'), 'microVolt'].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite measured observation')
    if frame.loc[frame['is_observed'].eq('False'), 'microVolt'].notna().any():
        raise ValueError('Unobserved quarters contain numeric values')
    return frame


def expected_dates(cutoff: pd.Timestamp) -> list[str]:
    return [(cutoff + pd.DateOffset(months=3 * step)).strftime('%Y-%m-%d') for step in range(1, HORIZON + 1)]


def training_dataset(frame: pd.DataFrame, cutoff: pd.Timestamp):
    from zenerestimation.data.dataset import BatteryDataset
    from zenerestimation.data.temporal import TemporalPreprocessor
    partition = frame.loc[frame['ds'] <= cutoff, ['ds', 'microVolt']].copy()
    if partition.empty or partition['ds'].iloc[-1] != cutoff:
        raise ValueError(f'Missing cutoff {cutoff.date()}')
    # Preprocessor is instantiated and fitted separately for each cutoff.
    prepared = TemporalPreprocessor().fit_transform(BatteryDataset(partition.reset_index(drop=True)))
    if prepared.data['microVolt'].isna().any():
        raise ValueError('Preprocessing left missing values')
    return prepared


def make_model(name: str, params: dict, seed: int):
    from zenerestimation.forecasting.neural.lstm import LSTMForecaster
    from zenerestimation.forecasting.neural.gru import GRUForecaster
    from zenerestimation.forecasting.hybrid.linear_trend_lstm import LinearTrendLSTMForecaster
    from zenerestimation.forecasting.hybrid.kalman_lstm import KalmanLSTMForecaster
    if name == 'LSTM':
        return LSTMForecaster(**params, seed=seed)
    if name == 'GRU':
        return GRUForecaster(**params, seed=seed)
    residual = LSTMForecaster(**params, seed=seed)
    if name == 'LinearTrendLSTM':
        return LinearTrendLSTMForecaster(window=params['window'], lstm_model=residual)
    if name == 'KalmanLSTM':
        return KalmanLSTMForecaster(window=params['window'], lstm_model=residual)
    raise ValueError(f'Unknown model {name}')


def forecast_one(model, dataset, cutoff: pd.Timestamp, *, fitted_callback=None) -> list[float]:
    model.fit(dataset)
    if fitted_callback is not None:
        fitted_callback(model, dataset)
    result = model.predict(HORIZON)
    dates = pd.DatetimeIndex(result.dates).strftime('%Y-%m-%d').tolist()
    if dates != expected_dates(cutoff) or int(result.horizon) != HORIZON:
        raise ValueError(f'Forecast alignment failed for {cutoff.date()}: {dates}')
    values = np.asarray(result.forecast, dtype=float).reshape(-1)
    if len(values) != HORIZON or not np.isfinite(values).all():
        raise ValueError('Invalid forecast values')
    return values.tolist()



def extract_historical_fitted(name: str, model, dataset, canonical: pd.DataFrame,
                              cutoff: pd.Timestamp) -> list[dict]:
    """Capture model-native in-sample fits; never relabel as external forecasts."""
    if hasattr(model, 'fitted_values'):
        fitted = model.fitted_values()
    else:
        fitted = getattr(model, 'fitted', None)
    if fitted is None:
        raise ValueError(f'{name}: model did not provide historical fitted values')
    values = np.asarray(fitted, dtype=float).reshape(-1)
    dates = pd.DatetimeIndex(pd.to_datetime(dataset.data['ds'], errors='raise'))
    expected = canonical.loc[canonical['ds'] <= cutoff].reset_index(drop=True)
    if (len(values) != len(dates) or len(values) != len(expected)
            or not dates.equals(pd.DatetimeIndex(expected['ds']))):
        raise ValueError(f'{name}: historical fitted dates/length do not align')
    if np.isinf(values).any():
        raise ValueError(f'{name}: infinite fitted values')
    window = int(getattr(model, 'window', 0))
    if window <= 0 or window >= len(values):
        raise ValueError(f'{name}: invalid model warm-up window')
    if not np.isnan(values[:window]).all():
        raise ValueError(f'{name}: expected NaN warm-up positions')
    if not np.isfinite(values[window:]).all():
        raise ValueError(f'{name}: missing fitted values after warm-up')
    rows = []
    for i, (date, fit, original) in enumerate(zip(dates, values, expected.itertuples(index=False), strict=True)):
        observed = original.is_observed == 'True'
        measured = float(original.microVolt) if observed else None
        rows.append({
            'date': date.strftime('%Y-%m-%d'),
            'model': name,
            'measured_microvolt': measured,
            'is_observed': observed,
            'was_interpolated': not observed,
            'is_warmup': i < window,
            'fitted_microvolt': None if np.isnan(fit) else float(fit),
            'error_fitted_minus_measured': (float(fit - measured)
                                            if observed and i >= window else None),
            'fit_scope': 'development_in_sample',
        })
    return rows


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    if len(actual) != HORIZON or len(predicted) != HORIZON or not np.isfinite(actual).all():
        raise ValueError('Invalid holdout measurements')
    error = predicted - actual
    return {'rmse': float(np.sqrt(np.mean(error ** 2))),
            'mae': float(np.mean(np.abs(error))),
            'first_step_signed_error_pred_minus_actual': float(error[0]),
            'first_step_absolute_error': float(abs(error[0]))}


def run(source: Path, dataset_path: Path, output: Path, seed: int, dry_run: bool = False, include_historical_fitted: bool = False) -> None:
    selected = load_selected(source)
    frame = load_canonical(dataset_path)
    holdout_dates = expected_dates(DEV_END)
    actual_frame = frame.loc[frame['ds'].isin(pd.to_datetime(holdout_dates))].copy()
    if (actual_frame['ds'].dt.strftime('%Y-%m-%d').tolist() != holdout_dates
            or not actual_frame['is_observed'].eq('True').all()):
        raise ValueError('Holdout contains missing or unobserved quarters')
    if frame.loc[frame['ds'] <= OP_END, 'ds'].iloc[-1] != OP_END:
        raise ValueError('Operational cutoff missing')
    if output.exists():
        raise FileExistsError(f'Output exists; choose a fresh --run-id: {output}')
    if dry_run:
        print('DRY RUN; no training, no writes')
        for name, item in selected.items():
            print(name, item['params'], 'internal RMSE', item['internal_rmse'])
        print('Holdout:', holdout_dates)
        print('Operational:', expected_dates(OP_END))
        return
    if type(seed) is not int or seed < 0:
        raise ValueError('Seed must be nonnegative integer')
    actual = actual_frame['microVolt'].to_numpy(dtype=float)
    history = training_dataset(frame, DEV_END)
    operational = training_dataset(frame, OP_END)
    holdout_rows = []
    future_rows = []
    metrics = {}
    fitted_rows = []
    for name, item in selected.items():
        print('Fitting', name, 'historical...', flush=True)
        callback = (lambda model, dataset, name=name: fitted_rows.extend(
            extract_historical_fitted(name, model, dataset, frame, DEV_END)
        )) if include_historical_fitted else None
        historical_pred = forecast_one(make_model(name, item['params'], seed), history, DEV_END,
                                       fitted_callback=callback)
        metrics[name] = evaluate(actual, np.asarray(historical_pred))
        for date, obs, pred in zip(holdout_dates, actual, historical_pred, strict=True):
            holdout_rows.append({'date': date, 'model': name, 'actual_microvolt': float(obs),
                                 'predicted_microvolt': float(pred), 'error_pred_minus_actual': float(pred - obs)})
        print('Fitting', name, 'operational...', flush=True)
        future_pred = forecast_one(make_model(name, item['params'], seed), operational, OP_END)
        for date, pred in zip(expected_dates(OP_END), future_pred, strict=True):
            future_rows.append({'date': date, 'model': name, 'predicted_microvolt': float(pred)})
    manifest = {'schema_version': 'optimized-neural-forecasts-16-4-v1', 'battery': BATTERY,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed,
                'source_directory': str(source), 'source_data': str(dataset_path),
                'dataset_sha256': sha256(dataset_path), 'development_cutoff': str(DEV_END.date()),
                'operational_cutoff': str(OP_END.date()), 'horizon': HORIZON,
                'holdout_dates': holdout_dates, 'operational_dates': expected_dates(OP_END),
                'evaluation_status': 'post_optimization_external_holdout_evaluation',
                'independence_qualification': 'source code excludes holdout from optimization; no audit of subsequent human selection',
                'operational_status': 'unvalidated_future_extrapolation_no_actuals_or_accuracy_metrics',
                'selected': selected, 'external_holdout_metrics': metrics,
                'files': {'holdout': 'holdout_predictions.csv', 'operational': 'operational_predictions.csv'}}
    if include_historical_fitted:
        manifest['files']['historical_fitted'] = 'historical_fitted_values.csv'
        manifest['historical_fitted'] = {
            'status': 'development_in_sample_diagnostic_not_external_validation',
            'cutoff': str(DEV_END.date()),
            'models': list(MODELS),
            'rows_per_model': len(history.data),
            'warmup': {name: int(selected[name]['params']['window']) for name in MODELS},
            'missing_observations': 'retained_as_unobserved; interpolated training targets are not measurements',
            'error_definition': 'fitted_minus_measured_only_for_observed_non_warmup_rows',
        }
        if len(fitted_rows) != len(MODELS) * len(history.data):
            raise ValueError('Incomplete historical fitted evidence')
    # Reserve the output path exclusively; never overwrite prior results.
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(holdout_rows).to_csv(output / 'holdout_predictions.csv', index=False, float_format='%.15g')
    pd.DataFrame(future_rows).to_csv(output / 'operational_predictions.csv', index=False, float_format='%.15g')
    if include_historical_fitted:
        pd.DataFrame(fitted_rows).to_csv(output / 'historical_fitted_values.csv',
                                         index=False, float_format='%.15g')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Saved:', output)
    for name, metric in metrics.items():
        print(f'{name:20s} external RMSE={metric["rmse"]:.6f} µV; MAE={metric["mae"]:.6f} µV')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE_DEFAULT)
    parser.add_argument('--dataset', type=Path, default=Path('datasets/processed') / f'{BATTERY}.csv')
    parser.add_argument('--run-id', required=True, help='Unique directory name (letters, digits, _ and -)')
    parser.add_argument('--seed', type=int, default=42, help='Must match the seed used during optimization')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--include-historical-fitted', action='store_true',
                        help='Save development-period in-sample fitted values (not holdout)')
    args = parser.parse_args()
    if not args.run_id or not all(c.isalnum() or c in '_-' for c in args.run_id) or args.run_id in ('.', '..'):
        parser.error('Invalid run-id')
    run(args.source, args.dataset, OUTPUT_BASE / args.run_id, args.seed, args.dry_run,
        args.include_historical_fitted)


if __name__ == '__main__':
    main()
