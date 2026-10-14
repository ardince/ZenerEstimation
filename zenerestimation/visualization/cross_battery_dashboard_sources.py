"""Read-only observed histories and operational forecasts for batteries 110 and 410."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BATTERY = '732B-5610110'
BATTERY_410 = '732B-5610410'
MODELS = ('ARIMA', 'Kalman', 'LSTM', 'GRU', 'LinearTrendLSTM', 'KalmanLSTM')
STATUS = 'unvalidated_future_operational_extrapolation'


def _dates(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, format='%Y-%m-%d', errors='raise')
    if dates.isna().any() or not dates.dt.month.isin([3, 6, 9, 12]).all() or not dates.dt.day.eq(1).all():
        raise ValueError('Dates must be quarterly month-start timestamps')
    return dates


def load_110_observed_history(repository_root: str | Path) -> pd.DataFrame:
    """Return only physically observed quarters; missing quarters remain absent."""
    path = Path(repository_root) / 'datasets' / 'processed' / f'{BATTERY}.csv'
    raw = pd.read_csv(path, dtype={'is_observed': 'string'})
    if not {'ds', 'microVolt', 'is_observed'}.issubset(raw.columns):
        raise ValueError('Canonical observed-history schema mismatch')
    if raw.empty or not raw['is_observed'].isin(['True', 'False']).all():
        raise ValueError('Invalid is_observed flags')
    raw['date'] = _dates(raw['ds'])
    if raw['date'].duplicated().any() or not raw['date'].is_monotonic_increasing:
        raise ValueError('Historical dates must be unique and increasing')
    measured = raw.loc[raw['is_observed'].eq('True'), ['date', 'microVolt']].copy()
    measured['measured_microvolt'] = pd.to_numeric(measured.pop('microVolt'), errors='raise')
    if measured.empty or not np.isfinite(measured['measured_microvolt'].to_numpy(dtype=float)).all():
        raise ValueError('Observed values must be finite')
    return measured.reset_index(drop=True)


def load_110_operational_forecasts(repository_root: str | Path) -> pd.DataFrame:
    """Load saved operational forecasts, validating CSV against JSON provenance."""
    folder = Path(repository_root) / 'results' / BATTERY / 'operational_extrapolation'
    metadata = json.loads((folder / 'future_extrapolation.json').read_text(encoding='utf-8'),
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Invalid JSON constant {value}')))
    required = {
        'kind': 'operational_extrapolation_not_holdout_evaluation',
        'battery': BATTERY, 'scientific_status': STATUS,
        'optimized_for_dataset_110': False, 'measured_actuals': None,
        'forecast_errors': None, 'evaluation_metrics': None,
        'uncertainty_intervals': None,
    }
    for key, expected in required.items():
        if key not in metadata or type(metadata[key]) is not type(expected) or metadata[key] != expected:
            raise ValueError(f'Operational provenance mismatch: {key}')
    if metadata.get('configuration_source_battery') != '732B-5610410':
        raise ValueError('Unexpected configuration source')
    if metadata.get('models') != list(MODELS):
        raise ValueError('Operational model list mismatch')
    csv = pd.read_csv(folder / 'future_extrapolation_full_precision.csv')
    if list(csv.columns) != ['date', *MODELS] or csv.empty:
        raise ValueError('Operational forecast CSV schema mismatch')
    csv['date'] = _dates(csv['date'])
    if csv['date'].duplicated().any() or not csv['date'].is_monotonic_increasing:
        raise ValueError('Operational forecast dates must be unique and increasing')
    dates = csv['date'].dt.strftime('%Y-%m-%d').tolist()
    if (dates != metadata.get('forecast_dates') or dates[0] != metadata.get('forecast_start')
            or dates[-1] != metadata.get('forecast_end') or len(csv) != metadata.get('horizon_quarters')
            or len(csv) * len(MODELS) != metadata.get('prediction_count')):
        raise ValueError('Operational horizon/date metadata mismatch')
    cutoff = pd.Timestamp(metadata['training_cutoff'])
    if cutoff != pd.Timestamp('2025-03-01') or csv['date'].min() != cutoff + pd.DateOffset(months=3):
        raise ValueError('Operational forecast cutoff mismatch')
    periods = csv['date'].dt.to_period('Q-FEB').astype('int64')
    if not periods.diff().dropna().eq(1).all():
        raise ValueError('Operational dates are not consecutive quarters')
    for model in MODELS:
        values = pd.to_numeric(csv[model], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite operational forecast: {model}')
        saved = metadata.get('predictions', {}).get(model)
        if not isinstance(saved, list) or len(saved) != len(values) or not np.allclose(values, saved, atol=1e-8, rtol=0):
            raise ValueError(f'CSV/JSON operational forecast disagreement: {model}')
    return csv


def load_410_observed_history(repository_root: str | Path) -> pd.DataFrame:
    """Return measured 410 history only, respecting observation flags."""
    path = Path(repository_root) / 'datasets' / 'processed' / f'{BATTERY_410}.csv'
    raw = pd.read_csv(path, dtype={'is_observed': 'string'})
    if not {'ds', 'microVolt', 'is_observed'}.issubset(raw.columns):
        raise ValueError('410 observed-history schema mismatch')
    if raw.empty or not raw['is_observed'].isin(['True', 'False']).all():
        raise ValueError('Invalid 410 observation flags')
    raw['date'] = _dates(raw['ds'])
    if raw['date'].duplicated().any() or not raw['date'].is_monotonic_increasing:
        raise ValueError('410 history dates must be unique and increasing')
    measured = raw.loc[raw['is_observed'].eq('True'), ['date', 'microVolt']].copy()
    measured['measured_microvolt'] = pd.to_numeric(measured.pop('microVolt'), errors='raise')
    if measured.empty or not np.isfinite(measured['measured_microvolt'].to_numpy(dtype=float)).all():
        raise ValueError('Invalid 410 measured values')
    return measured.reset_index(drop=True)


def load_410_operational_forecasts(repository_root: str | Path) -> pd.DataFrame:
    """Validate saved 410 CSV against higher-precision operational JSON."""
    folder = Path(repository_root) / 'results' / BATTERY_410 / 'operational_extrapolation'
    metadata = json.loads((folder / 'future_extrapolation.json').read_text(encoding='utf-8'),
                          parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f'Invalid JSON constant {v}')))
    expected = {
        'kind': 'operational_extrapolation_not_holdout_evaluation',
        'battery': BATTERY_410,
        'scientific_status': STATUS,
        'configuration_source': 'frozen_410_optimized_benchmark',
        'model_update_policy': 'refit_through_training_cutoff_without_reoptimization',
        'measured_actuals': None, 'forecast_errors': None,
        'evaluation_metrics': None, 'uncertainty_intervals': None,
    }
    for key, value in expected.items():
        if key not in metadata or type(metadata[key]) is not type(value) or metadata[key] != value:
            raise ValueError(f'410 operational provenance mismatch: {key}')
    if metadata.get('models') != list(MODELS):
        raise ValueError('410 operational model list mismatch')
    csv = pd.read_csv(folder / 'future_extrapolation.csv')
    if list(csv.columns) != ['date', *MODELS] or csv.empty:
        raise ValueError('410 operational CSV schema mismatch')
    csv['date'] = _dates(csv['date'])
    if csv['date'].duplicated().any() or not csv['date'].is_monotonic_increasing:
        raise ValueError('410 forecast dates must be unique and increasing')
    dates = csv['date'].dt.strftime('%Y-%m-%d').tolist()
    if (dates != metadata.get('forecast_dates') or dates[0] != metadata.get('forecast_start')
            or dates[-1] != metadata.get('forecast_end') or len(csv) != metadata.get('horizon_quarters')
            or len(csv) * len(MODELS) != metadata.get('prediction_count')):
        raise ValueError('410 forecast horizon/date mismatch')
    cutoff = pd.Timestamp(metadata['training_cutoff'])
    if cutoff != pd.Timestamp('2025-06-01') or csv['date'].min() != cutoff + pd.DateOffset(months=3):
        raise ValueError('410 forecast training cutoff mismatch')
    if not csv['date'].dt.to_period('Q-FEB').astype('int64').diff().dropna().eq(1).all():
        raise ValueError('410 forecast dates must be consecutive quarters')
    # CSV is rounded to 10 decimal places; JSON holds the authoritative precision.
    # Return the validated JSON predictions rather than reducing their precision.
    result = pd.DataFrame({'date': csv['date'].copy()})
    for model in MODELS:
        csv_values = pd.to_numeric(csv[model], errors='raise').to_numpy(dtype=float)
        saved = metadata.get('predictions', {}).get(model)
        if not isinstance(saved, list) or len(saved) != len(csv_values):
            raise ValueError(f'410 missing operational JSON predictions: {model}')
        json_values = np.asarray(saved, dtype=float)
        if (not np.isfinite(csv_values).all() or not np.isfinite(json_values).all()
                or not np.allclose(csv_values, json_values, rtol=0, atol=1e-9)):
            raise ValueError(f'410 CSV/JSON operational forecast disagreement: {model}')
        result[model] = json_values
    return result
