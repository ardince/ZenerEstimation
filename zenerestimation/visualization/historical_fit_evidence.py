"""Read-only historical fitted-value evidence. Never calls fit or predict."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = 'historical-fit-evidence-v1'
KINDS = {
    'LSTM': 'in_sample_trained_window_reconstruction',
    'GRU': 'in_sample_trained_window_reconstruction',
    'LinearTrendLSTM': 'in_sample_trend_plus_trained_residual_reconstruction',
    'KalmanLSTM': 'smoothed_kalman_plus_trained_residual_reconstruction',
    'Kalman': 'model_specific_filtered_or_predicted_unverified',
    'ARIMA': 'model_specific_in_sample_unverified',
}


def _date_index(values):
    dates = pd.DatetimeIndex(pd.to_datetime(values))
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Dates must be unique, finite and strictly increasing')
    return dates


def build_historical_fit_evidence(model_name, observed, fitted, *, cutoff,
                                  configuration=None, prediction_kind=None):
    """Validate exact dates and produce JSON-safe reconstruction evidence.

    ``fitted`` must be a date-indexed Series. Missing warm-up values are allowed.
    No interpolation or positional timestamp matching is performed.
    """
    if not isinstance(observed, pd.Series) or not isinstance(fitted, pd.Series):
        raise TypeError('observed and fitted must be pandas Series')
    dates = _date_index(observed.index)
    fit_dates = _date_index(fitted.index)
    if not dates.equals(fit_dates):
        raise ValueError('Fitted timestamps must exactly match observed timestamps')
    cutoff_ts = pd.Timestamp(cutoff)
    if len(dates) == 0 or dates[-1] > cutoff_ts:
        raise ValueError('Historical evidence includes dates after development cutoff')
    actual = observed.to_numpy(dtype=float)
    estimate = fitted.to_numpy(dtype=float)
    if np.isinf(actual).any():
        raise ValueError('Observed data must not contain infinity')
    if np.isinf(estimate).any():
        raise ValueError('Fitted data must not contain infinity')
    valid = np.isfinite(estimate) & np.isfinite(actual)
    residual = estimate[valid] - actual[valid]
    metrics = None
    if valid.any():
        metrics = {'rmse': float(np.sqrt(np.mean(residual ** 2))),
                   'mae': float(np.mean(np.abs(residual))),
                   'bias': float(np.mean(residual))}
    rows = [{'ds': d.strftime('%Y-%m-%d'), 'observed_microVolt': float(a) if np.isfinite(a) else None,
             'fitted_microVolt': float(f) if ok else None,
             'residual_microVolt': float(f-a) if ok else None}
            for d, a, f, ok in zip(dates, actual, estimate, valid)]
    payload = {'schema_version': SCHEMA, 'model': str(model_name),
               'scope': 'development_only_historical_reconstruction',
               'prediction_kind': prediction_kind or KINDS.get(model_name, 'unverified'),
               'development_cutoff': cutoff_ts.strftime('%Y-%m-%d'),
               'configuration': configuration or {},
               'coverage': {'total_observations': len(rows),
                            'valid_fitted_observations': int(valid.sum()),
                            'measured_observations': int(np.isfinite(actual).sum()),
                            'missing_measurements': int(np.isnan(actual).sum()),
                            'first_valid_date': dates[valid][0].strftime('%Y-%m-%d') if valid.any() else None,
                            'last_valid_date': dates[valid][-1].strftime('%Y-%m-%d') if valid.any() else None},
               'metrics': metrics, 'rows': rows,
               'interpretation': 'in_sample_fit_not_out_of_sample_forecast_accuracy'}
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    payload['sha256_payload'] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def write_historical_fit_evidence(payload, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return target


def common_interval_metrics(artifacts):
    """Compute a common measured-date interval among eligible fitted models.

    Models without usable fitted-and-measured pairs are reported as excluded,
    rather than collapsing the intersection for every other model. Prediction
    types may differ, so these are reconstruction diagnostics, not rankings.
    """
    if not artifacts:
        return {'dates': [], 'models': {}, 'included_models': [],
                'excluded_models': {}, 'warning': 'No artifacts supplied'}

    eligible = {}
    excluded = {}
    seen = set()
    for artifact in artifacts:
        name = artifact['model']
        if name in seen:
            raise ValueError('Duplicate model name')
        seen.add(name)
        rows = {}
        for row in artifact['rows']:
            if (row.get('observed_microVolt') is not None
                    and row.get('fitted_microVolt') is not None
                    and row.get('residual_microVolt') is not None):
                values = [row['observed_microVolt'], row['fitted_microVolt'],
                          row['residual_microVolt']]
                if not np.isfinite(np.asarray(values, dtype=float)).all():
                    raise ValueError(f'{name}: nonfinite common-interval row')
                date = row['ds']
                if date in rows:
                    raise ValueError(f'{name}: duplicate fitted date {date}')
                rows[date] = row
        if rows:
            eligible[name] = rows
        else:
            excluded[name] = 'no_measured_dates_with_fitted_predictions'

    dates = (sorted(set.intersection(*(set(rows) for rows in eligible.values())))
             if eligible else [])
    metrics = {}
    for name, rows in eligible.items():
        err = np.asarray([rows[d]['residual_microVolt'] for d in dates], dtype=float)
        metrics[name] = {
            'n': len(dates),
            'rmse': float(np.sqrt(np.mean(err ** 2))) if len(err) else None,
            'mae': float(np.mean(np.abs(err))) if len(err) else None,
            'bias': float(np.mean(err)) if len(err) else None,
        }
    return {
        'dates': dates,
        'models': metrics,
        'included_models': list(eligible),
        'excluded_models': excluded,
        'common_interval_status': ('available' if dates else 'no_shared_measured_dates'),
        'warning': 'Prediction types may differ; this is not a forecast ranking',
    }
