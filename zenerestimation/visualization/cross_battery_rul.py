"""Sprint 15.4.4: read-only, battery-specific *scenario* RUL diagnostics.

A forecast threshold crossing is not an observed failure, a calibrated probability,
or a validated RUL estimate. No extrapolation beyond the saved forecast horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd

MODELS = ('ARIMA', 'Kalman', 'LSTM', 'GRU', 'LinearTrendLSTM', 'KalmanLSTM')
BATTERIES = ('732B-5610410', '732B-5610110')
STATUS = 'unvalidated_future_operational_extrapolation'


@dataclass(frozen=True)
class BatteryRULDiagnostics:
    battery_id: str
    threshold_microvolt: float
    crossing_direction: str
    forecast_status: str
    last_observed_date: str
    last_observed_microvolt: float
    model_crossings: pd.DataFrame
    dispersion: pd.DataFrame
    interpretation: str


def analyze_battery_rul(*, battery_id: str, observed: pd.DataFrame,
                        forecasts: pd.DataFrame, threshold_microvolt: float,
                        crossing_direction: str = 'above') -> BatteryRULDiagnostics:
    """Return per-model discrete-quarter threshold scenarios and model spread.

    observed: columns date, measured_microvolt (physical observations only).
    forecasts: date and one column per model, all future quarterly starts.
    Threshold and direction are supplied explicitly, not inferred from the data.
    """
    if battery_id not in BATTERIES:
        raise ValueError('Unknown battery')
    if crossing_direction not in ('above', 'below'):
        raise ValueError('crossing_direction must be above or below')
    if not math.isfinite(float(threshold_microvolt)):
        raise ValueError('Threshold must be finite')
    if not isinstance(observed, pd.DataFrame) or not {'date', 'measured_microvolt'}.issubset(observed.columns):
        raise ValueError('Observed history schema mismatch')
    if not isinstance(forecasts, pd.DataFrame) or list(forecasts.columns) != ['date', *MODELS]:
        raise ValueError('Operational forecast schema mismatch')
    if observed.empty or forecasts.empty:
        raise ValueError('Observed history and forecast must be nonempty')
    obs = observed.copy(deep=True)
    fut = forecasts.copy(deep=True)
    for df in (obs, fut):
        df['date'] = pd.to_datetime(df['date'], errors='raise')
        if df['date'].isna().any() or df.date.duplicated().any() or not df.date.is_monotonic_increasing:
            raise ValueError('Dates must be valid, unique and increasing')
        if not df.date.dt.day.eq(1).all() or not df.date.dt.month.isin((3, 6, 9, 12)).all():
            raise ValueError('Expected quarterly month-start dates')
    values = pd.to_numeric(obs.measured_microvolt, errors='raise').to_numpy(dtype=float)
    matrix = fut[list(MODELS)].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(matrix).all():
        raise ValueError('Nonfinite observation or forecast')
    last_date = obs.date.iloc[-1]
    last_value = float(values[-1])
    expected = pd.date_range(last_date + pd.DateOffset(months=3), periods=len(fut), freq='3MS')
    if not fut.date.reset_index(drop=True).equals(pd.Series(expected)):
        raise ValueError('Forecast must begin next quarter and be consecutive')
    threshold = float(threshold_microvolt)
    crossed = (lambda v: v >= threshold) if crossing_direction == 'above' else (lambda v: v <= threshold)
    already = bool(crossed(last_value))
    crossings = []
    for col, model in enumerate(MODELS):
        indices = np.flatnonzero(crossed(matrix[:, col]))
        if already:
            state, date, quarters = 'already_crossed_observed', last_date, 0
        elif len(indices):
            quarters = int(indices[0]) + 1
            date = fut.date.iloc[quarters - 1]
            state = 'forecast_crossing_unvalidated'
        else:
            quarters, date, state = None, pd.NaT, 'not_crossed_within_horizon'
        crossings.append({'battery_id': battery_id, 'model': model, 'threshold_microvolt': threshold,
                          'crossing_direction': crossing_direction, 'status': state,
                          'crossing_date': date.strftime('%Y-%m-%d') if pd.notna(date) else None,
                          'rul_quarters': quarters, 'rul_years': quarters / 4 if quarters is not None else None,
                          'horizon_quarters': len(fut), 'forecast_status': STATUS})
    dispersion = []
    for i, row in fut.reset_index(drop=True).iterrows():
        predictions = matrix[i]
        minimum, maximum = float(predictions.min()), float(predictions.max())
        dispersion.append({'battery_id': battery_id, 'date': row.date.strftime('%Y-%m-%d'),
                           'forecast_quarter': i + 1, 'minimum_microvolt': minimum,
                           'maximum_microvolt': maximum, 'median_microvolt': float(np.median(predictions)),
                           'std_microvolt_population': float(np.std(predictions, ddof=0)),
                           'spread_microvolt': maximum - minimum,
                           'spread_percent_of_last_observed': (100 * (maximum-minimum) / abs(last_value)
                                                               if abs(last_value) > 1e-9 else np.nan),
                           'forecast_status': STATUS})
    return BatteryRULDiagnostics(
        battery_id=battery_id, threshold_microvolt=threshold, crossing_direction=crossing_direction,
        forecast_status=STATUS, last_observed_date=last_date.strftime('%Y-%m-%d'),
        last_observed_microvolt=last_value, model_crossings=pd.DataFrame(crossings),
        dispersion=pd.DataFrame(dispersion),
        interpretation=('Scenario-only first forecast-quarter threshold crossings; no interpolation or '
                        'extrapolation beyond the saved horizon. A non-crossing is right-censored '
                        'at the horizon, not infinite RUL. Forecasts are unvalidated; inter-model '
                        'spread is not a prediction interval. Battery-specific threshold direction '
                        'and value must be scientifically justified.'),
    )


def load_battery_rul(repository_root: str | Path, *, battery_id: str,
                     threshold_microvolt: float, crossing_direction: str = 'above') -> BatteryRULDiagnostics:
    """Load the existing validated operational forecast/observed-history sources."""
    from zenerestimation.visualization import cross_battery_dashboard_sources as sources
    if battery_id == BATTERIES[0]:
        observed = sources.load_410_observed_history(repository_root)
        forecasts = sources.load_410_operational_forecasts(repository_root)
    elif battery_id == BATTERIES[1]:
        observed = sources.load_110_observed_history(repository_root)
        forecasts = sources.load_110_operational_forecasts(repository_root)
    else:
        raise ValueError('Unknown battery')
    return analyze_battery_rul(battery_id=battery_id, observed=observed, forecasts=forecasts,
                               threshold_microvolt=threshold_microvolt,
                               crossing_direction=crossing_direction)
