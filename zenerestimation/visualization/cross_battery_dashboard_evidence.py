"""Sprint 15.3.3: read-only, provenance-aware dashboard evidence charts.

Pure chart constructors intentionally do not load files or train models.
The Streamlit renderer is optional and imported only when invoked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from zenerestimation.visualization.cross_battery_dashboard_sources import (
    load_110_observed_history, load_110_operational_forecasts,
    load_410_observed_history, load_410_operational_forecasts,
)

BATTERY_410 = '732B-5610410'
BATTERY_110 = '732B-5610110'
MODELS = ('ARIMA', 'Kalman', 'LSTM', 'GRU', 'LinearTrendLSTM', 'KalmanLSTM')
FROZEN_ORIGIN = 'frozen_transferred_benchmark'
DIAGNOSTIC_ORIGIN = 'regenerated_diagnostic_not_frozen'


def _subset(frame: pd.DataFrame, battery: str, models: Iterable[str], *, origin: str) -> pd.DataFrame:
    selected = tuple(models)
    if not selected or any(model not in MODELS for model in selected):
        raise ValueError('Select at least one recognized model')
    required = {'battery_id', 'date', 'model', 'origin'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing evidence columns: {sorted(required - set(frame.columns))}')
    rows = frame.loc[(frame['battery_id'] == battery) & frame['model'].isin(selected)].copy()
    if rows.empty:
        raise ValueError(f'No structured evidence available for {battery}')
    if not rows['origin'].eq(origin).all():
        raise ValueError('Evidence provenance does not match the requested chart')
    rows['date'] = pd.to_datetime(rows['date'], errors='raise')
    if rows[['date', 'model']].duplicated().any():
        raise ValueError('Duplicate model/date evidence rows')
    return rows.sort_values(['model', 'date'])


def make_110_holdout_chart(frame: pd.DataFrame, models: Iterable[str] = MODELS) -> go.Figure:
    """Render measured values and saved frozen predictions, never refit."""
    rows = _subset(frame, BATTERY_110, models, origin=FROZEN_ORIGIN)
    required = {'measured_microvolt', 'predicted_microvolt'}
    if not required.issubset(rows.columns):
        raise ValueError(f'Missing holdout columns: {sorted(required - set(rows.columns))}')
    fig = go.Figure()
    measured = rows[['date', 'measured_microvolt']].drop_duplicates()
    if measured.groupby('date')['measured_microvolt'].nunique(dropna=False).gt(1).any():
        raise ValueError('Inconsistent measured values between models')
    measured = measured.drop_duplicates('date').sort_values('date')
    fig.add_trace(go.Scatter(x=measured['date'], y=measured['measured_microvolt'], mode='lines+markers',
                             name='Measured (observed)', line={'color': '#111827', 'width': 3}))
    for model, part in rows.groupby('model', sort=False):
        fig.add_trace(go.Scatter(x=part['date'], y=part['predicted_microvolt'], mode='lines+markers', name=model))
    fig.update_layout(title='110 · Frozen transferred holdout (5 quarters)', xaxis_title='Quarter',
                      yaxis_title='microVolt (µV)', hovermode='x unified', legend_title='Evidence / model')
    return fig


def make_110_history_chart(frame: pd.DataFrame, models: Iterable[str] = MODELS,
                           observed: pd.DataFrame | None = None) -> go.Figure:
    """Render regenerated diagnostic fits; warmup NaNs remain missing."""
    rows = _subset(frame, BATTERY_110, models, origin=DIAGNOSTIC_ORIGIN)
    if 'fitted_microvolt' not in rows.columns:
        raise ValueError('Missing fitted_microvolt column')
    fig = go.Figure()
    if observed is not None:
        _validate_observed(observed)
        fig.add_trace(go.Scatter(x=observed['date'], y=observed['measured_microvolt'],
                                 mode='lines+markers', name='Measured (observed)',
                                 line={'color': '#111827', 'width': 3}, marker={'size': 5}))
    for model, part in rows.groupby('model', sort=False):
        # The first ARIMA fitted value is a known zero-initialization artifact.
        # Mask it only for display; preserve all source evidence unchanged.
        plotted = part['fitted_microvolt'].copy()
        if model == 'ARIMA' and not plotted.empty and plotted.iloc[0] == 0:
            plotted.iloc[0] = float('nan')
        fig.add_trace(go.Scatter(x=part['date'], y=plotted, mode='lines',
                                 connectgaps=False, name=model))
    fig.add_vline(x=pd.Timestamp('2022-12-01').timestamp() * 1000,
                  line_dash='dash', line_color='gray')
    fig.update_layout(title='110 · Observed history and regenerated fits (diagnostic only)',
                      xaxis_title='Quarter', yaxis_title='microVolt (µV)',
                      hovermode='x unified', legend_title='Regenerated model')
    return fig



def _validate_observed(observed: pd.DataFrame) -> None:
    if not {'date', 'measured_microvolt'}.issubset(observed.columns) or observed.empty:
        raise ValueError('Missing measured history')
    dates = pd.to_datetime(observed['date'], errors='raise')
    values = pd.to_numeric(observed['measured_microvolt'], errors='raise')
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Invalid measured history dates')
    if not values.notna().all() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError('Invalid measured history values')


def make_110_future_chart(forecasts: pd.DataFrame, observed: pd.DataFrame,
                          models: Iterable[str] = MODELS) -> go.Figure:
    """Operational forecasts only: append last observation as visual anchor."""
    _validate_observed(observed)
    selected = tuple(models)
    if not selected or len(set(selected)) != len(selected) or any(m not in MODELS for m in selected):
        raise ValueError('Invalid model selection')
    if list(forecasts.columns) != ['date', *MODELS] or forecasts.empty:
        raise ValueError('Invalid operational forecast schema')
    dates = pd.to_datetime(forecasts['date'], errors='raise')
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Invalid operational forecast dates')
    last = observed.iloc[-1]
    cutoff = pd.Timestamp(last['date'])
    if cutoff != pd.Timestamp('2025-03-01') or dates.min() != cutoff + pd.DateOffset(months=3):
        raise ValueError('Forecast does not follow measured cutoff')
    fig = go.Figure()
    recent = observed.tail(12)
    fig.add_trace(go.Scatter(x=recent['date'], y=recent['measured_microvolt'],
                             name='Measured (observed)', mode='lines+markers',
                             line={'color': '#111827', 'width': 3}))
    for model in selected:
        series = pd.to_numeric(forecasts[model], errors='raise')
        if not series.notna().all() or not np.isfinite(series.to_numpy(dtype=float)).all():
            raise ValueError(f'Invalid forecast values: {model}')
        fig.add_trace(go.Scatter(x=[cutoff, *dates.tolist()],
                                 y=[float(last['measured_microvolt']), *series.tolist()],
                                 name=f'{model} · unvalidated', mode='lines+markers',
                                 line={'dash': 'dash'}))
    fig.add_vline(x=cutoff.timestamp() * 1000, line_dash='dot', line_color='gray')
    fig.update_layout(title='110 · Operational forecasts (unvalidated; 2025-06 to 2026-09)',
                      xaxis_title='Quarter', yaxis_title='microVolt (µV)',
                      hovermode='x unified', legend_title='Observation / model')
    return fig

def make_410_future_chart(forecasts: pd.DataFrame, observed: pd.DataFrame,
                          models: Iterable[str] = MODELS) -> go.Figure:
    """Show saved 410 operational predictions anchored to last measurement."""
    _validate_observed(observed)
    selected = tuple(models)
    if not selected or len(set(selected)) != len(selected) or any(m not in MODELS for m in selected):
        raise ValueError('Invalid model selection')
    if list(forecasts.columns) != ['date', *MODELS] or forecasts.empty:
        raise ValueError('Invalid 410 operational forecast schema')
    dates = pd.to_datetime(forecasts['date'], errors='raise')
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Invalid 410 operational forecast dates')
    last = observed.iloc[-1]
    cutoff = pd.Timestamp(last['date'])
    if cutoff != pd.Timestamp('2025-06-01') or dates.min() != cutoff + pd.DateOffset(months=3):
        raise ValueError('410 forecast does not follow measured cutoff')
    fig = go.Figure()
    recent = observed.tail(12)
    fig.add_trace(go.Scatter(x=recent['date'], y=recent['measured_microvolt'],
                             name='Measured (observed)', mode='lines+markers',
                             line={'color': '#111827', 'width': 3}))
    for model in selected:
        values = pd.to_numeric(forecasts[model], errors='raise')
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f'Invalid 410 operational values: {model}')
        fig.add_trace(go.Scatter(x=[cutoff, *dates.tolist()],
                                 y=[float(last['measured_microvolt']), *values.tolist()],
                                 name=f'{model} · unvalidated', mode='lines+markers',
                                 line={'dash': 'dash'}))
    fig.add_vline(x=cutoff.timestamp() * 1000, line_dash='dot', line_color='gray')
    fig.update_layout(title='410 · Operational forecasts (unvalidated; 2025-09 to 2026-12)',
                      xaxis_title='Quarter', yaxis_title='microVolt (µV)',
                      hovermode='x unified', legend_title='Observation / model')
    return fig


def verified_410_visuals(repository_root: str | Path) -> dict[str, Path]:
    """Return existing frozen benchmark images; do not generate any files."""
    root = Path(repository_root) / 'results' / BATTERY_410 / 'optimized_benchmark'
    names = ('benchmark_history.png', 'benchmark_best_worst_history.png', 'benchmark_forecast.png')
    return {name: root / name for name in names if (root / name).is_file()}


def render_evidence_section(comparison, repository_root: str | Path = '.') -> None:
    """Optional Streamlit section. Pass the result of load_comparison()."""
    import streamlit as st

    st.subheader('Historical, holdout and future evidence')
    st.caption('Read-only saved evidence · no retraining · benchmark and operational predictions kept separate')
    battery = st.selectbox('Battery for evidence visualization', (BATTERY_410, BATTERY_110),
                           format_func=lambda b: f'{b} ({"optimized" if b == BATTERY_410 else "transferred"})',
                           key='s1533_battery')
    if battery == BATTERY_410:
        selected_410 = st.multiselect('Models shown', MODELS, default=MODELS, key='s1533_models_410')
        if not selected_410:
            st.info('Select at least one model to display evidence.')
            return
        benchmark, future_410 = st.tabs(['Saved optimized benchmark images', 'Future operational forecasts'])
        with benchmark:
            st.warning('410 historical fits and frozen holdout predictions are visual-only. No structured benchmark series are available.')
            visuals = verified_410_visuals(repository_root)
            for filename, label in (('benchmark_history.png', 'Historical benchmark'),
                                    ('benchmark_best_worst_history.png', 'Best/worst benchmark'),
                                    ('benchmark_forecast.png', 'Holdout forecast')):
                if filename in visuals:
                    st.image(str(visuals[filename]), caption=f'{label} · saved optimized benchmark image')
                else:
                    st.info(f'{label}: saved image unavailable ({filename}).')
        with future_410:
            st.warning('Operational extrapolations are unvalidated: no measured future actuals, errors, or uncertainty intervals.')
            try:
                observed = load_410_observed_history(repository_root)
                forecasts = load_410_operational_forecasts(repository_root)
                st.plotly_chart(make_410_future_chart(forecasts, observed, selected_410),
                                width="stretch")
                st.caption('Frozen 410 optimized configurations refitted through 2025-06 without reoptimization. Saved JSON predictions; CSV cross-checked.')
            except (OSError, ValueError, KeyError) as exc:
                st.error(f'Unable to validate saved 410 operational forecasts: {exc}')
        return

    selected = st.multiselect('Models shown', MODELS, default=MODELS, key='s1533_models')
    if not selected:
        st.info('Select at least one model to display evidence.')
        return
    holdout, history, future = st.tabs([
        'Frozen holdout forecasts', 'Regenerated historical fits', 'Future operational forecasts'
    ])
    with holdout:
        st.caption('Frozen transferred benchmark predictions versus measured holdout observations (2023-03 to 2024-03).')
        if comparison.holdout_predictions.empty:
            st.warning('Structured holdout evidence is unavailable.')
        else:
            st.plotly_chart(make_110_holdout_chart(comparison.holdout_predictions, selected), width="stretch")
    with history:
        st.warning('Historical fits were regenerated from frozen configurations. They are NOT original frozen benchmark fits.')
        if comparison.historical_fits.empty:
            st.info('Regenerated historical-fit evidence is unavailable.')
        else:
            try:
                observed = load_110_observed_history(repository_root)
                st.plotly_chart(make_110_history_chart(comparison.historical_fits, selected, observed),
                                width="stretch")
            except (OSError, ValueError, KeyError) as exc:
                st.error(f'Unable to validate observed historical evidence: {exc}')
    with future:
        st.warning('Operational extrapolations are unvalidated. No measured future actuals, errors, or uncertainty intervals are available.')
        try:
            observed = load_110_observed_history(repository_root)
            forecasts = load_110_operational_forecasts(repository_root)
            st.plotly_chart(make_110_future_chart(forecasts, observed, selected),
                            width="stretch")
            st.caption('Forecasts use frozen transferred configurations refitted through 2025-03 without reoptimization.')
        except (OSError, ValueError, KeyError) as exc:
            st.error(f'Unable to validate saved operational forecasts: {exc}')
