"""Dynamic, read-only numerical explorer for optimized neural evidence.

Source-precision exports and clear temporal/evidence separation.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from zenerestimation.visualization.cross_battery_dashboard_numerical_ui import csv_bytes, display_table

OPTIONS = (
    'Benchmark vs optimized · external RMSE and MAE',
    'Historical fitted values · development in-sample',
    'Historical fitted signed errors · development in-sample',
    'External holdout · measured and predictions',
    'External holdout · signed errors',
    'Operational forecasts · saved and optimized',
    'Operational forecasts · optimized minus saved',
)


def _wide(frame, date_col, value_col, models, *, label='Quarter'):
    selected = frame.loc[frame['model'].isin(models)].copy()
    selected[date_col] = pd.to_datetime(selected[date_col], errors='raise')
    if selected.duplicated([date_col, 'model']).any():
        raise ValueError('Duplicate date/model records')
    table = selected.pivot(index=date_col, columns='model', values=value_col)
    table = table.reindex(columns=list(models)).sort_index()
    return table.rename_axis(label).reset_index()


def _attach_actual(table, source, value_col, *, date_col='date', target='Measured (µV)'):
    selected = source.loc[source['model'].eq('LSTM'), [date_col, value_col]].copy()
    selected[date_col] = pd.to_datetime(selected[date_col])
    selected = selected.rename(columns={date_col: 'Quarter', value_col: target})
    if selected['Quarter'].duplicated().any():
        raise ValueError('Duplicate measurement dates')
    return table.merge(selected, on='Quarter', how='left', validate='one_to_one')[[
        'Quarter', target, *[c for c in table.columns if c != 'Quarter']]]


def _benchmark_metrics(root, manifest, models):
    from zenerestimation.visualization.model_performance_comparison import build_comparison
    external, _, _ = build_comparison(Path(root))
    needed = {'model', 'external_rmse_microvolt'}
    if not needed.issubset(external.columns):
        raise ValueError('Frozen benchmark RMSE columns missing')
    result = []
    for model in models:
        matching = external.loc[external['model'].eq(model)]
        if len(matching) != 1:
            raise ValueError(f'Frozen benchmark unavailable or ambiguous: {model}')
        benchmark = float(matching.iloc[0]['external_rmse_microvolt'])
        optimized = float(manifest['external_holdout_metrics'][model]['rmse'])
        result.append({'Model': model, 'Benchmark external RMSE (µV)': benchmark,
                       'Optimized external RMSE (µV)': optimized,
                       'Optimized − benchmark RMSE (µV)': optimized - benchmark,
                       'Optimized external MAE (µV)': float(manifest['external_holdout_metrics'][model]['mae']),
                       'Benchmark external MAE (µV)': float(matching.iloc[0]['external_mae_microvolt'])
                       if 'external_mae_microvolt' in matching and pd.notna(matching.iloc[0]['external_mae_microvolt']) else np.nan})
    return pd.DataFrame(result)


def numerical_result(root, manifest, holdout, future, historical, saved, *, kind, models):
    """Return (table, status), with dates as rows and models as columns."""
    if kind not in OPTIONS:
        raise ValueError('Unknown numerical result')
    from zenerestimation.visualization.optimized_neural_comparison import MODELS
    if len(set(models)) != len(models) or any(m not in MODELS for m in models):
        raise ValueError('Unknown or duplicate model selection')
    if not models:
        return pd.DataFrame(), 'Select at least one model.'
    if kind == OPTIONS[0]:
        return _benchmark_metrics(root, manifest, models), (
            'Descriptive six-quarter external RMSE/MAE comparison only. Earlier frozen benchmark '
            'and new optimized runs have different provenance; differences do not prove improvement.')
    if kind in OPTIONS[1:3]:
        if historical is None:
            return pd.DataFrame(), 'Historical fitted artifact is not available for this run.'
        col = 'fitted_microvolt' if kind == OPTIONS[1] else 'error_fitted_minus_measured'
        result = _wide(historical, 'date', col, models)
        if kind == OPTIONS[1]:
            result = _attach_actual(result, historical, 'measured_microvolt')
        return result, ('Development-period in-sample diagnostic only. NaN indicates warm-up or missing original measurement; '
                        'never compare its RMSE as if it were external holdout performance.')
    if kind in OPTIONS[3:5]:
        col = 'predicted_microvolt' if kind == OPTIONS[3] else 'error_pred_minus_actual'
        result = _wide(holdout, 'date', col, models)
        if kind == OPTIONS[3]:
            result = _attach_actual(result, holdout, 'actual_microvolt')
        return result, 'Six-quarter post-optimization external holdout; signed errors = prediction − measured (µV).'
    if kind == OPTIONS[5]:
        result = _wide(future, 'date', 'predicted_microvolt', models)
        if saved is not None:
            s = saved.copy()
            s['date'] = pd.to_datetime(s['date'], errors='raise')
            if s['date'].duplicated().any() or not pd.DatetimeIndex(s['date']).equals(pd.DatetimeIndex(result['Quarter'])):
                raise ValueError('Saved operational date alignment failed')
            for model in models:
                if model not in s:
                    raise ValueError(f'Missing saved operational model: {model}')
                result[f'{model} · saved'] = pd.to_numeric(s[model], errors='raise').to_numpy(dtype=float)
        return result, 'Future extrapolations without measured actuals. Saved and optimized values are separate predictions.'
    if saved is None:
        return pd.DataFrame(), 'Saved operational forecasts unavailable for difference calculation.'
    result = _wide(future, 'date', 'predicted_microvolt', models)
    s = saved.copy()
    s['date'] = pd.to_datetime(s['date'], errors='raise')
    if s['date'].duplicated().any() or not pd.DatetimeIndex(s['date']).equals(pd.DatetimeIndex(result['Quarter'])):
        raise ValueError('Saved operational date alignment failed')
    for model in models:
        if model not in s:
            raise ValueError(f'Missing saved operational model: {model}')
        result[model] = result[model] - pd.to_numeric(s[model], errors='raise').to_numpy(dtype=float)
    return result, 'Optimized − saved future prediction (µV); these are NOT errors against measurements.'


def orient_table(table, orientation):
    """Transpose only dated series; leave metric comparisons unchanged."""
    if orientation == 'Dates as rows' or 'Quarter' not in table:
        return table.copy(deep=True)
    if orientation != 'Dates as columns':
        raise ValueError('Unknown orientation')
    t = table.set_index('Quarter').T
    t.columns = pd.DatetimeIndex(t.columns).strftime('%Y-%m-%d')
    return t.rename_axis('Series').reset_index()


BATTERY_OPTIONS = ('732B-5610410', '732B-5610110', 'Both batteries', 'Current dashboard battery selection')


def resolve_batteries(choice, current):
    if choice not in BATTERY_OPTIONS:
        raise ValueError('Unknown battery selection')
    if choice == BATTERY_OPTIONS[2]:
        return ('732B-5610410', '732B-5610110')
    if choice == BATTERY_OPTIONS[3]:
        return tuple(b for b in ('732B-5610410', '732B-5610110') if b in current)
    return (choice,)


def benchmark_for_batteries(root, batteries, models):
    from zenerestimation.visualization.cross_battery_comparison import load_comparison
    from zenerestimation.visualization.cross_battery_dashboard_numerical_ui import numerical_result
    comparison = load_comparison(str(root))
    parts = []
    for battery in batteries:
        table, _ = numerical_result(comparison, root, battery_id=battery, kind='metrics', models=models)
        parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def render_optimized_numerical_explorer(root, manifest, holdout, future, historical, saved, models):
    import streamlit as st
    st.subheader('Numerical Results Explorer · optimized neural comparison')
    st.caption('Choose one numerical result; columns adapt automatically. Read-only; use the combined Excel workbook for export.')
    choice = st.selectbox('Battery set', BATTERY_OPTIONS,
                          key='optimized_numerical_battery')
    current = st.session_state.get('optimized_neural_dashboard_batteries', ('732B-5610410',))
    batteries = resolve_batteries(choice, current)
    a, b, c = st.columns([3, 2, 2])
    with a:
        kind = st.selectbox('Numerical result', OPTIONS, key='optimized_numerical_kind')
    with b:
        precision = st.selectbox('Display precision', ('Full', 2, 4, 6, 10),
                                 key='optimized_numerical_precision')
    with c:
        orientation = st.selectbox('Table orientation', ('Dates as rows', 'Dates as columns'),
                                   key='optimized_numerical_orientation')
    try:
        if batteries == ('732B-5610410',):
            table, status = numerical_result(root, manifest, holdout, future, historical, saved,
                                             kind=kind, models=models)
        elif kind == OPTIONS[0]:
            table = benchmark_for_batteries(root, batteries, models)
            status = ('Frozen benchmark metrics for selected batteries. Optimized neural '
                      'artifacts are currently available only for 732B-5610410; '
                      'no 110 optimization results are implied.')
        else:
            table = pd.DataFrame()
            status = ('Optimized neural evidence is currently available only for 732B-5610410. '
                      'Select that battery to inspect fitted, holdout or operational values; '
                      '110 optimization has not yet been completed.')
        table = orient_table(table, orientation)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f'Numerical evidence unavailable or invalid: {exc}')
        return
    st.info(status)
    if table.empty:
        st.warning('No structured numerical rows available for this selection.')
        return
    shown = display_table(table, decimals=None if precision == 'Full' else precision)
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption(f'{len(table)} rows · Numerical values shown at selected display precision')
