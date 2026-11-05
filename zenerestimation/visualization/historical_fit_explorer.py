"""Read-only C2 Historical Fit Explorer: consumes C1 JSON; never fits models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ('ARIMA', 'Kalman', 'LSTM', 'GRU', 'LinearTrendLSTM', 'KalmanLSTM')
RELATIVE = Path('results/732B-5610410/historical_fit_16_4_c1')


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path.name}: JSON root must be an object')
    return payload


def _verify_digest(payload: dict, name: str) -> None:
    recorded = payload.get('sha256_payload')
    if not isinstance(recorded, str):
        raise ValueError(f'{name}: missing SHA256 evidence digest')
    canonical = json.dumps({k: v for k, v in payload.items() if k != 'sha256_payload'},
                           sort_keys=True, separators=(',', ':'), allow_nan=False)
    expected = hashlib.sha256(canonical.encode()).hexdigest()
    if expected != recorded:
        raise ValueError(f'{name}: evidence SHA256 mismatch')


def load_historical_fit(root: Path) -> tuple[dict[str, dict], dict]:
    """Validate saved artifacts; no model imports or writes."""
    folder = Path(root) / RELATIVE
    artifacts = {}
    for model in MODELS:
        path = folder / f'{model}_historical_fit.json'
        payload = _load_json(path)
        _verify_digest(payload, model)
        if payload.get('model') != model or payload.get('schema_version') != 'historical-fit-evidence-v1':
            raise ValueError(f'{model}: unexpected model or schema')
        if payload.get('scope') != 'development_only_historical_reconstruction':
            raise ValueError(f'{model}: unexpected evidence scope')
        if payload.get('development_cutoff') != '2023-12-01':
            raise ValueError(f'{model}: unexpected development cutoff')
        rows = payload.get('rows')
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'{model}: no historical rows')
        dates = pd.to_datetime([r['ds'] for r in rows], errors='raise')
        if dates.has_duplicates or not dates.is_monotonic_increasing or dates.max() > pd.Timestamp('2023-12-01'):
            raise ValueError(f'{model}: invalid historical timestamps')
        for row in rows:
            actual, fitted, residual = (row.get('observed_microVolt'),
                                        row.get('fitted_microVolt'), row.get('residual_microVolt'))
            if any(v is not None and not np.isfinite(float(v)) for v in (actual, fitted, residual)):
                raise ValueError(f'{model}: nonfinite historical value')
            if residual is not None and (actual is None or fitted is None or not np.isclose(float(residual), float(fitted)-float(actual), atol=1e-9)):
                raise ValueError(f'{model}: inconsistent residual')
        artifacts[model] = payload
    common = _load_json(folder / 'common_interval_metrics.json')
    included = common.get('included_models', [])
    excluded = common.get('excluded_models', {})
    if set(included) | set(excluded) != set(MODELS) or set(included) & set(excluded):
        raise ValueError('Common comparison eligibility does not cover exactly six models')
    dates = common.get('dates', [])
    if dates != sorted(set(dates)):
        raise ValueError('Common comparison dates must be unique and sorted')
    for model in included:
        rows = {r['ds']: r for r in artifacts[model]['rows']}
        if any(d not in rows or rows[d]['observed_microVolt'] is None or rows[d]['residual_microVolt'] is None for d in dates):
            raise ValueError(f'{model}: common dates lack valid measured residuals')
        values = np.asarray([rows[d]['residual_microVolt'] for d in dates], dtype=float)
        expected = common['models'][model]
        for key, calculated in [('rmse', float(np.sqrt(np.mean(values**2))) if len(values) else None),
                                ('mae', float(np.mean(np.abs(values))) if len(values) else None),
                                ('bias', float(np.mean(values)) if len(values) else None)]:
            stored = expected.get(key)
            if (stored is None) != (calculated is None) or (stored is not None and not np.isclose(stored, calculated, atol=1e-9)):
                raise ValueError(f'{model}: common {key} does not match historical rows')
    return artifacts, common


def historical_frame(artifacts: dict, selected: list[str]) -> pd.DataFrame:
    rows = []
    for model in selected:
        for record in artifacts[model]['rows']:
            rows.append({'ds': pd.Timestamp(record['ds']), 'model': model,
                         'observed': record['observed_microVolt'],
                         'fitted': record['fitted_microVolt'],
                         'residual': record['residual_microVolt']})
    return pd.DataFrame(rows, columns=['ds', 'model', 'observed', 'fitted', 'residual'])


def render_historical_fit_explorer(repository_root: Path) -> None:
    """Streamlit is imported only on rendering, not while loading evidence."""
    import streamlit as st
    import plotly.graph_objects as go

    st.header('Historical Fit Explorer · 16.4-C2')
    st.caption('Saved development-only reconstructions; no model fitting, tuning or holdout evaluation.')
    try:
        artifacts, common = load_historical_fit(Path(repository_root))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        st.warning(f'Historical-fit evidence unavailable or invalid: {exc}')
        st.caption('Generate and verify evidence using examples/diagnostics/run_16_4_c1_historical_fit.py.')
        return

    available = list(common['included_models'])
    selected = st.multiselect('Historical fit models', MODELS, default=available,
                              key='historical_fit_models')
    st.info('These are historical reconstructions, not future forecast scores. KalmanLSTM includes a smoothed historical trend; prediction types differ.')
    st.write(f"**Common measured comparison:** {len(common['dates'])} dates · "
             f"**Included:** {', '.join(available) or 'none'}")
    for name, reason in common['excluded_models'].items():
        st.caption(f'{name}: unavailable for common comparison ({reason})')
    if not selected:
        st.info('Select a model to inspect its historical evidence.')
        return

    df = historical_frame(artifacts, selected)
    mode = st.radio('Date display', ['Full development history', 'Common measured dates'],
                    horizontal=True, key='historical_fit_date_mode')
    if mode == 'Common measured dates':
        df = df.loc[df.ds.isin(pd.to_datetime(common['dates']))]
    if df.empty:
        st.info('No historical points available for this selection.')
        return

    fig = go.Figure()
    observed = df[['ds', 'observed']].drop_duplicates('ds').sort_values('ds')
    fig.add_trace(go.Scatter(x=observed.ds, y=observed.observed, name='Measured voltage',
                             mode='markers+lines', line=dict(color='#333333', width=2),
                             connectgaps=False))
    for name in selected:
        series = df.loc[df.model == name].sort_values('ds')
        fig.add_trace(go.Scatter(x=series.ds, y=series.fitted, mode='lines', name=name,
                                 connectgaps=False))
    fig.update_layout(title='Measured and reconstructed voltage', xaxis_title='Date',
                      yaxis_title='Voltage (µV)', hovermode='x unified')
    st.plotly_chart(fig, width="stretch")

    residual_fig = go.Figure()
    for name in selected:
        series = df.loc[df.model == name].sort_values('ds')
        residual_fig.add_trace(go.Scatter(x=series.ds, y=series.residual, mode='lines+markers',
                                          name=name, connectgaps=False))
    residual_fig.add_hline(y=0, line_dash='dash', line_color='gray')
    residual_fig.update_layout(title='Historical residuals (fitted − measured)',
                               xaxis_title='Date', yaxis_title='Residual (µV)')
    st.plotly_chart(residual_fig, width="stretch")

    metrics = [{'model': name, **common['models'][name]} for name in selected if name in common['models']]
    st.subheader('Identical-date historical reconstruction metrics')
    if metrics:
        st.dataframe(pd.DataFrame(metrics), hide_index=True, width="stretch")
    else:
        st.info('No selected models have common-interval metrics.')
    st.caption('RMSE, MAE and bias are in µV. Missing measurements and model warm-up periods are excluded.')

    st.subheader('Coverage and prediction provenance')
    coverage = []
    for name in selected:
        payload = artifacts[name]
        coverage.append({'model': name, 'prediction_kind': payload['prediction_kind'],
                         **payload['coverage'], 'convergence': str(payload.get('optimizer_convergence',
                                                   payload.get('convergence', 'not recorded')))})
    st.dataframe(pd.DataFrame(coverage), hide_index=True, width="stretch")
    with st.expander('Evidence provenance and interpretation'):
        for name in selected:
            payload = artifacts[name]
            st.markdown(f'**{name}** · {payload["prediction_kind"]}')
            st.json({'configuration': payload.get('configuration', {}),
                     'preprocessing': payload.get('preprocessing', {}),
                     'development_cutoff': payload['development_cutoff'],
                     'sha256_payload': payload['sha256_payload'],
                     'interpretation': payload.get('interpretation'),
                     'optimizer_convergence': payload.get('optimizer_convergence',
                                             payload.get('convergence', 'not recorded'))})
    st.caption('ARIMA reported a convergence warning in the C1 console; inspect optimizer diagnostics before drawing conclusions. Historical fit is not out-of-sample accuracy.')
