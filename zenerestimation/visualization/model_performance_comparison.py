"""Read-only C4 overview: distinguish reconstruction, holdout, and operations.

Uses existing validated C2/C3 and operational loaders. No model execution or writes.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from zenerestimation.visualization.historical_fit_explorer import (
    MODELS, load_historical_fit, historical_frame,
)
from zenerestimation.visualization.external_accuracy_evidence import load_external_accuracy_evidence
from zenerestimation.visualization.cross_battery_dashboard_sources import (
    load_410_observed_history, load_410_operational_forecasts,
)
from zenerestimation.visualization.cross_battery_dashboard_evidence import (
    make_410_future_chart, verified_410_visuals,
)

BATTERY = '732B-5610410'


def build_comparison(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Read and validate independent evidence without treating it as matched experiments."""
    root = Path(root)
    artifacts, common = load_historical_fit(root)
    evidence_path = root / 'results' / BATTERY / 'external_accuracy_16_4_c3' / 'external_accuracy_evidence.json'
    external = load_external_accuracy_evidence(evidence_path)
    metrics = pd.DataFrame(external['metrics'])
    if metrics.empty or metrics.model.duplicated().any() or set(metrics.model) != set(MODELS):
        raise ValueError('External metrics must contain exactly one row for each model')
    if not {'model', 'rmse', 'mae', 'first_quarter_absolute_error'}.issubset(metrics.columns):
        raise ValueError('External metric schema mismatch')
    if set(common['included_models']) | set(common['excluded_models']) != set(MODELS):
        raise ValueError('Historical comparison model eligibility mismatch')
    rows = []
    for model in MODELS:
        ext = metrics.loc[metrics.model.eq(model)].iloc[0]
        hist = common['models'].get(model) if model in common['included_models'] else None
        rows.append({
            'model': model,
            'historical_rmse_microvolt': float(hist['rmse']) if hist else float('nan'),
            'external_rmse_microvolt': float(ext['rmse']),
            'first_quarter_absolute_error_microvolt': float(ext['first_quarter_absolute_error']),
            'historical_eligible': bool(hist),
        })
    return pd.DataFrame(rows), artifacts, common


def render_model_performance_comparison(repository_root: Path) -> None:
    import streamlit as st
    import plotly.graph_objects as go

    root = Path(repository_root)
    st.header('Model Performance & Forecast Comparison · 16.4-C4')
    st.caption('Dataset 410 · Saved evidence only · No training, tuning, forecasting, or artifact modification')
    st.info('Historical reconstruction, external holdout evaluation, and operational extrapolation answer different questions. Their errors are not interchangeable.')
    try:
        comparison, artifacts, common = build_comparison(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f'Unified comparison evidence unavailable or invalid: {exc}')
        return

    selected = st.multiselect('Models for unified comparison', MODELS,
                              default=list(MODELS), key='c4_models')
    if not selected:
        st.info('Select at least one model.')
        return
    overview, history, operational = st.tabs([
        'Performance overview', 'Historical vs saved benchmark', 'Future operational forecasts'
    ])
    with overview:
        st.subheader('Two evaluation periods · distinct metrics')
        st.caption(f"Historical: {len(common['dates'])} common genuinely measured development dates (through 2023-12). External: six measured holdout quarters (2024-03 to 2025-06).")
        display = comparison.loc[comparison.model.isin(selected)].copy()
        st.dataframe(display, hide_index=True, width="stretch")
        eligible = display.loc[display.historical_eligible]
        if not eligible.empty:
            fig = go.Figure()
            fig.add_bar(name='Historical reconstruction RMSE', x=eligible.model,
                        y=eligible.historical_rmse_microvolt)
            fig.add_bar(name='External six-quarter RMSE', x=eligible.model,
                        y=eligible.external_rmse_microvolt)
            fig.update_layout(barmode='group', yaxis_title='RMSE (µV)',
                              title='RMSE across different evaluation tasks (not optimization improvement)')
            st.plotly_chart(fig, width="stretch")
        for model in selected:
            if model in common['excluded_models']:
                st.caption(f"{model}: no comparable historical RMSE ({common['excluded_models'][model]}).")
        st.warning('Optimization improvement requires matched pre-optimization and post-optimization forecasts evaluated on the same holdout. Those baseline data are not established by these charts; no percentage gain is calculated.')
        st.caption('C3 forecasts are diagnostic regenerations reconciled to frozen aggregate benchmark metrics, not the original frozen prediction vectors.')

    with history:
        st.subheader('Saved optimized benchmark images · original evidence')
        st.caption('These PNGs are preserved exactly as saved; their plotted series are not used to calculate C2 historical errors.')
        visuals = verified_410_visuals(root)
        for filename, label in (
            ('benchmark_history.png', 'Historical benchmark'),
            ('benchmark_best_worst_history.png', 'Best/worst benchmark'),
            ('benchmark_forecast.png', 'Holdout forecast'),
        ):
            if filename in visuals:
                st.image(str(visuals[filename]), caption=f'{label} · saved benchmark PNG')
            else:
                st.info(f'Saved image unavailable: {filename}')
        st.subheader('C2 historical reconstruction · measured dates')
        chosen = [m for m in selected if m in common['included_models']]
        if not chosen:
            st.info('No selected model has eligible common-date historical reconstruction.')
        else:
            frame = historical_frame(artifacts, chosen)
            frame = frame.loc[frame.ds.isin(pd.to_datetime(common['dates']))]
            measured = frame[['ds', 'observed']].drop_duplicates('ds').sort_values('ds')
            fig = go.Figure()
            fig.add_scatter(x=measured.ds, y=measured.observed, mode='lines+markers',
                            name='Measured', line={'color':'#333333','width':3})
            for model in chosen:
                part = frame.loc[frame.model.eq(model)].sort_values('ds')
                fig.add_scatter(x=part.ds, y=part.fitted, mode='lines', name=model,
                                connectgaps=False)
            fig.update_layout(title='C2 · common measured dates (historical reconstruction)',
                              xaxis_title='Date', yaxis_title='Voltage (µV)')
            st.plotly_chart(fig, width="stretch")
            st.caption('C2 historical reconstructions may include smoothing; these are not external forecast predictions.')

    with operational:
        st.warning('Unvalidated future operational extrapolations. No future measured values or accuracy metrics are available for these dates.')
        try:
            observed = load_410_observed_history(root)
            forecasts = load_410_operational_forecasts(root)

            st.plotly_chart(
                make_410_future_chart(forecasts, observed, selected),
                width="stretch",
                key="c4_future_operational_forecasts_410",
            )
            st.caption('Existing validated operational evidence; refitted through 2025-06 without reoptimization. Forecasts cover 2025-09 through 2026-12.')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.error(f'Operational forecast evidence unavailable or invalid: {exc}')
