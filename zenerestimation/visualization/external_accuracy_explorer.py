"""Read-only Streamlit external forecast accuracy explorer (C3)."""
from __future__ import annotations

from pathlib import Path

from zenerestimation.visualization.external_accuracy_evidence import (
    BATTERY, MODELS, load_external_accuracy_evidence,
)


def render_external_accuracy_explorer(repository_root: Path) -> None:
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    st.header('External Forecast Accuracy Explorer · 16.4-C3')
    st.caption('Six measured holdout quarters · 2024-03 to 2025-06 · No model fitting or optimization')
    path = Path(repository_root) / 'results' / BATTERY / 'external_accuracy_16_4_c3' / 'external_accuracy_evidence.json'
    if not path.is_file():
        st.info('C3 evidence not generated yet. Run: python examples/diagnostics/run_16_4_c3_external_accuracy.py')
        return
    try:
        evidence = load_external_accuracy_evidence(path)
    except (ValueError, OSError, KeyError) as exc:
        st.error(f'External accuracy evidence failed validation: {exc}')
        return
    st.warning('These are separately regenerated diagnostic forecasts reconciled to frozen aggregate metrics, NOT the original frozen forecast vectors. The six holdout measurements are for evaluation only.')
    st.caption(f"Maximum absolute difference from frozen benchmark metrics: {evidence['max_abs_metric_delta']:.3g}; tolerance: {evidence['tolerance_abs']:.3g}.")
    selected = st.multiselect('External accuracy models', MODELS, default=list(MODELS), key='c3_external_models')
    if not selected:
        st.info('Select at least one model.')
        return
    summary = pd.DataFrame(evidence['metrics'])
    summary = summary[summary.model.isin(selected)]
    rows = pd.DataFrame(evidence['rows'])
    rows = rows[rows.model.isin(selected)].copy()
    st.subheader('Six-quarter external accuracy (RMSE ranking)')
    ranked = summary.sort_values('rmse')
    st.dataframe(ranked, hide_index=True, width='stretch')
    fig = px.bar(ranked, x='model', y='rmse', labels={'rmse': 'RMSE (µV)', 'model': 'Model'}, title='External six-quarter RMSE · lower is better')
    st.plotly_chart(fig, width='stretch')
    st.subheader('First-quarter absolute error (separate ranking)')
    first = summary.sort_values('first_quarter_absolute_error')
    fig = px.bar(first, x='model', y='first_quarter_absolute_error', labels={'first_quarter_absolute_error': 'Absolute error (µV)', 'model': 'Model'}, title='March 2024 · first holdout prediction')
    st.plotly_chart(fig, width='stretch')
    st.subheader('Measured versus forecast voltage')
    measured = rows[['date', 'measured_microVolt']].drop_duplicates('date').rename(columns={'measured_microVolt': 'value'})
    measured['model'] = 'Measured'
    predictions = rows[['date', 'model', 'predicted_microVolt']].rename(columns={'predicted_microVolt': 'value'})
    lines = pd.concat([measured, predictions], ignore_index=True)
    fig = px.line(lines, x='date', y='value', color='model', markers=True, labels={'value': 'Voltage (µV)', 'date': 'Holdout date'})
    st.plotly_chart(fig, width='stretch')
    st.subheader('Signed external forecast residuals')
    st.caption('Signed error = predicted − measured. Positive values indicate overestimation.')
    fig = px.line(rows, x='date', y='error_predicted_minus_measured', color='model', markers=True, labels={'error_predicted_minus_measured': 'Error (µV)', 'date': 'Holdout date'})
    fig.add_hline(y=0, line_dash='dash')
    st.plotly_chart(fig, width='stretch')
    with st.expander('Date-by-date evidence and provenance', expanded=False):
        st.dataframe(rows, hide_index=True, width='stretch')
        st.caption('Source kind: ' + evidence['prediction_source_kind'])
        for limitation in evidence['limitations']:
            st.caption('• ' + limitation)
