"""Concise, read-only presentation of existing Milestone 16.4 evidence."""
from __future__ import annotations
from pathlib import Path
from zenerestimation.visualization.optimized_neural_comparison import render_optimized_neural_comparison

from zenerestimation.visualization.model_performance_comparison import build_comparison
from zenerestimation.visualization.neural_optimization_explorer import (
    MODELS as NEURAL_MODELS, load_neural_run, render_neural_optimization_explorer,
)
from zenerestimation.visualization.external_accuracy_explorer import render_external_accuracy_explorer
from zenerestimation.visualization.historical_fit_explorer import render_historical_fit_explorer
from zenerestimation.visualization.cross_battery_dashboard_rul import render_rul_dashboard


def comparison_rows(root: Path, run_name: str) -> list[dict]:
    """Do not mix internal RMSE with external RMSE or calculate gains."""
    root = Path(root)
    external, _, _ = build_comparison(root)
    found = load_neural_run(root / 'results' / '732B-5610410' / run_name)
    rows = []
    for model in NEURAL_MODELS:
        match = external.loc[external.model.eq(model)]
        if len(match) != 1:
            raise ValueError(f'Missing or duplicate external metric for {model}')
        entry = found.get(model)
        rows.append({
            'Model': model,
            'Saved external RMSE (µV)': float(match.iloc[0]['external_rmse_microvolt']),
            'New internal RMSE (µV)': float(entry[1]['best_score']) if entry else None,
            'Optimized parameters': str(entry[1]['best_params']) if entry else 'Unavailable',
        })
    return rows


def render_concise_dashboard(repository_root: Path, *, benchmark_diagnostics=None) -> None:
    import streamlit as st
    root = Path(repository_root)
    st.header('Model performance and forecasts')
    st.caption('Concise view · existing evidence only · no training or new forecasts')
    performance, forecasts, diagnostics = st.tabs(['Performance', 'Forecasts', 'Diagnostics'])
    with performance:
        base = root / 'results' / '732B-5610410'
        runs = [p.name for p in (base / 'neural_expanded_16_4', base / 'neural_demo_16_4_a') if p.is_dir()]
        if not runs:
            st.info('No saved neural optimization runs found.')
        else:
            run = st.selectbox('Neural optimization run', runs, key='concise_neural_run')
            try:
                rows = comparison_rows(root, run)
            except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                st.error(f'Comparison evidence unavailable or invalid: {exc}')
            else:
                st.dataframe(rows, hide_index=True, width="stretch")
                st.caption('External: saved six-quarter benchmark, 2024-03 to 2025-06. Internal: temporal validation from the selected neural optimization run. These scores are NOT directly comparable.')
                st.caption('Original frozen benchmark is retained. New external holdout results appear in the optimized forecast comparison below.')
    with forecasts:
        st.subheader('Saved operational forecasts')
        st.info('Existing operational charts remain in “Historical, holdout and future evidence” above.')
        render_optimized_neural_comparison(root)
    with diagnostics:
        st.caption('Open detailed evidence only when needed. The original source artifacts remain unchanged.')
        with st.expander('External holdout diagnostics (C3)', expanded=False):
            render_external_accuracy_explorer(root)
        with st.expander('Neural optimization candidates and training histories', expanded=False):
            render_neural_optimization_explorer(root)
        with st.expander('Historical reconstruction (C2)', expanded=False):
            render_historical_fit_explorer(root)
        with st.expander('Remaining useful life (RUL)', expanded=False):
            render_rul_dashboard(root)
        if benchmark_diagnostics is not None:
            st.divider()
            benchmark_diagnostics()
