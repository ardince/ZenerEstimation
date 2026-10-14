"""Sprint 15.3.3: read-only Streamlit + Plotly cross-battery dashboard.

Run from repository root:
    python -m streamlit run examples/dashboard/cross_battery_dashboard.py
"""
from __future__ import annotations

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zenerestimation.visualization.cross_battery_dashboard_view import (
    BATTERY_IDS, METRIC_LABELS, dashboard_view,
)
from zenerestimation.visualization.cross_battery_dashboard_charts import (
    grouped_metric_chart, ranking_heatmap, battery_profile_chart,
)
from zenerestimation.visualization.cross_battery_artifacts import MODELS
from zenerestimation.visualization.cross_battery_comparison import load_comparison
from zenerestimation.visualization.cross_battery_dashboard_evidence import render_evidence_section
from zenerestimation.visualization.cross_battery_dashboard_provenance import (
    build_provenance,
    render_provenance_section,
)
from zenerestimation.visualization.cross_battery_dashboard_audit import (
    audit_dashboard_sources,
)
from zenerestimation.visualization.cross_battery_dashboard_numerical_ui import (
    render_numerical_explorer,
)
from zenerestimation.visualization.cross_battery_dashboard_rul import (
    render_rul_dashboard,
)

def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit('Streamlit is not installed. Run: python -m pip install streamlit') from exc

    st.set_page_config(page_title='ZenerEstimation | Cross-Battery', layout='wide')
    st.title('Cross-Battery Benchmark Explorer')
    st.caption('Sprint 15.3.3 extension · Saved evidence · Read-only · RMSE/MAE: µV · MAPE: %')

    try:
        comparison = load_comparison(str(REPOSITORY_ROOT))
    except Exception as exc:
        st.error(f'Unable to validate frozen benchmark artifacts: {exc}')
        st.stop()
        return

    with st.sidebar:
        st.header('Benchmark selection')
        selected = st.multiselect('Batteries', BATTERY_IDS, default=list(BATTERY_IDS))
        metric = st.selectbox('Metric', tuple(METRIC_LABELS), format_func=lambda x: METRIC_LABELS[x])
        models = st.multiselect('Models', MODELS, default=list(MODELS))
        st.caption('Selections filter saved evidence only; no models are executed.')

    if not selected:
        st.info('Select at least one battery to display results.')
        return
    if not models:
        st.info('Select at least one model to display charts.')
        return

    view = dashboard_view(comparison, selected, metric)
    st.warning(
        'Descriptive comparison only: dataset 410 is optimized (6-quarter holdout); '
        'dataset 110 is frozen parameter transfer (5-quarter holdout). '
        'Absolute metric differences do not establish generalization performance.'
    )

    st.subheader('Benchmark summary')
    st.dataframe(view.summary, hide_index=True, width="stretch",
                 column_config={'best_rmse_microvolt': st.column_config.NumberColumn('Best RMSE (µV)', format='%.6f')})

    st.subheader(f'Interactive model comparison · {METRIC_LABELS[metric]}')
    st.plotly_chart(grouped_metric_chart(comparison.model_metrics, selected, models, metric),
                    width="stretch")
    st.caption('Each bar is a frozen benchmark metric. Hover to inspect its value and within-battery rank.')

    st.subheader('Model ranking heatmap')
    st.plotly_chart(ranking_heatmap(comparison.model_metrics, selected, models, metric),
                    width="stretch")
    st.caption('Ranks are computed using all six models within each battery, even if only some are displayed. Rank 1 is best.')

    st.subheader('Within-battery performance profile')
    profile_battery = st.selectbox('Battery for detailed profile', selected,
                                   format_func=lambda b: f'{b} ({"optimized" if b.endswith("410") else "transferred"})')
    st.plotly_chart(battery_profile_chart(comparison.model_metrics, profile_battery, models, metric),
                    width="stretch")

    with st.expander('Numerical model comparison', expanded=False):
        filtered = view.metrics.loc[view.metrics.model.isin(models)].copy(deep=True)
        st.dataframe(filtered, hide_index=True, width="stretch",
                     column_config={metric: st.column_config.NumberColumn(METRIC_LABELS[metric], format='%.6f')})
    with st.expander('Evidence & provenance availability', expanded=False):
        st.dataframe(view.evidence, hide_index=True, width="stretch")
        st.caption(
            '410 holdout predictions and historical fits are visual-only. '
            '110 historical fits are regenerated diagnostics, not frozen historical predictions.'
        )

    st.divider()
    render_evidence_section(comparison, repository_root=REPOSITORY_ROOT)

    st.divider()

    render_numerical_explorer(
        comparison=comparison,
        repository_root=REPOSITORY_ROOT,
    )

    st.divider()

    provenance = build_provenance(REPOSITORY_ROOT, comparison)

    audit = audit_dashboard_sources(
        repository_root=REPOSITORY_ROOT,
        comparison=comparison,
    )

    render_provenance_section(provenance)

    with st.expander("Dashboard source-integrity audit", expanded=False):
        if audit.passed:
            st.success("All dashboard source-integrity checks passed.")
        else:
            st.error("One or more source-integrity checks failed.")

        st.dataframe(
            audit.checks,
            hide_index=True,
            width="stretch",
        )

        if not audit.passed:
            st.subheader("Failed checks")
            st.dataframe(
                audit.failures,
                hide_index=True,
                width="stretch",
            )

        st.caption(
            "Checks verify consistency of saved artifacts and provenance. "
            "They do not establish forecast accuracy or scientific validity."
        )

    st.divider()
    render_rul_dashboard(REPOSITORY_ROOT)

if __name__ == '__main__':
    main()
