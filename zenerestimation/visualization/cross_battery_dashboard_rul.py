"""Sprint 15.4.5: read-only, evidence-labeled Streamlit RUL panel.

No file writes, model fitting, probabilistic calibration or horizon extension.
The dashboard loads each battery independently and never pools microvolt values.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from zenerestimation.visualization.cross_battery_rul import (
    BATTERIES, MODELS, STATUS, BatteryRULDiagnostics, load_battery_rul,
)

ILLUSTRATIVE_THRESHOLDS = {BATTERIES[0]: 35.0, BATTERIES[1]: 175.0}


@dataclass(frozen=True)
class RULDashboardView:
    battery_id: str
    crossings: pd.DataFrame
    dispersion: pd.DataFrame
    crossing_count: int
    censored_count: int
    already_crossed_count: int
    threshold_microvolt: float
    interpretation: str


def build_rul_dashboard_view(result: BatteryRULDiagnostics) -> RULDashboardView:
    """Validate and copy the scientific outputs before presentation."""
    if result.battery_id not in BATTERIES or result.forecast_status != STATUS:
        raise ValueError('Unexpected battery or operational forecast provenance')
    crossings = result.model_crossings.copy(deep=True)
    dispersion = result.dispersion.copy(deep=True)
    expected = {'model', 'battery_id', 'status', 'threshold_microvolt', 'crossing_direction',
                'crossing_date', 'rul_quarters', 'rul_years', 'horizon_quarters', 'forecast_status'}
    if not expected.issubset(crossings.columns) or len(crossings) != len(MODELS):
        raise ValueError('Invalid model-crossing table')
    if set(crossings.model) != set(MODELS) or crossings.model.duplicated().any():
        raise ValueError('Missing or duplicated model')
    allowed = {'forecast_crossing_unvalidated', 'not_crossed_within_horizon', 'already_crossed_observed'}
    if not set(crossings.status).issubset(allowed):
        raise ValueError('Unknown threshold-crossing status')
    if not crossings.battery_id.eq(result.battery_id).all() or not crossings.forecast_status.eq(STATUS).all():
        raise ValueError('Inconsistent crossing provenance')
    if not crossings.threshold_microvolt.eq(result.threshold_microvolt).all():
        raise ValueError('Inconsistent threshold')
    if not crossings.crossing_direction.eq(result.crossing_direction).all():
        raise ValueError('Inconsistent crossing direction')
    if dispersion.empty or not {'battery_id', 'date', 'forecast_quarter', 'minimum_microvolt',
                                'maximum_microvolt', 'median_microvolt', 'std_microvolt_population',
                                'spread_microvolt', 'spread_percent_of_last_observed', 'forecast_status'}.issubset(dispersion.columns):
        raise ValueError('Invalid dispersion table')
    if not dispersion.battery_id.eq(result.battery_id).all() or not dispersion.forecast_status.eq(STATUS).all():
        raise ValueError('Inconsistent dispersion provenance')
    if list(dispersion.forecast_quarter) != list(range(1, len(dispersion) + 1)):
        raise ValueError('Forecast quarters must be consecutive')
    if not crossings.horizon_quarters.eq(len(dispersion)).all():
        raise ValueError('Forecast horizon mismatch')
    if not (dispersion.maximum_microvolt >= dispersion.minimum_microvolt).all():
        raise ValueError('Invalid forecast range')
    if not ((dispersion.maximum_microvolt - dispersion.minimum_microvolt - dispersion.spread_microvolt).abs() < 1e-8).all():
        raise ValueError('Invalid forecast spread')
    crossed = crossings.status.eq('forecast_crossing_unvalidated')
    censored = crossings.status.eq('not_crossed_within_horizon')
    already = crossings.status.eq('already_crossed_observed')
    if crossings.loc[censored, ['crossing_date', 'rul_quarters', 'rul_years']].notna().any().any():
        raise ValueError('Censored outcomes must not have crossing dates or RUL values')
    return RULDashboardView(result.battery_id, crossings, dispersion, int(crossed.sum()),
                            int(censored.sum()), int(already.sum()), result.threshold_microvolt,
                            result.interpretation)


def render_rul_dashboard(repository_root: str | Path = '.') -> None:
    """Render as an optional section inside the existing Streamlit dashboard."""
    import streamlit as st
    import plotly.graph_objects as go

    st.header('Battery-specific RUL & forecast robustness')
    st.warning('Exploratory threshold scenarios only. Neither threshold is a validated engineering '
               'failure limit. Operational forecasts are unvalidated; model disagreement is not '
               'a prediction interval or failure probability.')
    for battery in BATTERIES:
        with st.expander(f'{battery} — threshold scenario', expanded=False):
            st.caption('410: optimized holdout benchmark; 110: transferred holdout benchmark. '
                       'Operational extrapolations are separate from frozen holdout evidence.')
            threshold = st.number_input(
                f'Illustrative upper threshold (µV) — {battery}',
                value=ILLUSTRATIVE_THRESHOLDS[battery], step=0.5,
                key=f'rul_threshold_{battery}',
            )
            try:
                result = load_battery_rul(repository_root, battery_id=battery,
                                          threshold_microvolt=float(threshold), crossing_direction='above')
                view = build_rul_dashboard_view(result)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                st.error(f'Unable to validate RUL scenario: {exc}')
                continue
            st.caption(f'Last measured: {result.last_observed_date} · '
                       f'{result.last_observed_microvolt:.6f} µV · '
                       f'{len(view.dispersion)} forecast quarters · {STATUS}')
            c1, c2, c3 = st.columns(3)
            c1.metric('Forecast crossings', f'{view.crossing_count}/{len(MODELS)}')
            c2.metric('Right-censored', f'{view.censored_count}/{len(MODELS)}')
            c3.metric('Already crossed (observed)', f'{view.already_crossed_count}/{len(MODELS)}')
            st.dataframe(view.crossings, hide_index=True, width='stretch')
            st.caption('A missing crossing date means no crossing within the saved horizon; '
                       'it does not mean infinite RUL. RUL is measured from the last observation.')
            fig = go.Figure()
            d = view.dispersion
            fig.add_trace(go.Scatter(x=d.date, y=d.minimum_microvolt, name='Minimum', mode='lines'))
            fig.add_trace(go.Scatter(x=d.date, y=d.median_microvolt, name='Median', mode='lines+markers'))
            fig.add_trace(go.Scatter(x=d.date, y=d.maximum_microvolt, name='Maximum', mode='lines'))
            fig.add_hline(y=float(threshold), line_dash='dash', annotation_text='Illustrative threshold')
            fig.update_layout(title=f'{battery} · operational forecast dispersion',
                              xaxis_title='Forecast quarter', yaxis_title='Microvolt (µV)')
            st.plotly_chart(fig, width='stretch', key=f'rul_dispersion_chart_{battery}')
            st.dataframe(view.dispersion, hide_index=True, width='stretch')
            st.caption(view.interpretation)
            st.download_button('Export crossing scenarios (CSV)',
                               data=view.crossings.to_csv(index=False).encode('utf-8'),
                               file_name=f'{battery}_illustrative_rul_crossings.csv',
                               mime='text/csv', key=f'rul_crossings_export_{battery}')
            st.download_button('Export forecast dispersion (CSV)',
                               data=view.dispersion.to_csv(index=False).encode('utf-8'),
                               file_name=f'{battery}_operational_dispersion.csv',
                               mime='text/csv', key=f'rul_dispersion_export_{battery}')
