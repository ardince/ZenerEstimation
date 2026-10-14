"""Sprint 15.3.2: read-only Plotly figures from validated comparison tables.

No file I/O, training, forecast execution, or normalization across batteries.
All ranking calculations are *within each battery* for the selected metric.
"""
from __future__ import annotations

from typing import Sequence
import math

import pandas as pd
import plotly.graph_objects as go

from zenerestimation.visualization.cross_battery_dashboard_view import BATTERY_IDS, METRIC_LABELS
from zenerestimation.visualization.cross_battery_artifacts import MODELS

BENCHMARK_TYPES = {'732B-5610410': 'optimized', '732B-5610110': 'transferred'}
LABELS = {'732B-5610410': '410 · optimized', '732B-5610110': '110 · transferred'}


def _validated(metrics: pd.DataFrame, batteries: Sequence[str], models: Sequence[str], metric: str) -> pd.DataFrame:
    if metric not in METRIC_LABELS:
        raise ValueError(f'Unsupported metric: {metric}')
    if not batteries or len(set(batteries)) != len(batteries) or not set(batteries).issubset(BATTERY_IDS):
        raise ValueError('Select one or two distinct known batteries')
    if not models or len(set(models)) != len(models) or not set(models).issubset(MODELS):
        raise ValueError('Select distinct known models')
    required = {'battery_id', 'benchmark_type', 'model', metric}
    if not required.issubset(metrics.columns):
        raise ValueError('Missing required model metric columns')
    # Check the full source rows for each requested battery, including unselected
    # models, so a missing model cannot silently produce a misleading rank.
    source = metrics.loc[metrics['battery_id'].isin(batteries), list(required)].copy(deep=True)
    for battery in batteries:
        subset = source.loc[source.battery_id == battery]
        if len(subset) != len(MODELS) or set(subset.model) != set(MODELS):
            raise ValueError(f'Incomplete six-model benchmark for {battery}')
        if not (subset.benchmark_type == BENCHMARK_TYPES[battery]).all():
            raise ValueError(f'Unexpected benchmark protocol for {battery}')
    values = pd.to_numeric(source[metric], errors='coerce')
    if not values.map(lambda x: math.isfinite(x) and x >= 0).all():
        raise ValueError('Nonfinite or negative benchmark metric')
    source[metric] = values.astype(float)
    source['rank'] = source.groupby('battery_id')[metric].rank(method='min', ascending=True).astype(int)
    return source.loc[source.model.isin(models)].copy(deep=True)


def grouped_metric_chart(metrics: pd.DataFrame, batteries: Sequence[str], models: Sequence[str], metric: str = 'rmse') -> go.Figure:
    """Grouped bars: model vs metric, with benchmark protocol in each trace."""
    frame = _validated(metrics, batteries, models, metric)
    fig = go.Figure()
    for battery in batteries:
        rows = frame.loc[frame.battery_id == battery].set_index('model').reindex(models)
        fig.add_trace(go.Bar(
            name=LABELS[battery], x=list(models), y=rows[metric].tolist(),
            customdata=rows['rank'].tolist(),
            hovertemplate='<b>%{x}</b><br>' + LABELS[battery] + '<br>' + METRIC_LABELS[metric] + ': %{y:.6f}<br>Within-battery rank: %{customdata}<extra></extra>',
        ))
    fig.update_layout(barmode='group', xaxis_title='Model', yaxis_title=METRIC_LABELS[metric],
                      legend_title='Benchmark protocol', margin=dict(l=20, r=20, t=35, b=30),
                      hovermode='closest')
    fig.update_yaxes(rangemode='tozero')
    return fig


def ranking_heatmap(metrics: pd.DataFrame, batteries: Sequence[str], models: Sequence[str], metric: str = 'rmse') -> go.Figure:
    """Rank within each *complete* six-model benchmark; 1 is best."""
    frame = _validated(metrics, batteries, models, metric)
    z = []
    values = []
    for battery in batteries:
        rows = frame.loc[frame.battery_id == battery].set_index('model').reindex(models)
        z.append(rows['rank'].tolist())
        values.append(rows[metric].tolist())
    fig = go.Figure(go.Heatmap(
        x=list(models), y=[LABELS[b] for b in batteries], z=z,
        zmin=1, zmax=6, colorscale='Blues_r',
        text=[[str(v) for v in row] for row in z], texttemplate='%{text}',
        customdata=values,
        hovertemplate='%{y}<br>%{x}<br>Rank: %{z} / 6<br>' + METRIC_LABELS[metric] + ': %{customdata:.6f}<extra></extra>',
        colorbar=dict(title='Rank', tickvals=[1, 2, 3, 4, 5, 6]),
    ))
    fig.update_layout(xaxis_title='Model', yaxis_title='Benchmark',
                      margin=dict(l=20, r=20, t=30, b=30))
    fig.update_yaxes(autorange='reversed')
    return fig


def battery_profile_chart(metrics: pd.DataFrame, battery: str, models: Sequence[str], metric: str = 'rmse') -> go.Figure:
    """Single-battery ordered performance profile, avoiding cross-protocol scaling."""
    frame = _validated(metrics, [battery], models, metric).sort_values([metric, 'model'], kind='stable')
    fig = go.Figure(go.Bar(
        x=frame[metric].tolist(), y=frame.model.tolist(), orientation='h',
        customdata=frame['rank'].tolist(),
        hovertemplate='<b>%{y}</b><br>' + METRIC_LABELS[metric] + ': %{x:.6f}<br>Within-battery rank: %{customdata}<extra></extra>',
    ))
    fig.update_layout(xaxis_title=METRIC_LABELS[metric], yaxis_title='Model',
                      title=LABELS[battery], margin=dict(l=20, r=20, t=45, b=25))
    fig.update_yaxes(autorange='reversed')
    fig.update_xaxes(rangemode='tozero')
    return fig
