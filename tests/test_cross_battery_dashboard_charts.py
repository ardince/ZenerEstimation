"""Sprint 15.3.2: Plotly chart contracts, independent of Streamlit and real artifacts."""
import copy

import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_artifacts import MODELS
from zenerestimation.visualization.cross_battery_dashboard_charts import (
    grouped_metric_chart, ranking_heatmap, battery_profile_chart,
)

B410 = '732B-5610410'
B110 = '732B-5610110'


def metrics_fixture():
    values = {
        B410: [0.85, 0.94, 1.31, 1.40, 0.79, 0.67],
        B110: [0.31, 1.79, 0.26, 0.27, 1.55, 5.19],
    }
    return pd.DataFrame([
        dict(battery_id=b, benchmark_type='optimized' if b == B410 else 'transferred',
             model=m, rmse=v, mae=v / 2, mape=v / 3)
        for b, vals in values.items() for m, v in zip(MODELS, vals)
    ])


def test_grouped_chart_contains_two_protocols_and_six_models():
    fig = grouped_metric_chart(metrics_fixture(), [B410, B110], MODELS)
    assert len(fig.data) == 2
    assert list(fig.data[0].x) == list(MODELS)
    assert fig.data[0].name == '410 · optimized'
    assert fig.data[1].name == '110 · transferred'


def test_rank_reversal_in_heatmap():
    fig = ranking_heatmap(metrics_fixture(), [B410, B110], MODELS)
    ranks = fig.data[0].z
    assert ranks[0][-1] == 1  # KalmanLSTM, 410
    assert ranks[1][-1] == 6  # KalmanLSTM, 110
    assert ranks[1][2] == 1   # LSTM, 110


def test_hidden_models_do_not_change_rank():
    fig = ranking_heatmap(metrics_fixture(), [B410], ['ARIMA', 'KalmanLSTM'])
    assert list(fig.data[0].z[0]) == [3, 1]


def test_metric_switch_recomputes_rank():
    df = metrics_fixture()
    df.loc[(df.battery_id == B410) & (df.model == 'ARIMA'), 'mae'] = 0.01
    fig = ranking_heatmap(df, [B410], ['ARIMA'], 'mae')
    assert fig.data[0].z[0][0] == 1


def test_profile_orders_lowest_first():
    fig = battery_profile_chart(metrics_fixture(), B110, MODELS)
    assert fig.data[0].y[0] == 'LSTM'
    assert fig.data[0].y[-1] == 'KalmanLSTM'


def test_source_frame_unchanged():
    df = metrics_fixture()
    original = df.copy(deep=True)
    grouped_metric_chart(df, [B410, B110], MODELS, 'mape')
    ranking_heatmap(df, [B410], ['LSTM'])
    battery_profile_chart(df, B110, MODELS, 'mae')
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize('battery,models,metric', [
    (['unknown'], MODELS, 'rmse'),
    ([B410], ['unknown'], 'rmse'),
    ([B410], MODELS, 'unknown'),
    ([B410, B410], MODELS, 'rmse'),
    ([B410], [], 'rmse'),
])
def test_invalid_chart_selection(battery, models, metric):
    with pytest.raises(ValueError):
        grouped_metric_chart(metrics_fixture(), battery, models, metric)


def test_incomplete_benchmark_rejected():
    df = metrics_fixture()
    df = df.loc[~((df.battery_id == B410) & (df.model == 'ARIMA'))]
    with pytest.raises(ValueError, match='Incomplete'):
        ranking_heatmap(df, [B410], ['KalmanLSTM'])


def test_negative_or_nonfinite_metric_rejected():
    df = metrics_fixture()
    df.loc[0, 'rmse'] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        grouped_metric_chart(df, [B410], MODELS)
    df.loc[0, 'rmse'] = -1
    with pytest.raises(ValueError, match='Nonfinite'):
        grouped_metric_chart(df, [B410], MODELS)


def test_protocol_mismatch_rejected():
    df = metrics_fixture()
    df.loc[df.battery_id == B110, 'benchmark_type'] = 'optimized'
    with pytest.raises(ValueError, match='protocol'):
        ranking_heatmap(df, [B110], MODELS)
