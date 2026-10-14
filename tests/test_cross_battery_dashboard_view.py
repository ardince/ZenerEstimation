"""Sprint 15.3.1 presentation contract tests; no Streamlit dependency."""
import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_comparison import CrossBatteryComparison
from zenerestimation.visualization.cross_battery_dashboard_view import dashboard_view


def fixture_comparison():
    summary = pd.DataFrame([
        dict(battery_id='732B-5610410', benchmark_type='optimized', holdout_steps=6),
        dict(battery_id='732B-5610110', benchmark_type='transferred', holdout_steps=5),
    ])
    metrics = pd.DataFrame([
        dict(battery_id=b, benchmark_type=t, model=m, rmse=v, mae=v/2, mape=v/3)
        for b, t, m, v in [
            ('732B-5610410', 'optimized', 'ARIMA', 0.8),
            ('732B-5610410', 'optimized', 'KalmanLSTM', 0.6),
            ('732B-5610110', 'transferred', 'LSTM', 0.2),
        ]
    ])
    evidence = pd.DataFrame([
        dict(battery_id='732B-5610410', evidence_type='historical_fits', structured=False),
        dict(battery_id='732B-5610110', evidence_type='historical_fits', structured=True),
    ])
    return CrossBatteryComparison(summary, metrics, evidence, pd.DataFrame(), pd.DataFrame())


def test_all_batteries_retains_protocol_labels():
    result = dashboard_view(fixture_comparison(), ['732B-5610410', '732B-5610110'])
    assert set(result.summary.benchmark_type) == {'optimized', 'transferred'}
    assert len(result.metrics) == 3
    assert set(result.evidence.structured) == {True, False}


def test_filter_does_not_invent_other_battery():
    result = dashboard_view(fixture_comparison(), ['732B-5610410'])
    assert set(result.metrics.battery_id) == {'732B-5610410'}
    assert len(result.evidence) == 1


def test_metric_switch_and_sort():
    result = dashboard_view(fixture_comparison(), ['732B-5610410'], 'mae')
    assert list(result.metrics.columns) == ['battery_id', 'benchmark_type', 'model', 'mae']
    assert list(result.metrics.model) == ['KalmanLSTM', 'ARIMA']


def test_no_input_mutation():
    comparison = fixture_comparison()
    before = comparison.model_metrics.copy(deep=True)
    result = dashboard_view(comparison, ['732B-5610410'])
    result.metrics.loc[:, 'mae'] = 99
    pd.testing.assert_frame_equal(comparison.model_metrics, before)


@pytest.mark.parametrize('batteries,metric', [([], 'rmse'), (['unknown'], 'rmse'),
    (['732B-5610410', '732B-5610410'], 'rmse'), (['732B-5610410'], 'unknown')])
def test_invalid_selection_rejected(batteries, metric):
    with pytest.raises(ValueError):
        dashboard_view(fixture_comparison(), batteries, metric)
