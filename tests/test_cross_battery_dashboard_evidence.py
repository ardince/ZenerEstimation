from pathlib import Path

import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_dashboard_evidence import (
    BATTERY_110, FROZEN_ORIGIN, DIAGNOSTIC_ORIGIN,
    make_110_holdout_chart, make_110_history_chart, verified_410_visuals,
)


def _holdout():
    return pd.DataFrame([
        {'battery_id': BATTERY_110, 'date': date, 'model': model, 'origin': FROZEN_ORIGIN,
         'measured_microvolt': actual, 'predicted_microvolt': predicted}
        for date, actual in [('2023-03-01', 167.17), ('2023-06-01', 167.73)]
        for model, predicted in [('LSTM', actual + .1), ('ARIMA', actual + .2)]
    ])


def _history():
    return pd.DataFrame([
        {'battery_id': BATTERY_110, 'date': date, 'model': 'LSTM', 'origin': DIAGNOSTIC_ORIGIN,
         'fitted_microvolt': value}
        for date, value in [('2022-03-01', float('nan')), ('2022-06-01', 165.2)]
    ])


def test_holdout_contains_measured_and_selected_model():
    fig = make_110_holdout_chart(_holdout(), ['LSTM'])
    assert [trace.name for trace in fig.data] == ['Measured (observed)', 'LSTM']
    assert list(fig.data[0].y) == [167.17, 167.73]


def test_holdout_rejects_wrong_origin():
    frame = _holdout()
    frame.loc[0, 'origin'] = DIAGNOSTIC_ORIGIN
    with pytest.raises(ValueError, match='provenance'):
        make_110_holdout_chart(frame)


def test_holdout_rejects_inconsistent_observations():
    frame = _holdout()
    frame.loc[0, 'measured_microvolt'] = 999
    with pytest.raises(ValueError, match='Inconsistent'):
        make_110_holdout_chart(frame)


def test_history_keeps_warmup_missing():
    fig = make_110_history_chart(_history(), ['LSTM'])
    assert pd.isna(fig.data[0].y[0])
    assert fig.data[0].connectgaps is False


def test_history_rejects_frozen_origin():
    frame = _history()
    frame['origin'] = FROZEN_ORIGIN
    with pytest.raises(ValueError, match='provenance'):
        make_110_history_chart(frame)


def test_no_fabricated_410_visuals(tmp_path):
    assert verified_410_visuals(tmp_path) == {}
    folder = tmp_path / 'results' / '732B-5610410' / 'optimized_benchmark'
    folder.mkdir(parents=True)
    (folder / 'benchmark_history.png').write_bytes(b'png')
    assert list(verified_410_visuals(tmp_path)) == ['benchmark_history.png']


def test_input_is_unchanged():
    frame = _holdout()
    before = frame.copy(deep=True)
    make_110_holdout_chart(frame, ['ARIMA'])
    pd.testing.assert_frame_equal(frame, before)


def test_arima_initial_zero_masked_only_in_history_plot():
    frame = pd.DataFrame([
        {'battery_id': BATTERY_110, 'date': date, 'model': model,
         'origin': DIAGNOSTIC_ORIGIN, 'fitted_microvolt': value}
        for model, values in [('ARIMA', [0.0, 165.0]), ('Kalman', [0.0, 165.2])]
        for date, value in zip(['1998-03-01', '1998-06-01'], values)
    ])
    original = frame.copy(deep=True)
    fig = make_110_history_chart(frame, ['ARIMA', 'Kalman'])
    traces = {trace.name: trace for trace in fig.data}
    assert pd.isna(traces['ARIMA'].y[0])
    assert traces['ARIMA'].y[1] == 165.0
    assert traces['Kalman'].y[0] == 0.0
    assert traces['ARIMA'].connectgaps is False
    pd.testing.assert_frame_equal(frame, original)

def test_nonzero_first_arima_fit_is_not_masked():
    frame = pd.DataFrame([
        {'battery_id': BATTERY_110, 'date': '1998-03-01', 'model': 'ARIMA',
         'origin': DIAGNOSTIC_ORIGIN, 'fitted_microvolt': 69.03},
    ])
    fig = make_110_history_chart(frame, ['ARIMA'])
    assert fig.data[0].y[0] == 69.03
