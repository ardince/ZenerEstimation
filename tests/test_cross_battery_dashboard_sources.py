import json

import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_dashboard_sources import (
    MODELS, load_110_observed_history, load_110_operational_forecasts,
)
from zenerestimation.visualization.cross_battery_dashboard_evidence import (
    make_110_future_chart, make_110_history_chart, BATTERY_110, DIAGNOSTIC_ORIGIN,
)


def _files(tmp_path):
    dataset = tmp_path / 'datasets' / 'processed'
    dataset.mkdir(parents=True)
    pd.DataFrame({'ds':['2024-09-01','2024-12-01','2025-03-01'],
                  'microVolt':[None,None,171.6],
                  'is_observed':['False','False','True']}).to_csv(dataset/'732B-5610110.csv',index=False)
    folder = tmp_path/'results'/'732B-5610110'/'operational_extrapolation'
    folder.mkdir(parents=True)
    dates = ['2025-06-01','2025-09-01']
    predictions = {m:[172.0,173.0] for m in MODELS}
    pd.DataFrame({'date':dates,**predictions}).to_csv(folder/'future_extrapolation_full_precision.csv',index=False)
    meta = {'kind':'operational_extrapolation_not_holdout_evaluation',
            'battery':'732B-5610110',
            'scientific_status':'unvalidated_future_operational_extrapolation',
            'optimized_for_dataset_110':False, 'configuration_source_battery':'732B-5610410',
            'measured_actuals':None,'forecast_errors':None,'evaluation_metrics':None,
            'uncertainty_intervals':None, 'models':list(MODELS),
            'forecast_dates':dates,'forecast_start':dates[0],'forecast_end':dates[-1],
            'horizon_quarters':2,'prediction_count':12,'training_cutoff':'2025-03-01',
            'predictions':predictions}
    (folder/'future_extrapolation.json').write_text(json.dumps(meta),encoding='utf-8')
    return folder


def test_observed_mask(tmp_path):
    _files(tmp_path)
    actual = load_110_observed_history(tmp_path)
    assert actual['measured_microvolt'].tolist() == [171.6]
    assert actual['date'].dt.strftime('%Y-%m-%d').tolist() == ['2025-03-01']


def test_forecast_loader_validates_json_and_csv(tmp_path):
    _files(tmp_path)
    forecasts = load_110_operational_forecasts(tmp_path)
    assert len(forecasts) == 2
    assert forecasts['LSTM'].tolist() == [172.0,173.0]


def test_rejects_forecast_value_mismatch(tmp_path):
    folder = _files(tmp_path)
    csv = folder/'future_extrapolation_full_precision.csv'
    text = csv.read_text().replace('172.0','199.0',1)
    csv.write_text(text)
    with pytest.raises(ValueError,match='disagreement'):
        load_110_operational_forecasts(tmp_path)


def test_rejects_claimed_evaluation(tmp_path):
    folder = _files(tmp_path)
    path = folder/'future_extrapolation.json'
    meta = json.loads(path.read_text())
    meta['evaluation_metrics'] = {'rmse':0.0}
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='provenance'):
        load_110_operational_forecasts(tmp_path)


def test_future_chart_measured_anchor_not_claimed_future_actual(tmp_path):
    _files(tmp_path)
    observed = load_110_observed_history(tmp_path)
    forecasts = load_110_operational_forecasts(tmp_path)
    fig = make_110_future_chart(forecasts,observed,['LSTM'])
    assert [trace.name for trace in fig.data] == ['Measured (observed)','LSTM · unvalidated']
    assert list(fig.data[1].y) == [171.6,172.0,173.0]
    assert len(fig.data[0].y) == 1


def test_history_includes_actuals_and_does_not_mutate(tmp_path):
    _files(tmp_path)
    observed = load_110_observed_history(tmp_path)
    fitted = pd.DataFrame([{'battery_id':BATTERY_110,'date':'2022-12-01',
                            'model':'LSTM','origin':DIAGNOSTIC_ORIGIN,'fitted_microvolt':166.0}])
    original = fitted.copy(deep=True)
    fig = make_110_history_chart(fitted,['LSTM'],observed)
    assert [t.name for t in fig.data] == ['Measured (observed)','LSTM']
    assert list(fig.data[0].y) == [171.6]
    pd.testing.assert_frame_equal(fitted,original)
