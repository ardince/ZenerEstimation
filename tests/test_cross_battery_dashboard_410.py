"""Dataset 410 operational dashboard evidence: strictly read-only."""
import json

import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_dashboard_sources import (
    MODELS, load_410_observed_history, load_410_operational_forecasts,
)
from zenerestimation.visualization.cross_battery_dashboard_evidence import make_410_future_chart


def _fixtures(tmp_path):
    dataset = tmp_path / 'datasets' / 'processed'
    dataset.mkdir(parents=True)
    pd.DataFrame({'ds': ['2025-03-01', '2025-06-01'],
                  'microVolt': [31.932, 32.0], 'is_observed': ['True', 'True']}).to_csv(
                      dataset / '732B-5610410.csv', index=False)
    folder = tmp_path / 'results' / '732B-5610410' / 'operational_extrapolation'
    folder.mkdir(parents=True)
    dates = ['2025-09-01', '2025-12-01']
    predictions = {m: [32.1234567890123, 32.2234567890123] for m in MODELS}
    pd.DataFrame({'date': dates, **{m: [32.123456789, 32.223456789] for m in MODELS}}).to_csv(
        folder / 'future_extrapolation.csv', index=False)
    metadata = {'kind': 'operational_extrapolation_not_holdout_evaluation',
                'battery': '732B-5610410',
                'scientific_status': 'unvalidated_future_operational_extrapolation',
                'configuration_source': 'frozen_410_optimized_benchmark',
                'model_update_policy': 'refit_through_training_cutoff_without_reoptimization',
                'measured_actuals': None, 'forecast_errors': None,
                'evaluation_metrics': None, 'uncertainty_intervals': None,
                'models': list(MODELS), 'forecast_dates': dates,
                'forecast_start': dates[0], 'forecast_end': dates[-1],
                'horizon_quarters': 2, 'prediction_count': 12,
                'training_cutoff': '2025-06-01', 'predictions': predictions}
    (folder / 'future_extrapolation.json').write_text(json.dumps(metadata), encoding='utf-8')
    return folder


def test_410_observed_history(tmp_path):
    _fixtures(tmp_path)
    result = load_410_observed_history(tmp_path)
    assert result.measured_microvolt.tolist() == [31.932, 32.0]


def test_410_json_precision_retained(tmp_path):
    _fixtures(tmp_path)
    result = load_410_operational_forecasts(tmp_path)
    assert result.ARIMA.iloc[0] == 32.1234567890123


def test_410_csv_json_mismatch_rejected(tmp_path):
    folder = _fixtures(tmp_path)
    path = folder / 'future_extrapolation.csv'
    csv = pd.read_csv(path)
    csv.loc[0, 'ARIMA'] = 99.0
    csv.to_csv(path, index=False)
    with pytest.raises(ValueError, match='disagreement'):
        load_410_operational_forecasts(tmp_path)


def test_410_false_evaluation_rejected(tmp_path):
    folder = _fixtures(tmp_path)
    path = folder / 'future_extrapolation.json'
    metadata = json.loads(path.read_text())
    metadata['evaluation_metrics'] = {'rmse': 0.1}
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='provenance'):
        load_410_operational_forecasts(tmp_path)


def test_410_future_chart_anchor_and_input_unchanged(tmp_path):
    _fixtures(tmp_path)
    observed = load_410_observed_history(tmp_path)
    forecasts = load_410_operational_forecasts(tmp_path)
    original = forecasts.copy(deep=True)
    fig = make_410_future_chart(forecasts, observed, ['ARIMA'])
    assert [t.name for t in fig.data] == ['Measured (observed)', 'ARIMA · unvalidated']
    assert list(fig.data[1].y) == [32.0, 32.1234567890123, 32.2234567890123]
    pd.testing.assert_frame_equal(forecasts, original)
