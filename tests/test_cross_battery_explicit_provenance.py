"""Sprint 15.3.6: explicit, fail-closed numerical evidence provenance.

Pure adapter tests: no model execution, repository mutation, or Streamlit runtime.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_dashboard_numerical import numerical_table

B410 = '732B-5610410'
B110 = '732B-5610110'
REGENERATED = 'regenerated_diagnostic_not_frozen'
FROZEN_TRANSFER = 'frozen_transferred_benchmark'


def _comparison():
    holdout = pd.DataFrame([
        dict(battery_id=b, date='2024-03-01', model=m,
             measured_microvolt=30.0, predicted_microvolt=30.5,
             origin=origin)
        for b, origin in ((B410, REGENERATED), (B110, FROZEN_TRANSFER))
        for m in ('ARIMA', 'LSTM')
    ])
    fits = pd.DataFrame([
        dict(battery_id=b, date='2023-12-01', model=m,
             fitted_microvolt=29.8, origin=REGENERATED)
        for b in (B410, B110) for m in ('ARIMA', 'LSTM')
    ])
    metrics = pd.DataFrame([
        dict(battery_id=b, model='ARIMA', rmse=.5, mae=.4,
             mape=1.2, rmse_rank=1)
        for b in (B410, B110)
    ])
    return SimpleNamespace(holdout_predictions=holdout,
                           historical_fits=fits, model_metrics=metrics)


@pytest.mark.parametrize('battery,kind,expected_origin,expected_status', [
    (B410, 'holdout', REGENERATED, 'NOT original frozen'),
    (B110, 'holdout', FROZEN_TRANSFER, 'Frozen transferred'),
    (B410, 'historical_fits', REGENERATED, 'not frozen'),
    (B110, 'historical_fits', REGENERATED, 'not frozen'),
])
def test_expected_provenance_is_preserved(battery, kind, expected_origin, expected_status):
    result = numerical_table(_comparison(), battery_id=battery, kind=kind)
    assert result.available
    assert result.data['origin'].eq(expected_origin).all()
    assert expected_status in result.status


@pytest.mark.parametrize('battery,kind,bad_origin', [
    (B410, 'holdout', FROZEN_TRANSFER),
    (B410, 'holdout', 'frozen_optimized_benchmark'),
    (B110, 'holdout', REGENERATED),
    (B110, 'holdout', 'frozen_optimized_benchmark'),
    (B410, 'historical_fits', FROZEN_TRANSFER),
    (B110, 'historical_fits', FROZEN_TRANSFER),
    (B410, 'holdout', None),
    (B110, 'holdout', ''),
    (B410, 'historical_fits', None),
    (B110, 'historical_fits', ''),
])
def test_invalid_or_missing_origin_is_rejected(battery, kind, bad_origin):
    comparison = _comparison()
    source = comparison.holdout_predictions if kind == 'holdout' else comparison.historical_fits
    source.loc[source['battery_id'].eq(battery), 'origin'] = bad_origin
    with pytest.raises(ValueError, match='provenance'):
        numerical_table(comparison, battery_id=battery, kind=kind)


@pytest.mark.parametrize('battery,kind', [
    (B410, 'holdout'), (B110, 'holdout'),
    (B410, 'historical_fits'), (B110, 'historical_fits'),
])
def test_mixed_provenance_is_rejected(battery, kind):
    comparison = _comparison()
    source = comparison.holdout_predictions if kind == 'holdout' else comparison.historical_fits
    matching = source.index[source['battery_id'].eq(battery)]
    source.loc[matching[0], 'origin'] = 'unexpected_origin'
    with pytest.raises(ValueError, match='provenance'):
        numerical_table(comparison, battery_id=battery, kind=kind)


@pytest.mark.parametrize('battery', [B410, B110])
def test_derived_holdout_errors_do_not_mutate_source(battery):
    comparison = _comparison()
    original = comparison.holdout_predictions.copy(deep=True)
    result = numerical_table(comparison, battery_id=battery, kind='holdout')
    assert result.data['error_microvolt'].eq(.5).all()
    assert result.data['absolute_error_microvolt'].eq(.5).all()
    pd.testing.assert_frame_equal(comparison.holdout_predictions, original)
