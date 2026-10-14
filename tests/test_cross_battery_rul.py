"""Independent contract tests for battery-specific RUL scenarios."""
import numpy as np
import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_rul import (
    BATTERIES, MODELS, STATUS, analyze_battery_rul,
)


def fixtures():
    obs = pd.DataFrame({'date': ['2025-03-01'], 'measured_microvolt': [30.]})
    fut = pd.DataFrame({'date': ['2025-06-01', '2025-09-01', '2025-12-01']})
    for i, model in enumerate(MODELS):
        fut[model] = [31. + i / 10, 35. + i / 10, 36. + i / 10]
    return obs, fut


@pytest.mark.parametrize('battery', BATTERIES)
def test_first_crossing_and_spread(battery):
    obs, fut = fixtures()
    result = analyze_battery_rul(battery_id=battery, observed=obs, forecasts=fut, threshold_microvolt=35)
    assert len(result.model_crossings) == 6
    assert set(result.model_crossings.rul_quarters) == {2}
    assert set(result.model_crossings.crossing_date) == {'2025-09-01'}
    assert result.dispersion.iloc[0].spread_microvolt == pytest.approx(.5)
    assert result.dispersion.iloc[0].std_microvolt_population == pytest.approx(np.std(np.arange(6)/10))
    assert result.forecast_status == STATUS


def test_no_crossing_is_censored():
    obs, fut = fixtures()
    result = analyze_battery_rul(battery_id=BATTERIES[0], observed=obs, forecasts=fut, threshold_microvolt=100)
    assert result.model_crossings.rul_years.isna().all()
    assert set(result.model_crossings.status) == {'not_crossed_within_horizon'}


def test_already_crossed():
    obs, fut = fixtures()
    result = analyze_battery_rul(battery_id=BATTERIES[0], observed=obs, forecasts=fut, threshold_microvolt=29)
    assert set(result.model_crossings.rul_quarters) == {0}
    assert set(result.model_crossings.status) == {'already_crossed_observed'}


def test_below_direction():
    obs, fut = fixtures()
    result = analyze_battery_rul(battery_id=BATTERIES[1], observed=obs, forecasts=fut,
                                 threshold_microvolt=29, crossing_direction='below')
    assert set(result.model_crossings.status) == {'not_crossed_within_horizon'}


@pytest.mark.parametrize('change', ['bad_dates', 'missing_model', 'nan', 'bad_direction', 'bad_threshold'])
def test_invalid_inputs(change):
    obs, fut = fixtures()
    kwargs = {'battery_id': BATTERIES[0], 'observed': obs, 'forecasts': fut,
              'threshold_microvolt': 35.}
    if change == 'bad_dates':
        fut.loc[0, 'date'] = '2025-07-01'
    elif change == 'missing_model':
        fut.drop(columns=['GRU'], inplace=True)
    elif change == 'nan':
        fut.loc[0, 'GRU'] = np.nan
    elif change == 'bad_direction':
        kwargs['crossing_direction'] = 'upward'
    else:
        kwargs['threshold_microvolt'] = float('nan')
    with pytest.raises(ValueError):
        analyze_battery_rul(**kwargs)


def test_no_input_mutation():
    obs, fut = fixtures()
    before_obs, before_fut = obs.copy(deep=True), fut.copy(deep=True)
    analyze_battery_rul(battery_id=BATTERIES[0], observed=obs, forecasts=fut, threshold_microvolt=35)
    pd.testing.assert_frame_equal(obs, before_obs)
    pd.testing.assert_frame_equal(fut, before_fut)
