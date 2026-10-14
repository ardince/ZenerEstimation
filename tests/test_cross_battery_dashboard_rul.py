"""Sprint 15.4.5 pure-view regression tests (no Streamlit runtime required)."""
from dataclasses import replace
import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_rul import (
    BATTERIES, MODELS, STATUS, analyze_battery_rul,
)
from zenerestimation.visualization.cross_battery_dashboard_rul import (
    ILLUSTRATIVE_THRESHOLDS, build_rul_dashboard_view,
)


def make_result(battery=BATTERIES[0], threshold=35.0):
    observed = pd.DataFrame({'date': ['2025-03-01'], 'measured_microvolt': [30.0]})
    forecasts = pd.DataFrame({'date': ['2025-06-01', '2025-09-01']})
    for i, model in enumerate(MODELS):
        forecasts[model] = [31.0 + i, 32.0 + i]
    return analyze_battery_rul(battery_id=battery, observed=observed,
                               forecasts=forecasts, threshold_microvolt=threshold)


@pytest.mark.parametrize('battery', BATTERIES)
def test_view_counts_and_provenance(battery):
    view = build_rul_dashboard_view(make_result(battery))
    assert (view.crossing_count, view.censored_count, view.already_crossed_count) == (3, 3, 0)
    assert set(view.crossings.forecast_status) == {STATUS}
    assert set(view.dispersion.forecast_status) == {STATUS}
    assert len(view.dispersion) == 2


def test_nonmutation():
    result = make_result()
    crossings = result.model_crossings.copy(deep=True)
    dispersion = result.dispersion.copy(deep=True)
    build_rul_dashboard_view(result)
    pd.testing.assert_frame_equal(result.model_crossings, crossings)
    pd.testing.assert_frame_equal(result.dispersion, dispersion)


def test_threshold_defaults_are_battery_specific():
    assert ILLUSTRATIVE_THRESHOLDS == {BATTERIES[0]: 35.0, BATTERIES[1]: 175.0}


@pytest.mark.parametrize('defect', ['origin', 'model', 'status', 'horizon', 'spread', 'censor'])
def test_invalid_contract_rejected(defect):
    result = make_result()
    crossings, dispersion = result.model_crossings.copy(), result.dispersion.copy()
    if defect == 'origin':
        dispersion.loc[0, 'forecast_status'] = 'frozen_benchmark'
    elif defect == 'model':
        crossings.loc[0, 'model'] = 'GRU'
    elif defect == 'status':
        crossings.loc[0, 'status'] = 'validated_failure'
    elif defect == 'horizon':
        crossings.loc[0, 'horizon_quarters'] = 5
    elif defect == 'spread':
        dispersion.loc[0, 'spread_microvolt'] = -1
    else:
        idx = crossings.index[crossings.status.eq('not_crossed_within_horizon')][0]
        crossings.loc[idx, 'rul_years'] = 100
    with pytest.raises(ValueError):
        build_rul_dashboard_view(replace(result, model_crossings=crossings, dispersion=dispersion))
