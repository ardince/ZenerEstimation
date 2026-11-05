import pandas as pd
import pytest

from zenerestimation.visualization.historical_fit_evidence import (
    build_historical_fit_evidence,
    common_interval_metrics,
)


def artifact(name, observed, fitted):
    dates = pd.date_range('2023-03-01', periods=len(observed), freq='QS-MAR')
    return build_historical_fit_evidence(
        name, pd.Series(observed, index=dates), pd.Series(fitted, index=dates),
        cutoff='2023-12-01',
    )


def test_unavailable_model_does_not_erase_shared_dates():
    a = artifact('LSTM', [1, 2, None, 4], [None, 2.5, 3, 4.5])
    b = artifact('GRU', [1, 2, None, 4], [1.2, 2.2, 3, 3.8])
    missing = artifact('Kalman', [1, 2, None, 4], [None] * 4)
    result = common_interval_metrics([a, missing, b])
    assert result['dates'] == ['2023-06-01', '2023-12-01']
    assert result['included_models'] == ['LSTM', 'GRU']
    assert result['excluded_models'] == {'Kalman': 'no_measured_dates_with_fitted_predictions'}
    assert result['models']['LSTM']['n'] == 2
    assert result['models']['GRU']['rmse'] == pytest.approx(.2)


def test_all_unavailable():
    result = common_interval_metrics([artifact('Kalman', [1, 2], [None, None])])
    assert result['dates'] == []
    assert result['included_models'] == []
    assert result['common_interval_status'] == 'no_shared_measured_dates'


def test_duplicate_model_rejected():
    a = artifact('LSTM', [1, 2], [1, 2])
    with pytest.raises(ValueError, match='Duplicate'):
        common_interval_metrics([a, a])
