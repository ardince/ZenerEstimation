import pandas as pd
import pytest

from zenerestimation.visualization.historical_fit_evidence import (
    build_historical_fit_evidence, common_interval_metrics,
)


def test_warmup_and_metrics():
    idx = pd.date_range('2023-03-01', periods=4, freq='QS-MAR')
    actual = pd.Series([1., 2., 3., 4.], index=idx)
    fitted = pd.Series([float('nan'), 2.5, 2.5, 4.], index=idx)
    result = build_historical_fit_evidence('LSTM', actual, fitted, cutoff='2023-12-01')
    assert result['coverage']['valid_fitted_observations'] == 3
    assert result['rows'][0]['fitted_microVolt'] is None
    assert result['metrics']['mae'] == pytest.approx(1 / 3)


def test_shifted_dates_rejected():
    idx = pd.date_range('2023-03-01', periods=4, freq='QS-MAR')
    with pytest.raises(ValueError, match='timestamps'):
        build_historical_fit_evidence('LSTM', pd.Series([1]*4, index=idx),
            pd.Series([1]*4, index=idx + pd.DateOffset(months=3)), cutoff='2023-12-01')


def test_external_period_rejected():
    idx = pd.date_range('2023-03-01', periods=4, freq='QS-MAR')
    with pytest.raises(ValueError, match='cutoff'):
        build_historical_fit_evidence('LSTM', pd.Series([1]*4, index=idx),
            pd.Series([1]*4, index=idx), cutoff='2023-09-01')


def test_common_interval():
    idx = pd.date_range('2023-03-01', periods=4, freq='QS-MAR')
    actual = pd.Series([1., 2., 3., 4.], index=idx)
    a = build_historical_fit_evidence('LSTM', actual, pd.Series([None, 2., 3., 4.], index=idx), cutoff='2023-12-01')
    b = build_historical_fit_evidence('GRU', actual, pd.Series([None, None, 3., 4.], index=idx), cutoff='2023-12-01')
    comparison = common_interval_metrics([a, b])
    assert len(comparison['dates']) == 2
    assert comparison['models']['LSTM']['n'] == 2
