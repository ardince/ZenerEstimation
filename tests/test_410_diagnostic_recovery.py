"""Unit tests for Sprint 15.3.6.5; no model fitting."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from examples.optimization import run_410_diagnostic_recovery as recovery


def test_fitted_alignment_and_warmup():
    dates = pd.date_range('2023-03-01', periods=4, freq='QS-MAR')
    forecasts = {name: SimpleNamespace(model=name, fitted=pd.Series([np.nan, 1., 2., 3.], index=dates))
                 for name in recovery.MODELS}
    result = recovery.extract_fitted(forecasts, dates)
    assert list(result.columns) == ['date', *recovery.MODELS]
    assert pd.isna(result['LSTM'].iloc[0])


def test_fitted_rejects_internal_nan():
    dates = pd.date_range('2023-03-01', periods=4, freq='QS-MAR')
    forecasts = {name: SimpleNamespace(model=name, fitted=np.array([np.nan, 1., 2., 3.]))
                 for name in recovery.MODELS}
    forecasts['GRU'].fitted = np.array([np.nan, 1., np.nan, 3.])
    with pytest.raises(ValueError, match='GRU'):
        recovery.extract_fitted(forecasts, dates)


def test_reconcile_exact_and_mismatch():
    dates = pd.date_range('2024-03-01', periods=6, freq='QS-MAR')
    actual = np.array([30., 31., 32., 33., 34., 35.])
    predicted = actual + 0.5
    evaluations = {name: SimpleNamespace(model=name, dates=dates, actual=actual, predicted=predicted)
                   for name in recovery.MODELS}
    forecasts = {name: SimpleNamespace(model=name, dates=dates, forecast=predicted)
                 for name in recovery.MODELS}
    metrics = pd.DataFrame({'rmse': .5, 'mae': .5, 'mape': np.mean(.5 / actual) * 100},
                           index=recovery.MODELS)
    table, report = recovery.reconcile(evaluations, forecasts, dates, actual, metrics)
    assert len(table) == 6
    assert len(report) == 18
    assert report['within_1e_6'].all()
    metrics.loc['GRU', 'rmse'] = .7
    _, report = recovery.reconcile(evaluations, forecasts, dates, actual, metrics)
    assert not report.loc[(report.model == 'GRU') & (report.metric == 'rmse'), 'within_1e_6'].item()


def test_reconcile_rejects_forecast_date_shift():
    dates = pd.date_range('2024-03-01', periods=6, freq='QS-MAR')
    actual = np.ones(6)
    evaluations = {name: SimpleNamespace(model=name, dates=dates, actual=actual, predicted=actual)
                   for name in recovery.MODELS}
    forecasts = {name: SimpleNamespace(model=name, dates=dates, forecast=actual)
                 for name in recovery.MODELS}
    forecasts['ARIMA'].dates = dates + pd.offsets.QuarterBegin(startingMonth=3)
    metrics = pd.DataFrame(0., index=recovery.MODELS, columns=recovery.METRICS)
    with pytest.raises(ValueError, match='ARIMA'):
        recovery.reconcile(evaluations, forecasts, dates, actual, metrics)


def test_rejects_output_overlap_before_execution(tmp_path):
    benchmark = tmp_path / 'optimized_benchmark'
    with pytest.raises(ValueError, match='overlap'):
        recovery.run(benchmark_dir=benchmark, output_dir=benchmark / 'diagnostics')
