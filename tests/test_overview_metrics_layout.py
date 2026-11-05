import pandas as pd
import pytest
from zenerestimation.visualization.overview_metrics_layout import overview_with_metrics


def test_saved_metrics_horizontal_and_unrounded():
    notes = pd.DataFrame({'Property': ['Battery'], 'Value': ['110']})
    metrics = pd.DataFrame({'model': ['LSTM', 'GRU'], 'rmse': [0.123456789, 0.5],
                            'mae': [0.2, 0.3], 'mape': [1.2, 1.3], 'rmse_rank': [1, 2]})
    out = overview_with_metrics(notes, metrics, optimized=False)
    assert out.loc[0, 'Model'] == 'LSTM'
    assert out.loc[0, 'Saved RMSE (µV)'] == 0.123456789
    assert out.loc[1, 'Model'] == 'GRU'
    assert pd.isna(out.loc[1, 'Property'])
    assert list(metrics.columns)[0] == 'model'


def test_optimized_metrics_horizontal_and_provenance():
    notes = pd.DataFrame({'Property': ['Scope'], 'Value': ['Different protocols']})
    metrics = pd.DataFrame({
        'Model': ['LSTM'], 'Benchmark external RMSE (µV)': [1.2],
        'Optimized external RMSE (µV)': [0.8],
        'Optimized − benchmark RMSE (µV)': [-0.4],
        'Benchmark external MAE (µV)': [0.9], 'Optimized external MAE (µV)': [0.6],
    })
    out = overview_with_metrics(notes, metrics, optimized=True)
    assert out.loc[0, 'Saved benchmark RMSE (µV)'] == 1.2
    assert out.loc[0, 'Optimized RMSE (µV)'] == 0.8
    assert out.loc[0, 'Optimized − saved RMSE (µV)'] == -0.4
    assert out.loc[0, 'Value'] == 'Different protocols'


def test_missing_metric_schema_rejected():
    notes = pd.DataFrame({'Property': ['Battery'], 'Value': ['410']})
    with pytest.raises(ValueError):
        overview_with_metrics(notes, pd.DataFrame({'model': ['LSTM']}), optimized=False)
