"""Focused read-only display checks against representative saved artifacts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zenerestimation.visualization.optimized_neural_comparison import (
    MODELS, HOLDOUT, FUTURE, load_optimized_neural_forecasts,
    holdout_values_table, operational_values_table,
    make_holdout_figure, make_operational_figure,
)


def _fixture(tmp_path):
    folder = tmp_path / 'results' / '732B-5610410' / 'optimized_neural_forecasts_16_4' / 'test_run'
    folder.mkdir(parents=True)
    holdout = pd.DataFrame([
        {'date': str(date.date()), 'model': model, 'actual_microvolt': 30. + i,
         'predicted_microvolt': 30. + i + j * .1,
         'error_pred_minus_actual': j * .1}
        for j, model in enumerate(MODELS) for i, date in enumerate(HOLDOUT)
    ])
    future = pd.DataFrame([
        {'date': str(date.date()), 'model': model, 'predicted_microvolt': 40. + i + j * .1}
        for j, model in enumerate(MODELS) for i, date in enumerate(FUTURE)
    ])
    manifest = {
        'schema_version': 'optimized-neural-forecasts-16-4-v1',
        'battery': '732B-5610410', 'seed': 42,
        'development_cutoff': '2023-12-01', 'operational_cutoff': '2025-06-01',
        'horizon': 6, 'holdout_dates': HOLDOUT.strftime('%Y-%m-%d').tolist(),
        'operational_dates': FUTURE.strftime('%Y-%m-%d').tolist(),
        'evaluation_status': 'post_optimization_external_holdout_evaluation',
        'operational_status': 'unvalidated_future_extrapolation_no_actuals_or_accuracy_metrics',
        'files': {'holdout': 'holdout_predictions.csv', 'operational': 'operational_predictions.csv'},
        'selected': {m: {'internal_rmse': .3} for m in MODELS},
        'external_holdout_metrics': {m: {
            'rmse': float(j*.1), 'mae': float(j*.1),
            'first_step_signed_error_pred_minus_actual': float(j*.1),
            'first_step_absolute_error': float(j*.1),
        } for j,m in enumerate(MODELS)},
    }
    holdout.to_csv(folder / 'holdout_predictions.csv', index=False)
    future.to_csv(folder / 'operational_predictions.csv', index=False)
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    return tmp_path


def test_holdout_values_and_chart(tmp_path):
    root = _fixture(tmp_path)
    _, holdout, _ = load_optimized_neural_forecasts(root, 'test_run')
    table = holdout_values_table(holdout, ['LSTM', 'GRU'])
    assert len(table) == 12
    assert 'Measured (µV)' in table
    assert 'Prediction − measured (µV)' in table
    fig = make_holdout_figure(holdout, ['LSTM', 'GRU'])
    assert len(fig.data) == 3  # measured + two model curves


def test_future_values_and_saved_alignment(tmp_path):
    root = _fixture(tmp_path)
    _, _, future = load_optimized_neural_forecasts(root, 'test_run')
    saved = pd.DataFrame({'date': FUTURE, 'LSTM': np.arange(6.) + 39.})
    table = operational_values_table(future, ['LSTM'], saved)
    assert len(table) == 6
    assert np.allclose(table['New − saved (µV)'], 1.)
    fig = make_operational_figure(future, ['LSTM'], saved)
    assert len(fig.data) == 2
    assert fig.data[1].line.dash == 'dash'


def test_misaligned_saved_dates_rejected(tmp_path):
    root = _fixture(tmp_path)
    _, _, future = load_optimized_neural_forecasts(root, 'test_run')
    saved = pd.DataFrame({'date': FUTURE + pd.DateOffset(months=3), 'LSTM': np.arange(6.)})
    with pytest.raises(ValueError, match='dates'):
        operational_values_table(future, ['LSTM'], saved)


def test_new_forecasts_remain_visible_without_saved_comparison(tmp_path):
    root = _fixture(tmp_path)
    _, _, future = load_optimized_neural_forecasts(root, 'test_run')
    table = operational_values_table(future, list(MODELS))
    assert len(table) == 24
    assert 'Saved prediction (µV)' not in table
    assert len(make_operational_figure(future, list(MODELS)).data) == 4
