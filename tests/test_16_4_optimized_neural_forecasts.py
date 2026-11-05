"""Lightweight contract tests; no TensorFlow training required."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'examples/optimization/run_16_4_optimized_neural_forecasts.py'
spec = importlib.util.spec_from_file_location('optimized_neural_offline', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_quarterly_boundaries():
    assert runner.expected_dates(runner.DEV_END) == [
        '2024-03-01', '2024-06-01', '2024-09-01', '2024-12-01', '2025-03-01', '2025-06-01']
    assert runner.expected_dates(runner.OP_END) == [
        '2025-09-01', '2025-12-01', '2026-03-01', '2026-06-01', '2026-09-01', '2026-12-01']


def test_metrics_and_sign_convention():
    a = np.array([1., 2., 3., 4., 5., 6.])
    b = a + 0.5
    metrics = runner.evaluate(a, b)
    assert metrics['rmse'] == pytest.approx(.5)
    assert metrics['mae'] == pytest.approx(.5)
    assert metrics['first_step_signed_error_pred_minus_actual'] == pytest.approx(.5)


def test_alignment_rejects_shifted_dates():
    class Model:
        def fit(self, dataset): pass
        def predict(self, steps):
            return type('Result', (), {'dates': pd.date_range('2024-06-01', periods=6, freq='3MS'),
                                        'horizon': 6, 'forecast': np.arange(6)})()
    with pytest.raises(ValueError, match='alignment'):
        runner.forecast_one(Model(), None, runner.DEV_END)


def test_refuses_overwrite_before_training(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    for model in runner.MODELS:
        a = {'model': model, 'schema_version': 'neural-training-artifact-v1',
             'scope': 'internal_temporal_neural_training_only', 'selection_metric': 'rmse',
             'best_params': {'window': 4, 'units': 16, 'epochs': 24, 'batch_size': 8},
             'best_score': 1.0}
        b = {'model': model, 'schema_version': 'neural-candidate-report-v1',
             'best_params': a['best_params'], 'best_score': 1.0, 'selection_metric': 'rmse'}
        (source / f'{model}_neural_training.json').write_text(json.dumps(a))
        (source / f'{model}_learning_report.json').write_text(json.dumps(b))
    dates = pd.date_range('1998-03-01', '2025-06-01', freq='3MS')
    data = tmp_path / 'dataset.csv'
    pd.DataFrame({'ds': dates, 'microVolt': np.arange(len(dates), dtype=float),
                  'is_observed': 'True'}).to_csv(data, index=False)
    output = tmp_path / 'existing'
    output.mkdir()
    with pytest.raises(FileExistsError):
        runner.run(source, data, output, seed=42, dry_run=True)
    runner.run(source, data, tmp_path / 'unused', seed=42, dry_run=True)
    assert not (tmp_path / 'unused').exists()
