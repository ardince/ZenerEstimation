"""G.5 transition diagnostics regression tests; all use stored synthetic evidence."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from examples.optimization import run_110_transition_diagnostics as tr


def fixture_evidence(tmp_path):
    data = tmp_path / 'data.csv'
    benchmark_dir = tmp_path / 'benchmark'
    operational_dir = tmp_path / 'operational'
    benchmark_dir.mkdir()
    operational_dir.mkdir()
    dates = pd.date_range('1998-03-01', '2025-03-01', freq='QS-MAR')
    measured = ~dates.isin(tr.UNOBSERVED)
    values = np.linspace(100, 171.6, len(dates))
    values[~measured] = np.nan
    for date, value in zip(tr.BENCHMARK_DATES, tr.MEASURED_HOLDOUT):
        values[dates == date] = value
    values[-1] = 171.6
    pd.DataFrame({'ds': dates.strftime('%Y-%m-%d'),
                  'microVolt': values,
                  'is_observed': measured}).to_csv(data, index=False)
    benchmark = pd.DataFrame({'date': tr.BENCHMARK_DATES.strftime('%Y-%m-%d'),
                              'measured_microVolt': tr.MEASURED_HOLDOUT})
    for i, name in enumerate(tr.MODELS):
        benchmark[name] = tr.MEASURED_HOLDOUT + (i + 1) * .1
    benchmark.to_csv(benchmark_dir / 'holdout_predictions_full_precision.csv', index=False)
    operational = {
        'battery': tr.BATTERY,
        'kind': 'operational_extrapolation_not_holdout_evaluation',
        'scientific_status': 'unvalidated_future_operational_extrapolation',
        'training_cutoff': '2025-03-01',
        'optimized_for_dataset_110': False,
        'models': list(tr.MODELS),
        'measured_actuals': None,
        'evaluation_metrics': None,
        'forecast_dates': list(tr.OPERATIONAL_DATES.strftime('%Y-%m-%d')),
        'predictions': {name: list(np.linspace(172 + i, 174 + i, 6))
                        for i, name in enumerate(tr.MODELS)},
    }
    (operational_dir / 'future_extrapolation.json').write_text(json.dumps(operational))
    return data, benchmark_dir, operational_dir


def test_load_evidence_frozen_contract(tmp_path):
    data, bench, op = fixture_evidence(tmp_path)
    history, benchmark, forecasts = tr.load_evidence(
        dataset_path=data, benchmark_dir=bench, operational_dir=op)
    assert len(history) == 109
    assert len(benchmark) == 5
    assert set(forecasts) == set(tr.MODELS)
    assert all(len(values) == 6 for values in forecasts.values())


def test_transition_creates_chart_without_models(tmp_path):
    data, bench, op = fixture_evidence(tmp_path)
    result = tr.run_transition(dataset_path=data, benchmark_dir=bench,
                               operational_dir=op, output_dir=tmp_path / 'out')
    assert result.is_file() and result.stat().st_size > 1000


def test_rounded_csv_is_not_accepted_as_benchmark_source(tmp_path):
    data, bench, op = fixture_evidence(tmp_path)
    original = bench / 'holdout_predictions_full_precision.csv'
    original.rename(bench / 'holdout_predictions.csv')
    with pytest.raises(FileNotFoundError, match='Full-precision'):
        tr.load_evidence(dataset_path=data, benchmark_dir=bench, operational_dir=op)


def test_benchmark_actuals_cannot_be_changed(tmp_path):
    data, bench, op = fixture_evidence(tmp_path)
    path = bench / 'holdout_predictions_full_precision.csv'
    frame = pd.read_csv(path)
    frame.loc[0, 'measured_microVolt'] += 1
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError, match='holdout values'):
        tr.load_evidence(dataset_path=data, benchmark_dir=bench, operational_dir=op)


def test_operational_dates_cannot_be_changed(tmp_path):
    data, bench, op = fixture_evidence(tmp_path)
    path = op / 'future_extrapolation.json'
    content = json.loads(path.read_text())
    content['forecast_dates'][0] = '2025-03-01'
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='operational: dates'):
        tr.load_evidence(dataset_path=data, benchmark_dir=bench, operational_dir=op)


def test_operational_status_must_be_unvalidated(tmp_path):
    data, bench, op = fixture_evidence(tmp_path)
    path = op / 'future_extrapolation.json'
    content = json.loads(path.read_text())
    content['evaluation_metrics'] = {'rmse': 0.01}
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='scientific provenance'):
        tr.load_evidence(dataset_path=data, benchmark_dir=bench, operational_dir=op)


def test_unobserved_quarters_cannot_become_measurements(tmp_path):
    data, bench, op = fixture_evidence(tmp_path)
    frame = pd.read_csv(data)
    frame.loc[frame['ds'] == '2024-09-01', 'is_observed'] = True
    frame.to_csv(data, index=False)
    with pytest.raises(ValueError, match='provenance'):
        tr.load_evidence(dataset_path=data, benchmark_dir=bench, operational_dir=op)


def test_operational_model_set_must_match(tmp_path):
    data, bench, op = fixture_evidence(tmp_path)
    path = op / 'future_extrapolation.json'
    content = json.loads(path.read_text())
    del content['predictions']['GRU']
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='model set'):
        tr.load_evidence(dataset_path=data, benchmark_dir=bench, operational_dir=op)
