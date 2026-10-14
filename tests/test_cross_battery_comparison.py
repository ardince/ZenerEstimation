"""Focused read-only comparison adapter tests with synthetic records."""
import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_artifacts import (
    BatteryBenchmarkRecord, EvidenceAvailability, ArtifactContractError, MODELS,
)
from zenerestimation.visualization.cross_battery_comparison import build_comparison


def _record(battery, transferred):
    values = [0.7, 0.9, 1.3, 1.4, 0.8, 0.6] if not transferred else [0.3, 1.8, 0.26, 0.27, 1.5, 5.2]
    metrics = pd.DataFrame({'rmse': values, 'mae': values, 'mape': values}, index=MODELS)
    rankings = {'rmse': [dict(rank=i+1, model=m, value=metrics.loc[m, 'rmse'])
                         for i, m in enumerate(sorted(MODELS, key=lambda x: metrics.loc[x, 'rmse']))]}
    predictions = pd.DataFrame([dict(date='2023-03-01', measured_microVolt=167.17,
                                     **{m: 167.2 for m in MODELS})]) if transferred else None
    fits = pd.DataFrame([dict(date='2022-12-01', **{m: 166.61 for m in MODELS})]) if transferred else None
    return BatteryBenchmarkRecord(battery_id=battery, benchmark_type='transferred' if transferred else 'optimized',
        development_end='2022-12-01' if transferred else '2023-12-01',
        holdout_start='2023-03-01' if transferred else '2024-03-01',
        holdout_end='2024-03-01' if transferred else '2025-06-01',
        holdout_steps=5 if transferred else 6, metrics=metrics, rankings=rankings, provenance={},
        holdout_predictions=predictions, historical_fits=fits,
        evidence={'historical_fits': EvidenceAvailability(transferred, True, 'diagnostic.csv' if transferred else None,
                    'regenerated_diagnostic_not_frozen' if transferred else 'frozen_benchmark_plot_only')},
        source_hashes={})


def _records():
    return {'732B-5610410': _record('732B-5610410', False),
            '732B-5610110': _record('732B-5610110', True)}


def test_summary_and_rankings():
    result = build_comparison(_records())
    assert len(result.summary) == 2
    assert len(result.model_metrics) == 12
    assert result.summary['best_rmse_model'].tolist() == ['KalmanLSTM', 'LSTM']
    assert result.model_metrics.query("battery_id == '732B-5610110' and model == 'LSTM'")['rmse_rank'].item() == 1


def test_structured_evidence_never_invented():
    result = build_comparison(_records())
    assert set(result.holdout_predictions.battery_id) == {'732B-5610110'}
    assert set(result.historical_fits.battery_id) == {'732B-5610110'}
    assert result.evidence.query("battery_id == '732B-5610410'")['structured'].eq(False).all()
    assert set(result.historical_fits.origin) == {'regenerated_diagnostic_not_frozen'}


def test_long_format_row_counts():
    result = build_comparison(_records())
    assert len(result.holdout_predictions) == 6
    assert len(result.historical_fits) == 6
    assert set(result.holdout_predictions.model) == set(MODELS)


def test_does_not_mutate_input_frames():
    records = _records()
    before = {key: record.metrics.copy(deep=True) for key, record in records.items()}
    build_comparison(records)
    for key, record in records.items():
        pd.testing.assert_frame_equal(record.metrics, before[key])


def test_rejects_incomplete_battery_pair():
    records = _records()
    del records['732B-5610110']
    with pytest.raises(ArtifactContractError, match='exactly batteries'):
        build_comparison(records)


def test_rejects_identity_mismatch():
    records = _records()
    records['732B-5610410'] = records['732B-5610110']
    with pytest.raises(ArtifactContractError, match='identifier mismatch'):
        build_comparison(records)
