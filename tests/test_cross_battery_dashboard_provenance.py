"""Sprint 15.3.4.1 provenance adapter unit tests (no repository files required)."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import pandas as pd
import pytest

from zenerestimation.visualization import cross_battery_dashboard_provenance as p


def _fixture(tmp_path, monkeypatch):
    summaries = []
    evidences = []
    for battery, benchmark_type, cutoff, start, end, anchor in (
        (p.BATTERY_410, 'optimized', '2025-06-01', '2025-09-01', '2026-12-01', 32.0),
        (p.BATTERY_110, 'transferred', '2025-03-01', '2025-06-01', '2026-09-01', 171.6),
    ):
        summaries.append(dict(battery_id=battery, benchmark_type=benchmark_type,
                              development_end='2023-12-01', holdout_start='2024-03-01',
                              holdout_end='2025-06-01', holdout_steps=6))
        evidences.append(dict(battery_id=battery, evidence_type='holdout', structured=False,
                              visual=True, origin='visual_only', source='saved image'))
        folder = tmp_path / 'results' / battery / 'operational_extrapolation'
        folder.mkdir(parents=True)
        metadata = dict(scientific_status=p.STATUS, measured_actuals=None,
                        evaluation_metrics=None, models=list(p.MODELS), training_cutoff=cutoff,
                        model_update_policy='refit_through_training_cutoff_without_reoptimization')
        if battery == p.BATTERY_110:
            metadata.update(optimized_for_dataset_110=False, configuration_source_battery=p.BATTERY_410)
            csv_name = 'future_extrapolation_full_precision.csv'
            benchmark_folder = tmp_path / 'results' / battery / 'transferred_benchmark'
            provenance_name = 'transfer_provenance_validated.json'
        else:
            metadata['configuration_source'] = 'frozen_410_optimized_benchmark'
            csv_name = 'future_extrapolation.csv'
            benchmark_folder = tmp_path / 'results' / battery / 'optimized_benchmark'
            provenance_name = 'provenance.json'
        (folder / 'future_extrapolation.json').write_text(json.dumps(metadata), encoding='utf-8')
        (folder / csv_name).write_text('date,ARIMA\n', encoding='utf-8')
        benchmark_folder.mkdir(parents=True)
        for filename in ('benchmark.json', provenance_name):
            (benchmark_folder / filename).write_text('{}', encoding='utf-8')
        dataset = tmp_path / 'datasets' / 'processed'
        dataset.mkdir(parents=True, exist_ok=True)
        (dataset / f'{battery}.csv').write_text('ds,microVolt,is_observed\n', encoding='utf-8')
        hist = pd.DataFrame({'date': pd.to_datetime([cutoff]), 'measured_microvolt': [anchor]})
        fcst = pd.DataFrame({'date': pd.date_range(start, periods=6, freq='QS-MAR')})
        if battery == p.BATTERY_410:
            monkeypatch.setattr(p, 'load_410_observed_history', lambda root, h=hist: h.copy())
            monkeypatch.setattr(p, 'load_410_operational_forecasts', lambda root, f=fcst: f.copy())
        else:
            monkeypatch.setattr(p, 'load_110_observed_history', lambda root, h=hist: h.copy())
            monkeypatch.setattr(p, 'load_110_operational_forecasts', lambda root, f=fcst: f.copy())
    return SimpleNamespace(summary=pd.DataFrame(summaries), evidence=pd.DataFrame(evidences))


def test_protocols_distinguish_optimized_and_transfer(tmp_path, monkeypatch):
    comparison = _fixture(tmp_path, monkeypatch)
    result = p.build_provenance(tmp_path, comparison)
    assert len(result.protocols) == 2
    rows = result.protocols.set_index('battery_id')
    assert 'transferred' in rows.loc[p.BATTERY_110, 'configuration_source'].lower()
    assert rows.loc[p.BATTERY_410, 'operational_end'] == '2026-12-01'
    assert rows.loc[p.BATTERY_110, 'operational_start'] == '2025-06-01'
    assert not rows.operational_evaluated.any()


def test_sha256_is_of_exact_file_bytes(tmp_path, monkeypatch):
    result = p.build_provenance(tmp_path, _fixture(tmp_path, monkeypatch))
    for row in result.sources.itertuples():
        assert row.sha256 == hashlib.sha256((tmp_path / row.source).read_bytes()).hexdigest()
    assert len(result.sources) == 10


def test_does_not_mutate_comparison(tmp_path, monkeypatch):
    comparison = _fixture(tmp_path, monkeypatch)
    summary = comparison.summary.copy(deep=True)
    evidence = comparison.evidence.copy(deep=True)
    p.build_provenance(tmp_path, comparison)
    pd.testing.assert_frame_equal(comparison.summary, summary)
    pd.testing.assert_frame_equal(comparison.evidence, evidence)


def test_rejects_false_operational_evaluation(tmp_path, monkeypatch):
    comparison = _fixture(tmp_path, monkeypatch)
    path = tmp_path / 'results' / p.BATTERY_410 / 'operational_extrapolation' / 'future_extrapolation.json'
    metadata = json.loads(path.read_text())
    metadata['evaluation_metrics'] = {'rmse': 0.1}
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Operational status'):
        p.build_provenance(tmp_path, comparison)


def test_rejects_cutoff_mismatch(tmp_path, monkeypatch):
    comparison = _fixture(tmp_path, monkeypatch)
    path = tmp_path / 'results' / p.BATTERY_110 / 'operational_extrapolation' / 'future_extrapolation.json'
    metadata = json.loads(path.read_text())
    metadata['training_cutoff'] = '2024-12-01'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='cutoff'):
        p.build_provenance(tmp_path, comparison)


def test_rejects_missing_provenance_file(tmp_path, monkeypatch):
    comparison = _fixture(tmp_path, monkeypatch)
    path = tmp_path / 'results' / p.BATTERY_110 / 'transferred_benchmark' / 'transfer_provenance_validated.json'
    path.unlink()
    with pytest.raises(FileNotFoundError, match='Provenance source'):
        p.build_provenance(tmp_path, comparison)
