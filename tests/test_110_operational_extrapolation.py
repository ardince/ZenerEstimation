"""Sprint 14.9G.4: operational dataset-110 regression contracts (no neural fits)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from examples.optimization import run_110_operational_extrapolation as op


MISSING = pd.to_datetime([
    '2002-09-01', '2010-03-01', '2010-06-01',
    '2024-06-01', '2024-09-01', '2024-12-01',
])


def dataset_110(*, future=False):
    dates = pd.date_range('1998-03-01', '2025-06-01' if future else '2025-03-01', freq='QS-MAR')
    values = np.linspace(70.0, 171.60, len(dates))
    flags = ~dates.isin(MISSING)
    values[~flags] = np.nan
    values[dates == pd.Timestamp('2025-03-01')] = 171.60
    return BatteryDataset(pd.DataFrame({'ds': dates, 'microVolt': values, 'is_observed': flags}))


def test_cutoff_and_unobserved_provenance_preserved():
    prepared = op.prepare_operational_training(dataset_110(future=True))
    assert pd.Timestamp(prepared.data['ds'].iloc[-1]) == op.CUTOFF
    assert len(prepared.data) == len(pd.date_range('1998-03-01', op.CUTOFF, freq='QS-MAR'))
    assert pd.DatetimeIndex(prepared.data.loc[~prepared.data['is_observed'], 'ds']).equals(pd.DatetimeIndex(MISSING))
    assert prepared.data['microVolt'].notna().all()


def test_missing_cutoff_measurement_rejected():
    dataset = dataset_110()
    dataset.data.loc[dataset.data['ds'] == op.CUTOFF, 'microVolt'] = np.nan
    dataset.data.loc[dataset.data['ds'] == op.CUTOFF, 'is_observed'] = False
    with pytest.raises(ValueError, match='Cutoff measurement'):
        op.prepare_operational_training(dataset)


def test_wrong_cutoff_measurement_rejected():
    dataset = dataset_110()
    dataset.data.loc[dataset.data['ds'] == op.CUTOFF, 'microVolt'] = 172.0
    with pytest.raises(ValueError, match='171.60'):
        op.prepare_operational_training(dataset)


def test_provenance_flag_disagreement_rejected():
    dataset = dataset_110()
    dataset.data.loc[dataset.data['ds'] == pd.Timestamp('2024-06-01'), 'is_observed'] = True
    with pytest.raises(ValueError, match='Observation flags'):
        op.prepare_operational_training(dataset)


class DummyModel:
    def __init__(self, name, wrong_dates=False):
        self.name = name
        self.wrong_dates = wrong_dates
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, dataset):
        self.fit_calls += 1
        assert dataset.data['ds'].iloc[-1] == op.CUTOFF
        return self

    def predict(self, steps):
        self.predict_calls += 1
        assert steps == 6
        dates = op.EXPECTED_DATES + (pd.DateOffset(months=3) if self.wrong_dates else pd.DateOffset(days=0))
        return SimpleNamespace(model=self.name, dates=dates, forecast=np.arange(steps) + 172.0)


def test_each_model_fits_and_predicts_once():
    training = op.prepare_operational_training(dataset_110())
    models = [DummyModel('A'), DummyModel('B')]
    specs = [SimpleNamespace(name=m.name, create_model=lambda m=m: m) for m in models]
    result = op.compute_forecasts(training, specs)
    assert tuple(result) == ('A', 'B')
    assert all(m.fit_calls == 1 and m.predict_calls == 1 for m in models)


def test_wrong_forecast_dates_rejected():
    training = op.prepare_operational_training(dataset_110())
    m = DummyModel('A', wrong_dates=True)
    with pytest.raises(ValueError, match='forecast dates'):
        op.compute_forecasts(training, [SimpleNamespace(name='A', create_model=lambda: m)])


def test_saved_outputs_unvalidated_and_precision_preserved(tmp_path):
    training = op.prepare_operational_training(dataset_110())
    specs = [SimpleNamespace(name='A', params={'seed': 42})]
    values = np.arange(6) + 172.123456789
    paths = op.save_outputs(training, {'A': values}, specs, output_dir=tmp_path)
    assert all(p.exists() for p in paths.values())
    readable = pd.read_csv(paths['csv'])
    precise = pd.read_csv(paths['full_precision_csv'])
    assert readable['A'].iloc[0] == 172.12
    assert abs(precise['A'].iloc[0] - values[0]) < 1e-8
    metadata = json.loads(paths['json'].read_text(encoding='utf-8'))
    assert metadata['evaluation_metrics'] is None
    assert metadata['measured_actuals'] is None
    assert metadata['training_cutoff'] == '2025-03-01'
    assert metadata['forecast_end'] == '2026-09-01'
    assert metadata['optimized_for_dataset_110'] is False
    assert metadata['unobserved_training_dates'] == [d.strftime('%Y-%m-%d') for d in MISSING]
    assert 'NOT A HOLDOUT BENCHMARK' in paths['report'].read_text(encoding='utf-8')


def test_operational_contract_rejects_model_order():
    specs = [SimpleNamespace(name='A'), SimpleNamespace(name='B')]
    table = op.build_forecast_table({'B': np.ones(6), 'A': np.ones(6)})
    with pytest.raises(ValueError, match='columns'):
        op.validate_operational_contract(table, specs)


def test_operational_dates_and_model_count():
    assert list(op.EXPECTED_DATES.strftime('%Y-%m-%d')) == [
        '2025-06-01', '2025-09-01', '2025-12-01',
        '2026-03-01', '2026-06-01', '2026-09-01',
    ]
    assert len(op.EXPECTED_MODELS) == 6
