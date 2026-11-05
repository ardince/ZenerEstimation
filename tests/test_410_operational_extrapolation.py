"""Regression tests for the independent 410 operational extrapolation."""
from __future__ import annotations

from types import SimpleNamespace

import json

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from examples.optimization.run_410_operational_extrapolation import (
    CUTOFF, EXPECTED_DATES, EXPECTED_MODELS, HORIZON,
    build_forecast_table, build_provenance, compute_forecasts,
    prepare_operational_training, render_report, save_outputs,
    validate_operational_contract,
)


def sample_dataset(extra=False):
    dates = pd.date_range('2024-03-01', periods=8 if extra else 6, freq='QS-MAR')
    return BatteryDataset(pd.DataFrame({
        'ds': dates,
        'microVolt': np.arange(len(dates), dtype=float) + 29.0,
    }))


def test_training_excludes_all_later_observations():
    prepared = prepare_operational_training(sample_dataset(extra=True))
    dates = pd.DatetimeIndex(prepared.data['ds'])
    assert dates[-1] == CUTOFF
    assert len(prepared.data) == 6
    assert (dates <= CUTOFF).all()


def test_missing_cutoff_is_rejected():
    dataset = sample_dataset()
    dataset.data = dataset.data.iloc[:-1].copy()
    with pytest.raises(ValueError, match='2025-06-01'):
        prepare_operational_training(dataset)


class DummyModel:
    def __init__(self, name, wrong_dates=False):
        self.name = name
        self.wrong_dates = wrong_dates
        self.fits = 0

    def fit(self, training):
        self.fits += 1
        assert pd.Timestamp(training.data['ds'].iloc[-1]) == CUTOFF
        return self

    def predict(self, steps):
        assert self.fits == 1
        assert steps == HORIZON
        return SimpleNamespace(
            model=self.name,
            dates=EXPECTED_DATES + (pd.DateOffset(months=3) if self.wrong_dates else pd.DateOffset(days=0)),
            forecast=np.arange(steps, dtype=float) + 30.0,
        )


def test_each_model_fits_only_once():
    models = [DummyModel('A'), DummyModel('B')]
    specs = [SimpleNamespace(name=model.name, create_model=lambda m=model: m)
             for model in models]
    training = prepare_operational_training(sample_dataset())
    results = compute_forecasts(training, specs)
    assert list(results) == ['A', 'B']
    assert all(model.fits == 1 for model in models)


def test_wrong_future_dates_are_rejected():
    model = DummyModel('A', wrong_dates=True)
    specs = [SimpleNamespace(name='A', create_model=lambda: model)]
    training = prepare_operational_training(sample_dataset())
    with pytest.raises(ValueError, match='forecast dates'):
        compute_forecasts(training, specs)


def test_outputs_are_independent_and_unvalidated(tmp_path):
    training = prepare_operational_training(sample_dataset())
    specs = [SimpleNamespace(name='A', params={'seed': 42})]
    paths = save_outputs(training, {'A': np.arange(6, dtype=float) + 30}, specs, tmp_path)
    assert all(path.exists() for path in paths.values())
    text = paths['report'].read_text(encoding='utf-8')
    assert 'NOT A HOLDOUT BENCHMARK' in text
    assert '2026-12-01' in text
    assert '30.000000' in text


def test_strict_contract_accepts_exactly_36_predictions():
    forecasts = {name: np.arange(HORIZON, dtype=float) + 30.0 + i for i, name in enumerate(EXPECTED_MODELS)}
    specs = [SimpleNamespace(name=name, params={"seed": 42}) for name in EXPECTED_MODELS]
    table = build_forecast_table(forecasts)
    validate_operational_contract(table, specs, require_expected_models=True)
    assert table.drop(columns="date").size == 36


def test_strict_contract_rejects_incomplete_model_set():
    table = build_forecast_table({"ARIMA": np.arange(6, dtype=float) + 30})
    specs = [SimpleNamespace(name="ARIMA", params={})]
    with pytest.raises(ValueError, match="six frozen 410"):
        validate_operational_contract(table, specs, require_expected_models=True)


def test_provenance_records_refit_without_reoptimization():
    training = prepare_operational_training(sample_dataset())
    specs = [SimpleNamespace(name="LSTM", params={"seed": 42, "units": 32}), SimpleNamespace(name="GRU", params={"seed": 42, "units": 24})]
    table = build_forecast_table({"LSTM": np.arange(6, dtype=float) + 30, "GRU": np.arange(6, dtype=float) + 31})
    provenance = build_provenance(training, table, specs)
    assert provenance["configuration_source"] == "frozen_410_optimized_benchmark"
    assert provenance["model_update_policy"] == "refit_through_training_cutoff_without_reoptimization"
    assert provenance["training_cutoff"] == "2025-06-01"
    assert provenance["forecast_start"] == "2025-09-01"
    assert provenance["forecast_end"] == "2026-12-01"
    assert provenance["neural_seeds"] == {"LSTM": 42, "GRU": 42}
    assert provenance["prediction_count"] == 12


def test_provenance_contains_no_pseudo_evaluation_evidence():
    training = prepare_operational_training(sample_dataset())
    specs = [SimpleNamespace(name="A", params={"seed": 42})]
    table = build_forecast_table({"A": np.arange(6, dtype=float) + 30})
    provenance = build_provenance(training, table, specs)
    assert provenance["measured_actuals"] is None
    assert provenance["forecast_errors"] is None
    assert provenance["evaluation_metrics"] is None
    assert provenance["uncertainty_intervals"] is None
    serialized = json.dumps(provenance).lower()
    assert '"rmse"' not in serialized and '"mae"' not in serialized and '"mape"' not in serialized


def test_report_states_operational_scientific_status():
    training = prepare_operational_training(sample_dataset())
    specs = [SimpleNamespace(name="A", params={"seed": 42})]
    table = build_forecast_table({"A": np.arange(6, dtype=float) + 30})
    provenance = build_provenance(training, table, specs)
    text = render_report(table, provenance)
    assert "NOT A HOLDOUT BENCHMARK" in text
    assert "without reoptimization" in text
    assert "unvalidated" in text
    assert "2025-09-01" in text and "2026-12-01" in text


def test_saved_artifacts_share_same_predictions(tmp_path):
    training = prepare_operational_training(sample_dataset())
    specs = [SimpleNamespace(name="A", params={"seed": 42}), SimpleNamespace(name="B", params={"seed": 42})]
    forecasts = {"A": np.arange(6, dtype=float) + 30, "B": np.arange(6, dtype=float) + 31}
    paths = save_outputs(training, forecasts, specs, tmp_path)
    csv_table = pd.read_csv(paths["csv"])
    provenance = json.loads(paths["json"].read_text(encoding="utf-8"))
    report = paths["report"].read_text(encoding="utf-8")
    assert csv_table["date"].tolist() == provenance["forecast_dates"]
    for model in ("A", "B"):
        np.testing.assert_allclose(csv_table[model].to_numpy(dtype=float), np.asarray(provenance["predictions"][model], dtype=float), rtol=0.0, atol=1e-10)
        for value in forecasts[model]:
            assert f"{value:.6f}" in report
