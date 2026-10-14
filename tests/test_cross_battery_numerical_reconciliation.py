"""Sprint 15.3.5.3 regression: audit exact adapter outputs and source exports."""
from types import SimpleNamespace

import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_numerical_reconciliation import (
    _export_roundtrip, reconcile_numerical_results,
)
from zenerestimation.visualization.cross_battery_dashboard_numerical_ui import (
    BATTERY_110, BATTERY_410,
)


@pytest.fixture
def comparison():
    return SimpleNamespace(
        model_metrics=pd.DataFrame([
            dict(battery_id=BATTERY_410, model="ARIMA", rmse=.849441641452755,
                 mae=.7625092812293447, mape=2.427532722653989, rmse_rank=2),
            dict(battery_id=BATTERY_110, model="LSTM", rmse=.2605953745203285,
                 mae=.23046032714843817, mape=.13687791706633257, rmse_rank=1),
        ]),
        holdout_predictions=pd.DataFrame([
            dict(battery_id=BATTERY_110, date=pd.Timestamp("2023-03-01"), model="LSTM",
                 measured_microvolt=167.17, predicted_microvolt=167.2,
                 origin="frozen_transferred_benchmark"),
        ]),
        historical_fits=pd.DataFrame([
            dict(battery_id=BATTERY_110, date=pd.Timestamp("2022-12-01"), model="LSTM",
                 fitted_microvolt=float("nan"), origin="regenerated_diagnostic_not_frozen"),
        ]),
    )


@pytest.fixture
def stub_loaders(monkeypatch):
    import zenerestimation.visualization.cross_battery_numerical_reconciliation as mod
    import zenerestimation.visualization.cross_battery_dashboard_numerical_ui as ui
    for battery, suffix in ((BATTERY_410, "410"), (BATTERY_110, "110")):
        history = pd.DataFrame({"date": [pd.Timestamp("2025-03-01")],
                                "measured_microvolt": [32.0 if suffix == "410" else 171.6]})
        future = pd.DataFrame({"date": [pd.Timestamp("2025-06-01")],
                               "ARIMA": [32.1627520918 if suffix == "410" else 172.032076528]})
        monkeypatch.setattr(mod, f"load_{suffix}_observed_history", lambda root, df=history: df.copy())
        monkeypatch.setattr(ui, f"load_{suffix}_observed_history", lambda root, df=history: df.copy())
        monkeypatch.setattr(mod, f"load_{suffix}_operational_forecasts", lambda root, df=future: df.copy())
        monkeypatch.setattr(ui, f"load_{suffix}_operational_forecasts", lambda root, df=future: df.copy())
    # The UI requests every model column, so stub full model schemas.
    from zenerestimation.visualization.cross_battery_dashboard_sources import MODELS
    for suffix in ("410", "110"):
        f = getattr(mod, f"load_{suffix}_operational_forecasts")
        df = f(None)
        for name in MODELS:
            if name not in df:
                df[name] = df["ARIMA"]
        monkeypatch.setattr(mod, f"load_{suffix}_operational_forecasts", lambda root, df=df: df.copy())
        monkeypatch.setattr(ui, f"load_{suffix}_operational_forecasts", lambda root, df=df: df.copy())


def test_all_ten_sections_reconcile(comparison, stub_loaders):
    audit = reconcile_numerical_results(comparison, ".")
    assert len(audit.checks) == 10
    assert audit.passed, audit.failures.to_string(index=False)
    assert audit.failures.empty


def test_410_missing_structured_evidence_is_not_fabricated(comparison, stub_loaders):
    audit = reconcile_numerical_results(comparison, ".")
    for kind in ("holdout", "historical_fits"):
        row = audit.checks.loc[(audit.checks.battery_id == BATTERY_410)
                               & (audit.checks.kind == kind)].iloc[0]
        assert row.passed and "0 rows" in row.detail


def test_frozen_holdout_provenance_mismatch_is_reported(comparison, stub_loaders):
    comparison.holdout_predictions.loc[0, "origin"] = "unvalidated"
    audit = reconcile_numerical_results(comparison, ".")
    assert not audit.passed
    assert "holdout" in set(audit.failures.kind)


def test_csv_full_precision_and_nan():
    frame = pd.DataFrame({"date": [pd.Timestamp("2023-03-01"), pd.Timestamp("2023-06-01")],
                          "prediction": [.2605953745203285, float("nan")],
                          "rank": [1, 2]})
    _export_roundtrip(frame)


def test_input_frames_not_mutated(comparison, stub_loaders):
    original = {name: getattr(comparison, name).copy(deep=True)
                for name in ("model_metrics", "holdout_predictions", "historical_fits")}
    reconcile_numerical_results(comparison, ".")
    for name, frame in original.items():
        pd.testing.assert_frame_equal(getattr(comparison, name), frame)
