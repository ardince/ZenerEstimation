"""Sprint 15.3.5.2 tests: pure numerical UI helpers, no Streamlit dependency."""
from types import SimpleNamespace
import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_dashboard_numerical_ui import (
    BATTERY_110, BATTERY_410, csv_bytes, display_table, numerical_result,
)


@pytest.fixture
def comparison():
    return SimpleNamespace(
        model_metrics=pd.DataFrame([
            dict(battery_id=BATTERY_110, benchmark_type="transferred", model="LSTM",
                 rmse=0.2605953745203285, mae=0.23046032714843817,
                 mape=0.13687791706633257, rmse_rank=1),
        ]),
        holdout_predictions=pd.DataFrame([
            dict(battery_id=BATTERY_110, benchmark_type="transferred",
                 date=pd.Timestamp("2023-03-01"), model="LSTM",
                 measured_microvolt=167.17, predicted_microvolt=167.20,
                 origin="frozen_transferred_benchmark"),
        ]),
        historical_fits=pd.DataFrame([
            dict(battery_id=BATTERY_110, date=pd.Timestamp("2022-12-01"),
                 model="LSTM", fitted_microvolt=166.4,
                 origin="regenerated_diagnostic_not_frozen"),
        ]),
    )


def test_metrics_preserve_full_precision(comparison):
    table, status = numerical_result(comparison, ".", battery_id=BATTERY_110,
                                     kind="metrics", models=["LSTM"])
    assert table.loc[0, "rmse"] == 0.2605953745203285
    assert "benchmark" in status.lower()


def test_holdout_errors(comparison):
    table, status = numerical_result(comparison, ".", battery_id=BATTERY_110,
                                     kind="holdout", models=["LSTM"])
    assert table.loc[0, "error_microvolt"] == pytest.approx(.03)
    assert "frozen" in status.lower()


def test_410_holdout_not_reconstructed(comparison):
    table, status = numerical_result(comparison, ".", battery_id=BATTERY_410,
                                     kind="holdout")
    assert table.empty and "unavailable" in status.lower()


def test_diagnostic_status(comparison):
    _, status = numerical_result(comparison, ".", battery_id=BATTERY_110,
                                 kind="historical_fits")
    assert "not frozen" in status.lower()


def test_observed_delegates_to_validated_loader(monkeypatch, comparison):
    import zenerestimation.visualization.cross_battery_dashboard_numerical_ui as mod
    monkeypatch.setattr(mod, "load_110_observed_history",
                        lambda root: pd.DataFrame({"date": [pd.Timestamp("2025-03-01")],
                                                   "measured_microvolt": [171.6]}))
    table, status = numerical_result(comparison, ".", battery_id=BATTERY_110,
                                     kind="observed")
    assert table.iloc[0].measured_microvolt == 171.6
    assert "Measured" in status


def test_operational_model_selection(monkeypatch, comparison):
    import zenerestimation.visualization.cross_battery_dashboard_numerical_ui as mod
    monkeypatch.setattr(mod, "load_410_operational_forecasts",
                        lambda root: pd.DataFrame({"date": [pd.Timestamp("2025-09-01")],
                                                   "ARIMA": [32.1627520918],
                                                   "LSTM": [32.0196380615]}))
    table, status = numerical_result(comparison, ".", battery_id=BATTERY_410,
                                     kind="operational", models=["ARIMA"])
    assert list(table) == ["date", "ARIMA"]
    assert "Unvalidated" in status


def test_empty_operational_selection_keeps_date(monkeypatch, comparison):
    import zenerestimation.visualization.cross_battery_dashboard_numerical_ui as mod
    monkeypatch.setattr(mod, "load_410_operational_forecasts",
                        lambda root: pd.DataFrame({"date": [pd.Timestamp("2025-09-01")],
                                                   "ARIMA": [32.16]}))
    table, _ = numerical_result(comparison, ".", battery_id=BATTERY_410,
                                kind="operational", models=[])
    assert list(table) == ["date"]


@pytest.mark.parametrize("battery,kind,models", [
    ("unknown", "metrics", None),
    (BATTERY_110, "unknown", None),
    (BATTERY_110, "metrics", ["NotAModel"]),
])
def test_invalid_selections(comparison, battery, kind, models):
    with pytest.raises(ValueError):
        numerical_result(comparison, ".", battery_id=battery, kind=kind, models=models)


def test_rounding_is_display_only():
    original = pd.DataFrame({"rmse": [0.2605953745203285], "rank": [1]})
    shown = display_table(original, decimals=4)
    assert shown.loc[0, "rmse"] == pytest.approx(0.2606)
    assert original.loc[0, "rmse"] == 0.2605953745203285
    assert shown.loc[0, "rank"] == 1


def test_csv_uses_unrounded_values_and_dates():
    original = pd.DataFrame({"date": [pd.Timestamp("2025-06-01")],
                             "forecast": [172.032076528]})
    csv = csv_bytes(original).decode("utf-8-sig")
    assert "2025-06-01" in csv
    assert "172.032076528" in csv


def test_csv_rejects_non_dataframe():
    with pytest.raises(TypeError):
        csv_bytes([])


def test_bad_precision_rejected():
    with pytest.raises(ValueError):
        display_table(pd.DataFrame({"x": [1.0]}), decimals=17)
