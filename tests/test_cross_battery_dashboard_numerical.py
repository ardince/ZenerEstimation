"""Sprint 15.3.5.1: regression tests for read-only numerical tables."""
from types import SimpleNamespace

import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_dashboard_numerical import (
    numerical_table,
)

BATTERY_410 = "732B-5610410"
BATTERY_110 = "732B-5610110"


@pytest.fixture
def comparison():
    return SimpleNamespace(
        model_metrics=pd.DataFrame([
            {"battery_id": BATTERY_410, "benchmark_type": "optimized_benchmark",
             "model": "KalmanLSTM", "rmse": 0.6721611750049541,
             "mae": 0.5700114302412848, "mape": 1.8602854625097138,
             "rmse_rank": 1},
            {"battery_id": BATTERY_110, "benchmark_type": "transferred_benchmark",
             "model": "LSTM", "rmse": 0.2605953745203285,
             "mae": 0.23046032714843817, "mape": 0.13687791706633257,
             "rmse_rank": 1},
            {"battery_id": BATTERY_110, "benchmark_type": "transferred_benchmark",
             "model": "GRU", "rmse": 0.2717940008743964,
             "mae": 0.2657478027343757, "mape": 0.15798571885560114,
             "rmse_rank": 2},
        ]),
        holdout_predictions=pd.DataFrame([
            {"battery_id": BATTERY_110, "benchmark_type": "transferred_benchmark",
             "date": pd.Timestamp("2023-03-01"), "model": "LSTM",
             "measured_microvolt": 167.17, "predicted_microvolt": 167.2,
             "origin": "frozen_transferred_benchmark"},
            {"battery_id": BATTERY_110, "benchmark_type": "transferred_benchmark",
             "date": pd.Timestamp("2023-06-01"), "model": "GRU",
             "measured_microvolt": 167.73, "predicted_microvolt": 167.6,
             "origin": "frozen_transferred_benchmark"},
        ]),
        historical_fits=pd.DataFrame([
            {"battery_id": BATTERY_110, "date": pd.Timestamp("2022-09-01"),
             "model": "LSTM", "fitted_microvolt": float("nan"),
             "origin": "regenerated_diagnostic_not_frozen"},
            {"battery_id": BATTERY_110, "date": pd.Timestamp("2022-12-01"),
             "model": "LSTM", "fitted_microvolt": 166.4,
             "origin": "regenerated_diagnostic_not_frozen"},
        ]),
    )


def test_metrics_preserve_source_precision(comparison):
    table = numerical_table(comparison, battery_id=BATTERY_110, kind="metrics")
    assert table.available
    assert table.data.loc[0, "rmse"] == 0.2605953745203285
    assert table.data.loc[1, "rmse_rank"] == 2


def test_410_metrics_are_available(comparison):
    table = numerical_table(comparison, battery_id=BATTERY_410, kind="metrics")
    assert len(table.data) == 1
    assert table.data.iloc[0]["model"] == "KalmanLSTM"


def test_110_holdout_errors_are_signed_prediction_minus_measurement(comparison):
    table = numerical_table(comparison, battery_id=BATTERY_110, kind="holdout")
    assert table.available
    assert table.data["error_microvolt"].tolist() == pytest.approx([0.03, -0.13])
    assert table.data["absolute_error_microvolt"].tolist() == pytest.approx([0.03, 0.13])


def test_410_holdout_is_explicitly_unavailable(comparison):
    table = numerical_table(comparison, battery_id=BATTERY_410, kind="holdout")
    assert not table.available
    assert table.data.empty
    assert "unavailable" in table.status.lower()


def test_410_fitted_history_is_explicitly_unavailable(comparison):
    table = numerical_table(comparison, battery_id=BATTERY_410, kind="historical_fits")
    assert not table.available
    assert table.data.empty


def test_historical_diagnostic_nan_warmup_is_preserved(comparison):
    table = numerical_table(comparison, battery_id=BATTERY_110, kind="historical_fits")
    assert table.available
    assert pd.isna(table.data.loc[0, "fitted_microvolt"])
    assert table.data.loc[1, "fitted_microvolt"] == 166.4
    assert "not frozen" in table.status.lower()


def test_model_selection_does_not_recalculate_ranks(comparison):
    table = numerical_table(
        comparison, battery_id=BATTERY_110, kind="metrics", models=["GRU"]
    )
    assert table.data["model"].tolist() == ["GRU"]
    assert table.data.iloc[0]["rmse_rank"] == 2


def test_empty_model_selection_returns_no_rows(comparison):
    table = numerical_table(
        comparison, battery_id=BATTERY_110, kind="metrics", models=[]
    )
    assert not table.available
    assert table.data.empty


@pytest.mark.parametrize("battery,kind", [
    ("unknown", "metrics"),
    (BATTERY_110, "future"),
])
def test_unknown_selection_is_rejected(comparison, battery, kind):
    with pytest.raises(ValueError, match="Unknown"):
        numerical_table(comparison, battery_id=battery, kind=kind)


@pytest.mark.parametrize("source,removed", [
    ("model_metrics", "rmse"),
    ("holdout_predictions", "origin"),
    ("historical_fits", "fitted_microvolt"),
])
def test_required_schema_columns_are_enforced(comparison, source, removed):
    setattr(comparison, source, getattr(comparison, source).drop(columns=[removed]))
    kind = {"model_metrics": "metrics", "holdout_predictions": "holdout",
            "historical_fits": "historical_fits"}[source]
    with pytest.raises(ValueError, match="schema"):
        numerical_table(comparison, battery_id=BATTERY_110, kind=kind)


@pytest.mark.parametrize("source,kind", [
    ("holdout_predictions", "holdout"),
    ("historical_fits", "historical_fits"),
])
def test_incorrect_evidence_origin_is_rejected(comparison, source, kind):
    getattr(comparison, source).loc[0, "origin"] = "operational_unvalidated"
    with pytest.raises(ValueError, match="provenance|frozen"):
        numerical_table(comparison, battery_id=BATTERY_110, kind=kind)


def test_source_dataframes_remain_unchanged_and_output_is_detached(comparison):
    originals = {name: getattr(comparison, name).copy(deep=True)
                 for name in ("model_metrics", "holdout_predictions", "historical_fits")}
    for kind in ("metrics", "holdout", "historical_fits"):
        table = numerical_table(comparison, battery_id=BATTERY_110, kind=kind)
        if not table.data.empty:
            table.data.loc[0, "model"] = "MUTATED"
    for name, original in originals.items():
        pd.testing.assert_frame_equal(getattr(comparison, name), original)


def test_missing_dataframe_is_rejected(comparison):
    comparison.model_metrics = None
    with pytest.raises(ValueError, match="schema"):
        numerical_table(comparison, battery_id=BATTERY_110, kind="metrics")
