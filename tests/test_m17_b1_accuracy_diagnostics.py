"""M17-B1 contract tests (RED until the additive diagnostics module exists).

Proposed public API:
    build_accuracy_diagnostics(folds, *, dataset, model, scope) -> dict

Each fold is a mapping with fold_index (0-based), dates, actual,
predicted, alignment_status. The result contains rows, summary,
horizon_metrics, fold_metrics, and provenance.
"""

import math

import numpy as np
import pandas as pd
import pytest

from zenerestimation.evaluation.accuracy_diagnostics import build_accuracy_diagnostics


DATES = pd.to_datetime(["2025-03-01", "2025-06-01", "2025-09-01"])


def fold(index, actual, predicted, *, dates=DATES, status="verified"):
    return {
        "fold_index": index,
        "dates": dates,
        "actual": actual,
        "predicted": predicted,
        "alignment_status": status,
    }


def diagnose(folds, scope="internal_expanding_window"):
    return build_accuracy_diagnostics(
        folds, dataset="732B-5610410", model="SyntheticModel", scope=scope
    )


def test_signed_errors_units_and_first_step():
    report = diagnose([fold(0, [10.0, 20.0, 30.0], [11.0, 18.0, 33.0])])
    rows = report["rows"]
    assert [r["horizon_step"] for r in rows] == [1, 2, 3]
    assert [r["date"] for r in rows] == ["2025-03-01", "2025-06-01", "2025-09-01"]
    assert [r["error_microVolt"] for r in rows] == pytest.approx([1.0, -2.0, 3.0])
    assert [r["absolute_error_microVolt"] for r in rows] == pytest.approx([1, 2, 3])
    assert [r["squared_error_microVolt2"] for r in rows] == pytest.approx([1, 4, 9])
    assert report["summary"]["first_step_error_microVolt"] == pytest.approx(1)
    assert report["summary"]["first_step_absolute_error_microVolt"] == pytest.approx(1)
    assert report["summary"]["bias_microVolt"] == pytest.approx(2 / 3)
    assert report["summary"]["mae_microVolt"] == pytest.approx(2)
    assert report["summary"]["rmse_microVolt"] == pytest.approx(math.sqrt(14 / 3))


def test_horizon_metrics_pool_same_horizon_across_folds():
    report = diagnose([
        fold(0, [10, 20, 30], [11, 18, 33]),
        fold(1, [40, 50, 60], [39, 54, 60]),
    ])
    horizons = report["horizon_metrics"]
    assert [h["horizon_step"] for h in horizons] == [1, 2, 3]
    assert [h["n"] for h in horizons] == [2, 2, 2]
    assert [h["bias_microVolt"] for h in horizons] == pytest.approx([0, 1, 1.5])
    assert [h["mae_microVolt"] for h in horizons] == pytest.approx([1, 3, 1.5])
    assert [h["rmse_microVolt"] for h in horizons] == pytest.approx(
        [1, math.sqrt(10), math.sqrt(4.5)]
    )


def test_fold_metrics_and_population_stability_are_explicit():
    report = diagnose([
        fold(0, [10, 20, 30], [11, 21, 31]),
        fold(1, [10, 20, 30], [12, 22, 32]),
    ])
    assert [x["rmse_microVolt"] for x in report["fold_metrics"]] == pytest.approx([1, 2])
    stability = report["summary"]["fold_stability"]
    assert stability["fold_count"] == 2
    assert stability["rmse_mean_microVolt"] == pytest.approx(1.5)
    assert stability["rmse_std_population_microVolt"] == pytest.approx(0.5)
    assert stability["worst_fold_index"] == 1
    assert stability["worst_fold_rmse_microVolt"] == pytest.approx(2)


def test_single_fold_stability_is_defined_without_nan():
    report = diagnose([fold(0, [10, 20, 30], [11, 21, 31])])
    assert report["summary"]["fold_stability"]["rmse_std_population_microVolt"] == 0.0


def test_provenance_preserves_scope_and_alignment_status():
    report = diagnose([fold(0, [10, 20, 30], [11, 21, 31])])
    assert report["provenance"] == {
        "dataset": "732B-5610410",
        "model": "SyntheticModel",
        "scope": "internal_expanding_window",
        "units": "microVolt",
        "alignment_statuses": ["verified"],
    }


def test_positional_only_is_never_promoted_to_verified():
    report = diagnose([fold(0, [10, 20, 30], [11, 21, 31], status="positional_only")])
    assert report["provenance"]["alignment_statuses"] == ["positional_only"]
    assert report["provenance"].get("alignment_verified", False) is False


@pytest.mark.parametrize(
    "bad_fold",
    [
        fold(0, [1, 2, 3], [1, 2]),
        fold(0, [1, 2, 3], [1, 2, np.nan]),
        fold(0, [1, 2, 3], [1, 2, np.inf]),
        fold(0, [1, 2, 3], [1, 2, 3], dates=pd.to_datetime(["2025-03-01"] * 3)),
        fold(0, [1, 2, 3], [1, 2, 3], dates=pd.to_datetime(["2025-06-01", "2025-03-01", "2025-09-01"])),
        fold(0, [], [], dates=pd.DatetimeIndex([])),
    ],
)
def test_invalid_fold_inputs_fail_closed(bad_fold):
    with pytest.raises(ValueError):
        diagnose([bad_fold])


def test_unknown_alignment_status_is_rejected():
    with pytest.raises(ValueError, match="alignment"):
        diagnose([fold(0, [1, 2, 3], [1, 2, 3], status="verified-ish")])


def test_empty_folds_are_rejected():
    with pytest.raises(ValueError):
        diagnose([])


def test_operational_forecast_cannot_be_scored_as_measured_accuracy():
    with pytest.raises(ValueError, match="scope|operational|measured"):
        diagnose([fold(0, [10, 20, 30], [11, 21, 31])], scope="operational_forecast")


def test_input_arrays_are_not_mutated():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([11.0, 21.0, 31.0])
    actual_before = actual.copy()
    predicted_before = predicted.copy()
    diagnose([fold(0, actual, predicted)])
    np.testing.assert_array_equal(actual, actual_before)
    np.testing.assert_array_equal(predicted, predicted_before)
