"""Sprint 14.9G.3.1: no-training transfer benchmark preflight tests."""
import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from examples.optimization import run_110_transferred_benchmark as runner


@pytest.fixture
def canonical_110():
    dates = pd.date_range("1998-03-01", "2025-03-01", freq="QS-MAR")
    values = np.linspace(70, 171.60, len(dates))
    missing = pd.to_datetime([
        "2002-09-01", "2010-03-01", "2010-06-01",
        "2024-06-01", "2024-09-01", "2024-12-01",
    ])
    values[dates.isin(missing)] = np.nan
    for date, value in zip(runner.HOLDOUT_DATES, runner.HOLDOUT_VALUES):
        values[dates.get_loc(date)] = value
    values[dates.get_loc(pd.Timestamp("2025-03-01"))] = 171.60
    return BatteryDataset(pd.DataFrame({
        "ds": dates, "microVolt": values,
        "is_observed": ~dates.isin(missing),
    }))


def test_exact_frozen_holdout_and_training_only_interpolation(canonical_110):
    training, holdout, prepared = runner.verify_preflight(canonical_110)
    assert training.data["ds"].iloc[-1] == runner.TRAINING_END
    assert holdout["ds"].tolist() == list(runner.HOLDOUT_DATES)
    np.testing.assert_allclose(holdout["microVolt"], runner.HOLDOUT_VALUES)
    assert prepared.data["microVolt"].notna().all()
    assert prepared.data.loc[~prepared.data["is_observed"], "microVolt"].notna().all()
    assert len(training) == 100


def test_future_measurement_changes_do_not_change_preflight(canonical_110):
    original = runner.verify_preflight(canonical_110)
    changed = canonical_110.data.copy()
    changed.loc[changed["ds"] > runner.EVALUATION_END, "microVolt"] = [999., 999., 999., 999.]
    changed.loc[changed["ds"] > runner.EVALUATION_END, "is_observed"] = True
    other = runner.verify_preflight(BatteryDataset(changed))
    pd.testing.assert_frame_equal(original[0].data, other[0].data)
    pd.testing.assert_frame_equal(original[1], other[1])
    pd.testing.assert_frame_equal(original[2].data, other[2].data)


def test_rejects_changed_measured_holdout(canonical_110):
    frame = canonical_110.data.copy()
    frame.loc[frame["ds"] == "2023-09-01", "microVolt"] += 0.5
    with pytest.raises(AssertionError, match="Frozen measured holdout values changed"):
        runner.verify_preflight(BatteryDataset(frame))


def test_rejects_unobserved_holdout(canonical_110):
    frame = canonical_110.data.copy()
    mask = frame["ds"] == "2023-09-01"
    frame.loc[mask, "microVolt"] = np.nan
    frame.loc[mask, "is_observed"] = False
    with pytest.raises(ValueError, match="unmeasured targets"):
        runner.verify_preflight(BatteryDataset(frame))


def test_rejects_provenance_mismatch(canonical_110):
    frame = canonical_110.data.copy()
    frame.loc[frame["ds"] == "2002-09-01", "is_observed"] = True
    with pytest.raises(ValueError, match="provenance disagrees"):
        runner.verify_preflight(BatteryDataset(frame))


def test_rejects_duplicate_dates(canonical_110):
    frame = canonical_110.data.copy()
    frame.loc[1, "ds"] = frame.loc[0, "ds"]
    with pytest.raises(ValueError, match="sorted and unique"):
        runner.verify_preflight(BatteryDataset(frame))


def test_rejects_missing_canonical_quarter(canonical_110):
    frame = canonical_110.data.loc[canonical_110.data["ds"] != "2002-09-01"].copy()
    with pytest.raises(ValueError, match="complete and quarterly"):
        runner.verify_preflight(BatteryDataset(frame))


def test_frozen_spec_names_and_provenance():
    specs = runner.create_transferred_specs()
    assert tuple(spec.name for spec in specs) == runner.EXPECTED_MODELS
    manifest = runner.preflight_manifest(specs)
    assert manifest["optimized_for_dataset_110"] is False
    assert manifest["benchmark_executed"] is False
    assert manifest["configuration_source_battery"] == "732B-5610410"
    assert manifest["neural_benchmark_seed"] == 42
    assert manifest["holdout_steps"] == 5
