"""M19-B3-A2 data isolation and adversarial partition tests."""
from copy import deepcopy

import pandas as pd
import pytest

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION,
)
from zenerestimation.assessment.holdout_isolation import (
    isolate_declared_holdout, require_isolated_training_dataset,
)
from zenerestimation.data.dataset import BatteryDataset


@pytest.fixture
def declaration():
    return AssessmentDeclaration(
        contract_version=CONTRACT_VERSION,
        assessment_id="isolation-410", assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id="410", development_cutoff="2023-12-01",
        forecast_dates=("2024-03-01", "2024-06-01"),
    )


@pytest.fixture
def frame():
    return pd.DataFrame({
        "ds": pd.to_datetime(["2023-03-01", "2023-06-01", "2023-09-01",
                              "2023-12-01", "2024-03-01", "2024-06-01"]),
        "microVolt": [27.0, 28.0, 29.0, 29.55, 29.9, 30.506],
    })


def _split(frame, declaration):
    return isolate_declared_holdout(BatteryDataset(frame), declaration=declaration)


def test_declared_partition_and_guard(frame, declaration):
    isolated = _split(frame, declaration)
    assert isolated.evidence["status"] == "PARTITION_VERIFIED"
    assert isolated.evidence["scientific_independence"] == "not_established"
    assert len(isolated.development.data) == 4
    assert len(isolated.holdout) == 2
    require_isolated_training_dataset(isolated.development, isolated=isolated)


def test_holdout_perturbation_does_not_change_development(frame, declaration):
    original = _split(frame, declaration)
    altered = frame.copy(deep=True)
    altered.loc[altered["ds"] >= pd.Timestamp("2024-03-01"), "microVolt"] = [999., -999.]
    changed = _split(altered, declaration)
    pd.testing.assert_frame_equal(original.development.data, changed.development.data)
    assert original.holdout["microVolt"].tolist() != changed.holdout["microVolt"].tolist()
    assert original.evidence == changed.evidence


def test_unsorted_input_is_sorted_before_partition(frame, declaration):
    shuffled = frame.iloc[[5, 1, 3, 0, 4, 2]].reset_index(drop=True)
    isolated = _split(shuffled, declaration)
    assert isolated.development.data["ds"].is_monotonic_increasing
    assert isolated.holdout["ds"].is_monotonic_increasing


def test_full_dataset_training_is_rejected(frame, declaration):
    isolated = _split(frame, declaration)
    with pytest.raises(ValueError, match="training input differs"):
        require_isolated_training_dataset(BatteryDataset(frame), isolated=isolated)


def test_modified_development_training_is_rejected(frame, declaration):
    isolated = _split(frame, declaration)
    altered = isolated.development.data.copy(deep=True)
    altered.loc[0, "microVolt"] += 1.0
    with pytest.raises(ValueError, match="training input differs"):
        require_isolated_training_dataset(BatteryDataset(altered), isolated=isolated)


def test_duplicate_timestamp_rejected(frame, declaration):
    frame.loc[0, "ds"] = frame.loc[1, "ds"]
    with pytest.raises(ValueError, match="duplicate"):
        _split(frame, declaration)


def test_missing_holdout_timestamp_rejected(frame, declaration):
    frame.loc[5, "ds"] = pd.Timestamp("2024-09-01")
    with pytest.raises(ValueError, match="holdout calendar"):
        _split(frame, declaration)


def test_extra_future_timestamp_rejected(frame, declaration):
    extra = pd.DataFrame({"ds": [pd.Timestamp("2024-09-01")], "microVolt": [31.]})
    with pytest.raises(ValueError, match="holdout calendar"):
        _split(pd.concat([frame, extra], ignore_index=True), declaration)


def test_development_cutoff_mismatch_rejected(frame, declaration):
    frame.loc[3, "ds"] = pd.Timestamp("2023-10-01")
    with pytest.raises(ValueError, match="development cutoff"):
        _split(frame, declaration)


def test_missing_development_measurement_rejected(frame, declaration):
    frame.loc[0, "microVolt"] = float("nan")
    with pytest.raises(ValueError, match="missing measurements"):
        _split(frame, declaration)


def test_detached_partitions(frame, declaration):
    isolated = _split(frame, declaration)
    frame.loc[0, "microVolt"] = 777.
    frame.loc[5, "microVolt"] = 888.
    assert isolated.development.data["microVolt"].iloc[0] == 27.
    assert isolated.holdout["microVolt"].iloc[-1] == 30.506
