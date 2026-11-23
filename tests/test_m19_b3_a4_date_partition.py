"""M19-B3-A4: date-defined holdout partition hardening tests."""
from copy import deepcopy

import pandas as pd
import pytest

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION,
)
from zenerestimation.assessment.holdout_isolation import (
    ISOLATION_VERSION, isolate_declared_holdout, require_isolated_training_dataset,
)
from zenerestimation.data.dataset import BatteryDataset


@pytest.fixture
def declaration():
    return AssessmentDeclaration(
        contract_version=CONTRACT_VERSION,
        assessment_id="a4-date-boundary", assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id="410", development_cutoff="2023-12-01",
        forecast_dates=("2024-03-01", "2024-06-01"),
    )


@pytest.fixture
def frame():
    return pd.DataFrame({
        "ds": pd.to_datetime(["2023-03-01", "2023-06-01", "2023-09-01",
                              "2023-12-01", "2024-03-01", "2024-06-01"]),
        "microVolt": [27., 28., 29., 29.55, 29.9, 30.506],
    })


def isolate(frame, declaration):
    return isolate_declared_holdout(BatteryDataset(frame), declaration=declaration)


def test_explicit_date_boundary_and_evidence(frame, declaration):
    result = isolate(frame, declaration)
    assert result.evidence["isolation_version"] == ISOLATION_VERSION
    assert result.evidence["status"] == "PARTITION_VERIFIED"
    assert result.evidence["scientific_independence"] == "not_established"
    assert result.development.data["ds"].dt.strftime("%Y-%m-%d").tolist() == [
        "2023-03-01", "2023-06-01", "2023-09-01", "2023-12-01"]
    assert result.holdout["ds"].dt.strftime("%Y-%m-%d").tolist() == list(declaration.forecast_dates)
    require_isolated_training_dataset(result.development, isolated=result)


def test_out_of_order_dates_partition_by_date_not_position(frame, declaration):
    shuffled = frame.iloc[[5, 0, 4, 2, 3, 1]].reset_index(drop=True)
    result = isolate(shuffled, declaration)
    assert result.development.data["microVolt"].tolist() == [27., 28., 29., 29.55]
    assert result.holdout["microVolt"].tolist() == [29.9, 30.506]


def test_extra_future_row_rejected_without_shift(frame, declaration):
    extended = pd.concat([frame, pd.DataFrame({
        "ds": [pd.Timestamp("2024-09-01")], "microVolt": [31.]
    })], ignore_index=True)
    with pytest.raises(ValueError, match="holdout calendar"):
        isolate(extended, declaration)


def test_missing_declared_holdout_row_rejected(frame, declaration):
    shortened = frame.loc[frame["ds"] != pd.Timestamp("2024-03-01")]
    with pytest.raises(ValueError, match="holdout calendar"):
        isolate(shortened, declaration)


def test_unexpected_between_cutoff_and_first_forecast_rejected(frame, declaration):
    extra = pd.DataFrame({"ds": [pd.Timestamp("2024-01-01")], "microVolt": [29.7]})
    with pytest.raises(ValueError, match="holdout calendar"):
        isolate(pd.concat([frame, extra], ignore_index=True), declaration)


def test_missing_cutoff_row_rejected(frame, declaration):
    changed = frame.loc[frame["ds"] != pd.Timestamp("2023-12-01")]
    with pytest.raises(ValueError, match="development cutoff"):
        isolate(changed, declaration)


def test_holdout_value_changes_cannot_change_development(frame, declaration):
    first = isolate(frame, declaration)
    changed = frame.copy(deep=True)
    changed.loc[changed["ds"] > pd.Timestamp(declaration.development_cutoff), "microVolt"] = [999., -999.]
    second = isolate(changed, declaration)
    pd.testing.assert_frame_equal(first.development.data, second.development.data)
    assert first.evidence == second.evidence
    assert first.holdout["microVolt"].tolist() != second.holdout["microVolt"].tolist()


def test_duplicate_development_date_rejected(frame, declaration):
    changed = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        isolate(changed, declaration)


def test_training_guard_still_rejects_complete_input(frame, declaration):
    result = isolate(frame, declaration)
    with pytest.raises(ValueError, match="training input differs"):
        require_isolated_training_dataset(BatteryDataset(frame), isolated=result)


def test_result_is_detached_from_input(frame, declaration):
    result = isolate(frame, declaration)
    frame.loc[0, "microVolt"] = 1234.
    frame.loc[5, "microVolt"] = 4321.
    assert result.development.data["microVolt"].iloc[0] == 27.
    assert result.holdout["microVolt"].iloc[-1] == 30.506
