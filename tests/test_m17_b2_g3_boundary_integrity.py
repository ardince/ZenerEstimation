
"""M17-B2-G3: additional structural and numerical boundary tests."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from zenerestimation.evaluation.accuracy_diagnostics import (
    build_accuracy_diagnostics,
)
from zenerestimation.evaluation.accuracy_profile import (
    build_accuracy_profile,
)


@pytest.fixture
def valid_report():
    folds = [
        {
            "fold_index": 0,
            "dates": pd.to_datetime(
                ["2022-03-01", "2022-06-01", "2022-09-01"]
            ),
            "actual": np.array([10.0, 11.0, 12.0]),
            "predicted": np.array([11.0, 12.0, 14.0]),
            "alignment_status": "verified",
        },
        {
            "fold_index": 1,
            "dates": pd.to_datetime(
                ["2023-03-01", "2023-06-01", "2023-09-01"]
            ),
            "actual": np.array([20.0, 21.0, 22.0]),
            "predicted": np.array([19.0, 23.0, 25.0]),
            "alignment_status": "positional_only",
        },
    ]

    return build_accuracy_diagnostics(
        folds,
        dataset="synthetic_b2_boundary",
        model="SyntheticForecaster",
        scope="internal_expanding_window",
    )


def test_g3_boundary_accepts_valid_three_step_report(valid_report):
    profile = build_accuracy_profile(valid_report)

    assert profile["overall"]["n"] == 6
    assert len(profile["by_horizon"]) == 3
    assert profile["first_step"]["n"] == 2


def test_g3_boundary_rejects_nonconsecutive_horizon(valid_report):
    report = deepcopy(valid_report)
    report["rows"][1]["horizon_step"] = 4

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_boundary_rejects_reordered_horizon(valid_report):
    report = deepcopy(valid_report)
    report["rows"][0], report["rows"][1] = (
        report["rows"][1],
        report["rows"][0],
    )

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_boundary_rejects_duplicate_dates(valid_report):
    report = deepcopy(valid_report)
    report["rows"][1]["date"] = report["rows"][0]["date"]

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_boundary_rejects_decreasing_dates(valid_report):
    report = deepcopy(valid_report)
    report["rows"][1]["date"] = "2021-06-01"

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_boundary_rejects_mixed_alignment_within_fold(valid_report):
    report = deepcopy(valid_report)
    report["rows"][1]["alignment_status"] = "positional_only"

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


@pytest.mark.parametrize(
    "field",
    [
        "fold_index",
        "horizon_step",
    ],
)
def test_g3_boundary_rejects_boolean_indices(valid_report, field):
    report = deepcopy(valid_report)
    report["rows"][0][field] = True

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


@pytest.mark.parametrize(
    "field",
    [
        "actual_microVolt",
        "predicted_microVolt",
        "error_microVolt",
        "absolute_error_microVolt",
        "squared_error_microVolt2",
    ],
)
def test_g3_boundary_rejects_boolean_numeric_fields(valid_report, field):
    report = deepcopy(valid_report)
    report["rows"][0][field] = True

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_boundary_rejects_missing_row_field(valid_report):
    report = deepcopy(valid_report)
    del report["rows"][0]["actual_microVolt"]

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_boundary_rejects_invalid_date(valid_report):
    report = deepcopy(valid_report)
    report["rows"][0]["date"] = "not-a-date"

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_boundary_rejects_noncanonical_date(valid_report):
    report = deepcopy(valid_report)
    report["rows"][0]["date"] = "2022-3-1"

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_boundary_rejects_missing_provenance(valid_report):
    report = deepcopy(valid_report)
    del report["provenance"]["dataset"]

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_boundary_rejects_wrong_units(valid_report):
    report = deepcopy(valid_report)
    report["provenance"]["units"] = "milliVolt"

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_boundary_rejects_invalid_scope(valid_report):
    report = deepcopy(valid_report)
    report["provenance"]["scope"] = "operational_future"

    with pytest.raises(ValueError, match="scope"):
        build_accuracy_profile(report)
