
"""M17-B2-G3: scientific consistency and evidence integrity."""

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


def make_valid_report():
    folds = [
        {
            "fold_index": 0,
            "dates": pd.to_datetime(
                ["2022-03-01", "2022-06-01"]
            ),
            "actual": np.array([10.0, 11.0]),
            "predicted": np.array([11.0, 13.0]),
            "alignment_status": "verified",
        },
        {
            "fold_index": 1,
            "dates": pd.to_datetime(
                ["2023-03-01", "2023-06-01"]
            ),
            "actual": np.array([20.0, 21.0]),
            "predicted": np.array([17.0, 25.0]),
            "alignment_status": "positional_only",
        },
    ]

    return build_accuracy_diagnostics(
        folds,
        dataset="synthetic_b2_g3",
        model="SyntheticForecaster",
        scope="internal_expanding_window",
    )


def test_g3_accepts_valid_b1_evidence():
    report = make_valid_report()
    profile = build_accuracy_profile(report)

    assert profile["overall"]["n"] == 4
    assert profile["alignment_summary"]["verified_rows"] == 2
    assert profile["alignment_summary"]["positional_only_rows"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("error_microVolt", 99.0),
        ("absolute_error_microVolt", 99.0),
        ("squared_error_microVolt2", 99.0),
    ],
)
def test_g3_rejects_inconsistent_observation_errors(field, value):
    report = make_valid_report()
    report["rows"][0][field] = value

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


@pytest.mark.parametrize(
    "field",
    [
        "rmse_microVolt",
        "mae_microVolt",
        "bias_microVolt",
        "n",
    ],
)
def test_g3_rejects_inconsistent_overall_metrics(field):
    report = make_valid_report()
    report["summary"][field] += 1

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_rejects_inconsistent_horizon_metrics():
    report = make_valid_report()
    report["horizon_metrics"][0]["rmse_microVolt"] += 1.0

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_rejects_inconsistent_fold_metrics():
    report = make_valid_report()
    report["fold_metrics"][0]["mae_microVolt"] += 1.0

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_rejects_inconsistent_fold_stability():
    report = make_valid_report()
    report["summary"]["fold_stability"][
        "worst_fold_rmse_microVolt"
    ] += 1.0

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_rejects_inconsistent_alignment_provenance():
    report = make_valid_report()
    report["provenance"]["alignment_statuses"] = ["verified"]

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_g3_rejects_nonfinite_observation_values(value):
    report = make_valid_report()
    report["rows"][0]["predicted_microVolt"] = value

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_rejects_duplicate_observation_identity():
    report = make_valid_report()
    duplicate = deepcopy(report["rows"][0])
    report["rows"].append(duplicate)

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_rejects_missing_horizon_evidence():
    report = make_valid_report()
    report["horizon_metrics"] = []

    with pytest.raises(ValueError):
        build_accuracy_profile(report)


def test_g3_does_not_modify_valid_source():
    report = make_valid_report()
    original = deepcopy(report)

    build_accuracy_profile(report)

    assert report == original
