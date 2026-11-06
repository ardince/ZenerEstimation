
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


def make_diagnostics(
    *,
    scope="internal_expanding_window",
    second_alignment="verified",
):
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
            "alignment_status": second_alignment,
        },
    ]

    return build_accuracy_diagnostics(
        folds,
        dataset="synthetic_b2_fixture",
        model="SyntheticForecaster",
        scope=scope,
    )


def test_g2_profile_has_complete_schema():
    profile = build_accuracy_profile(make_diagnostics())

    assert profile["profile_version"] == "m17_b2_v1"

    assert set(profile) == {
        "profile_version",
        "provenance",
        "overall",
        "first_step",
        "by_horizon",
        "by_date",
        "fold_stability",
        "alignment_summary",
    }


def test_g2_preserves_b1_metrics_and_provenance():
    diagnostics = make_diagnostics()
    profile = build_accuracy_profile(diagnostics)

    assert profile["provenance"] == diagnostics["provenance"]

    for key in (
        "n",
        "rmse_microVolt",
        "mae_microVolt",
        "bias_microVolt",
    ):
        assert profile["overall"][key] == pytest.approx(
            diagnostics["summary"][key]
        )

    assert (
        profile["fold_stability"]
        == diagnostics["summary"]["fold_stability"]
    )
    assert profile["by_horizon"] == diagnostics["horizon_metrics"]
    assert profile["by_date"] == diagnostics["rows"]


def test_g2_first_step_accuracy_is_mathematically_correct():
    profile = build_accuracy_profile(make_diagnostics())
    first = profile["first_step"]

    # First-step errors are +1 and -3 microvolts.
    assert first["n"] == 2
    assert first["mean_error_microVolt"] == pytest.approx(-1.0)
    assert first["mae_microVolt"] == pytest.approx(2.0)
    assert first["rmse_microVolt"] == pytest.approx(
        np.sqrt(5.0)
    )


def test_g2_alignment_summary_preserves_partial_evidence():
    diagnostics = make_diagnostics(
        second_alignment="positional_only"
    )
    profile = build_accuracy_profile(diagnostics)

    alignment = profile["alignment_summary"]

    assert alignment["verified_rows"] == 2
    assert alignment["positional_only_rows"] == 2
    assert alignment["total_rows"] == 4
    assert alignment["fully_verified"] is False

    assert profile["provenance"]["alignment_statuses"] == [
        "verified",
        "positional_only",
    ]


def test_g2_external_holdout_remains_separate():
    profile = build_accuracy_profile(
        make_diagnostics(scope="external_holdout")
    )

    assert profile["provenance"]["scope"] == "external_holdout"
    assert profile["overall"]["n"] == 4


def test_g2_does_not_mutate_source_diagnostics():
    diagnostics = make_diagnostics()
    original = deepcopy(diagnostics)

    profile = build_accuracy_profile(diagnostics)

    assert diagnostics == original

    # Mutating the returned profile must not change B1 evidence.
    profile["by_date"][0]["error_microVolt"] = 999.0
    assert diagnostics == original


def test_g2_rejects_operational_scope():
    diagnostics = make_diagnostics()
    diagnostics["provenance"]["scope"] = "operational_future"

    with pytest.raises(ValueError, match="scope"):
        build_accuracy_profile(diagnostics)


def test_g2_rejects_missing_observation_evidence():
    diagnostics = make_diagnostics()
    diagnostics["rows"] = []

    with pytest.raises(ValueError):
        build_accuracy_profile(diagnostics)
