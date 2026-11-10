
"""M17-C4: read-only acceptance presentation contract."""

from copy import deepcopy

import pandas as pd
import pytest

from zenerestimation.evaluation.accuracy_diagnostics import (
    build_accuracy_diagnostics,
)
from zenerestimation.evaluation.accuracy_profile import (
    build_accuracy_profile,
)
from zenerestimation.visualization.m17_accuracy_acceptance_view import (
    build_accuracy_acceptance_view,
)


def make_profile(
    model,
    *,
    error=0.5,
    dataset="synthetic_m17_c4",
    alignment="verified",
):
    diagnostics = build_accuracy_diagnostics(
        [
            {
                "fold_index": 0,
                "dates": pd.to_datetime(
                    ["2022-03-01", "2022-06-01"]
                ),
                "actual": [10.0, 11.0],
                "predicted": [
                    10.0 + error,
                    11.0 + error,
                ],
                "alignment_status": alignment,
            }
        ],
        dataset=dataset,
        model=model,
        scope="external_holdout",
    )

    return build_accuracy_profile(diagnostics)


def make_profiles():
    return {
        "ModelA": make_profile("ModelA", error=0.25),
        "ModelB": make_profile("ModelB", error=0.75),
        "ModelC": make_profile("ModelC", error=0.5),
    }


def test_c4_acceptance_view_pass_fail_boundary():
    table = build_accuracy_acceptance_view(
        make_profiles(),
        models=["ModelA", "ModelB", "ModelC"],
    )

    assert list(table["model"]) == [
        "ModelA", "ModelB", "ModelC"
    ]
    assert list(table["status"]) == [
        "pass", "fail", "pass"
    ]
    assert set(table["threshold_microVolt"]) == {0.5}
    assert set(table["comparison_status"]) == {
        "directly_comparable"
    }


def test_c4_acceptance_view_preserves_selected_order():
    table = build_accuracy_acceptance_view(
        make_profiles(),
        models=["ModelC", "ModelA"],
    )

    assert list(table["model"]) == ["ModelC", "ModelA"]


def test_c4_acceptance_view_custom_threshold():
    table = build_accuracy_acceptance_view(
        make_profiles(),
        models=["ModelA", "ModelB"],
        threshold_microVolt=1.0,
    )

    assert list(table["status"]) == ["pass", "pass"]


def test_c4_acceptance_view_rejects_unknown_model():
    with pytest.raises(ValueError):
        build_accuracy_acceptance_view(
            make_profiles(),
            models=["Unknown"],
        )


def test_c4_acceptance_view_rejects_duplicate_selection():
    with pytest.raises(ValueError):
        build_accuracy_acceptance_view(
            make_profiles(),
            models=["ModelA", "ModelA"],
        )


def test_c4_acceptance_view_does_not_mutate_profiles():
    profiles = make_profiles()
    original = deepcopy(profiles)

    build_accuracy_acceptance_view(
        profiles,
        models=["ModelA", "ModelB"],
    )

    assert profiles == original


def test_c4_acceptance_view_rejects_corrupted_profile():
    profiles = make_profiles()

    profiles["ModelA"]["by_date"][0][
        "error_microVolt"
    ] += 5e-13

    with pytest.raises(ValueError):
        build_accuracy_acceptance_view(
            profiles,
            models=["ModelA"],
        )


def test_c4_acceptance_view_full_collection_governs_eligibility():
    profiles = make_profiles()
    profiles["ModelC"] = make_profile(
        "ModelC",
        error=0.5,
        alignment="positional_only",
    )

    table = build_accuracy_acceptance_view(
        profiles,
        models=["ModelA", "ModelB"],
    )

    assert set(table["status"]) == {"not_eligible"}
    assert set(table["comparison_status"]) == {
        "descriptive_only"
    }


def test_c4_acceptance_view_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        build_accuracy_acceptance_view(
            make_profiles(),
            models=["ModelA"],
            threshold_microVolt=-0.1,
        )
