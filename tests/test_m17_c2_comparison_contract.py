
"""M17-C2: contract-first tests for scientific accuracy comparisons."""

from copy import deepcopy

import pandas as pd
import pytest

from zenerestimation.evaluation.accuracy_diagnostics import (
    build_accuracy_diagnostics,
)
from zenerestimation.evaluation.accuracy_profile import (
    build_accuracy_profile,
)
from zenerestimation.evaluation.accuracy_comparison import (
    compare_accuracy_profiles,
)


def make_profile(
    model,
    *,
    dataset="synthetic_m17_c",
    scope="internal_expanding_window",
    offset=0.0,
    dates=None,
    actual=None,
    alignment="verified",
):
    if dates is None:
        dates = [
            ["2022-03-01", "2022-06-01"],
            ["2023-03-01", "2023-06-01"],
        ]

    if actual is None:
        actual = [
            [10.0, 11.0],
            [20.0, 21.0],
        ]

    folds = []

    for index, (fold_dates, values) in enumerate(
        zip(dates, actual)
    ):
        predictions = [
            value + offset + (1.0 if step == 0 else -1.0)
            for step, value in enumerate(values)
        ]

        folds.append(
            {
                "fold_index": index,
                "dates": pd.to_datetime(fold_dates),
                "actual": values,
                "predicted": predictions,
                "alignment_status": alignment,
            }
        )

    diagnostics = build_accuracy_diagnostics(
        folds,
        dataset=dataset,
        model=model,
        scope=scope,
    )

    return build_accuracy_profile(diagnostics)


def test_matched_profiles_are_directly_comparable():
    profiles = {
        "ModelA": make_profile("ModelA", offset=0.0),
        "ModelB": make_profile("ModelB", offset=0.5),
    }

    result = compare_accuracy_profiles(profiles)

    assert result["comparison_version"] == "m17_c_v1"
    assert result["comparability"]["status"] == "directly_comparable"
    assert result["comparability"]["reasons"] == []
    assert result["ranking"] is None


def test_different_datasets_are_descriptive_only():
    profiles = {
        "ModelA": make_profile("ModelA", dataset="battery_410"),
        "ModelB": make_profile("ModelB", dataset="battery_110"),
    }

    result = compare_accuracy_profiles(profiles)

    assert result["comparability"]["status"] == "descriptive_only"
    assert result["ranking"] is None
    assert result["comparability"]["reasons"]


def test_different_scopes_are_descriptive_only():
    profiles = {
        "ModelA": make_profile(
            "ModelA", scope="internal_expanding_window"
        ),
        "ModelB": make_profile(
            "ModelB", scope="external_holdout"
        ),
    }

    result = compare_accuracy_profiles(profiles)

    assert result["comparability"]["status"] == "descriptive_only"
    assert result["ranking"] is None


def test_different_observation_dates_are_descriptive_only():
    profiles = {
        "ModelA": make_profile("ModelA"),
        "ModelB": make_profile(
            "ModelB",
            dates=[
                ["2022-03-01", "2022-06-01"],
                ["2024-03-01", "2024-06-01"],
            ],
        ),
    }

    result = compare_accuracy_profiles(profiles)

    assert result["comparability"]["status"] == "descriptive_only"
    assert result["ranking"] is None


def test_conflicting_measurements_are_rejected():
    profiles = {
        "ModelA": make_profile("ModelA"),
        "ModelB": make_profile(
            "ModelB",
            actual=[
                [10.0, 11.0],
                [20.0, 99.0],
            ],
        ),
    }

    with pytest.raises(ValueError, match="measurement"):
        compare_accuracy_profiles(profiles)


def test_positional_alignment_prevents_direct_comparison():
    profiles = {
        "ModelA": make_profile("ModelA"),
        "ModelB": make_profile(
            "ModelB",
            alignment="positional_only",
        ),
    }

    result = compare_accuracy_profiles(profiles)

    assert result["comparability"]["status"] == "descriptive_only"
    assert result["ranking"] is None


def test_model_identity_mismatch_is_rejected():
    profiles = {
        "ModelA": make_profile("DifferentModel"),
        "ModelB": make_profile("ModelB"),
    }

    with pytest.raises(ValueError, match="model"):
        compare_accuracy_profiles(profiles)


def test_comparison_does_not_mutate_profiles():
    profiles = {
        "ModelA": make_profile("ModelA"),
        "ModelB": make_profile("ModelB", offset=0.5),
    }

    before = deepcopy(profiles)

    compare_accuracy_profiles(profiles)

    assert profiles == before


def test_comparison_preserves_signed_metrics():
    profiles = {
        "ModelA": make_profile("ModelA"),
        "ModelB": make_profile("ModelB", offset=0.5),
    }

    result = compare_accuracy_profiles(profiles)

    for model in profiles:
        assert (
            result["profiles"][model]["overall"]["bias_microVolt"]
            == pytest.approx(
                profiles[model]["overall"]["bias_microVolt"]
            )
        )


def test_rejects_invalid_profile_version():
    profiles = {
        "ModelA": make_profile("ModelA"),
        "ModelB": make_profile("ModelB"),
    }

    profiles["ModelB"]["profile_version"] = "unknown_version"

    with pytest.raises(ValueError, match="version"):
        compare_accuracy_profiles(profiles)



# ============================================================
# M17-C2-G2: scientific integrity hardening
# ============================================================


def _valid_pair():
    return {
        "ModelA": make_profile("ModelA"),
        "ModelB": make_profile("ModelB", offset=0.5),
    }


def test_g2_rejects_tampered_overall_rmse():
    profiles = _valid_pair()

    profiles["ModelA"]["overall"]["rmse_microVolt"] += 5.0

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)


def test_g2_rejects_tampered_signed_bias():
    profiles = _valid_pair()

    profiles["ModelB"]["overall"]["bias_microVolt"] *= -1
    profiles["ModelB"]["overall"]["bias_microVolt"] += 3.0

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)


def test_g2_rejects_inconsistent_observation_error():
    profiles = _valid_pair()

    profiles["ModelA"]["by_date"][0]["error_microVolt"] = 999.0

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)


def test_g2_rejects_falsified_alignment_summary():
    profiles = _valid_pair()

    profiles["ModelB"]["by_date"][0]["alignment_status"] = (
        "positional_only"
    )

    # The summary falsely continues to claim full verification.
    assert profiles["ModelB"]["alignment_summary"]["fully_verified"]

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)


def test_g2_rejects_duplicate_reference_observation():
    profiles = _valid_pair()

    profiles["ModelA"]["by_date"][1]["fold_index"] = 0
    profiles["ModelA"]["by_date"][1]["horizon_step"] = 1
    profiles["ModelA"]["by_date"][1]["date"] = (
        profiles["ModelA"]["by_date"][0]["date"]
    )

    with pytest.raises(ValueError, match="Duplicate|duplicate"):
        compare_accuracy_profiles(profiles)


def test_g2_rejects_nonconsecutive_horizon():
    profiles = _valid_pair()

    profiles["ModelB"]["by_date"][1]["horizon_step"] = 4

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)


def test_g2_preserves_inputs_when_validation_fails():
    profiles = _valid_pair()

    profiles["ModelA"]["overall"]["rmse_microVolt"] = 999.0

    original = deepcopy(profiles)

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)

    assert profiles == original



# ============================================================
# M17-C2-G4: boundary and adversarial comparison tests
# ============================================================


def test_g4_rejects_tampered_first_step_metrics():
    profiles = _valid_pair()

    profiles["ModelA"]["first_step"]["mae_microVolt"] += 2.0

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)


def test_g4_rejects_tampered_horizon_metrics():
    profiles = _valid_pair()

    profiles["ModelB"]["by_horizon"][0]["rmse_microVolt"] += 1.0

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)


def test_g4_rejects_tampered_fold_stability():
    profiles = _valid_pair()

    profiles["ModelA"]["fold_stability"][
        "rmse_mean_microVolt"
    ] += 1.0

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)


def test_g4_rejects_missing_profile_section():
    profiles = _valid_pair()

    del profiles["ModelA"]["first_step"]

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)


def test_g4_rejects_nonfinite_prediction():
    profiles = _valid_pair()

    profiles["ModelB"]["by_date"][0][
        "predicted_microVolt"
    ] = float("nan")

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)


def test_g4_rejects_noncanonical_observation_date():
    profiles = _valid_pair()

    profiles["ModelA"]["by_date"][0]["date"] = "2022-3-1"

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)


def test_g4_rejects_falsified_alignment_counts():
    profiles = _valid_pair()

    profiles["ModelA"]["alignment_summary"][
        "verified_rows"
    ] = 3

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)


def test_g4_three_matched_models_are_comparable():
    profiles = {
        "ModelA": make_profile("ModelA"),
        "ModelB": make_profile("ModelB", offset=0.5),
        "ModelC": make_profile("ModelC", offset=-0.5),
    }

    result = compare_accuracy_profiles(profiles)

    assert result["comparability"]["status"] == (
        "directly_comparable"
    )
    assert result["ranking"] is None
    assert result["models"] == list(profiles)


def test_g4_third_model_with_different_dates_is_descriptive():
    profiles = {
        "ModelA": make_profile("ModelA"),
        "ModelB": make_profile("ModelB", offset=0.5),
        "ModelC": make_profile(
            "ModelC",
            dates=[
                ["2022-03-01", "2022-06-01"],
                ["2024-03-01", "2024-06-01"],
            ],
        ),
    }

    result = compare_accuracy_profiles(profiles)

    assert result["comparability"]["status"] == (
        "descriptive_only"
    )
    assert result["ranking"] is None
    assert result["comparability"]["reasons"]


def test_g4_rejects_unsupported_comparison_policy():
    profiles = _valid_pair()

    with pytest.raises(ValueError, match="policy"):
        compare_accuracy_profiles(
            profiles,
            comparison_policy="permissive",
        )


def test_g4_rejects_single_profile():
    profiles = {
        "ModelA": make_profile("ModelA"),
    }

    with pytest.raises(ValueError, match="two"):
        compare_accuracy_profiles(profiles)


def test_g4_rejects_extra_profile_field():
    profiles = _valid_pair()

    profiles["ModelA"]["unverified_accuracy"] = {
        "rmse_microVolt": 0.0,
    }

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)



# ============================================================
# M17-C2-G5: numerical and order-independence contract
# ============================================================


def test_g5_rejects_tiny_observation_error_corruption():
    profiles = _valid_pair()

    profiles["ModelA"]["by_date"][0][
        "error_microVolt"
    ] += 5e-13

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)


def test_g5_rejects_tiny_prediction_corruption():
    profiles = _valid_pair()

    profiles["ModelB"]["by_date"][0][
        "predicted_microVolt"
    ] += 5e-13

    with pytest.raises(ValueError):
        compare_accuracy_profiles(profiles)


def test_g5_comparability_status_is_order_independent():
    profiles = {
        "ModelA": make_profile("ModelA"),
        "ModelB": make_profile(
            "ModelB",
            alignment="positional_only",
        ),
        "ModelC": make_profile("ModelC", offset=0.5),
    }

    forward = compare_accuracy_profiles(profiles)

    reverse = compare_accuracy_profiles(
        dict(reversed(list(profiles.items())))
    )

    assert forward["comparability"]["status"] == (
        reverse["comparability"]["status"]
    )
    assert forward["comparability"]["status"] == (
        "descriptive_only"
    )



def test_g5_comparability_reasons_are_order_independent():
    profiles = {
        "ModelA": make_profile("ModelA"),
        "ModelB": make_profile(
            "ModelB",
            alignment="positional_only",
        ),
        "ModelC": make_profile(
            "ModelC",
            dates=[
                ["2022-03-01", "2022-06-01"],
                ["2024-03-01", "2024-06-01"],
            ],
        ),
    }

    forward = compare_accuracy_profiles(profiles)

    reverse = compare_accuracy_profiles(
        dict(reversed(list(profiles.items())))
    )

    assert (
        forward["comparability"]
        == reverse["comparability"]
    )

    assert (
        forward["comparability"]["status"]
        == "descriptive_only"
    )

    assert forward["ranking"] is None
    assert reverse["ranking"] is None
