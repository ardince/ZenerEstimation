
"""M17-B2-G5: external measured holdout integration and isolation."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from zenerestimation.visualization.optimized_neural_comparison import (
    FUTURE,
    HOLDOUT,
    MODELS,
    build_external_holdout_accuracy_profiles,
    load_optimized_neural_forecasts,
)

from tests.test_16_4_neural_values_dashboard import _fixture


@pytest.fixture
def evidence(tmp_path):
    root = _fixture(tmp_path)
    manifest, holdout, future = load_optimized_neural_forecasts(
        root, "test_run"
    )
    return manifest, holdout, future


def build(manifest, holdout):
    return build_external_holdout_accuracy_profiles(
        manifest, holdout
    )


def test_g5_creates_one_profile_per_model(evidence):
    manifest, holdout, _ = evidence

    profiles = build(manifest, holdout)

    assert set(profiles) == set(MODELS)

    for model in MODELS:
        profile = profiles[model]
        assert profile["profile_version"] == "m17_b2_v1"
        assert profile["overall"]["n"] == 6
        assert profile["first_step"]["n"] == 1
        assert len(profile["by_horizon"]) == 6
        assert len(profile["by_date"]) == 6


def test_g5_external_provenance_is_explicit(evidence):
    manifest, holdout, _ = evidence

    profiles = build(manifest, holdout)

    for model, profile in profiles.items():
        provenance = profile["provenance"]

        assert provenance["dataset"] == manifest["battery"]
        assert provenance["model"] == model
        assert provenance["scope"] == "external_holdout"
        assert provenance["units"] == "microVolt"


def test_g5_metrics_match_existing_manifest(evidence):
    manifest, holdout, _ = evidence

    profiles = build(manifest, holdout)

    for model in MODELS:
        profile = profiles[model]
        saved = manifest["external_holdout_metrics"][model]

        assert profile["overall"]["rmse_microVolt"] == pytest.approx(
            saved["rmse"], abs=1e-8
        )
        assert profile["overall"]["mae_microVolt"] == pytest.approx(
            saved["mae"], abs=1e-8
        )
        assert profile["first_step"]["mean_error_microVolt"] == pytest.approx(
            saved["first_step_signed_error_pred_minus_actual"],
            abs=1e-8,
        )
        assert profile["first_step"]["mae_microVolt"] == pytest.approx(
            saved["first_step_absolute_error"],
            abs=1e-8,
        )


def test_g5_signed_bias_matches_predictions_minus_actual(evidence):
    manifest, holdout, _ = evidence

    profiles = build(manifest, holdout)

    for model in MODELS:
        rows = holdout.loc[holdout["model"].eq(model)]
        expected = float(
            (
                rows["predicted_microvolt"]
                - rows["actual_microvolt"]
            ).mean()
        )

        assert profiles[model]["overall"]["bias_microVolt"] == pytest.approx(
            expected
        )


def test_g5_all_external_dates_are_verified(evidence):
    manifest, holdout, _ = evidence

    profiles = build(manifest, holdout)

    for profile in profiles.values():
        alignment = profile["alignment_summary"]

        assert alignment["total_rows"] == 6
        assert alignment["verified_rows"] == 6
        assert alignment["positional_only_rows"] == 0
        assert alignment["fully_verified"] is True

        assert [row["date"] for row in profile["by_date"]] == (
            HOLDOUT.strftime("%Y-%m-%d").tolist()
        )


def test_g5_rejects_shifted_holdout_dates(evidence):
    manifest, holdout, _ = evidence
    corrupted = holdout.copy(deep=True)

    corrupted["date"] = (
        pd.to_datetime(corrupted["date"])
        + pd.DateOffset(months=3)
    )

    with pytest.raises(ValueError):
        build(manifest, corrupted)


def test_g5_rejects_duplicate_model_quarter(evidence):
    manifest, holdout, _ = evidence
    corrupted = holdout.copy(deep=True)

    corrupted.loc[1, "date"] = corrupted.loc[0, "date"]

    with pytest.raises(ValueError):
        build(manifest, corrupted)


def test_g5_rejects_missing_measurement(evidence):
    manifest, holdout, _ = evidence
    corrupted = holdout.copy(deep=True)

    corrupted.loc[0, "actual_microvolt"] = np.nan

    with pytest.raises(ValueError):
        build(manifest, corrupted)


def test_g5_rejects_corrupted_signed_error(evidence):
    manifest, holdout, _ = evidence
    corrupted = holdout.copy(deep=True)

    corrupted.loc[0, "error_pred_minus_actual"] += 1.0

    with pytest.raises(ValueError):
        build(manifest, corrupted)


def test_g5_rejects_manifest_metric_mismatch(evidence):
    manifest, holdout, _ = evidence
    corrupted = deepcopy(manifest)

    corrupted["external_holdout_metrics"]["LSTM"]["rmse"] += 1.0

    with pytest.raises(ValueError):
        build(corrupted, holdout)


def test_g5_does_not_mutate_sources(evidence):
    manifest, holdout, _ = evidence

    original_manifest = deepcopy(manifest)
    original_holdout = holdout.copy(deep=True)

    build(manifest, holdout)

    assert manifest == original_manifest
    pd.testing.assert_frame_equal(holdout, original_holdout)


def test_g5_does_not_modify_internal_selection(evidence):
    manifest, holdout, _ = evidence

    original_selection = deepcopy(manifest["selected"])

    profiles = build(manifest, holdout)

    assert manifest["selected"] == original_selection
    assert set(profiles) == set(MODELS)


def test_g5_operational_predictions_remain_unscored(evidence):
    manifest, holdout, future = evidence

    profiles = build(manifest, holdout)

    future_dates = set(
        FUTURE.strftime("%Y-%m-%d").tolist()
    )

    for profile in profiles.values():
        dates = {row["date"] for row in profile["by_date"]}
        assert dates.isdisjoint(future_dates)

    assert len(future) == len(MODELS) * len(FUTURE)


def test_g5_rejects_incomplete_model_evidence(evidence):
    manifest, holdout, _ = evidence

    corrupted = holdout.loc[
        ~(
            holdout["model"].eq("LSTM")
            & holdout["date"].eq(HOLDOUT[0])
        )
    ].copy()

    with pytest.raises(ValueError):
        build(manifest, corrupted)
