
"""M17-B2-G4: opt-in accuracy profile integration contracts."""

import pytest

from zenerestimation.optimization.evaluator import (
    ForecastAlignmentError,
    OptimizationEvaluator,
)
from zenerestimation.optimization.validation import (
    ExpandingWindowSplitter,
)

from tests.test_m17_b1_g4_evaluator_integration import (
    CountingForecaster,
    evaluate,
    make_dataset,
)


def make_evaluator(
    *,
    profile=False,
    diagnostics=False,
    strict=False,
    metric="rmse",
):
    return OptimizationEvaluator(
        ExpandingWindowSplitter(
            folds=3,
            validation_steps=2,
        ),
        metric=metric,
        require_verified_alignment=strict,
        record_accuracy_diagnostics=diagnostics,
        record_accuracy_profile=profile,
        accuracy_dataset_id="synthetic_b2_g4",
    )


def test_g4_profile_disabled_by_default():
    result = evaluate(make_evaluator())

    assert "accuracy_profile" not in result.metadata
    assert "accuracy_diagnostics" not in result.metadata


def test_g4_profile_opt_in_creates_both_reports():
    result = evaluate(make_evaluator(profile=True))

    assert "accuracy_profile" in result.metadata
    assert "accuracy_diagnostics" in result.metadata

    profile = result.metadata["accuracy_profile"]
    diagnostics = result.metadata["accuracy_diagnostics"]

    assert profile["profile_version"] == "m17_b2_v1"
    assert profile["overall"]["n"] == 6
    assert profile["first_step"]["n"] == 3

    assert profile["overall"]["rmse_microVolt"] == pytest.approx(
        diagnostics["summary"]["rmse_microVolt"]
    )

    assert profile["provenance"]["dataset"] == "synthetic_b2_g4"
    assert profile["provenance"]["scope"] == "internal_expanding_window"


def test_g4_b1_only_remains_unchanged():
    result = evaluate(make_evaluator(diagnostics=True))

    assert "accuracy_diagnostics" in result.metadata
    assert "accuracy_profile" not in result.metadata


def test_g4_both_flags_produce_single_profile():
    result = evaluate(
        make_evaluator(
            profile=True,
            diagnostics=True,
        )
    )

    assert "accuracy_profile" in result.metadata
    assert "accuracy_diagnostics" in result.metadata


@pytest.mark.parametrize("metric", ["rmse", "mae", "mape"])
def test_g4_profile_does_not_change_scoring(metric):
    baseline = evaluate(make_evaluator(metric=metric))
    enabled = evaluate(
        make_evaluator(profile=True, metric=metric)
    )

    assert enabled.score == pytest.approx(baseline.score)
    assert enabled.metrics == pytest.approx(baseline.metrics)

    for key, value in baseline.metadata.items():
        assert enabled.metadata[key] == value

    assert set(enabled.metadata) == (
        set(baseline.metadata)
        | {"accuracy_diagnostics", "accuracy_profile"}
    )


def test_g4_profile_does_not_add_fit_or_predict_calls():
    CountingForecaster.fit_calls = 0
    CountingForecaster.predict_calls = 0

    evaluate(make_evaluator(profile=True))

    assert CountingForecaster.fit_calls == 3
    assert CountingForecaster.predict_calls == 3


def test_g4_profile_preserves_verified_alignment():
    result = evaluate(make_evaluator(profile=True))

    profile = result.metadata["accuracy_profile"]

    assert profile["alignment_summary"]["fully_verified"] is True
    assert profile["alignment_summary"]["verified_rows"] == 6


def test_g4_profile_preserves_positional_alignment():
    result = evaluate(
        make_evaluator(profile=True),
        with_dates=False,
    )

    profile = result.metadata["accuracy_profile"]

    assert profile["alignment_summary"]["fully_verified"] is False
    assert profile["alignment_summary"]["positional_only_rows"] == 6


def test_g4_strict_alignment_remains_independent():
    evaluator = make_evaluator(
        profile=True,
        strict=True,
    )

    with pytest.raises(
        ForecastAlignmentError,
        match="positional_only",
    ):
        evaluate(evaluator, with_dates=False)


def test_g4_rejects_nonboolean_profile_flag():
    with pytest.raises(
        TypeError,
        match="record_accuracy_profile",
    ):
        make_evaluator(profile="yes")


def test_g4_profile_does_not_mutate_dataset():
    dataset = make_dataset()
    original = dataset.data.copy(deep=True)

    evaluator = make_evaluator(profile=True)
    evaluator.evaluate(
        dataset,
        CountingForecaster,
        {"offset": 0.0, "with_dates": True},
    )

    import pandas as pd

    pd.testing.assert_frame_equal(dataset.data, original)
