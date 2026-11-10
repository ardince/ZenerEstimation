
"""M17-C3: contract tests for read-only RMSE acceptance."""

from copy import deepcopy

import pandas as pd
import pytest
import math

from zenerestimation.evaluation.accuracy_diagnostics import (
    build_accuracy_diagnostics,
)
from zenerestimation.evaluation.accuracy_profile import (
    build_accuracy_profile,
)
from zenerestimation.evaluation.accuracy_comparison import (
    compare_accuracy_profiles,
)
from zenerestimation.evaluation.accuracy_acceptance import (
    evaluate_accuracy_threshold,
)


def make_profile(
    model,
    *,
    error=0.5,
    dataset="synthetic_m17_c3",
    scope="internal_expanding_window",
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
        scope=scope,
    )

    return build_accuracy_profile(diagnostics)


def make_comparison(
    *,
    error_a=0.4,
    error_b=0.7,
    dataset_b="synthetic_m17_c3",
    alignment_b="verified",
):
    return compare_accuracy_profiles(
        {
            "ModelA": make_profile(
                "ModelA",
                error=error_a,
            ),
            "ModelB": make_profile(
                "ModelB",
                error=error_b,
                dataset=dataset_b,
                alignment=alignment_b,
            ),
        }
    )


def test_c3_acceptance_pass_and_fail():
    comparison = make_comparison()

    result = evaluate_accuracy_threshold(comparison)

    assert result["acceptance_version"] == "m17_c3_v1"
    assert result["metric"] == "overall_rmse"
    assert result["threshold_microVolt"] == 0.5
    assert result["comparison_status"] == "directly_comparable"
    assert result["acceptance"]["ModelA"]["status"] == "pass"
    assert result["acceptance"]["ModelB"]["status"] == "fail"
    assert result["ranking"] is None


def test_c3_threshold_is_inclusive():
    # Exactly representable binary fraction at the boundary.
    comparison = make_comparison(
        error_a=0.5,
        error_b=0.75,
    )

    result = evaluate_accuracy_threshold(comparison)

    assert result["acceptance"]["ModelA"]["status"] == "pass"
    assert result["acceptance"]["ModelB"]["status"] == "fail"


def test_c3_custom_threshold():
    comparison = make_comparison(
        error_a=0.25,
        error_b=0.75,
    )

    result = evaluate_accuracy_threshold(
        comparison,
        threshold_microVolt=1.0,
    )

    assert result["threshold_microVolt"] == 1.0
    assert all(
        item["status"] == "pass"
        for item in result["acceptance"].values()
    )


@pytest.mark.parametrize(
    "threshold",
    [-0.1, float("nan"), float("inf"), True, "0.5"],
)
def test_c3_rejects_invalid_threshold(threshold):
    comparison = make_comparison()

    with pytest.raises((TypeError, ValueError)):
        evaluate_accuracy_threshold(
            comparison,
            threshold_microVolt=threshold,
        )


def test_c3_rejects_unsupported_metric():
    comparison = make_comparison()

    with pytest.raises(ValueError, match="metric"):
        evaluate_accuracy_threshold(
            comparison,
            metric="future_rmse",
        )


def test_c3_descriptive_only_is_not_eligible():
    comparison = make_comparison(
        dataset_b="different_dataset",
    )

    result = evaluate_accuracy_threshold(comparison)

    assert result["comparison_status"] == "descriptive_only"
    assert result["ranking"] is None

    assert all(
        item["status"] == "not_eligible"
        for item in result["acceptance"].values()
    )


def test_c3_positional_alignment_is_not_eligible():
    comparison = make_comparison(
        alignment_b="positional_only",
    )

    result = evaluate_accuracy_threshold(comparison)

    assert result["comparison_status"] == "descriptive_only"

    assert all(
        item["status"] == "not_eligible"
        for item in result["acceptance"].values()
    )


def test_c3_rejects_tampered_comparison_metrics():
    comparison = make_comparison()

    comparison["profiles"]["ModelA"]["overall"][
        "rmse_microVolt"
    ] = 0.0

    with pytest.raises(ValueError):
        evaluate_accuracy_threshold(comparison)


def test_c3_rejects_tampered_comparability():
    comparison = make_comparison()

    comparison["comparability"]["status"] = "descriptive_only"

    with pytest.raises(ValueError):
        evaluate_accuracy_threshold(comparison)


def test_c3_does_not_mutate_comparison():
    comparison = make_comparison()

    original = deepcopy(comparison)

    evaluate_accuracy_threshold(comparison)

    assert comparison == original


def test_c3_does_not_create_ranking():
    comparison = make_comparison()

    result = evaluate_accuracy_threshold(comparison)

    assert result["ranking"] is None
    assert comparison["ranking"] is None



# M17-C3-C: adversarial integrity and boundary tests.


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(models=["ModelB", "ModelA"]),
        lambda c: c.update(ranking=["ModelA", "ModelB"]),
        lambda c: c.update(comparison_version="forged_version"),
        lambda c: c.pop("comparison_version"),
        lambda c: c.update(unexpected_field=True),
        lambda c: c["comparability"].update(
            reasons=["forged reason"]
        ),
    ],
)
def test_c3_rejects_tampered_comparison_envelope(mutation):
    comparison = make_comparison()
    original = deepcopy(comparison)

    mutation(comparison)
    tampered = deepcopy(comparison)

    with pytest.raises(ValueError):
        evaluate_accuracy_threshold(comparison)

    # Failure must not mutate the supplied report.
    assert comparison == tampered

    # Confirm that the mutation actually changed the report.
    assert comparison != original


def test_c3_acceptance_boundary_uses_no_rounding():
    comparison = make_comparison(
        error_a=0.5,
        error_b=0.75,
    )

    exact = evaluate_accuracy_threshold(
        comparison,
        threshold_microVolt=0.5,
    )

    below = evaluate_accuracy_threshold(
        comparison,
        threshold_microVolt=math.nextafter(0.5, 0.0),
    )

    above = evaluate_accuracy_threshold(
        comparison,
        threshold_microVolt=math.nextafter(0.5, math.inf),
    )

    assert exact["acceptance"]["ModelA"]["status"] == "pass"
    assert below["acceptance"]["ModelA"]["status"] == "fail"
    assert above["acceptance"]["ModelA"]["status"] == "pass"


def test_c3_rejects_tampered_observation_evidence():
    comparison = make_comparison()

    comparison["profiles"]["ModelA"]["by_date"][0][
        "error_microVolt"
    ] += 5e-13

    with pytest.raises(ValueError):
        evaluate_accuracy_threshold(comparison)


def test_c3_preserves_descriptive_only_evidence():
    comparison = make_comparison(
        dataset_b="different_dataset",
    )

    original = deepcopy(comparison)

    result = evaluate_accuracy_threshold(comparison)

    assert comparison == original
    assert result["comparison_status"] == "descriptive_only"
    assert result["ranking"] is None

    for model in comparison["models"]:
        assert result["acceptance"][model]["status"] == "not_eligible"
        assert (
            result["acceptance"][model]["rmse_microVolt"]
            == comparison["profiles"][model]["overall"][
                "rmse_microVolt"
            ]
        )
