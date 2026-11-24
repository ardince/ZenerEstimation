
"""M19-D3-B1: diagnostic-only internal selection robustness contracts."""

from __future__ import annotations

from copy import deepcopy

from unittest.mock import patch

import pytest

from tests.test_m18_accuracy_profile_ranking import (
    make_result,
    trusted_context,
)
from zenerestimation.assessment.selection_robustness import (
    assess_internal_selection_robustness,
)


def _real(errors):
    trusted = trusted_context()
    results = {
        model: make_result(model, trusted, errors=folds)
        for model, folds in errors.items()
    }
    return results, trusted


def _assess(errors):
    results, trusted = _real(errors)
    return assess_internal_selection_robustness(
        results, validation_context=trusted
    )


def test_stable_unique_recommendation():
    report = _assess({
        "LSTM": ((0.4, 0.4), (0.6, 0.6)),
        "GRU": ((0.1, 0.1), (0.2, 0.2)),
    })

    assert report["assessment_version"] == (
        "m19_d3_b1_selection_robustness_v1"
    )
    assert report["scope"] == "internal_expanding_window"
    assert report["status"] == "available"
    assert report["baseline_top_models"] == ["GRU"]
    assert report["baseline_recommended_model"] == "GRU"
    assert report["recommendation_consistency_count"] == 2
    assert report["recommendation_consistency_fraction"] == 1.0
    assert report["ambiguous_omission_count"] == 0
    assert report["alternative_winner_models"] == []
    assert report["selection_authorized"] is False
    assert all(
        fold["recommended_model"] == "GRU"
        for fold in report["fold_omissions"]
    )


def test_unique_recommendation_replaced_by_another_model():
    report = _assess({
        "LSTM": ((0.1, 0.1), (0.8, 0.8)),
        "GRU": ((0.6, 0.6), (0.4, 0.4)),
    })

    assert report["baseline_recommended_model"] == "LSTM"
    assert [
        fold["recommended_model"]
        for fold in report["fold_omissions"]
    ] == ["GRU", "LSTM"]
    assert report["recommendation_consistency_count"] == 1
    assert report["recommendation_consistency_fraction"] == 0.5
    assert report["alternative_winner_models"] == ["GRU"]


def test_exact_tie_under_omission_is_ambiguous():
    report = _assess({
        "LSTM": ((0.1, 0.1), (0.3, 0.3)),
        "GRU": ((0.2, 0.2), (0.3, 0.3)),
    })

    assert report["baseline_recommended_model"] == "LSTM"
    assert [
        fold["recommended_model"]
        for fold in report["fold_omissions"]
    ] == [None, "LSTM"]
    assert report["ambiguous_omission_count"] == 1
    assert report["recommendation_consistency_count"] == 1
    assert report["recommendation_consistency_fraction"] == 0.5
    assert report["alternative_winner_models"] == []


def test_baseline_exact_tie_remains_unresolved():
    report = _assess({
        "LSTM": ((0.2, 0.2), (0.2, 0.2)),
        "GRU": ((0.2, 0.2), (0.2, 0.2)),
        "ARIMA": ((0.6, 0.6), (0.6, 0.6)),
    })

    assert report["status"] == "available"
    assert report["baseline_top_models"] == ["GRU", "LSTM"]
    assert report["baseline_recommended_model"] is None
    assert report["recommendation_consistency_count"] is None
    assert report["recommendation_consistency_fraction"] is None
    assert report["ambiguous_omission_count"] == 2
    assert report["selection_authorized"] is False


def test_baseline_tie_does_not_create_implicit_recommendation():
    report = _assess({
        "LSTM": ((0.125, 0.125), (0.375, 0.375)),
        "GRU": ((0.25, 0.25), (0.25, 0.25)),
    })

    assert report["baseline_top_models"] == ["GRU", "LSTM"]
    assert report["baseline_recommended_model"] is None
    assert report["recommendation_consistency_fraction"] is None
    assert report["selection_authorized"] is False


def test_one_model_is_unavailable():
    report = _assess({
        "LSTM": ((0.2, 0.2), (0.6, 0.6)),
    })

    assert report["status"] == "unavailable_insufficient_models"
    assert report["fold_omissions"] == []
    assert report["recommendation_consistency_count"] is None
    assert report["recommendation_consistency_fraction"] is None
    assert report["ambiguous_omission_count"] is None
    assert report["alternative_winner_models"] is None
    assert report["selection_authorized"] is False


def test_deterministic_and_read_only():
    results, trusted = _real({
        "LSTM": ((0.2, 0.2), (0.6, 0.6)),
        "GRU": ((0.1, 0.1), (0.4, 0.4)),
    })
    before_results = {
        model: result.to_dict()
        for model, result in results.items()
    }
    before_context = deepcopy(trusted)

    first = assess_internal_selection_robustness(
        results, validation_context=trusted
    )
    second = assess_internal_selection_robustness(
        dict(reversed(list(results.items()))),
        validation_context=trusted,
    )

    assert first == second
    assert {k: v.to_dict() for k, v in results.items()} == before_results
    assert trusted == before_context


@pytest.mark.parametrize(
    "tamper",
    ["negative", "nan", "calendar", "alignment", "profile"],
)
def test_rejects_corrupted_m18_evidence(tamper):
    results, trusted = _real({
        "LSTM": ((0.2, 0.2), (0.6, 0.6)),
        "GRU": ((0.1, 0.1), (0.4, 0.4)),
    })
    candidate = results["LSTM"].candidates[0]

    if tamper == "negative":
        candidate.metadata["fold_metrics"][0]["rmse"] = -1
    elif tamper == "nan":
        candidate.metadata["fold_metrics"][0]["rmse"] = float("nan")
    elif tamper == "calendar":
        candidate.metadata["fold_alignment"][0][
            "forecast_dates"
        ][0] = "2030-03-01"
    elif tamper == "alignment":
        candidate.metadata["fold_alignment"][0][
            "status"
        ] = "positional_only"
    else:
        candidate.metadata["accuracy_profile"]["by_date"][0][
            "predicted_microVolt"
        ] += 1

    with pytest.raises(ValueError):
        assess_internal_selection_robustness(
            results, validation_context=trusted
        )


def test_rejects_external_holdout_context():
    results, trusted = _real({
        "LSTM": ((0.2, 0.2), (0.6, 0.6)),
        "GRU": ((0.1, 0.1), (0.4, 0.4)),
    })
    trusted["scope"] = "external_holdout"

    with pytest.raises(ValueError):
        assess_internal_selection_robustness(
            results, validation_context=trusted
        )


def test_missing_trusted_context_rejected():
    results, _ = _real({
        "LSTM": ((0.2, 0.2), (0.6, 0.6)),
        "GRU": ((0.1, 0.1), (0.4, 0.4)),
    })

    with pytest.raises((TypeError, ValueError)):
        assess_internal_selection_robustness(
            results, validation_context=None
        )



# M19-D3-B1-H1: additional diagnostic boundary tests.

from zenerestimation.assessment.ranking_sensitivity import (
    assess_internal_ranking_sensitivity,
)


def test_near_equal_scores_remain_distinct():
    report = _assess({
        "LSTM": ((0.2, 0.2), (0.40000000000001, 0.40000000000001)),
        "GRU": ((0.2, 0.2), (0.4, 0.4)),
    })

    assert report["status"] == "available"
    assert report["baseline_top_models"] == ["GRU"]
    assert report["baseline_recommended_model"] == "GRU"
    assert report["selection_authorized"] is False


def test_one_fold_unavailability_is_not_zero_robustness():
    results, trusted = _real({
        "LSTM": ((0.2, 0.2), (0.6, 0.6)),
        "GRU": ((0.1, 0.1), (0.4, 0.4)),
    })

    source = assess_internal_ranking_sensitivity(
        results, validation_context=trusted
    )

    # Simulate the D2-C1 verified one-fold unavailable contract.
    source["fold_count"] = 1
    source["fold_validation_dates"] = source["fold_validation_dates"][:1]
    source["status"] = "unavailable_insufficient_folds"
    source["fold_omissions"] = []
    source["any_rank_change"] = None
    source["any_top_model_set_change"] = None

    with patch(
        "zenerestimation.assessment.selection_robustness."
        "assess_internal_ranking_sensitivity",
        return_value=source,
    ):
        report = assess_internal_selection_robustness(
            results, validation_context=trusted
        )

    assert report["status"] == "unavailable_insufficient_folds"
    assert report["fold_omissions"] == []
    assert report["recommendation_consistency_count"] is None
    assert report["recommendation_consistency_fraction"] is None
    assert report["ambiguous_omission_count"] is None
    assert report["alternative_winner_models"] is None
    assert report["selection_authorized"] is False


@pytest.mark.parametrize(
    "tamper",
    [
        "wrong_version",
        "duplicate_fold",
        "missing_fold",
        "wrong_calendar",
        "wrong_baseline_winners",
        "wrong_omission_winners",
    ],
)
def test_rejects_inconsistent_intermediate_evidence(tamper):
    results, trusted = _real({
        "LSTM": ((0.1, 0.1), (0.8, 0.8)),
        "GRU": ((0.6, 0.6), (0.4, 0.4)),
    })

    source = assess_internal_ranking_sensitivity(
        results, validation_context=trusted
    )
    source = deepcopy(source)

    if tamper == "wrong_version":
        source["assessment_version"] = "unsupported"
    elif tamper == "duplicate_fold":
        source["fold_omissions"][1]["fold_index"] = 0
    elif tamper == "missing_fold":
        source["fold_omissions"].pop()
    elif tamper == "wrong_calendar":
        source["fold_omissions"][0]["validation_dates"] = [
            "2030-03-01"
        ]
    elif tamper == "wrong_baseline_winners":
        source["baseline_top_models"] = ["GRU"]
    elif tamper == "wrong_omission_winners":
        source["fold_omissions"][0]["top_models"] = ["LSTM"]

    with patch(
        "zenerestimation.assessment.selection_robustness."
        "assess_internal_ranking_sensitivity",
        return_value=source,
    ):
        with pytest.raises(ValueError):
            assess_internal_selection_robustness(
                results, validation_context=trusted
            )



def test_rejects_duplicate_model_in_omission_ranking():
    results, trusted = _real({
        "LSTM": ((0.1, 0.1), (0.8, 0.8)),
        "GRU": ((0.6, 0.6), (0.4, 0.4)),
    })

    source = deepcopy(
        assess_internal_ranking_sensitivity(
            results, validation_context=trusted
        )
    )

    omission = source["fold_omissions"][0]
    ranking = omission["ranking"]

    # Preserve row count and the existing rank-1 winner,
    # but duplicate one model identity.
    ranking[1]["model"] = ranking[0]["model"]

    with patch(
        "zenerestimation.assessment.selection_robustness."
        "assess_internal_ranking_sensitivity",
        return_value=source,
    ):
        with pytest.raises(ValueError):
            assess_internal_selection_robustness(
                results, validation_context=trusted
            )



@pytest.mark.parametrize(
    "tamper",
    [
        "reversed_scores",
        "false_exact_tie",
        "incorrect_competition_rank",
        "nonfinite_score",
        "boolean_score",
        "negative_score",
    ],
)
def test_rejects_numerically_inconsistent_omission_ranking(tamper):
    results, trusted = _real({
        "LSTM": ((0.1, 0.1), (0.8, 0.8)),
        "GRU": ((0.6, 0.6), (0.4, 0.4)),
        "ARIMA": ((0.7, 0.7), (0.9, 0.9)),
    })

    source = deepcopy(
        assess_internal_ranking_sensitivity(
            results, validation_context=trusted
        )
    )

    ranking = source["fold_omissions"][0]["ranking"]

    if tamper == "reversed_scores":
        ranking[0]["score"], ranking[1]["score"] = (
            ranking[1]["score"],
            ranking[0]["score"],
        )
    elif tamper == "false_exact_tie":
        ranking[1]["score"] = ranking[0]["score"]
    elif tamper == "incorrect_competition_rank":
        ranking[2]["rank"] = 2
    elif tamper == "nonfinite_score":
        ranking[1]["score"] = float("nan")
    elif tamper == "boolean_score":
        ranking[1]["score"] = True
    else:
        ranking[1]["score"] = -1.0

    with patch(
        "zenerestimation.assessment.selection_robustness."
        "assess_internal_ranking_sensitivity",
        return_value=source,
    ):
        with pytest.raises(ValueError):
            assess_internal_selection_robustness(
                results, validation_context=trusted
            )


def test_rejects_numerically_inconsistent_baseline_ranking():
    results, trusted = _real({
        "LSTM": ((0.1, 0.1), (0.8, 0.8)),
        "GRU": ((0.6, 0.6), (0.4, 0.4)),
    })

    source = deepcopy(
        assess_internal_ranking_sensitivity(
            results, validation_context=trusted
        )
    )
    source["baseline_ranking"][0]["score"] = 100.0

    with patch(
        "zenerestimation.assessment.selection_robustness."
        "assess_internal_ranking_sensitivity",
        return_value=source,
    ):
        with pytest.raises(ValueError):
            assess_internal_selection_robustness(
                results, validation_context=trusted
            )
