"""M19-D3-B2 RED: diagnostic cross-contract selection consistency."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from tests.test_m18_accuracy_profile_ranking import make_result, trusted_context
from zenerestimation.assessment.ranking_stability import assess_internal_ranking_stability
from zenerestimation.assessment.selection_robustness import assess_internal_selection_robustness
from zenerestimation.assessment.selection_consistency import assess_internal_selection_consistency

MODULE = "zenerestimation.assessment.selection_consistency"


def _real(errors):
    trusted = trusted_context()
    return ({model: make_result(model, trusted, errors=folds)
             for model, folds in errors.items()}, trusted)


def _assess(errors):
    results, trusted = _real(errors)
    return assess_internal_selection_consistency(results, validation_context=trusted)


def _sources(errors=None):
    if errors is None:
        errors = {"LSTM": ((0.1, 0.1), (0.8, 0.8)),
                  "GRU": ((0.6, 0.6), (0.4, 0.4))}
    results, trusted = _real(errors)
    return (results, trusted,
            assess_internal_selection_robustness(results, validation_context=trusted),
            assess_internal_ranking_stability(results, validation_context=trusted))


def _mocked_assess(results, trusted, robustness, stability):
    with patch(f"{MODULE}.assess_internal_selection_robustness", return_value=robustness), \
         patch(f"{MODULE}.assess_internal_ranking_stability", return_value=stability):
        return assess_internal_selection_consistency(results, validation_context=trusted)


def test_stable_unique_winner_and_ranking():
    report = _assess({"LSTM": ((0.4, 0.4), (0.6, 0.6)),
                      "GRU": ((0.1, 0.1), (0.2, 0.2))})
    assert report["assessment_version"] == "m19_d3_b2_selection_consistency_v1"
    assert report["scope"] == "internal_expanding_window"
    assert report["status"] == "available"
    assert report["baseline_recommended_model"] == "GRU"
    assert report["recommendation_consistency_fraction"] == 1.0
    assert report["winner_set_stability_fraction"] == 1.0
    assert report["rank_stability_fraction"] == 1.0
    assert report["recommendation_winner_gap"] == 0.0
    assert report["winner_rank_gap"] == 0.0
    assert report["consistency_relationship_verified"] is True
    assert report["selection_authorized"] is False


def test_lower_rank_changes_without_winner_change():
    report = _assess({"GRU": ((0.1, 0.1), (0.1, 0.1)),
                      "LSTM": ((0.2, 0.2), (0.8, 0.8)),
                      "ARIMA": ((0.8, 0.8), (0.2, 0.2))})
    assert report["baseline_recommended_model"] == "GRU"
    assert report["recommendation_consistency_fraction"] == 1.0
    assert report["winner_set_stability_fraction"] == 1.0
    assert report["rank_stability_fraction"] == 0.0
    assert report["recommendation_winner_gap"] == 0.0
    assert report["winner_rank_gap"] == 1.0


def test_replacement_winner():
    report = _assess({"LSTM": ((0.1, 0.1), (0.8, 0.8)),
                      "GRU": ((0.6, 0.6), (0.4, 0.4))})
    assert report["baseline_recommended_model"] == "LSTM"
    assert report["recommendation_consistency_fraction"] == 0.5
    assert report["winner_set_stability_fraction"] == 0.5
    assert report["rank_stability_fraction"] == 0.5
    assert report["recommendation_winner_gap"] == 0.0
    assert report["winner_rank_gap"] == 0.0


def test_omission_tie_is_not_consistent():
    report = _assess({"LSTM": ((0.1, 0.1), (0.3, 0.3)),
                      "GRU": ((0.2, 0.2), (0.3, 0.3))})
    assert report["baseline_recommended_model"] == "LSTM"
    assert report["recommendation_consistency_fraction"] == 0.5
    assert report["winner_set_stability_fraction"] == 0.5


def test_tied_baseline_does_not_imply_recommendation():
    report = _assess({"LSTM": ((0.2, 0.2), (0.2, 0.2)),
                      "GRU": ((0.2, 0.2), (0.2, 0.2)),
                      "ARIMA": ((0.6, 0.6), (0.6, 0.6))})
    assert report["status"] == "available"
    assert report["baseline_recommended_model"] is None
    assert report["recommendation_consistency_fraction"] is None
    assert report["winner_set_stability_fraction"] == 1.0
    assert report["rank_stability_fraction"] == 1.0
    assert report["recommendation_winner_gap"] is None
    assert report["winner_rank_gap"] == 0.0
    assert report["selection_authorized"] is False


def test_insufficient_models_is_not_zero_consistency():
    report = _assess({"LSTM": ((0.2, 0.2), (0.6, 0.6))})
    assert report["status"] == "unavailable_insufficient_models"
    for key in ("recommendation_consistency_fraction", "winner_set_stability_fraction",
                "rank_stability_fraction", "recommendation_winner_gap", "winner_rank_gap"):
        assert report[key] is None
    assert report["selection_authorized"] is False


def test_deterministic_read_only():
    results, trusted = _real({"LSTM": ((0.2, 0.2), (0.6, 0.6)),
                              "GRU": ((0.1, 0.1), (0.4, 0.4))})
    before_results = {k: v.to_dict() for k, v in results.items()}
    before_context = deepcopy(trusted)
    first = assess_internal_selection_consistency(results, validation_context=trusted)
    second = assess_internal_selection_consistency(
        dict(reversed(list(results.items()))), validation_context=trusted)
    assert first == second
    assert {k: v.to_dict() for k, v in results.items()} == before_results
    assert trusted == before_context


@pytest.mark.parametrize("tamper", ["negative", "nan", "calendar", "alignment", "profile"])
def test_rejects_corrupted_m18_evidence(tamper):
    results, trusted = _real({"LSTM": ((0.2, 0.2), (0.6, 0.6)),
                              "GRU": ((0.1, 0.1), (0.4, 0.4))})
    candidate = results["LSTM"].candidates[0]
    if tamper == "negative":
        candidate.metadata["fold_metrics"][0]["rmse"] = -1
    elif tamper == "nan":
        candidate.metadata["fold_metrics"][0]["rmse"] = float("nan")
    elif tamper == "calendar":
        candidate.metadata["fold_alignment"][0]["forecast_dates"][0] = "2030-03-01"
    elif tamper == "alignment":
        candidate.metadata["fold_alignment"][0]["status"] = "positional_only"
    else:
        candidate.metadata["accuracy_profile"]["by_date"][0]["predicted_microVolt"] += 1
    with pytest.raises(ValueError):
        assess_internal_selection_consistency(results, validation_context=trusted)


def test_rejects_external_holdout_context():
    results, trusted = _real({"LSTM": ((0.2, 0.2), (0.6, 0.6)),
                              "GRU": ((0.1, 0.1), (0.4, 0.4))})
    trusted["scope"] = "external_holdout"
    with pytest.raises(ValueError):
        assess_internal_selection_consistency(results, validation_context=trusted)


def test_missing_trusted_context_rejected():
    results, _ = _real({"LSTM": ((0.2, 0.2), (0.6, 0.6)),
                        "GRU": ((0.1, 0.1), (0.4, 0.4))})
    with pytest.raises((TypeError, ValueError)):
        assess_internal_selection_consistency(results, validation_context=None)


@pytest.mark.parametrize("tamper", [
    "robustness_version", "stability_version", "scope", "dataset_id", "fingerprint",
    "fold_count", "calendar", "model_count", "baseline_winners", "status",
    "recommendation_fraction", "winner_fraction", "ranking_fraction",
    "recommendation_count", "winner_count", "rank_count", "selection_authority",
    "omission_winner", "omission_calendar", "omission_consistency", "omission_duplicate",
])
def test_rejects_inconsistent_upstream_reports(tamper):
    results, trusted, robustness, stability = _sources()
    robustness, stability = deepcopy(robustness), deepcopy(stability)
    if tamper == "robustness_version":
        robustness["assessment_version"] = "unsupported"
    elif tamper == "stability_version":
        stability["assessment_version"] = "unsupported"
    elif tamper == "scope":
        stability["scope"] = "external_holdout"
    elif tamper == "dataset_id":
        stability["dataset_id"] = "different"
    elif tamper == "fingerprint":
        stability["dataset_fingerprint"] = "sha256:" + "0" * 64
    elif tamper == "fold_count":
        stability["fold_count"] += 1
    elif tamper == "calendar":
        stability["fold_validation_dates"][0] = ["2030-03-01"]
    elif tamper == "model_count":
        stability["model_count"] += 1
    elif tamper == "baseline_winners":
        stability["baseline_top_models"] = ["other"]
    elif tamper == "status":
        stability["status"] = "unavailable_insufficient_folds"
    elif tamper == "recommendation_fraction":
        robustness["recommendation_consistency_fraction"] = 1.0
    elif tamper == "winner_fraction":
        stability["winner_set_stability_fraction"] = 1.0
    elif tamper == "ranking_fraction":
        stability["rank_stability_fraction"] = 1.0
    elif tamper == "recommendation_count":
        robustness["recommendation_consistency_count"] = 2
    elif tamper == "winner_count":
        stability["winner_set_change_count"] = 0
    elif tamper == "rank_count":
        stability["rank_change_count"] = 0
    elif tamper == "selection_authority":
        robustness["selection_authorized"] = True
    elif tamper == "omission_winner":
        robustness["fold_omissions"][0]["top_models"] = ["LSTM"]
    elif tamper == "omission_calendar":
        robustness["fold_omissions"][0]["validation_dates"] = ["2030-03-01"]
    elif tamper == "omission_consistency":
        robustness["fold_omissions"][0]["recommendation_consistent"] = True
    else:
        robustness["fold_omissions"][1]["fold_index"] = 0
    with pytest.raises(ValueError):
        _mocked_assess(results, trusted, robustness, stability)


def test_mocked_matching_upstream_reports_accepted():
    results, trusted, robustness, stability = _sources()
    report = _mocked_assess(results, trusted, robustness, stability)
    assert report["status"] == "available"
    assert report["recommendation_consistency_fraction"] == 0.5
    assert report["winner_set_stability_fraction"] == 0.5
    assert report["rank_stability_fraction"] == 0.5
    assert report["consistency_relationship_verified"] is True
    assert report["selection_authorized"] is False


def test_insufficient_folds_remain_unavailable():
    """Matching one-fold upstream contracts must not report zero robustness."""
    results, trusted, robustness, stability = _sources()
    robustness, stability = deepcopy(robustness), deepcopy(stability)

    for source in (robustness, stability):
        source["fold_count"] = 1
        source["fold_validation_dates"] = source["fold_validation_dates"][:1]
        source["status"] = "unavailable_insufficient_folds"

    robustness.update({
        "fold_omissions": [],
        "recommendation_consistency_count": None,
        "recommendation_consistency_fraction": None,
        "ambiguous_omission_count": None,
        "alternative_winner_models": None,
    })
    stability.update({
        "rank_change_count": None,
        "winner_set_change_count": None,
        "rank_stability_fraction": None,
        "winner_set_stability_fraction": None,
        "max_absolute_rank_displacement": None,
        "per_model_rank_change_count": None,
    })

    report = _mocked_assess(results, trusted, robustness, stability)
    assert report["status"] == "unavailable_insufficient_folds"
    assert report["fold_count"] == 1
    for key in (
        "recommendation_consistency_fraction",
        "winner_set_stability_fraction",
        "rank_stability_fraction",
        "recommendation_winner_gap",
        "winner_rank_gap",
    ):
        assert report[key] is None
    assert report["selection_authorized"] is False


def test_matching_aggregates_do_not_prove_fold_level_agreement():
    """Characterize aggregate-only B2 verification, not fold-level certification.

    A mocked B1 report can swap which fold changes the winner, preserving its
    own counts and the C2 aggregate counts. C2 does not expose fold-level
    winner sets, so B2 cannot independently detect the disagreement.
    """
    results, trusted, robustness, stability = _sources()
    baseline = _mocked_assess(
        results, trusted, deepcopy(robustness), deepcopy(stability)
    )
    robustness = deepcopy(robustness)
    stability = deepcopy(stability)

    first, second = robustness["fold_omissions"]
    assert first["recommended_model"] != second["recommended_model"]
    assert first["recommendation_consistent"] != second["recommendation_consistent"]

    # Swap only the winner-related payload, not fold indexes or calendars.
    # Each B1 omission remains internally coherent; aggregate totals stay fixed.
    for key in (
        "top_models", "recommended_model", "ambiguous",
        "recommendation_consistent",
    ):
        first[key], second[key] = second[key], first[key]

    report = _mocked_assess(results, trusted, robustness, stability)
    assert report["status"] == "available"
    assert report["recommendation_consistency_fraction"] == baseline[
        "recommendation_consistency_fraction"
    ]
    assert report["winner_set_stability_fraction"] == baseline[
        "winner_set_stability_fraction"
    ]
    assert report["rank_stability_fraction"] == baseline[
        "rank_stability_fraction"
    ]
    assert report["consistency_relationship_verified"] is True
    assert report["selection_authorized"] is False
