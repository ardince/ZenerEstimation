"""M19-D2-C2: deterministic internal ranking stability regression."""
from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import pytest

from tests.test_m18_accuracy_profile_ranking import make_result, trusted_context
from zenerestimation.assessment.ranking_stability import assess_internal_ranking_stability


def _real(errors):
    trusted = trusted_context()
    results = {name: make_result(name, trusted, errors=folds)
               for name, folds in errors.items()}
    return results, trusted


def _assess(errors):
    results, trusted = _real(errors)
    return assess_internal_ranking_stability(results, validation_context=trusted)


def test_stable_rankings():
    report = _assess({"LSTM": ((0.4, 0.4), (0.6, 0.6)),
                      "GRU": ((0.1, 0.1), (0.2, 0.2))})
    assert report["status"] == "available"
    assert report["rank_change_count"] == 0
    assert report["winner_set_change_count"] == 0
    assert report["rank_stability_fraction"] == 1.0
    assert report["winner_set_stability_fraction"] == 1.0
    assert report["max_absolute_rank_displacement"] == 0
    assert report["per_model_rank_change_count"] == {"GRU": 0, "LSTM": 0}


def test_partial_rank_instability():
    report = _assess({"LSTM": ((0.1, 0.1), (0.8, 0.8)),
                      "GRU": ((0.6, 0.6), (0.4, 0.4))})
    assert report["rank_change_count"] == 1
    assert report["winner_set_change_count"] == 1
    assert report["rank_stability_fraction"] == 0.5
    assert report["winner_set_stability_fraction"] == 0.5
    assert report["max_absolute_rank_displacement"] == 1
    assert report["per_model_rank_change_count"] == {"GRU": 1, "LSTM": 1}


def test_shared_winner_set_changes():
    report = _assess({"LSTM": ((0.125, 0.125), (0.375, 0.375)),
                      "GRU": ((0.25, 0.25), (0.25, 0.25))})
    assert report["baseline_top_models"] == ["GRU", "LSTM"]
    assert report["rank_change_count"] == 2
    assert report["winner_set_change_count"] == 2
    assert report["max_absolute_rank_displacement"] == 1


def test_exact_ties_and_competition_rank_gap():
    report = _assess({"LSTM": ((0.25, 0.25), (0.25, 0.25)),
                      "GRU": ((0.25, 0.25), (0.25, 0.25)),
                      "ARIMA": ((0.75, 0.75), (0.75, 0.75))})
    assert report["rank_change_count"] == 0
    assert report["baseline_top_models"] == ["GRU", "LSTM"]
    assert report["per_model_rank_change_count"] == {"ARIMA": 0, "GRU": 0, "LSTM": 0}


def test_deterministic_order_and_read_only():
    results, trusted = _real({"LSTM": ((0.2, 0.2), (0.6, 0.6)),
                              "GRU": ((0.1, 0.1), (0.4, 0.4))})
    original = {name: result.to_dict() for name, result in results.items()}
    original_context = deepcopy(trusted)
    a = assess_internal_ranking_stability(results, validation_context=trusted)
    b = assess_internal_ranking_stability(
        dict(reversed(list(results.items()))), validation_context=trusted)
    assert a == b == assess_internal_ranking_stability(results, validation_context=trusted)
    assert {name: result.to_dict() for name, result in results.items()} == original
    assert trusted == original_context


@pytest.mark.parametrize("tamper", ["negative", "nan", "calendar", "alignment", "profile"])
def test_m18_evidence_tampering_rejected(tamper):
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
        assess_internal_ranking_stability(results, validation_context=trusted)


def test_external_holdout_rejected():
    results, trusted = _real({"LSTM": ((0.2, 0.2), (0.6, 0.6)),
                              "GRU": ((0.1, 0.1), (0.4, 0.4))})
    trusted["scope"] = "external_holdout"
    with pytest.raises(ValueError):
        assess_internal_ranking_stability(results, validation_context=trusted)


def test_missing_context_rejected():
    results, _ = _real({"LSTM": ((0.2, 0.2), (0.6, 0.6)),
                        "GRU": ((0.1, 0.1), (0.4, 0.4))})
    with pytest.raises(ValueError):
        assess_internal_ranking_stability(results, validation_context=None)


def test_one_model_unavailable():
    report = _assess({"LSTM": ((0.2, 0.2), (0.6, 0.6))})
    assert report["status"] == "unavailable_insufficient_models"
    assert report["rank_change_count"] is None
    assert report["winner_set_stability_fraction"] is None
    assert report["per_model_rank_change_count"] is None


def _mock_report():
    return {
        "assessment_version": "m19_d2_c1_ranking_sensitivity_v1",
        "scope": "internal_expanding_window",
        "source_assessment_version": "m19_d2_b2_fold_sensitivity_v1",
        "dataset_id": "synthetic", "dataset_fingerprint": "sha256:" + "a" * 64,
        "fold_validation_dates": [["2022-03-01"], ["2022-06-01"]],
        "fold_count": 2, "model_count": 2, "status": "available",
        "baseline_ranking": [
            {"model": "GRU", "score": 0.2, "rank": 1},
            {"model": "LSTM", "score": 0.4, "rank": 2}],
        "baseline_top_models": ["GRU"],
        "fold_omissions": [
            {"fold_index": 0, "validation_dates": ["2022-03-01"],
             "ranking": [{"model": "LSTM", "score": 0.1, "rank": 1},
                         {"model": "GRU", "score": 0.3, "rank": 2}],
             "top_models": ["LSTM"],
             "rank_changes": [
                 {"model": "GRU", "baseline_rank": 1, "lofo_rank": 2, "rank_delta": 1},
                 {"model": "LSTM", "baseline_rank": 2, "lofo_rank": 1, "rank_delta": -1}],
             "rank_changed": True, "top_model_set_changed": True},
            {"fold_index": 1, "validation_dates": ["2022-06-01"],
             "ranking": [{"model": "GRU", "score": 0.1, "rank": 1},
                         {"model": "LSTM", "score": 0.5, "rank": 2}],
             "top_models": ["GRU"], "rank_changes": [],
             "rank_changed": False, "top_model_set_changed": False}],
        "any_rank_change": True, "any_top_model_set_change": True,
    }


@pytest.mark.parametrize("tamper", ["duplicate_fold", "missing_fold", "dates", "winner",
                                    "rank", "delta", "flag", "aggregate", "scope", "version"])
def test_inconsistent_c1_reports_rejected(tamper):
    stub = _mock_report()
    if tamper == "duplicate_fold":
        stub["fold_omissions"][1]["fold_index"] = 0
    elif tamper == "missing_fold":
        stub["fold_omissions"].pop()
    elif tamper == "dates":
        stub["fold_omissions"][0]["validation_dates"] = ["2030-03-01"]
    elif tamper == "winner":
        stub["fold_omissions"][0]["top_models"] = ["GRU"]
    elif tamper == "rank":
        stub["fold_omissions"][0]["ranking"][0]["rank"] = 2
    elif tamper == "delta":
        stub["fold_omissions"][0]["rank_changes"][0]["rank_delta"] = 3
    elif tamper == "flag":
        stub["fold_omissions"][0]["rank_changed"] = False
    elif tamper == "aggregate":
        stub["any_top_model_set_change"] = False
    elif tamper == "scope":
        stub["scope"] = "external_holdout"
    else:
        stub["assessment_version"] = "unsupported"
    with patch("zenerestimation.assessment.ranking_stability.assess_internal_ranking_sensitivity",
               return_value=stub):
        with pytest.raises(ValueError):
            assess_internal_ranking_stability({}, validation_context={})


def test_mocked_valid_report_aggregation():
    with patch("zenerestimation.assessment.ranking_stability.assess_internal_ranking_sensitivity",
               return_value=_mock_report()):
        report = assess_internal_ranking_stability({}, validation_context={})
    assert report["rank_change_count"] == 1
    assert report["winner_set_change_count"] == 1
    assert report["per_model_rank_change_count"] == {"GRU": 1, "LSTM": 1}


def test_one_fold_unavailable_policy():
    stub = _mock_report()
    stub.update({"fold_validation_dates": [["2022-03-01"]], "fold_count": 1,
                 "status": "unavailable_insufficient_folds", "fold_omissions": [],
                 "any_rank_change": None, "any_top_model_set_change": None})
    with patch("zenerestimation.assessment.ranking_stability.assess_internal_ranking_sensitivity",
               return_value=stub):
        report = assess_internal_ranking_stability({}, validation_context={})
    assert report["status"] == "unavailable_insufficient_folds"
    assert report["rank_stability_fraction"] is None
