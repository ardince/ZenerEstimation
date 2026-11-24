"""M19-D2-C1: common-fold ranking sensitivity regression."""
from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import pytest

from tests.test_m18_accuracy_profile_ranking import make_result, trusted_context
from zenerestimation.assessment.ranking_sensitivity import (
    assess_internal_ranking_sensitivity,
)


def _real(errors):
    trusted = trusted_context()
    results = {name: make_result(name, trusted, errors=folds)
               for name, folds in errors.items()}
    return results, trusted


def _assess(errors):
    results, trusted = _real(errors)
    return assess_internal_ranking_sensitivity(results, validation_context=trusted)


def test_stable_order_and_winner():
    report = _assess({"LSTM": ((0.4, 0.4), (0.6, 0.6)),
                      "GRU": ((0.1, 0.1), (0.2, 0.2))})
    assert report["status"] == "available"
    assert report["baseline_top_models"] == ["GRU"]
    assert [r["model"] for r in report["baseline_ranking"]] == ["GRU", "LSTM"]
    assert report["any_rank_change"] is False
    assert report["any_top_model_set_change"] is False


def test_rank_reversal_on_common_fold_omission():
    report = _assess({"LSTM": ((0.1, 0.1), (0.8, 0.8)),
                      "GRU": ((0.6, 0.6), (0.4, 0.4))})
    assert report["baseline_top_models"] == ["LSTM"]
    assert [f["top_models"] for f in report["fold_omissions"]] == [["GRU"], ["LSTM"]]
    assert report["any_rank_change"] is True
    assert report["any_top_model_set_change"] is True
    changes = report["fold_omissions"][0]["rank_changes"]
    assert [(c["model"], c["rank_delta"]) for c in changes] == [("GRU", -1), ("LSTM", 1)]


def test_exact_ties_and_competition_rank_gap():
    report = _assess({"LSTM": ((0.2, 0.2), (0.2, 0.2)),
                      "GRU": ((0.2, 0.2), (0.2, 0.2)),
                      "ARIMA": ((0.6, 0.6), (0.6, 0.6))})
    assert [(r["model"], r["rank"]) for r in report["baseline_ranking"]] == [
        ("GRU", 1), ("LSTM", 1), ("ARIMA", 3)]
    assert report["baseline_top_models"] == ["GRU", "LSTM"]
    assert report["any_top_model_set_change"] is False


def test_shared_winner_set_change_without_new_sole_winner():
    report = _assess({"LSTM": ((0.125, 0.125), (0.375, 0.375)),
                      "GRU": ((0.25, 0.25), (0.25, 0.25))})
    # Both baseline scores equal exactly 0.25 in binary floating point;
    # omitting either fold produces a sole winner.
    assert report["baseline_top_models"] == ["GRU", "LSTM"]
    assert report["any_top_model_set_change"] is True
    assert all(len(f["top_models"]) == 1 for f in report["fold_omissions"])


def test_deterministic_order_and_read_only():
    results, trusted = _real({"LSTM": ((0.2, 0.2), (0.6, 0.6)),
                              "GRU": ((0.1, 0.1), (0.4, 0.4))})
    original = {k: v.to_dict() for k, v in results.items()}
    context_copy = deepcopy(trusted)
    a = assess_internal_ranking_sensitivity(results, validation_context=trusted)
    b = assess_internal_ranking_sensitivity(
        dict(reversed(list(results.items()))), validation_context=trusted)
    assert a == b
    assert a == assess_internal_ranking_sensitivity(results, validation_context=trusted)
    assert {k: v.to_dict() for k, v in results.items()} == original
    assert trusted == context_copy


@pytest.mark.parametrize("tamper", ["negative", "nan", "calendar", "alignment", "profile"])
def test_rejects_m18_evidence_tampering(tamper):
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
        assess_internal_ranking_sensitivity(results, validation_context=trusted)


def test_external_holdout_rejected():
    results, trusted = _real({"LSTM": ((0.2, 0.2), (0.6, 0.6)),
                              "GRU": ((0.1, 0.1), (0.4, 0.4))})
    trusted["scope"] = "external_holdout"
    with pytest.raises(ValueError):
        assess_internal_ranking_sensitivity(results, validation_context=trusted)


def test_missing_context_rejected():
    results, _ = _real({"LSTM": ((0.2, 0.2), (0.6, 0.6)),
                        "GRU": ((0.1, 0.1), (0.4, 0.4))})
    with pytest.raises(ValueError):
        assess_internal_ranking_sensitivity(results, validation_context=None)


def test_one_model_unavailable():
    report = _assess({"LSTM": ((0.2, 0.2), (0.6, 0.6))})
    assert report["status"] == "unavailable_insufficient_models"
    assert report["fold_omissions"] == []
    assert report["any_rank_change"] is None


def test_one_fold_unavailable_policy():
    results, trusted = _real({"LSTM": ((0.2, 0.2), (0.6, 0.6)),
                              "GRU": ((0.1, 0.1), (0.4, 0.4))})
    stub = {
        "assessment_version": "m19_d2_b2_fold_sensitivity_v1",
        "scope": "internal_expanding_window",
        "method": "leave_one_fold_out_reaggregation",
        "dataset_id": "synthetic", "dataset_fingerprint": "sha256:" + "a" * 64,
        "fold_validation_dates": [["2022-03-01"]], "fold_count": 1,
        "models": [
            {"model": "GRU", "baseline_rmse": 0.1,
             "status": "unavailable_insufficient_folds", "leave_one_fold_out": []},
            {"model": "LSTM", "baseline_rmse": 0.2,
             "status": "unavailable_insufficient_folds", "leave_one_fold_out": []},
        ],
    }
    with patch("zenerestimation.assessment.ranking_sensitivity.assess_internal_fold_sensitivity",
               return_value=stub):
        report = assess_internal_ranking_sensitivity(results, validation_context=trusted)
    assert report["status"] == "unavailable_insufficient_folds"
    assert report["any_top_model_set_change"] is None


def test_rejects_incomplete_common_fold_comparison():
    results, trusted = _real({"LSTM": ((0.2, 0.2), (0.6, 0.6)),
                              "GRU": ((0.1, 0.1), (0.4, 0.4))})
    good = {
        "assessment_version": "m19_d2_b2_fold_sensitivity_v1",
        "scope": "internal_expanding_window", "method": "leave_one_fold_out_reaggregation",
        "dataset_id": "synthetic", "dataset_fingerprint": "sha256:" + "a" * 64,
        "fold_validation_dates": [["2022-03-01"], ["2022-06-01"]], "fold_count": 2,
        "models": [{"model": "LSTM", "baseline_rmse": 0.2, "status": "available",
                    "leave_one_fold_out": [{"fold_index": 0, "validation_dates": ["2022-03-01"],
                                            "lofo_rmse": 0.2, "influence": 0.0}]}],
    }
    with patch("zenerestimation.assessment.ranking_sensitivity.assess_internal_fold_sensitivity",
               return_value=good):
        with pytest.raises(ValueError):
            assess_internal_ranking_sensitivity(results, validation_context=trusted)
