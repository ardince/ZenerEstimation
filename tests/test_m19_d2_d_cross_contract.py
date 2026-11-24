"""M19-D2-D: integration of the verified internal ranking sensitivity chain.

The fixture builds genuine M18 accuracy profiles; no intermediate D2 layer is mocked.
This suite verifies software contracts, not external dataset authentication.
"""
from __future__ import annotations

from copy import deepcopy
from math import fsum

import pytest

from tests.test_m18_accuracy_profile_ranking import make_result, trusted_context
from zenerestimation.optimization.internal_ranking import (
    extract_verified_internal_fold_scores,
    rank_internal_models,
)
from zenerestimation.assessment.fold_sensitivity import assess_internal_fold_sensitivity
from zenerestimation.assessment.ranking_sensitivity import assess_internal_ranking_sensitivity
from zenerestimation.assessment.ranking_stability import assess_internal_ranking_stability


def _case():
    trusted = trusted_context()
    errors = {
        "LSTM": ((0.125, 0.125), (0.375, 0.375)),
        "GRU": ((0.25, 0.25), (0.25, 0.25)),
        "ARIMA": ((0.75, 0.75), (0.75, 0.75)),
    }
    return {model: make_result(model, trusted, errors=folds)
            for model, folds in errors.items()}, trusted


def _chain(results, trusted):
    return (
        extract_verified_internal_fold_scores(results, validation_context=trusted),
        assess_internal_fold_sensitivity(results, validation_context=trusted),
        assess_internal_ranking_sensitivity(results, validation_context=trusted),
        assess_internal_ranking_stability(results, validation_context=trusted),
    )


def _competition_rank(scores):
    ordered = sorted(scores.items(), key=lambda pair: (pair[1], pair[0]))
    ranks = {}
    previous = None
    for position, (model, score) in enumerate(ordered, 1):
        if previous is None or score != previous:
            rank = position
        ranks[model] = rank
        previous = score
    return ranks


def test_complete_chain_identity_scope_calendar_and_versions():
    results, trusted = _case()
    b1, b2, c1, c2 = _chain(results, trusted)
    assert [b1["evidence_version"], b2["assessment_version"],
            c1["assessment_version"], c2["assessment_version"]] == [
        "m19_d2_verified_internal_folds_v1",
        "m19_d2_b2_fold_sensitivity_v1",
        "m19_d2_c1_ranking_sensitivity_v1",
        "m19_d2_c2_ranking_stability_v1",
    ]
    for report in (b1, b2, c1, c2):
        assert report["scope"] == "internal_expanding_window"
        assert report["dataset_id"] == trusted["dataset_id"]
        assert report["dataset_fingerprint"] == trusted["dataset_fingerprint"]
        assert report["fold_validation_dates"] == trusted["fold_validation_dates"]
    assert b2["evidence_version"] == b1["evidence_version"]
    assert c1["source_assessment_version"] == b2["assessment_version"]
    assert c2["source_assessment_version"] == c1["assessment_version"]
    assert c1["status"] == c2["status"] == "available"


def test_independent_fold_reaggregation_and_ranking():
    results, trusted = _case()
    b1, b2, c1, _ = _chain(results, trusted)
    scores = {row["model"]: row["fold_rmse"] for row in b1["models"]}
    baseline = {model: fsum(folds) / len(folds) for model, folds in scores.items()}
    assert {row["model"]: row["baseline_rmse"] for row in b2["models"]} == baseline
    assert {row["model"]: row["rank"] for row in c1["baseline_ranking"]} == _competition_rank(baseline)
    for omission in c1["fold_omissions"]:
        index = omission["fold_index"]
        recomputed = {model: fsum(folds[:index] + folds[index + 1:]) / (len(folds) - 1)
                      for model, folds in scores.items()}
        assert {row["model"]: row["score"] for row in omission["ranking"]} == recomputed
        assert {row["model"]: row["rank"] for row in omission["ranking"]} == _competition_rank(recomputed)


def test_independent_stability_aggregates_from_c1_rankings():
    results, trusted = _case()
    _, _, c1, c2 = _chain(results, trusted)
    baseline = {row["model"]: row["rank"] for row in c1["baseline_ranking"]}
    baseline_winners = {model for model, rank in baseline.items() if rank == 1}
    per_model = {model: 0 for model in sorted(baseline)}
    rank_change_count = winner_change_count = maximum = 0
    for omission in c1["fold_omissions"]:
        ranks = {row["model"]: row["rank"] for row in omission["ranking"]}
        changed = {model: abs(rank - baseline[model]) for model, rank in ranks.items()
                   if rank != baseline[model]}
        rank_change_count += bool(changed)
        winner_change_count += ({model for model, rank in ranks.items() if rank == 1}
                                != baseline_winners)
        for model, displacement in changed.items():
            per_model[model] += 1
            maximum = max(maximum, displacement)
    count = len(c1["fold_omissions"])
    assert c2["rank_change_count"] == rank_change_count
    assert c2["winner_set_change_count"] == winner_change_count
    assert c2["rank_stability_fraction"] == (count - rank_change_count) / count
    assert c2["winner_set_stability_fraction"] == (count - winner_change_count) / count
    assert c2["max_absolute_rank_displacement"] == maximum
    assert c2["per_model_rank_change_count"] == per_model
    assert c2["baseline_top_models"] == sorted(baseline_winners)


def test_exact_tie_winner_set_change_through_all_layers():
    results, trusted = _case()
    _, _, c1, c2 = _chain(results, trusted)
    assert c1["baseline_top_models"] == ["GRU", "LSTM"]
    assert c2["rank_change_count"] == 2
    assert c2["winner_set_change_count"] == 2
    assert c2["rank_stability_fraction"] == 0.0
    assert c2["winner_set_stability_fraction"] == 0.0
    assert c2["max_absolute_rank_displacement"] == 1


def test_chain_is_deterministic_read_only_and_preserves_m18_ranking():
    results, trusted = _case()
    before_results = {model: result.to_dict() for model, result in results.items()}
    before_context = deepcopy(trusted)
    original_ranking = rank_internal_models(results, validation_context=trusted)
    first = _chain(results, trusted)
    reversed_results = dict(reversed(list(results.items())))
    assert first == _chain(reversed_results, trusted)
    assert first == _chain(results, trusted)
    assert rank_internal_models(results, validation_context=trusted) == original_ranking
    assert original_ranking["selected_model"] is None
    assert {model: result.to_dict() for model, result in results.items()} == before_results
    assert trusted == before_context


@pytest.mark.parametrize("tamper", [
    "fold_metric", "nonfinite_fold_metric", "alignment_dates", "alignment_status",
    "profile_prediction", "profile_aggregate", "candidate_metric",
    "provenance_id", "provenance_fingerprint", "context_scope", "context_fingerprint",
])
def test_all_four_layers_reject_corrupted_source_evidence(tamper):
    results, trusted = _case()
    candidate = results["LSTM"].candidates[0]
    metadata = candidate.metadata
    if tamper == "fold_metric":
        metadata["fold_metrics"][0]["rmse"] += 1
    elif tamper == "nonfinite_fold_metric":
        metadata["fold_metrics"][0]["rmse"] = float("nan")
    elif tamper == "alignment_dates":
        metadata["fold_alignment"][0]["forecast_dates"][0] = "2030-03-01"
    elif tamper == "alignment_status":
        metadata["fold_alignment"][0]["status"] = "positional_only"
    elif tamper == "profile_prediction":
        metadata["accuracy_profile"]["by_date"][0]["predicted_microVolt"] += 1
    elif tamper == "profile_aggregate":
        metadata["accuracy_profile"]["overall"]["rmse_microVolt"] += 1
    elif tamper == "candidate_metric":
        candidate.metrics["rmse"] += 1
    elif tamper == "provenance_id":
        metadata["internal_provenance"]["dataset_id"] = "other"
    elif tamper == "provenance_fingerprint":
        metadata["internal_provenance"]["dataset_fingerprint"] = "sha256:" + "0" * 64
    elif tamper == "context_scope":
        trusted["scope"] = "external_holdout"
    else:
        trusted["dataset_fingerprint"] = "sha256:" + "0" * 64
    for assessment in (
        extract_verified_internal_fold_scores,
        assess_internal_fold_sensitivity,
        assess_internal_ranking_sensitivity,
        assess_internal_ranking_stability,
    ):
        with pytest.raises(ValueError):
            assessment(results, validation_context=trusted)


def test_one_model_unavailable_propagates_without_fabricated_stability():
    trusted = trusted_context()
    results = {"LSTM": make_result("LSTM", trusted)}
    _, b2, c1, c2 = _chain(results, trusted)
    assert b2["models"][0]["status"] == "available"
    assert c1["status"] == c2["status"] == "unavailable_insufficient_models"
    assert c1["any_rank_change"] is None
    for field in ("rank_change_count", "winner_set_change_count",
                  "rank_stability_fraction", "winner_set_stability_fraction",
                  "max_absolute_rank_displacement", "per_model_rank_change_count"):
        assert c2[field] is None


def test_missing_trusted_context_rejected_at_every_layer():
    results, _ = _case()
    for assessment in (
        extract_verified_internal_fold_scores,
        assess_internal_fold_sensitivity,
        assess_internal_ranking_sensitivity,
        assess_internal_ranking_stability,
    ):
        with pytest.raises((ValueError, TypeError)):
            assessment(results, validation_context=None)
