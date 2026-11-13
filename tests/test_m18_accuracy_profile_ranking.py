"""M18-B4 RED: ranking must verify canonical M17 measured accuracy profiles.

Run against the current M18-B3 ranking implementation first. The positive
cases should pass; the tampering cases should fail until B4 is implemented.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.evaluation.accuracy_diagnostics import build_accuracy_diagnostics
from zenerestimation.evaluation.accuracy_profile import build_accuracy_profile
from zenerestimation.optimization.candidate import CandidateResult
from zenerestimation.optimization.internal_provenance import build_internal_validation_context
from zenerestimation.optimization.internal_ranking import rank_internal_models
from zenerestimation.optimization.result import OptimizationResult
from zenerestimation.optimization.validation import ExpandingWindowSplitter

DATASET_ID = "m18-b4-synthetic-development"


def trusted_context():
    development = BatteryDataset(pd.DataFrame({
        "ds": pd.date_range("2020-03-01", periods=12, freq="QS-MAR"),
        "microVolt": np.arange(1, 13, dtype=float),
    }))
    return build_internal_validation_context(
        development, ExpandingWindowSplitter(folds=2, validation_steps=2),
        dataset_id=DATASET_ID,
    )


def make_result(model, trusted, *, errors=((0.2, 0.2), (0.4, 0.4))):
    folds = []
    for i, (date_strings, fold_errors) in enumerate(zip(trusted["fold_validation_dates"], errors)):
        actual = np.array([10.0, 11.0], dtype=float)
        predicted = actual + np.array(fold_errors, dtype=float)
        folds.append({
            "fold_index": i,
            "dates": pd.DatetimeIndex(date_strings),
            "actual": actual,
            "predicted": predicted,
            "alignment_status": "verified",
        })
    diagnostics = build_accuracy_diagnostics(
        folds, dataset=DATASET_ID, model=model, scope="internal_expanding_window",
    )
    profile = build_accuracy_profile(diagnostics)
    fold_rmse = [item["rmse_microVolt"] for item in diagnostics["fold_metrics"]]
    mean_rmse = sum(fold_rmse) / len(fold_rmse)
    metadata = {
        "validation": "expanding_window",
        "selection_metric": "rmse",
        "folds": len(folds),
        "validation_steps": 2,
        "fold_metrics": [{"rmse": score} for score in fold_rmse],
        "fold_alignment": [{
            "fold_index": i,
            "status": "verified",
            "validation_steps": len(dates),
            "forecast_horizon_provided": True,
            "forecast_dates_provided": True,
            "timestamp_match": True,
            "validation_dates": list(dates),
            "forecast_dates": list(dates),
        } for i, dates in enumerate(trusted["fold_validation_dates"])],
        "internal_provenance": deepcopy(trusted),
        "accuracy_profile": profile,
    }
    candidate = CandidateResult(
        params={"window": 4}, score=mean_rmse,
        metrics={"rmse": mean_rmse}, metadata=metadata,
    )
    return OptimizationResult(
        model=model, metric="rmse", best_params=candidate.params,
        best_score=mean_rmse, candidates=(candidate,),
        metadata={"validation": "internal_temporal"},
    )


def rank(results, trusted):
    return rank_internal_models(results, validation_context=trusted)


def profile_of(result):
    return result.candidates[0].metadata["accuracy_profile"]


def test_canonical_m17_profiles_rank_multiple_models():
    trusted = trusted_context()
    results = {
        "LSTM": make_result("LSTM", trusted, errors=((0.3, 0.3), (0.5, 0.5))),
        "GRU": make_result("GRU", trusted, errors=((0.1, 0.1), (0.3, 0.3))),
    }
    report = rank(results, trusted)
    assert [item["model"] for item in report["ranking"]] == ["GRU", "LSTM"]
    assert report["selected_model"] is None


def test_canonical_profile_ranking_is_read_only():
    trusted = trusted_context()
    results = {"LSTM": make_result("LSTM", trusted)}
    before_context = deepcopy(trusted)
    before_results = {name: item.to_dict() for name, item in results.items()}
    rank(results, trusted)
    assert trusted == before_context
    assert {name: item.to_dict() for name, item in results.items()} == before_results


@pytest.mark.parametrize("field", [
    "overall", "first_step", "by_horizon", "by_date", "fold_stability", "alignment_summary",
])
def test_missing_canonical_profile_field_rejected(field):
    trusted = trusted_context()
    result = make_result("LSTM", trusted)
    profile_of(result).pop(field)
    with pytest.raises(ValueError):
        rank({"LSTM": result}, trusted)


@pytest.mark.parametrize("field", [
    "model", "units", "alignment_statuses",
])
def test_tampered_profile_provenance_rejected(field):
    trusted = trusted_context()
    result = make_result("LSTM", trusted)
    replacements = {"model": "GRU", "units": "Volt", "alignment_statuses": ["positional_only"]}
    profile_of(result)["provenance"][field] = replacements[field]
    with pytest.raises(ValueError):
        rank({"LSTM": result}, trusted)


@pytest.mark.parametrize("path", [
    ("overall", "rmse_microVolt"),
    ("overall", "n"),
    ("first_step", "rmse_microVolt"),
    ("by_horizon", 0, "rmse_microVolt"),
    ("fold_stability", "rmse_mean_microVolt"),
    ("alignment_summary", "verified_rows"),
])
def test_corrupted_aggregate_rejected(path):
    trusted = trusted_context()
    result = make_result("LSTM", trusted)
    target = profile_of(result)
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] += 1
    with pytest.raises(ValueError):
        rank({"LSTM": result}, trusted)


@pytest.mark.parametrize("field", [
    "date", "horizon_step", "actual_microVolt", "predicted_microVolt",
    "error_microVolt", "absolute_error_microVolt", "squared_error_microVolt2",
    "alignment_status",
])
def test_tampered_observation_rejected(field):
    trusted = trusted_context()
    result = make_result("LSTM", trusted)
    row = profile_of(result)["by_date"][0]
    if field == "date":
        row[field] = "2029-03-01"
    elif field == "alignment_status":
        row[field] = "positional_only"
    else:
        row[field] += 1
    with pytest.raises(ValueError):
        rank({"LSTM": result}, trusted)


def test_missing_observation_rejected():
    trusted = trusted_context()
    result = make_result("LSTM", trusted)
    profile_of(result)["by_date"].pop()
    with pytest.raises(ValueError):
        rank({"LSTM": result}, trusted)


def test_duplicate_observation_rejected():
    trusted = trusted_context()
    result = make_result("LSTM", trusted)
    profile_of(result)["by_date"].append(deepcopy(profile_of(result)["by_date"][0]))
    with pytest.raises(ValueError):
        rank({"LSTM": result}, trusted)
