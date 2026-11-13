"""M18-B3 red-phase: ranking requires evaluation-bound development provenance.

No external holdout, model fitting, or changes to optimizer selection are needed.
Fixtures use canonical M17 accuracy profiles to preserve B4 ranking checks.
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


DATASET_ID = "synthetic-m18-development"


def development_dataset():
    return BatteryDataset(pd.DataFrame({
        "ds": pd.date_range("2020-03-01", periods=16, freq="QS-MAR"),
        "microVolt": np.arange(1, 17, dtype=float),
    }))


def trusted_context():
    return build_internal_validation_context(
        development_dataset(), ExpandingWindowSplitter(folds=2, validation_steps=2),
        dataset_id=DATASET_ID,
    )


def make_result(model, trusted, *, fold_rmse=(0.2, 0.4), provenance=None):
    dates = trusted["fold_validation_dates"]
    # Use constant within-fold errors to realize each requested RMSE exactly.
    measured_folds = []
    for index, (block, fold_score) in enumerate(zip(dates, fold_rmse)):
        actual = np.full(len(block), 10.0)
        measured_folds.append({
            "fold_index": index,
            "dates": pd.DatetimeIndex(block),
            "actual": actual,
            "predicted": actual + float(fold_score),
            "alignment_status": "verified",
        })
    diagnostics = build_accuracy_diagnostics(
        measured_folds, dataset=DATASET_ID, model=model,
        scope="internal_expanding_window",
    )
    profile = build_accuracy_profile(diagnostics)
    score = sum(fold_rmse) / len(fold_rmse)
    metadata = {
        "validation": "expanding_window",
        "selection_metric": "rmse",
        "folds": len(dates),
        "validation_steps": len(dates[0]),
        "fold_metrics": [{"rmse": value} for value in fold_rmse],
        "fold_alignment": [{
            "fold_index": index,
            "status": "verified",
            "validation_steps": len(block),
            "forecast_horizon_provided": True,
            "forecast_dates_provided": True,
            "timestamp_match": True,
            "validation_dates": list(block),
            "forecast_dates": list(block),
        } for index, block in enumerate(dates)],
        "accuracy_profile": profile,
    }
    if provenance is not None:
        metadata["internal_provenance"] = deepcopy(provenance)
    selected = CandidateResult(
        params={"window": 4}, score=score, metrics={"rmse": score}, metadata=metadata,
    )
    return OptimizationResult(
        model=model, metric="rmse", best_params=selected.params,
        best_score=score, candidates=(selected,),
        metadata={"validation": "internal_temporal"},
    )


def rank_with(trusted, *, model="LSTM", provenance=None):
    return rank_internal_models(
        {model: make_result(model, trusted, provenance=provenance)},
        validation_context=trusted,
    )


def test_matching_provenance_allows_multi_model_ranking():
    trusted = trusted_context()
    report = rank_internal_models({
        "LSTM": make_result("LSTM", trusted, fold_rmse=(0.3, 0.5), provenance=trusted),
        "GRU": make_result("GRU", trusted, fold_rmse=(0.1, 0.3), provenance=trusted),
    }, validation_context=trusted)
    assert [(r["model"], r["rank"]) for r in report["ranking"]] == [
        ("GRU", 1), ("LSTM", 2),
    ]
    assert report["selected_model"] is None


def test_missing_candidate_provenance_is_rejected():
    trusted = trusted_context()
    with pytest.raises(ValueError, match="provenance"):
        rank_with(trusted)


def test_missing_trusted_fingerprint_is_rejected():
    trusted = trusted_context()
    recorded = deepcopy(trusted)
    trusted.pop("dataset_fingerprint")
    with pytest.raises(ValueError, match="fingerprint|provenance"):
        rank_with(trusted, provenance=recorded)


def test_historical_measurement_mismatch_is_rejected():
    trusted = trusted_context()
    changed = development_dataset()
    changed.data.loc[0, "microVolt"] += 0.125
    recorded = build_internal_validation_context(
        changed, ExpandingWindowSplitter(folds=2, validation_steps=2),
        dataset_id=DATASET_ID,
    )
    assert recorded["fold_validation_dates"] == trusted["fold_validation_dates"]
    assert recorded["dataset_fingerprint"] != trusted["dataset_fingerprint"]
    with pytest.raises(ValueError, match="fingerprint|provenance"):
        rank_with(trusted, provenance=recorded)


@pytest.mark.parametrize("field,value", [
    ("dataset_id", "other-development"),
    ("scope", "external_holdout"),
    ("protocol_version", "unsupported-protocol"),
    ("dataset_fingerprint", "sha256:" + "0" * 64),
    ("folds", 3),
    ("validation_steps", 3),
])
def test_mismatched_candidate_provenance_rejected(field, value):
    trusted = trusted_context()
    recorded = deepcopy(trusted)
    recorded[field] = value
    with pytest.raises(ValueError, match="provenance|fingerprint|protocol|fold|scope|dataset"):
        rank_with(trusted, provenance=recorded)


@pytest.mark.parametrize("field", [
    "dataset_fingerprint", "protocol_version", "fold_training_end_dates",
])
def test_missing_candidate_provenance_field_rejected(field):
    trusted = trusted_context()
    recorded = deepcopy(trusted)
    recorded.pop(field)
    with pytest.raises(ValueError, match="provenance|fingerprint|protocol|training"):
        rank_with(trusted, provenance=recorded)


@pytest.mark.parametrize("field", ["fold_validation_dates", "fold_training_end_dates"])
def test_mismatched_fold_boundary_rejected(field):
    trusted = trusted_context()
    recorded = deepcopy(trusted)
    if field == "fold_validation_dates":
        recorded[field][0][0] = "2000-03-01"
    else:
        recorded[field][0] = "2000-03-01"
    with pytest.raises(ValueError, match="provenance|fold|date|training"):
        rank_with(trusted, provenance=recorded)


def test_mixed_valid_and_invalid_models_cannot_be_ranked_together():
    trusted = trusted_context()
    wrong = deepcopy(trusted)
    wrong["dataset_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="fingerprint|provenance"):
        rank_internal_models({
            "LSTM": make_result("LSTM", trusted, provenance=trusted),
            "GRU": make_result("GRU", trusted, provenance=wrong),
        }, validation_context=trusted)


def test_exact_ties_remain_shared_and_deterministic():
    trusted = trusted_context()
    results = {
        "LSTM": make_result("LSTM", trusted, provenance=trusted),
        "GRU": make_result("GRU", trusted, provenance=trusted),
    }
    first = rank_internal_models(results, validation_context=trusted)
    reverse = rank_internal_models(dict(reversed(list(results.items()))), validation_context=trusted)
    assert first["ranking"] == reverse["ranking"]
    assert [row["rank"] for row in first["ranking"]] == [1, 1]
    assert first["selected_model"] is None


def test_ranking_does_not_mutate_context_or_results():
    trusted = trusted_context()
    results = {"LSTM": make_result("LSTM", trusted, provenance=trusted)}
    before_context = deepcopy(trusted)
    before_results = {key: value.to_dict() for key, value in results.items()}
    rank_internal_models(results, validation_context=trusted)
    assert trusted == before_context
    assert {key: value.to_dict() for key, value in results.items()} == before_results
