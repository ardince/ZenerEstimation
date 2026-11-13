"""M18-C1 RED phase: controlled selection from verified internal rankings.

Expected initial failure: ModuleNotFoundError for internal_selection.
The test module intentionally does not modify production code.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.evaluation.accuracy_diagnostics import build_accuracy_diagnostics
from zenerestimation.evaluation.accuracy_profile import build_accuracy_profile
from zenerestimation.optimization.internal_provenance import build_internal_validation_context
from zenerestimation.optimization.validation import ExpandingWindowSplitter
from zenerestimation.optimization.candidate import CandidateResult
from zenerestimation.optimization.result import OptimizationResult
from zenerestimation.optimization.internal_ranking import rank_internal_models
from zenerestimation.optimization.internal_selection import select_internal_model

DATES = (
    ("2022-03-01", "2022-06-01"),
    ("2022-09-01", "2022-12-01"),
)


def context(*, dataset_id="synthetic-dev-v1", dates=DATES):
    """Trusted context supplied independently of candidate metadata."""
    development = BatteryDataset(pd.DataFrame({
        "ds": pd.date_range("2020-03-01", periods=12, freq="QS-MAR"),
        "microVolt": np.arange(1, 13, dtype=float),
    }))
    trusted = build_internal_validation_context(
        development,
        ExpandingWindowSplitter(folds=2, validation_steps=2),
        dataset_id=dataset_id,
    )
    # Preserve the original B1 test helper's ability to supply deliberately
    # inconsistent validation dates for negative contract tests.
    if dates != DATES:
        trusted["fold_validation_dates"] = [list(fold) for fold in dates]
    return trusted


def candidate(
    rmse=(0.2, 0.4),
    *,
    dates=DATES,
    statuses=("verified", "verified"),
    selection_metric="rmse",
    score=None,
    scope="internal_expanding_window",
    model="LSTM",
    dataset_id="synthetic-dev-v1",
):
    mean_rmse = sum(rmse) / len(rmse)

    if score is None:
        score = mean_rmse

    measured_folds = []
    for index, (fold_dates, fold_rmse) in enumerate(zip(dates, rmse)):
        actual = np.full(len(fold_dates), 10.0)
        measured_folds.append({
            "fold_index": index,
            "dates": pd.DatetimeIndex(fold_dates),
            "actual": actual,
            "predicted": actual + float(fold_rmse),
            "alignment_status": "verified",
        })
    diagnostics = build_accuracy_diagnostics(
        measured_folds, dataset=dataset_id, model=model,
        scope=scope,
    )
    profile = build_accuracy_profile(diagnostics)

    return CandidateResult(
        params={"window": 4},
        score=score,
        metrics={"rmse": mean_rmse},
        metadata={
            "validation": "expanding_window",
            "folds": len(rmse),
            "validation_steps": 2,
            "selection_metric": selection_metric,
            "fold_metrics": [
                {"rmse": value}
                for value in rmse
            ],
            "fold_alignment": [
                {
                    "fold_index": index,
                    "status": status,
                    "validation_steps": len(fold_dates),
                    "forecast_horizon_provided": True,
                    "forecast_dates_provided": status == "verified",
                    "validation_dates": list(fold_dates),
                    "forecast_dates": (
                        list(fold_dates)
                        if status == "verified"
                        else None
                    ),
                    "timestamp_match": (
                        True if status == "verified" else None
                    ),
                }
                for index, (fold_dates, status)
                in enumerate(zip(dates, statuses))
            ],
            "internal_provenance": deepcopy(context(dataset_id=dataset_id)),
            "accuracy_profile": profile,
        },
    )


def result(model, *, rmse=(0.2, 0.4), **candidate_overrides):
    selected = candidate(rmse, model=model, **candidate_overrides)

    return OptimizationResult(
        model=model,
        metric="rmse",
        best_params=selected.params,
        best_score=selected.score,
        candidates=(selected,),
        metadata={"validation": "internal_temporal"},
    )



def select(results, *, authorized=False, validation_context=None):
    return select_internal_model(
        results,
        validation_context=context() if validation_context is None else validation_context,
        authorized=authorized,
    )


def unique_results():
    return {
        "LSTM": result("LSTM", rmse=(0.1, 0.6)),
        "GRU": result("GRU", rmse=(0.2, 0.4)),
    }


def assert_common_contract(decision):
    assert decision["selection_version"] == "m18_internal_selection_v1"
    assert decision["ranking_version"] == "m18_internal_ranking_v1"
    assert decision["selection_metric"] == "rmse"
    assert decision["ranking"]


def test_unique_winner_requires_authorization():
    decision = select(unique_results())
    assert_common_contract(decision)
    assert decision["status"] == "PENDING_AUTHORIZATION"
    assert decision["selected_model"] is None
    assert decision["recommended_model"] == "GRU"
    assert decision["selection_score"] == pytest.approx(0.3)
    assert decision["authorized"] is False


def test_explicit_authorization_selects_unique_winner():
    decision = select(unique_results(), authorized=True)
    assert_common_contract(decision)
    assert decision["status"] == "SELECTED"
    assert decision["selected_model"] == "GRU"
    assert decision["recommended_model"] == "GRU"
    assert decision["selection_score"] == pytest.approx(0.3)
    assert decision["authorized"] is True


@pytest.mark.parametrize("authorized", [False, True])
def test_exact_two_way_tie_is_unresolved(authorized):
    decision = select({"LSTM": result("LSTM"), "GRU": result("GRU")}, authorized=authorized)
    assert_common_contract(decision)
    assert decision["status"] == "TIE_UNRESOLVED"
    assert decision["selected_model"] is None
    assert decision["recommended_model"] is None
    assert decision["selection_score"] is None
    assert [row["rank"] for row in decision["ranking"]] == [1, 1]


def test_three_way_exact_tie_is_unresolved():
    models = {name: result(name) for name in ("LSTM", "GRU", "Kalman")}
    decision = select(models, authorized=True)
    assert decision["status"] == "TIE_UNRESOLVED"
    assert decision["selected_model"] is None
    assert [row["rank"] for row in decision["ranking"]] == [1, 1, 1]


def test_nearly_equal_but_distinct_scores_do_not_tie():
    models = {
        "LSTM": result("LSTM", rmse=(0.2, 0.40000000000001)),
        "GRU": result("GRU", rmse=(0.2, 0.4)),
    }
    decision = select(models, authorized=True)
    assert decision["status"] == "SELECTED"
    assert decision["selected_model"] == "GRU"
    assert [row["rank"] for row in decision["ranking"]] == [1, 2]


def test_input_order_does_not_change_decision():
    models = unique_results()
    first = select(models, authorized=True)
    second = select(dict(reversed(list(models.items()))), authorized=True)
    assert first == second


def test_authorization_does_not_change_scientific_ranking():
    models = unique_results()
    pending = select(models)
    selected = select(models, authorized=True)
    trusted_ranking = rank_internal_models(models, validation_context=context())
    assert pending["ranking"] == selected["ranking"] == trusted_ranking["ranking"]


@pytest.mark.parametrize("authorized", [None, 0, 1, "true", [], {}])
def test_authorization_requires_literal_boolean(authorized):
    with pytest.raises((TypeError, ValueError)):
        select(unique_results(), authorized=authorized)


def test_empty_results_rejected_by_existing_ranking_contract():
    with pytest.raises(ValueError):
        select({}, authorized=True)


def test_dataset_fingerprint_or_identity_mismatch_rejected():
    with pytest.raises(ValueError):
        select(unique_results(), authorized=True, validation_context=context(dataset_id="other-dev"))


def test_corrupted_canonical_accuracy_profile_rejected():
    models = unique_results()
    models["GRU"].candidates[0].metadata["accuracy_profile"]["provenance"]["model"] = "LSTM"
    with pytest.raises(ValueError):
        select(models, authorized=True)


def test_external_validation_scope_rejected():
    invalid = context()
    invalid["scope"] = "external_holdout"
    with pytest.raises(ValueError):
        select(unique_results(), authorized=True, validation_context=invalid)


def test_new_dataset_identifier_uses_same_selection_contract():
    dataset_id = "new-battery-510"
    models = {"GRU": result("GRU", dataset_id=dataset_id)}
    decision = select(models, authorized=True, validation_context=context(dataset_id=dataset_id))
    assert decision["status"] == "SELECTED"
    assert decision["selected_model"] == "GRU"


def test_selection_does_not_mutate_results_or_context():
    models = unique_results()
    trusted = context()
    before_results = {name: obj.to_dict() for name, obj in models.items()}
    before_context = deepcopy(trusted)
    select(models, authorized=True, validation_context=trusted)
    assert {name: obj.to_dict() for name, obj in models.items()} == before_results
    assert trusted == before_context


def test_failed_nonselected_candidate_does_not_block_selection():
    good = candidate(model="LSTM")
    failed = CandidateResult.failed(params={"window": 99}, error=RuntimeError("fit failed"))
    models = {"LSTM": OptimizationResult(
        model="LSTM", metric="rmse", best_params=good.params,
        best_score=good.score, candidates=(failed, good),
        metadata={"validation": "internal_temporal"},
    )}
    decision = select(models, authorized=True)
    assert decision["status"] == "SELECTED"
    assert decision["selected_model"] == "LSTM"
