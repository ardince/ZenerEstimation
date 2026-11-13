
"""M18-B1 contract tests for strict internal-validation model ranking.

Proposed API:

    rank_internal_models(results, *, validation_context) -> report

Results map model labels to OptimizationResult objects. The trusted
validation_context supplies the development-dataset identity and exact
fold validation dates.

No external holdout observations, training, or repository artifacts
are needed.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import numpy as np
import pandas as pd

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.evaluation.accuracy_diagnostics import build_accuracy_diagnostics
from zenerestimation.evaluation.accuracy_profile import build_accuracy_profile
from zenerestimation.optimization.internal_provenance import build_internal_validation_context
from zenerestimation.optimization.validation import ExpandingWindowSplitter

from zenerestimation.optimization.candidate import CandidateResult
from zenerestimation.optimization.result import OptimizationResult
from zenerestimation.optimization.internal_ranking import rank_internal_models


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
        measured_folds, dataset="synthetic-dev-v1", model=model,
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
            "internal_provenance": deepcopy(context()),
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


def rank(results, *, validation_context=None):
    return rank_internal_models(
        results,
        validation_context=(
            context()
            if validation_context is None
            else validation_context
        ),
    )


def test_ranks_mean_fold_rmse_ascending_not_pooled_rmse():
    # Mean fold RMSE: GRU=.30; LSTM=.35.
    report = rank({
        "LSTM": result("LSTM", rmse=(0.1, 0.6)),
        "GRU": result("GRU", rmse=(0.2, 0.4)),
    })

    assert [row["model"] for row in report["ranking"]] == [
        "GRU", "LSTM"
    ]

    assert [row["rank"] for row in report["ranking"]] == [1, 2]

    assert [
        row["score"] for row in report["ranking"]
    ] == pytest.approx([0.3, 0.35])


def test_order_invariance():
    a = {
        "LSTM": result("LSTM", rmse=(0.1, 0.6)),
        "GRU": result("GRU"),
    }

    assert rank(a)["ranking"] == rank(
        dict(reversed(list(a.items())))
    )["ranking"]


def test_equal_scores_share_rank_without_implicit_winner():
    report = rank({
        "LSTM": result("LSTM"),
        "GRU": result("GRU"),
    })

    assert {row["rank"] for row in report["ranking"]} == {1}
    assert {row["model"] for row in report["ranking"]} == {
        "LSTM", "GRU"
    }
    assert report.get("selected_model") is None


def test_failed_candidate_reported_ineligible():
    failed = CandidateResult.failed(
        params={"window": 4},
        error=RuntimeError("fit failed"),
    )

    failed_result = OptimizationResult(
        model="LSTM",
        metric="rmse",
        best_params={"window": 4},
        best_score=0.3,
        candidates=(failed, candidate()),
        metadata={"validation": "internal_temporal"},
    )

    # A failed nonselected candidate does not invalidate
    # the successfully evaluated selected candidate.
    report = rank({"LSTM": failed_result})

    assert report["ranking"][0]["model"] == "LSTM"


@pytest.mark.parametrize("change", [
    {"statuses": ("verified", "positional_only")},
    {
        "dates": (
            ("2022-03-01", "2022-06-01"),
            ("2023-03-01", "2023-06-01"),
        )
    },
    {"score": 0.9},
    {"scope": "external_holdout"},
])
def test_rejects_invalid_or_incomparable_evidence(change):
    with pytest.raises(ValueError):
        rank({"LSTM": result("LSTM", **change)})


def test_rejects_missing_fold_dates():
    selected = candidate()
    selected.metadata["fold_alignment"][0].pop("validation_dates")

    res = OptimizationResult(
        model="LSTM",
        metric="rmse",
        best_params=selected.params,
        best_score=selected.score,
        candidates=(selected,),
    )

    with pytest.raises(ValueError):
        rank({"LSTM": res})


def test_rejects_mismatched_development_dataset_context():
    with pytest.raises(ValueError):
        rank(
            {"LSTM": result("LSTM")},
            validation_context=context(dataset_id="other-dev"),
        )


def test_rejects_missing_trusted_context():
    with pytest.raises((TypeError, ValueError)):
        rank_internal_models(
            {"LSTM": result("LSTM")},
            validation_context=None,
        )


def test_rejects_external_holdout_context():
    invalid = context()
    invalid["scope"] = "external_holdout"

    with pytest.raises(ValueError):
        rank(
            {"LSTM": result("LSTM")},
            validation_context=invalid,
        )


def test_does_not_mutate_results_or_context():
    results = {
        "LSTM": result("LSTM"),
        "GRU": result("GRU", rmse=(0.3, 0.5)),
    }

    trusted = context()

    before_results = {
        name: item.to_dict()
        for name, item in results.items()
    }
    before_context = deepcopy(trusted)

    rank(results, validation_context=trusted)

    assert {
        name: item.to_dict()
        for name, item in results.items()
    } == before_results

    assert trusted == before_context


def test_rejects_duplicate_model_identity():
    with pytest.raises(ValueError):
        rank({"LSTM": result("GRU")})
