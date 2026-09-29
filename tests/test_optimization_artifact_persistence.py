"""Persistence tests for standardized optimization artifacts."""

from __future__ import annotations

import json

from zenerestimation.optimization import (
    CandidateResult,
    OptimizationArtifact,
    OptimizationResult,
)

# IMPORTANT:
# Replace this import only if save_metadata lives elsewhere
# in the current repository.
from zenerestimation.utils.results import save_metadata


def _optimization_result() -> OptimizationResult:
    evaluated = CandidateResult(
        params={"order": (5, 1, 3)},
        score=0.5,
        metrics={
            "rmse": 0.5,
            "mae": 0.4,
            "mape": 0.25,
        },
        metadata={
            "validation": "expanding_window",
            "folds": 3,
            "validation_steps": 5,
            "selection_metric": "rmse",
            "fold_metrics": [
                {
                    "rmse": 0.7,
                    "mae": 0.5,
                    "mape": 0.3,
                },
                {
                    "rmse": 0.5,
                    "mae": 0.4,
                    "mape": 0.25,
                },
                {
                    "rmse": 0.3,
                    "mae": 0.3,
                    "mape": 0.2,
                },
            ],
            "preprocessing": {
                "enabled": True,
                "name": "TemporalPreprocessor",
                "method": "linear",
                "fill_edges": True,
            },
        },
    )

    failed = CandidateResult.failed(
        params={"order": (4, 0, 0)},
        error=RuntimeError(
            "synthetic numerical failure"
        ),
        metadata={
            "validation": "internal_temporal",
        },
    )

    return OptimizationResult(
        model="ARIMA",
        metric="rmse",
        best_params={"order": (5, 1, 3)},
        best_score=0.5,
        candidates=(
            evaluated,
            failed,
        ),
        metadata={
            "search": "explicit_candidates",
            "objective": "minimize",
            "candidate_count": 2,
            "validation": "internal_temporal",
        },
    )


def test_optimization_artifact_can_be_persisted(
    tmp_path,
):
    result = _optimization_result()

    artifact = OptimizationArtifact(
        result,
        benchmark_steps=5,
    )

    path = (
        tmp_path
        / "optimization.json"
    )

    save_metadata(
        path,
        artifact.to_dict(),
    )

    assert path.exists()

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    assert payload["status"] == "optimized"
    assert payload["model"] == "ARIMA"
    assert payload["metric"] == "rmse"

    assert payload["best"]["params"] == {
        "order": [5, 1, 3],
    }

    assert payload["best"]["score"] == 0.5

    assert (
        payload["search"]["candidate_count"]
        == 2
    )

    assert (
        payload["search"]["successful_count"]
        == 1
    )

    assert (
        payload["search"]["failed_count"]
        == 1
    )

    assert (
        payload["validation"][
            "validation_steps"
        ]
        == 5
    )

    assert (
        payload["validation"][
            "benchmark_steps"
        ]
        == 5
    )

    assert (
        payload["validation"][
            "horizon_aligned"
        ]
        is True
    )


def test_failed_candidate_is_persisted(
    tmp_path,
):
    artifact = OptimizationArtifact(
        _optimization_result(),
        benchmark_steps=5,
    )

    path = (
        tmp_path
        / "optimization.json"
    )

    save_metadata(
        path,
        artifact.to_dict(),
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    failed = payload[
        "candidates"
    ][1]

    assert failed["status"] == "failed"
    assert failed["params"] == {
        "order": [4, 0, 0],
    }

    assert failed["score"] is None
    assert failed["metrics"] == {}

    assert (
        failed["metadata"]["error_type"]
        == "RuntimeError"
    )

    assert (
        failed["metadata"]["error_message"]
        == "synthetic numerical failure"
    )


def test_optimization_json_contains_no_benchmark_results(
    tmp_path,
):
    artifact = OptimizationArtifact(
        _optimization_result(),
        benchmark_steps=5,
    )

    path = (
        tmp_path
        / "optimization.json"
    )

    save_metadata(
        path,
        artifact.to_dict(),
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    serialized = json.dumps(
        payload
    )

    forbidden = (
        "benchmark_rmse",
        "benchmark_mae",
        "benchmark_mape",
        '"actual"',
        '"predicted"',
        '"predictions"',
        '"errors"',
        '"evaluation"',
    )

    for token in forbidden:
        assert token not in serialized


def test_optimization_write_does_not_modify_evaluation_file(
    tmp_path,
):
    evaluation_path = (
        tmp_path
        / "evaluation.json"
    )

    evaluation_payload = {
        "status": "evaluated",
        "rmse": 0.447650,
        "mae": 0.353932,
        "mape": 0.209373,
    }

    save_metadata(
        evaluation_path,
        evaluation_payload,
    )

    before = evaluation_path.read_text(
        encoding="utf-8"
    )

    artifact = OptimizationArtifact(
        _optimization_result(),
        benchmark_steps=5,
    )

    optimization_path = (
        tmp_path
        / "optimization.json"
    )

    save_metadata(
        optimization_path,
        artifact.to_dict(),
    )

    after = evaluation_path.read_text(
        encoding="utf-8"
    )

    assert before == after