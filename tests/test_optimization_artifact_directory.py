"""Regression tests for optimization artifacts in result directories."""

from __future__ import annotations

import json

from zenerestimation.optimization import (
    CandidateResult,
    OptimizationArtifact,
    OptimizationResult,
)

# Use the same confirmed import that is already working in:
# tests/test_optimization_artifact_persistence.py
from zenerestimation.utils.results import save_metadata


def _optimization_result() -> OptimizationResult:
    """Build a small deterministic optimization result."""

    successful = CandidateResult(
        params={
            "order": (5, 1, 3),
        },
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
        },
    )

    failed = CandidateResult.failed(
        params={
            "order": (4, 0, 0),
        },
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
        best_params={
            "order": (5, 1, 3),
        },
        best_score=0.5,
        candidates=(
            successful,
            failed,
        ),
        metadata={
            "search": "explicit_candidates",
            "objective": "minimize",
            "candidate_count": 2,
            "validation": "internal_temporal",
        },
    )


def test_optimization_and_evaluation_artifacts_remain_separate(
    tmp_path,
):
    """Optimization and benchmark evidence remain separate files."""

    result_directory = (
        tmp_path
        / "732B-5610110"
        / "arima"
        / "test_run"
    )

    result_directory.mkdir(
        parents=True
    )

    optimization_path = (
        result_directory
        / "optimization.json"
    )

    evaluation_path = (
        result_directory
        / "evaluation.json"
    )

    optimization = OptimizationArtifact(
        _optimization_result(),
        benchmark_steps=5,
    )

    evaluation_payload = {
        "status": "evaluated",
        "model": "ARIMA",
        "evaluation_steps": 5,
        "rmse": 0.447650,
        "mae": 0.353932,
        "mape": 0.209373,
        "actual": [
            167.17,
            167.73,
            168.30,
            168.86,
            169.90,
        ],
        "predicted": [
            166.975089,
            167.690539,
            167.980270,
            168.498578,
            169.045862,
        ],
    }

    save_metadata(
        optimization_path,
        optimization.to_dict(),
    )

    save_metadata(
        evaluation_path,
        evaluation_payload,
    )

    assert optimization_path.exists()
    assert evaluation_path.exists()

    with optimization_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        optimization_payload = json.load(
            handle
        )

    with evaluation_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        persisted_evaluation = json.load(
            handle
        )

    # -----------------------------------------------------
    # Optimization artifact contract
    # -----------------------------------------------------

    assert (
        optimization_payload["status"]
        == "optimized"
    )

    assert (
        optimization_payload["model"]
        == "ARIMA"
    )

    assert (
        optimization_payload["best"]["params"]
        == {
            "order": [5, 1, 3],
        }
    )

    assert (
        optimization_payload["best"]["score"]
        == 0.5
    )

    assert (
        optimization_payload["search"][
            "candidate_count"
        ]
        == 2
    )

    assert (
        optimization_payload["search"][
            "successful_count"
        ]
        == 1
    )

    assert (
        optimization_payload["search"][
            "failed_count"
        ]
        == 1
    )

    assert (
        optimization_payload["validation"][
            "horizon_aligned"
        ]
        is True
    )

    # -----------------------------------------------------
    # Evaluation artifact contract
    # -----------------------------------------------------

    assert (
        persisted_evaluation["status"]
        == "evaluated"
    )

    assert (
        persisted_evaluation["rmse"]
        == 0.447650
    )

    assert "actual" in persisted_evaluation
    assert "predicted" in persisted_evaluation

    # -----------------------------------------------------
    # Scientific boundary regression
    # -----------------------------------------------------

    assert "rmse" not in optimization_payload
    assert "mae" not in optimization_payload
    assert "mape" not in optimization_payload

    assert "actual" not in optimization_payload
    assert "predicted" not in optimization_payload
    assert "evaluation" not in optimization_payload

    assert "best" not in persisted_evaluation
    assert "candidates" not in persisted_evaluation
    assert "search" not in persisted_evaluation

    # Internal model-selection score and final benchmark
    # score are intentionally different concepts.
    assert (
        optimization_payload["best"]["score"]
        != persisted_evaluation["rmse"]
    )


def test_optimization_artifact_uses_standard_filename(
    tmp_path,
):
    """Optimized runs use the canonical optimization filename."""

    result_directory = (
        tmp_path
        / "battery"
        / "arima"
        / "test_run"
    )

    result_directory.mkdir(
        parents=True
    )

    optimization_path = (
        result_directory
        / "optimization.json"
    )

    artifact = OptimizationArtifact(
        _optimization_result(),
        benchmark_steps=5,
    )

    save_metadata(
        optimization_path,
        artifact.to_dict(),
    )

    files = {
        path.name
        for path in result_directory.iterdir()
    }

    assert "optimization.json" in files