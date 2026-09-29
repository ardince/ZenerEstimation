"""Tests for the standardized optimization artifact contract."""

from __future__ import annotations

import json

import pytest

from zenerestimation.optimization import (
    CandidateResult,
    OptimizationArtifact,
    OptimizationResult,
)


@pytest.fixture
def optimization_result() -> OptimizationResult:
    """Return a result containing evaluated and failed candidates."""

    first = CandidateResult(
        params={
            "order": (1, 1, 0),
        },
        score=0.8,
        metrics={
            "rmse": 0.8,
            "mae": 0.6,
            "mape": 0.4,
        },
        metadata={
            "validation": "expanding_window",
            "folds": 3,
            "validation_steps": 5,
            "selection_metric": "rmse",
            "fold_metrics": [
                {
                    "rmse": 1.0,
                    "mae": 0.8,
                    "mape": 0.5,
                },
                {
                    "rmse": 0.8,
                    "mae": 0.6,
                    "mape": 0.4,
                },
                {
                    "rmse": 0.6,
                    "mae": 0.4,
                    "mape": 0.3,
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

    best = CandidateResult(
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
            "preprocessing": {
                "enabled": True,
                "name": "TemporalPreprocessor",
                "method": "linear",
                "fill_edges": True,
            },
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
            first,
            failed,
            best,
        ),
        metadata={
            "search": "explicit_candidates",
            "objective": "minimize",
            "candidate_count": 3,
            "validation": "internal_temporal",
        },
    )


def test_requires_optimization_result():
    with pytest.raises(
        TypeError,
        match="OptimizationResult",
    ):
        OptimizationArtifact(
            "not a result"
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_rejects_nonpositive_benchmark_steps(
    optimization_result,
    value,
):
    with pytest.raises(ValueError):
        OptimizationArtifact(
            optimization_result,
            benchmark_steps=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        5.0,
        "5",
    ],
)
def test_rejects_invalid_benchmark_step_type(
    optimization_result,
    value,
):
    with pytest.raises(TypeError):
        OptimizationArtifact(
            optimization_result,
            benchmark_steps=value,
        )


def test_preserves_model_metric_and_status(
    optimization_result,
):
    artifact = OptimizationArtifact(
        optimization_result,
        benchmark_steps=5,
    )

    payload = artifact.to_dict()

    assert payload["status"] == "optimized"
    assert payload["model"] == "ARIMA"
    assert payload["metric"] == "rmse"


def test_preserves_candidate_counts(
    optimization_result,
):
    artifact = OptimizationArtifact(
        optimization_result,
        benchmark_steps=5,
    )

    search = artifact.to_dict()["search"]

    assert search["candidate_count"] == 3
    assert search["successful_count"] == 2
    assert search["failed_count"] == 1

    assert (
        search["successful_count"]
        + search["failed_count"]
        == search["candidate_count"]
    )


def test_preserves_candidate_order(
    optimization_result,
):
    artifact = OptimizationArtifact(
        optimization_result,
        benchmark_steps=5,
    )

    candidates = artifact.to_dict()[
        "candidates"
    ]

    assert candidates[0]["params"] == {
        "order": (1, 1, 0),
    }

    assert candidates[1]["params"] == {
        "order": (4, 0, 0),
    }

    assert candidates[2]["params"] == {
        "order": (5, 1, 3),
    }


def test_preserves_failed_candidate_metadata(
    optimization_result,
):
    artifact = OptimizationArtifact(
        optimization_result,
        benchmark_steps=5,
    )

    failed = artifact.to_dict()[
        "candidates"
    ][1]

    assert failed["status"] == "failed"
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


def test_best_candidate_contains_complete_metrics(
    optimization_result,
):
    artifact = OptimizationArtifact(
        optimization_result,
        benchmark_steps=5,
    )

    best = artifact.to_dict()["best"]

    assert best["params"] == {
        "order": (5, 1, 3),
    }

    assert best["score"] == pytest.approx(
        0.5
    )

    assert best["metrics"] == {
        "rmse": 0.5,
        "mae": 0.4,
        "mape": 0.25,
    }


def test_validation_contract(
    optimization_result,
):
    artifact = OptimizationArtifact(
        optimization_result,
        benchmark_steps=5,
    )

    validation = artifact.to_dict()[
        "validation"
    ]

    assert (
        validation["method"]
        == "expanding_window"
    )

    assert validation["folds"] == 3
    assert validation["validation_steps"] == 5
    assert validation["benchmark_steps"] == 5
    assert validation["horizon_aligned"] is True


def test_horizon_alignment_false(
    optimization_result,
):
    artifact = OptimizationArtifact(
        optimization_result,
        benchmark_steps=6,
    )

    assert artifact.horizon_aligned is False

    assert (
        artifact.to_dict()[
            "validation"
        ]["horizon_aligned"]
        is False
    )


def test_horizon_alignment_unknown_without_benchmark(
    optimization_result,
):
    artifact = OptimizationArtifact(
        optimization_result
    )

    validation = artifact.to_dict()[
        "validation"
    ]

    assert artifact.horizon_aligned is None
    assert validation["benchmark_steps"] is None
    assert validation["horizon_aligned"] is None


def test_artifact_does_not_mutate_result(
    optimization_result,
):
    original_params = dict(
        optimization_result.best_params
    )

    artifact = OptimizationArtifact(
        optimization_result,
        benchmark_steps=5,
    )

    artifact.to_dict()

    assert (
        optimization_result.best_params
        == original_params
    )


def test_returned_payload_is_defensively_independent(
    optimization_result,
):
    artifact = OptimizationArtifact(
        optimization_result,
        benchmark_steps=5,
    )

    payload = artifact.to_dict()

    payload["best"]["params"][
        "order"
    ] = (99, 99, 99)

    payload["candidates"][0][
        "metrics"
    ]["rmse"] = 999.0

    fresh = artifact.to_dict()

    assert fresh["best"]["params"] == {
        "order": (5, 1, 3),
    }

    assert (
        fresh["candidates"][0][
            "metrics"
        ]["rmse"]
        == 0.8
    )


def test_payload_is_json_serializable(
    optimization_result,
):
    artifact = OptimizationArtifact(
        optimization_result,
        benchmark_steps=5,
    )

    payload = artifact.to_dict()

    serialized = json.dumps(
        payload
    )

    assert isinstance(
        serialized,
        str,
    )


def test_artifact_contains_no_benchmark_evidence(
    optimization_result,
):
    artifact = OptimizationArtifact(
        optimization_result,
        benchmark_steps=5,
    )

    payload = artifact.to_dict()

    forbidden_keys = {
        "benchmark_rmse",
        "benchmark_mae",
        "benchmark_mape",
        "actual",
        "predicted",
        "predictions",
        "errors",
        "evaluation",
    }

    def collect_keys(value):
        keys = set()

        if isinstance(value, dict):
            for key, item in value.items():
                keys.add(key)
                keys.update(
                    collect_keys(item)
                )

        elif isinstance(value, list):
            for item in value:
                keys.update(
                    collect_keys(item)
                )

        return keys

    all_keys = collect_keys(
        payload
    )

    assert forbidden_keys.isdisjoint(
        all_keys
    )


def test_repr_contains_useful_summary(
    optimization_result,
):
    artifact = OptimizationArtifact(
        optimization_result,
        benchmark_steps=5,
    )

    text = repr(
        artifact
    )

    assert "OptimizationArtifact" in text
    assert "ARIMA" in text
    assert "rmse" in text
    assert "benchmark_steps=5" in text