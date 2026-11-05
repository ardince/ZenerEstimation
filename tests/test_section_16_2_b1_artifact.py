"""Regression: optimization artifacts exclude full temporal evidence."""
from copy import deepcopy

from zenerestimation.optimization.artifact import OptimizationArtifact
from zenerestimation.optimization.candidate import CandidateResult
from zenerestimation.optimization.result import OptimizationResult


def test_artifact_alignment_summary_preserves_internal_evidence():
    full = {
        "fold_index": 0,
        "status": "verified",
        "validation_steps": 2,
        "forecast_horizon_provided": True,
        "forecast_dates_provided": True,
        "timestamp_match": True,
        "validation_dates": ["2025-03-01", "2025-06-01"],
        "forecast_dates": ["2025-03-01", "2025-06-01"],
    }
    candidate = CandidateResult(
        params={"window": 4}, score=0.1, metrics={"rmse": 0.1},
        metadata={"validation": "expanding_window", "folds": 1,
                  "validation_steps": 2, "fold_alignment": [deepcopy(full)]},
    )
    result = OptimizationResult(
        model="LSTM", metric="rmse", best_params={"window": 4},
        best_score=0.1, candidates=(candidate,),
    )
    artifact = OptimizationArtifact(result, benchmark_steps=2)
    payload = artifact.to_dict()
    exported = payload["candidates"][0]["metadata"]["fold_alignment"][0]
    assert exported == {
        key: full[key] for key in (
            "fold_index", "status", "validation_steps",
            "forecast_horizon_provided", "forecast_dates_provided",
            "timestamp_match",
        )
    }
    assert "forecast_dates" not in exported
    assert "validation_dates" not in exported
    assert candidate.metadata["fold_alignment"][0] == full
    assert artifact.to_dict() == payload


def test_artifact_legacy_candidate_without_alignment():
    candidate = CandidateResult(
        params={"window": 4}, score=0.1, metrics={"rmse": 0.1},
        metadata={"validation_steps": 2},
    )
    result = OptimizationResult(
        model="LSTM", metric="rmse", best_params={"window": 4},
        best_score=0.1, candidates=(candidate,),
    )
    payload = OptimizationArtifact(result).to_dict()
    assert "fold_alignment" not in payload["candidates"][0]["metadata"]
