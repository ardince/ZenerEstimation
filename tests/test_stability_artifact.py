"""Regression tests for repeated-seed stability artifacts."""

import json
from pathlib import Path

import numpy as np
import pytest

from zenerestimation.optimization.stability import (
    FailedSeedResult,
    SeedResult,
    StabilityResult,
)
from zenerestimation.optimization.stability_artifact import (
    StabilityArtifact,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _seed_result(
    seed: int = 0,
    score: float = 0.5,
) -> SeedResult:
    """Create successful repeated-seed evidence."""

    return SeedResult(
        seed=seed,
        score=score,
        metrics={
            "rmse": score,
            "mae": score * 0.8,
            "mape": score * 2.0,
        },
        metadata={
            "validation": {
                "method": "expanding_window",
                "folds": 3,
                "validation_steps": 5,
            },
        },
    )


def _failed_seed_result(
    seed: int = 1,
) -> FailedSeedResult:
    """Create failed repeated-seed evidence."""

    return FailedSeedResult(
        seed=seed,
        error_type="ValueError",
        error_message=(
            "predicted must contain only finite values"
        ),
        metadata={
            "analysis": "repeated_seed",
        },
    )


def _stability_result() -> StabilityResult:
    """Create representative successful stability evidence."""

    return StabilityResult(
        model="LSTM",
        metric="rmse",
        params={
            "window": 4,
            "units": 32,
            "epochs": 100,
            "batch_size": 8,
        },
        seeds=(
            _seed_result(
                seed=0,
                score=0.5,
            ),
            _seed_result(
                seed=1,
                score=0.7,
            ),
            _seed_result(
                seed=2,
                score=0.6,
            ),
        ),
        metadata={
            "analysis": "repeated_seed",
            "seed_count": 3,
            "selection": "none",
        },
    )


def _mixed_stability_result() -> StabilityResult:
    """Create stability evidence containing a failed seed."""

    return StabilityResult(
        model="GRU",
        metric="rmse",
        params={
            "window": 4,
            "units": 32,
            "epochs": 100,
            "batch_size": 8,
        },
        seeds=(
            _seed_result(
                seed=0,
                score=0.5,
            ),
            _failed_seed_result(
                seed=1,
            ),
            _seed_result(
                seed=2,
                score=0.7,
            ),
        ),
        metadata={
            "analysis": "repeated_seed",
            "seed_count": 3,
            "selection": "none",
        },
    )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_stability_artifact_accepts_stability_result():
    """A StabilityResult should be accepted."""

    result = _stability_result()

    artifact = StabilityArtifact(
        result,
    )

    assert artifact.result is result


def test_stability_artifact_rejects_invalid_result():
    """Artifact construction must require StabilityResult."""

    with pytest.raises(
        TypeError,
        match="StabilityResult",
    ):
        StabilityArtifact(
            result="invalid",
        )


def test_stability_artifact_filename_is_standardized():
    """Stability evidence should use one canonical filename."""

    assert (
        StabilityArtifact.FILENAME
        == "stability.json"
    )


# ---------------------------------------------------------------------
# Serialization delegation
# ---------------------------------------------------------------------


def test_stability_artifact_to_dict_matches_result():
    """Artifact payload must exactly match StabilityResult."""

    result = _stability_result()

    artifact = StabilityArtifact(
        result,
    )

    assert (
        artifact.to_dict()
        == result.to_dict()
    )


def test_stability_artifact_preserves_frozen_params():
    """Frozen optimized parameters must survive serialization."""

    payload = StabilityArtifact(
        _stability_result()
    ).to_dict()

    assert payload["params"] == {
        "window": 4,
        "units": 32,
        "epochs": 100,
        "batch_size": 8,
    }


def test_stability_artifact_does_not_add_seed_to_params():
    """Seed must remain a replication dimension."""

    payload = StabilityArtifact(
        _stability_result()
    ).to_dict()

    assert "seed" not in payload["params"]


def test_stability_artifact_has_no_best_seed():
    """Repeated-seed evidence must not imply seed selection."""

    payload = StabilityArtifact(
        _stability_result()
    ).to_dict()

    assert "best_seed" not in payload


def test_stability_artifact_preserves_selection_none():
    """Metadata must explicitly preserve non-selection semantics."""

    payload = StabilityArtifact(
        _stability_result()
    ).to_dict()

    assert (
        payload["metadata"]["selection"]
        == "none"
    )


# ---------------------------------------------------------------------
# Summary evidence
# ---------------------------------------------------------------------


def test_stability_artifact_preserves_summary_statistics():
    """Aggregate statistics must come from StabilityResult."""

    result = _stability_result()

    payload = StabilityArtifact(
        result
    ).to_dict()

    assert np.isclose(
        payload["summary"]["mean"],
        result.mean_score,
    )

    assert np.isclose(
        payload["summary"]["std"],
        result.std_score,
    )

    assert np.isclose(
        payload["summary"]["min"],
        result.min_score,
    )

    assert np.isclose(
        payload["summary"]["max"],
        result.max_score,
    )


def test_stability_artifact_preserves_run_counts():
    """Attempted, successful, and failed counts must survive."""

    result = _mixed_stability_result()

    payload = StabilityArtifact(
        result
    ).to_dict()

    assert payload["run_count"] == 3
    assert payload["successful_count"] == 2
    assert payload["failed_count"] == 1


# ---------------------------------------------------------------------
# Failed-seed evidence
# ---------------------------------------------------------------------


def test_stability_artifact_preserves_failed_seed():
    """Failed seed evidence must not be silently discarded."""

    payload = StabilityArtifact(
        _mixed_stability_result()
    ).to_dict()

    failed = payload["seeds"][1]

    assert failed["seed"] == 1
    assert failed["status"] == "failed"
    assert failed["error_type"] == "ValueError"

    assert (
        failed["error_message"]
        == "predicted must contain only finite values"
    )


def test_stability_artifact_preserves_seed_order():
    """Seed execution order must survive serialization."""

    payload = StabilityArtifact(
        _mixed_stability_result()
    ).to_dict()

    assert [
        entry["seed"]
        for entry in payload["seeds"]
    ] == [
        0,
        1,
        2,
    ]


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------


def test_stability_artifact_save_creates_json(
    tmp_path,
):
    """Saving should create stability.json."""

    artifact = StabilityArtifact(
        _stability_result()
    )

    path = artifact.save(
        tmp_path
    )

    assert path == (
        tmp_path
        / "stability.json"
    )

    assert path.exists()
    assert path.is_file()


def test_stability_artifact_save_creates_directory(
    tmp_path,
):
    """Missing parent result directory should be created."""

    directory = (
        tmp_path
        / "battery"
        / "lstm"
        / "run"
    )

    artifact = StabilityArtifact(
        _stability_result()
    )

    path = artifact.save(
        directory
    )

    assert directory.exists()
    assert path.exists()


def test_stability_artifact_saved_json_matches_result(
    tmp_path,
):
    """Persisted JSON must equal the canonical result payload."""

    result = _stability_result()

    artifact = StabilityArtifact(
        result
    )

    path = artifact.save(
        tmp_path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    assert payload == result.to_dict()


def test_stability_artifact_saved_json_is_valid(
    tmp_path,
):
    """The written artifact must be valid JSON."""

    path = StabilityArtifact(
        _stability_result()
    ).save(
        tmp_path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    assert isinstance(
        payload,
        dict,
    )

    assert payload["status"] == "evaluated"


def test_stability_artifact_saved_json_preserves_failure(
    tmp_path,
):
    """Failed seed evidence must survive disk persistence."""

    path = StabilityArtifact(
        _mixed_stability_result()
    ).save(
        tmp_path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    assert payload["failed_count"] == 1

    assert (
        payload["seeds"][1]["status"]
        == "failed"
    )


def test_stability_artifact_save_accepts_string_path(
    tmp_path,
):
    """String result-directory paths should be accepted."""

    artifact = StabilityArtifact(
        _stability_result()
    )

    path = artifact.save(
        str(tmp_path)
    )

    assert isinstance(
        path,
        Path,
    )

    assert path.exists()


def test_stability_artifact_rejects_invalid_directory_type():
    """Unsupported destination types should fail explicitly."""

    artifact = StabilityArtifact(
        _stability_result()
    )

    with pytest.raises(
        TypeError,
        match="directory",
    ):
        artifact.save(
            123,
        )


def test_stability_artifact_rejects_existing_file_as_directory(
    tmp_path,
):
    """An existing regular file cannot be used as a directory."""

    existing_file = (
        tmp_path
        / "existing.txt"
    )

    existing_file.write_text(
        "not a directory",
        encoding="utf-8",
    )

    artifact = StabilityArtifact(
        _stability_result()
    )

    with pytest.raises(
        ValueError,
        match="directory",
    ):
        artifact.save(
            existing_file
        )


# ---------------------------------------------------------------------
# Scientific boundary
# ---------------------------------------------------------------------


def test_stability_artifact_contains_no_benchmark_evidence():
    """Stability artifact must remain separate from final evaluation."""

    payload = StabilityArtifact(
        _stability_result()
    ).to_dict()

    forbidden_keys = {
        "benchmark",
        "evaluation",
        "evaluation_result",
        "benchmark_rmse",
        "benchmark_mae",
        "benchmark_mape",
    }

    assert forbidden_keys.isdisjoint(
        payload.keys()
    )


def test_stability_artifact_contains_no_optimization_selection():
    """Stability persistence must not become parameter selection."""

    payload = StabilityArtifact(
        _stability_result()
    ).to_dict()

    forbidden_keys = {
        "best_seed",
        "best_params",
        "best_score",
        "candidate_count",
        "candidates",
    }

    assert forbidden_keys.isdisjoint(
        payload.keys()
    )


def test_stability_artifact_is_publicly_exported():
    """StabilityArtifact should be available from optimization."""

    from zenerestimation.optimization import (
        StabilityArtifact as PublicStabilityArtifact,
    )

    assert (
        PublicStabilityArtifact
        is StabilityArtifact
    )