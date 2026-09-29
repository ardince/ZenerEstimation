"""Contract tests for repeated-seed stability result objects."""

import json

import numpy as np
import pytest

from zenerestimation.optimization.stability import SeedResult, StabilityResult, FailedSeedResult


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _seed_result(
    seed: int = 0,
    score: float = 0.5,
) -> SeedResult:
    """Create a valid SeedResult for contract tests."""

    return SeedResult(
        seed=seed,
        score=score,
        metrics={
            "rmse": score,
            "mae": score * 0.8,
            "mape": score * 2.0,
        },
        metadata={
            "validation": "expanding_window",
        },
    )


def _stability_result() -> StabilityResult:
    """Create a valid StabilityResult for contract tests."""

    return StabilityResult(
        model="LSTM",
        metric="rmse",
        params={
            "window": 4,
            "units": 32,
        },
        seeds=(
            _seed_result(seed=0, score=0.5),
            _seed_result(seed=1, score=0.7),
            _seed_result(seed=2, score=0.6),
        ),
        metadata={
            "analysis": "repeated_seed",
        },
    )


# ---------------------------------------------------------------------
# SeedResult — valid construction
# ---------------------------------------------------------------------


def test_seed_result_accepts_valid_values():
    """A valid repeated-seed result should be accepted."""

    result = _seed_result()

    assert result.seed == 0
    assert result.score == 0.5

    assert result.metrics == {
        "rmse": 0.5,
        "mae": 0.4,
        "mape": 1.0,
    }

    assert result.metadata == {
        "validation": "expanding_window",
    }


def test_seed_result_accepts_seed_zero():
    """Seed zero is a valid random seed."""

    result = _seed_result(seed=0)

    assert result.seed == 0


# ---------------------------------------------------------------------
# SeedResult — validation
# ---------------------------------------------------------------------


def test_seed_result_rejects_boolean_seed():
    """Boolean values must not be accepted as integer seeds."""

    with pytest.raises(TypeError):
        SeedResult(
            seed=True,
            score=0.5,
            metrics={"rmse": 0.5},
        )


def test_seed_result_rejects_noninteger_seed():
    """Seeds must be integers."""

    with pytest.raises(TypeError):
        SeedResult(
            seed=1.5,
            score=0.5,
            metrics={"rmse": 0.5},
        )


@pytest.mark.parametrize(
    "score",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_seed_result_rejects_nonfinite_score(score):
    """Seed scores must be finite."""

    with pytest.raises(ValueError):
        SeedResult(
            seed=0,
            score=score,
            metrics={"rmse": 0.5},
        )


def test_seed_result_rejects_empty_metrics():
    """At least one metric must be recorded."""

    with pytest.raises(ValueError):
        SeedResult(
            seed=0,
            score=0.5,
            metrics={},
        )


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_seed_result_rejects_nonfinite_metric(value):
    """All recorded metric values must be finite."""

    with pytest.raises(ValueError):
        SeedResult(
            seed=0,
            score=0.5,
            metrics={
                "rmse": value,
            },
        )


# ---------------------------------------------------------------------
# SeedResult — defensive copying
# ---------------------------------------------------------------------


def test_seed_result_normalizes_none_metadata():
    """None metadata should be normalized to an empty dictionary."""

    result = SeedResult(
        seed=0,
        score=0.5,
        metrics={
            "rmse": 0.5,
        },
        metadata=None,
    )

    assert result.metadata == {}


def test_seed_result_defensively_copies_metrics():
    """External metric mutation must not change the result."""

    metrics = {
        "rmse": 0.5,
        "mae": 0.4,
    }

    result = SeedResult(
        seed=0,
        score=0.5,
        metrics=metrics,
    )

    metrics["rmse"] = 999.0
    metrics["new_metric"] = 123.0

    assert result.metrics == {
        "rmse": 0.5,
        "mae": 0.4,
    }


def test_seed_result_defensively_copies_metadata():
    """External metadata mutation must not change the result."""

    metadata = {
        "validation": {
            "folds": 3,
        },
    }

    result = SeedResult(
        seed=0,
        score=0.5,
        metrics={
            "rmse": 0.5,
        },
        metadata=metadata,
    )

    metadata["validation"]["folds"] = 999

    assert result.metadata == {
        "validation": {
            "folds": 3,
        },
    }


# ---------------------------------------------------------------------
# StabilityResult — valid construction
# ---------------------------------------------------------------------


def test_stability_result_accepts_valid_values():
    """A valid repeated-seed stability result should be accepted."""

    result = _stability_result()

    assert result.model == "LSTM"
    assert result.metric == "rmse"

    assert result.params == {
        "window": 4,
        "units": 32,
    }

    assert len(result.seeds) == 3

    assert result.metadata == {
        "analysis": "repeated_seed",
    }


# ---------------------------------------------------------------------
# StabilityResult — validation
# ---------------------------------------------------------------------


def test_stability_result_rejects_empty_model():
    """Model identity must be non-empty."""

    with pytest.raises(ValueError):
        StabilityResult(
            model="",
            metric="rmse",
            params={
                "window": 4,
                "units": 32,
            },
            seeds=(
                _seed_result(),
            ),
        )


def test_stability_result_rejects_empty_metric():
    """Selection metric identity must be non-empty."""

    with pytest.raises(ValueError):
        StabilityResult(
            model="LSTM",
            metric="",
            params={
                "window": 4,
                "units": 32,
            },
            seeds=(
                _seed_result(),
            ),
        )


def test_stability_result_rejects_empty_params():
    """The frozen neural configuration must be recorded."""

    with pytest.raises(ValueError):
        StabilityResult(
            model="LSTM",
            metric="rmse",
            params={},
            seeds=(
                _seed_result(),
            ),
        )


def test_stability_result_rejects_non_dictionary_params():
    """Frozen parameters must be represented by a dictionary."""

    with pytest.raises(TypeError):
        StabilityResult(
            model="LSTM",
            metric="rmse",
            params=[("window", 4)],
            seeds=(
                _seed_result(),
            ),
        )


def test_stability_result_rejects_empty_seed_tuple():
    """At least one repeated-seed run must be recorded."""

    with pytest.raises(ValueError):
        StabilityResult(
            model="LSTM",
            metric="rmse",
            params={
                "window": 4,
                "units": 32,
            },
            seeds=(),
        )


def test_stability_result_requires_seed_tuple():
    """Seed results should use the immutable tuple contract."""

    with pytest.raises(TypeError):
        StabilityResult(
            model="LSTM",
            metric="rmse",
            params={
                "window": 4,
                "units": 32,
            },
            seeds=[
                _seed_result(),
            ],
        )


def test_stability_result_rejects_non_seed_result_element():
    """Every entry must be successful or failed seed evidence."""

    with pytest.raises(TypeError):
        StabilityResult(
            model="LSTM",
            metric="rmse",
            params={
                "window": 4,
                "units": 32,
            },
            seeds=(
                _seed_result(),
                "invalid",
            ),
        )


def test_stability_result_rejects_duplicate_seeds():
    """Duplicate seeds do not provide independent seed-sensitivity evidence."""

    with pytest.raises(ValueError):
        StabilityResult(
            model="LSTM",
            metric="rmse",
            params={
                "window": 4,
                "units": 32,
            },
            seeds=(
                _seed_result(
                    seed=42,
                    score=0.5,
                ),
                _seed_result(
                    seed=42,
                    score=0.6,
                ),
            ),
        )


# ---------------------------------------------------------------------
# StabilityResult — defensive copying
# ---------------------------------------------------------------------


def test_stability_result_defensively_copies_params():
    """External parameter mutation must not change the result."""

    params = {
        "window": 4,
        "units": 32,
    }

    result = StabilityResult(
        model="LSTM",
        metric="rmse",
        params=params,
        seeds=(
            _seed_result(),
        ),
    )

    params["window"] = 999
    params["units"] = 999

    assert result.params == {
        "window": 4,
        "units": 32,
    }


def test_stability_result_defensively_copies_metadata():
    """External metadata mutation must not change the result."""

    metadata = {
        "analysis": {
            "method": "repeated_seed",
        },
    }

    result = StabilityResult(
        model="LSTM",
        metric="rmse",
        params={
            "window": 4,
            "units": 32,
        },
        seeds=(
            _seed_result(),
        ),
        metadata=metadata,
    )

    metadata["analysis"]["method"] = "changed"

    assert result.metadata == {
        "analysis": {
            "method": "repeated_seed",
        },
    }


# ---------------------------------------------------------------------
# StabilityResult — summary statistics
# ---------------------------------------------------------------------


def test_stability_result_reports_run_count():
    """run_count should equal the number of seed replications."""

    result = _stability_result()

    assert result.run_count == 3


def test_stability_result_reports_scores():
    """scores should preserve seed-run order."""

    result = _stability_result()

    assert result.scores == (
        0.5,
        0.7,
        0.6,
    )


def test_stability_result_reports_mean_score():
    """mean_score should use all seed scores."""

    result = _stability_result()

    assert np.isclose(
        result.mean_score,
        np.mean([0.5, 0.7, 0.6]),
    )


def test_stability_result_reports_population_std():
    """std_score should use population standard deviation."""

    result = _stability_result()

    assert np.isclose(
        result.std_score,
        np.std(
            [0.5, 0.7, 0.6],
            ddof=0,
        ),
    )


def test_stability_result_reports_min_score():
    """min_score should report the lowest observed seed score."""

    result = _stability_result()

    assert np.isclose(
        result.min_score,
        0.5,
    )


def test_stability_result_reports_max_score():
    """max_score should report the highest observed seed score."""

    result = _stability_result()

    assert np.isclose(
        result.max_score,
        0.7,
    )


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


def test_seed_result_to_dict():
    """SeedResult should serialize into the stability artifact schema."""

    result = _seed_result(
        seed=3,
        score=0.75,
    )

    payload = result.to_dict()

    assert payload["seed"] == 3
    assert payload["score"] == 0.75


    assert np.isclose(
        payload["metrics"]["rmse"],
        0.75,
    )

    assert np.isclose(
        payload["metrics"]["mae"],
        0.6,
    )

    assert np.isclose(
        payload["metrics"]["mape"],
        1.5,
    )
    

    assert payload["metadata"] == {
        "validation": "expanding_window",
    }


def test_stability_result_to_dict():
    """StabilityResult should expose the standardized schema."""

    result = _stability_result()

    payload = result.to_dict()

    assert payload["status"] == "evaluated"
    assert payload["model"] == "LSTM"
    assert payload["metric"] == "rmse"

    assert payload["params"] == {
        "window": 4,
        "units": 32,
    }

    assert payload["run_count"] == 3

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

    assert len(payload["seeds"]) == 3

    assert payload["seeds"][0]["seed"] == 0
    assert payload["seeds"][1]["seed"] == 1
    assert payload["seeds"][2]["seed"] == 2

    assert payload["metadata"] == {
        "analysis": "repeated_seed",
    }


def test_stability_result_is_json_serializable():
    """The standardized stability payload must be JSON serializable."""

    payload = _stability_result().to_dict()

    serialized = json.dumps(payload)

    assert isinstance(serialized, str)
    assert serialized


# ---------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------


def test_seed_result_repr_contains_identity():
    """repr should expose useful seed-result identity."""

    result = _seed_result(
        seed=42,
        score=0.5,
    )

    representation = repr(result)

    assert "SeedResult" in representation
    assert "42" in representation


def test_stability_result_repr_contains_identity():
    """repr should expose useful stability-result identity."""

    result = _stability_result()

    representation = repr(result)

    assert "StabilityResult" in representation
    assert "LSTM" in representation
    assert "rmse" in representation


# ---------------------------------------------------------------------
# FailedSeedResult
# ---------------------------------------------------------------------


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


def test_failed_seed_result_accepts_valid_values():
    result = _failed_seed_result()

    assert result.seed == 1
    assert result.status == "failed"
    assert result.error_type == "ValueError"


def test_failed_seed_result_to_dict():
    result = _failed_seed_result()

    payload = result.to_dict()

    assert payload["seed"] == 1
    assert payload["status"] == "failed"
    assert payload["error_type"] == "ValueError"
    assert (
        payload["error_message"]
        == "predicted must contain only finite values"
    )


# ---------------------------------------------------------------------
# Mixed successful / failed stability evidence
# ---------------------------------------------------------------------


def test_stability_result_accepts_failed_seed_results():
    result = StabilityResult(
        model="LSTM",
        metric="rmse",
        params={
            "window": 4,
            "units": 32,
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
    )

    assert result.run_count == 3
    assert result.successful_count == 2
    assert result.failed_count == 1


def test_stability_result_scores_exclude_failed_seeds():
    result = StabilityResult(
        model="LSTM",
        metric="rmse",
        params={
            "window": 4,
            "units": 32,
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
    )

    assert result.scores == (
        0.5,
        0.7,
    )

    assert np.isclose(
        result.mean_score,
        0.6,
    )


def test_stability_result_serializes_failed_seed_evidence():
    result = StabilityResult(
        model="LSTM",
        metric="rmse",
        params={
            "window": 4,
            "units": 32,
        },
        seeds=(
            _seed_result(
                seed=0,
                score=0.5,
            ),
            _failed_seed_result(
                seed=1,
            ),
        ),
    )

    payload = result.to_dict()

    assert payload["run_count"] == 2
    assert payload["successful_count"] == 1
    assert payload["failed_count"] == 1

    assert payload["seeds"][0]["status"] == "evaluated"
    assert payload["seeds"][1]["status"] == "failed"

    assert (
        payload["seeds"][1]["error_type"]
        == "ValueError"
    )


def test_stability_result_rejects_seed_in_frozen_params():
    with pytest.raises(
        ValueError,
        match="must not contain seed",
    ):
        StabilityResult(
            model="LSTM",
            metric="rmse",
            params={
                "window": 4,
                "units": 32,
                "seed": 42,
            },
            seeds=(
                _seed_result(),
            ),
        )


def test_stability_result_rejects_duplicate_seed_across_statuses():
    with pytest.raises(
        ValueError,
        match="duplicate seeds",
    ):
        StabilityResult(
            model="LSTM",
            metric="rmse",
            params={
                "window": 4,
                "units": 32,
            },
            seeds=(
                _seed_result(
                    seed=1,
                ),
                _failed_seed_result(
                    seed=1,
                ),
            ),
        )


def test_stability_result_has_no_best_seed_semantics():
    payload = _stability_result().to_dict()

    assert "best_seed" not in payload


def test_stability_result_mixed_payload_is_json_serializable():
    result = StabilityResult(
        model="LSTM",
        metric="rmse",
        params={
            "window": 4,
            "units": 32,
        },
        seeds=(
            _seed_result(
                seed=0,
                score=0.5,
            ),
            _failed_seed_result(
                seed=1,
            ),
        ),
    )

    serialized = json.dumps(
        result.to_dict()
    )

    assert isinstance(
        serialized,
        str,
    )
    assert serialized