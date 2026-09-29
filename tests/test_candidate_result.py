import pytest

from zenerestimation.optimization import CandidateResult


def make_candidate():
    return CandidateResult(
        params={
            "window": 4,
            "units": 32,
        },
        score=0.25,
        metrics={
            "rmse": 0.25,
            "mae": 0.20,
            "mape": 0.12,
        },
        metadata={
            "folds": 3,
        },
    )


def test_candidate_result_creation():

    result = make_candidate()

    assert result.params == {
        "window": 4,
        "units": 32,
    }

    assert result.score == 0.25

    assert result.metrics["rmse"] == 0.25

    assert result.status == "evaluated"

    assert result.metadata["folds"] == 3


def test_candidate_result_to_dict():

    result = make_candidate()

    data = result.to_dict()

    assert data["status"] == "evaluated"
    assert data["score"] == 0.25
    assert data["params"]["window"] == 4
    assert data["metrics"]["rmse"] == 0.25
    assert data["metadata"]["folds"] == 3


def test_candidate_result_rejects_empty_params():

    with pytest.raises(ValueError):
        CandidateResult(
            params={},
            score=0.25,
            metrics={"rmse": 0.25},
        )


def test_candidate_result_rejects_non_finite_score():

    with pytest.raises(ValueError):
        CandidateResult(
            params={"window": 4},
            score=float("nan"),
            metrics={"rmse": 0.25},
        )


def test_candidate_result_rejects_empty_metrics():

    with pytest.raises(ValueError):
        CandidateResult(
            params={"window": 4},
            score=0.25,
            metrics={},
        )


def test_candidate_result_rejects_non_finite_metric():

    with pytest.raises(ValueError):
        CandidateResult(
            params={"window": 4},
            score=0.25,
            metrics={
                "rmse": float("inf"),
            },
        )


def test_candidate_result_defensively_copies_inputs():

    params = {
        "window": 4,
    }

    metrics = {
        "rmse": 0.25,
    }

    metadata = {
        "folds": 3,
    }

    result = CandidateResult(
        params=params,
        score=0.25,
        metrics=metrics,
        metadata=metadata,
    )

    params["window"] = 12
    metrics["rmse"] = 99.0
    metadata["folds"] = 99

    assert result.params["window"] == 4
    assert result.metrics["rmse"] == 0.25
    assert result.metadata["folds"] == 3