import pytest

from zenerestimation.optimization import (
    CandidateResult,
    OptimizationResult,
)


def make_candidates():

    first = CandidateResult(
        params={"window": 4},
        score=0.30,
        metrics={"rmse": 0.30},
    )

    second = CandidateResult(
        params={"window": 6},
        score=0.20,
        metrics={"rmse": 0.20},
    )

    return first, second


def make_result():

    first, second = make_candidates()

    return OptimizationResult(
        model="LSTM",
        metric="rmse",
        best_params={"window": 6},
        best_score=0.20,
        candidates=(first, second),
        metadata={
            "validation": "temporal",
        },
    )


def test_optimization_result_creation():

    result = make_result()

    assert result.model == "LSTM"
    assert result.metric == "rmse"

    assert result.best_params == {
        "window": 6,
    }

    assert result.best_score == 0.20

    assert result.candidate_count == 2


def test_optimization_result_to_dict():

    result = make_result()

    data = result.to_dict()

    assert data["status"] == "optimized"
    assert data["model"] == "LSTM"
    assert data["metric"] == "rmse"
    assert data["best_params"]["window"] == 6
    assert data["best_score"] == 0.20
    assert data["candidate_count"] == 2

    assert len(
        data["candidates"]
    ) == 2


def test_optimization_result_rejects_empty_candidates():

    with pytest.raises(ValueError):
        OptimizationResult(
            model="LSTM",
            metric="rmse",
            best_params={"window": 4},
            best_score=0.20,
            candidates=(),
        )


def test_optimization_result_rejects_invalid_candidate_type():

    with pytest.raises(TypeError):
        OptimizationResult(
            model="LSTM",
            metric="rmse",
            best_params={"window": 4},
            best_score=0.20,
            candidates=(
                "not-a-candidate",
            ),
        )


def test_optimization_result_rejects_non_finite_best_score():

    first, _ = make_candidates()

    with pytest.raises(ValueError):
        OptimizationResult(
            model="LSTM",
            metric="rmse",
            best_params={"window": 4},
            best_score=float("nan"),
            candidates=(first,),
        )


def test_optimization_result_requires_matching_best_candidate():

    first, second = make_candidates()

    with pytest.raises(ValueError):
        OptimizationResult(
            model="LSTM",
            metric="rmse",
            best_params={"window": 8},
            best_score=0.10,
            candidates=(
                first,
                second,
            ),
        )


def test_optimization_result_defensively_copies_best_params():

    first, second = make_candidates()

    best_params = {
        "window": 6,
    }

    result = OptimizationResult(
        model="LSTM",
        metric="rmse",
        best_params=best_params,
        best_score=0.20,
        candidates=(
            first,
            second,
        ),
    )

    best_params["window"] = 12

    assert result.best_params["window"] == 6