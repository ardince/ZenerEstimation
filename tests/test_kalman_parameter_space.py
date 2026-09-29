"""Tests for the Kalman optimization parameter-space contract."""

from __future__ import annotations

import math

import pytest

from zenerestimation.optimization import (
    KalmanParameterSpace,
)


def test_kalman_parameter_space_builds_cartesian_product():
    space = KalmanParameterSpace(
        process_noise=(
            1e-4,
            1e-3,
        ),
        drift_noise=(
            1e-5,
            1e-4,
        ),
        regime_factor=(
            2.0,
            2.5,
        ),
        regime_multiplier=(
            5.0,
            10.0,
        ),
    )

    candidates = space.candidates()

    assert len(candidates) == 16
    assert space.candidate_count == 16


def test_kalman_parameter_space_preserves_deterministic_order():
    space = KalmanParameterSpace(
        process_noise=(
            1e-4,
            1e-3,
        ),
        drift_noise=(
            1e-5,
        ),
        regime_factor=(
            2.0,
            2.5,
        ),
        regime_multiplier=(
            5.0,
            10.0,
        ),
    )

    candidates = space.candidates()

    assert candidates[0] == {
        "process_noise": 1e-4,
        "drift_noise": 1e-5,
        "regime_factor": 2.0,
        "regime_multiplier": 5.0,
    }

    assert candidates[1] == {
        "process_noise": 1e-4,
        "drift_noise": 1e-5,
        "regime_factor": 2.0,
        "regime_multiplier": 10.0,
    }

    assert candidates[2] == {
        "process_noise": 1e-4,
        "drift_noise": 1e-5,
        "regime_factor": 2.5,
        "regime_multiplier": 5.0,
    }

    assert candidates[-1] == {
        "process_noise": 1e-3,
        "drift_noise": 1e-5,
        "regime_factor": 2.5,
        "regime_multiplier": 10.0,
    }


def test_kalman_candidates_contain_only_tunable_parameters():
    space = KalmanParameterSpace(
        process_noise=(1e-3,),
        drift_noise=(1e-4,),
        regime_factor=(2.5,),
        regime_multiplier=(10.0,),
    )

    candidate = space.candidates()[0]

    assert set(candidate) == {
        "process_noise",
        "drift_noise",
        "regime_factor",
        "regime_multiplier",
    }

    assert "dt" not in candidate
    assert "adaptive" not in candidate


def test_kalman_parameter_space_normalizes_integers_to_float():
    space = KalmanParameterSpace(
        process_noise=(1,),
        drift_noise=(2,),
        regime_factor=(3,),
        regime_multiplier=(10,),
    )

    candidate = space.candidates()[0]

    assert candidate == {
        "process_noise": 1.0,
        "drift_noise": 2.0,
        "regime_factor": 3.0,
        "regime_multiplier": 10.0,
    }

    assert all(
        isinstance(value, float)
        for value in candidate.values()
    )


def test_kalman_parameter_space_preserves_duplicates():
    space = KalmanParameterSpace(
        process_noise=(
            1e-3,
            1e-3,
        ),
        drift_noise=(1e-4,),
        regime_factor=(2.5,),
        regime_multiplier=(10.0,),
    )

    candidates = space.candidates()

    assert space.candidate_count == 2
    assert len(candidates) == 2
    assert candidates[0] == candidates[1]


@pytest.mark.parametrize(
    "field",
    [
        "process_noise",
        "drift_noise",
        "regime_factor",
        "regime_multiplier",
    ],
)
def test_kalman_parameter_space_rejects_empty_dimension(
    field,
):
    kwargs = {
        "process_noise": (1e-3,),
        "drift_noise": (1e-4,),
        "regime_factor": (2.5,),
        "regime_multiplier": (10.0,),
    }

    kwargs[field] = ()

    with pytest.raises(ValueError):
        KalmanParameterSpace(**kwargs)


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        0.0,
        -1,
        -1.0,
    ],
)
@pytest.mark.parametrize(
    "field",
    [
        "process_noise",
        "drift_noise",
        "regime_factor",
        "regime_multiplier",
    ],
)
def test_kalman_parameter_space_rejects_nonpositive_values(
    field,
    invalid_value,
):
    kwargs = {
        "process_noise": (1e-3,),
        "drift_noise": (1e-4,),
        "regime_factor": (2.5,),
        "regime_multiplier": (10.0,),
    }

    kwargs[field] = (
        invalid_value,
    )

    with pytest.raises(ValueError):
        KalmanParameterSpace(**kwargs)


@pytest.mark.parametrize(
    "invalid_value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
@pytest.mark.parametrize(
    "field",
    [
        "process_noise",
        "drift_noise",
        "regime_factor",
        "regime_multiplier",
    ],
)
def test_kalman_parameter_space_rejects_nonfinite_values(
    field,
    invalid_value,
):
    kwargs = {
        "process_noise": (1e-3,),
        "drift_noise": (1e-4,),
        "regime_factor": (2.5,),
        "regime_multiplier": (10.0,),
    }

    kwargs[field] = (
        invalid_value,
    )

    with pytest.raises(ValueError):
        KalmanParameterSpace(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "process_noise",
        "drift_noise",
        "regime_factor",
        "regime_multiplier",
    ],
)
def test_kalman_parameter_space_rejects_boolean_values(
    field,
):
    kwargs = {
        "process_noise": (1e-3,),
        "drift_noise": (1e-4,),
        "regime_factor": (2.5,),
        "regime_multiplier": (10.0,),
    }

    kwargs[field] = (
        True,
    )

    with pytest.raises(TypeError):
        KalmanParameterSpace(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "process_noise",
        "drift_noise",
        "regime_factor",
        "regime_multiplier",
    ],
)
def test_kalman_parameter_space_rejects_string_values(
    field,
):
    kwargs = {
        "process_noise": (1e-3,),
        "drift_noise": (1e-4,),
        "regime_factor": (2.5,),
        "regime_multiplier": (10.0,),
    }

    kwargs[field] = (
        "2.5",
    )

    with pytest.raises(TypeError):
        KalmanParameterSpace(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "process_noise",
        "drift_noise",
        "regime_factor",
        "regime_multiplier",
    ],
)
def test_kalman_parameter_space_rejects_string_dimension(
    field,
):
    kwargs = {
        "process_noise": (1e-3,),
        "drift_noise": (1e-4,),
        "regime_factor": (2.5,),
        "regime_multiplier": (10.0,),
    }

    kwargs[field] = "invalid"

    with pytest.raises(TypeError):
        KalmanParameterSpace(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "process_noise",
        "drift_noise",
        "regime_factor",
        "regime_multiplier",
    ],
)
def test_kalman_parameter_space_rejects_noniterable_dimension(
    field,
):
    kwargs = {
        "process_noise": (1e-3,),
        "drift_noise": (1e-4,),
        "regime_factor": (2.5,),
        "regime_multiplier": (10.0,),
    }

    kwargs[field] = 1e-3

    with pytest.raises(TypeError):
        KalmanParameterSpace(**kwargs)


def test_kalman_parameter_space_candidate_count_matches_product():
    space = KalmanParameterSpace(
        process_noise=(
            1e-4,
            1e-3,
            1e-2,
        ),
        drift_noise=(
            1e-5,
            1e-4,
            1e-3,
        ),
        regime_factor=(
            2.0,
            2.5,
            3.0,
        ),
        regime_multiplier=(
            5.0,
            10.0,
        ),
    )

    assert space.candidate_count == 54
    assert len(space.candidates()) == 54


def test_kalman_parameter_space_repr_is_informative():
    space = KalmanParameterSpace(
        process_noise=(1e-3,),
        drift_noise=(1e-4,),
        regime_factor=(2.5,),
        regime_multiplier=(10.0,),
    )

    text = repr(space)

    assert "KalmanParameterSpace" in text
    assert "process_noise" in text
    assert "drift_noise" in text
    assert "regime_factor" in text
    assert "regime_multiplier" in text
    assert "candidate_count=1" in text