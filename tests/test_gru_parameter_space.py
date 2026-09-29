"""Tests for the GRU optimization parameter-space contract."""

import pytest

from zenerestimation.optimization import GRUParameterSpace


def test_gru_parameter_space_generates_cartesian_product():
    """All window/unit combinations should be generated."""

    space = GRUParameterSpace(
        window=(4, 6),
        units=(24, 32),
    )

    assert space.candidates() == [
        {
            "window": 4,
            "units": 24,
        },
        {
            "window": 4,
            "units": 32,
        },
        {
            "window": 6,
            "units": 24,
        },
        {
            "window": 6,
            "units": 32,
        },
    ]


def test_gru_parameter_space_preserves_deterministic_order():
    """Candidate order should follow the supplied dimension order."""

    space = GRUParameterSpace(
        window=(8, 4),
        units=(32, 24),
    )

    assert space.candidates() == [
        {
            "window": 8,
            "units": 32,
        },
        {
            "window": 8,
            "units": 24,
        },
        {
            "window": 4,
            "units": 32,
        },
        {
            "window": 4,
            "units": 24,
        },
    ]


def test_gru_candidates_contain_only_tunable_parameters():
    """Training-protocol parameters must not enter the search space."""

    space = GRUParameterSpace(
        window=(4,),
        units=(32,),
    )

    candidate = space.candidates()[0]

    assert set(candidate) == {
        "window",
        "units",
    }

    assert "epochs" not in candidate
    assert "batch_size" not in candidate
    assert "seed" not in candidate


@pytest.mark.parametrize(
    "window, units",
    [
        ((1,), (1,)),
        ((4,), (24,)),
        ((4, 6, 8), (24, 32)),
        ((12,), (64,)),
    ],
)
def test_gru_parameter_space_accepts_positive_integers(
    window,
    units,
):
    """Positive integer dimensions should be accepted."""

    space = GRUParameterSpace(
        window=window,
        units=units,
    )

    assert space.candidate_count == (
        len(window) * len(units)
    )


def test_gru_parameter_space_preserves_duplicates():
    """Duplicate values should remain explicit search candidates."""

    space = GRUParameterSpace(
        window=(4, 4),
        units=(24, 32),
    )

    assert space.candidate_count == 4

    assert space.candidates() == [
        {
            "window": 4,
            "units": 24,
        },
        {
            "window": 4,
            "units": 32,
        },
        {
            "window": 4,
            "units": 24,
        },
        {
            "window": 4,
            "units": 32,
        },
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "window": (),
            "units": (24, 32),
        },
        {
            "window": (4, 6),
            "units": (),
        },
    ],
)
def test_gru_parameter_space_rejects_empty_dimensions(
    kwargs,
):
    """Every search dimension must contain at least one value."""

    with pytest.raises(ValueError):
        GRUParameterSpace(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "window": (0,),
            "units": (24,),
        },
        {
            "window": (-1,),
            "units": (24,),
        },
        {
            "window": (4,),
            "units": (0,),
        },
        {
            "window": (4,),
            "units": (-32,),
        },
    ],
)
def test_gru_parameter_space_rejects_nonpositive_values(
    kwargs,
):
    """Window and unit counts must be strictly positive."""

    with pytest.raises(ValueError):
        GRUParameterSpace(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "window": (4.0,),
            "units": (24,),
        },
        {
            "window": (4,),
            "units": (32.0,),
        },
    ],
)
def test_gru_parameter_space_rejects_float_values(
    kwargs,
):
    """Window and unit counts must be integers."""

    with pytest.raises(TypeError):
        GRUParameterSpace(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "window": (True,),
            "units": (24,),
        },
        {
            "window": (4,),
            "units": (False,),
        },
    ],
)
def test_gru_parameter_space_rejects_boolean_values(
    kwargs,
):
    """Booleans must not be accepted as integer hyperparameters."""

    with pytest.raises(TypeError):
        GRUParameterSpace(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "window": "468",
            "units": (24,),
        },
        {
            "window": (4,),
            "units": "2432",
        },
        {
            "window": b"468",
            "units": (24,),
        },
        {
            "window": (4,),
            "units": b"2432",
        },
    ],
)
def test_gru_parameter_space_rejects_string_dimensions(
    kwargs,
):
    """Strings and bytes must not be interpreted as dimensions."""

    with pytest.raises(TypeError):
        GRUParameterSpace(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "window": 4,
            "units": (24,),
        },
        {
            "window": (4,),
            "units": 32,
        },
        {
            "window": None,
            "units": (24,),
        },
        {
            "window": (4,),
            "units": None,
        },
    ],
)
def test_gru_parameter_space_rejects_noniterable_dimensions(
    kwargs,
):
    """Search dimensions themselves must be iterable."""

    with pytest.raises(TypeError):
        GRUParameterSpace(**kwargs)


def test_gru_parameter_space_candidate_count():
    """Candidate count should equal the Cartesian-product size."""

    space = GRUParameterSpace(
        window=(4, 6, 8),
        units=(24, 32),
    )

    assert space.candidate_count == 6
    assert len(space.candidates()) == 6


def test_gru_production_space_has_ten_candidates():
    """The frozen Sprint 14.4 production grid contains ten candidates."""

    space = GRUParameterSpace(
        window=(4, 6, 8, 10, 12),
        units=(24, 32),
    )

    assert space.candidate_count == 10
    assert len(space.candidates()) == 10


def test_gru_parameter_space_repr():
    """repr should expose the class and configured dimensions."""

    space = GRUParameterSpace(
        window=(4, 6),
        units=(24, 32),
    )

    representation = repr(space)

    assert "GRUParameterSpace" in representation
    assert "window=(4, 6)" in representation
    assert "units=(24, 32)" in representation