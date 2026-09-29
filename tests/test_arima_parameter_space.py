import pytest

from zenerestimation.optimization import (
    ARIMAParameterSpace,
)


def test_arima_parameter_space_generates_cartesian_product():

    space = ARIMAParameterSpace(
        p=(0, 1),
        d=(0, 1),
        q=(0, 1),
    )

    candidates = space.candidates()

    assert candidates == (
        {"order": (0, 0, 0)},
        {"order": (0, 0, 1)},
        {"order": (0, 1, 0)},
        {"order": (0, 1, 1)},
        {"order": (1, 0, 0)},
        {"order": (1, 0, 1)},
        {"order": (1, 1, 0)},
        {"order": (1, 1, 1)},
    )


def test_arima_parameter_space_candidate_count():

    space = ARIMAParameterSpace(
        p=(0, 1, 2),
        d=(0, 1),
        q=(0, 1, 2, 3),
    )

    assert space.candidate_count == 24

    assert len(
        space.candidates()
    ) == 24


def test_arima_parameter_space_preserves_requested_values():

    space = ARIMAParameterSpace(
        p=(0, 2, 20),
        d=(1,),
        q=(0, 3),
    )

    assert space.p == (
        0,
        2,
        20,
    )

    assert space.d == (
        1,
    )

    assert space.q == (
        0,
        3,
    )


@pytest.mark.parametrize(
    "name",
    [
        "p",
        "d",
        "q",
    ],
)
def test_arima_parameter_space_rejects_empty_dimension(
    name,
):

    kwargs = {
        "p": (0, 1),
        "d": (0, 1),
        "q": (0, 1),
    }

    kwargs[name] = ()

    with pytest.raises(
        ValueError,
        match=f"{name} cannot be empty",
    ):
        ARIMAParameterSpace(
            **kwargs
        )


@pytest.mark.parametrize(
    "name",
    [
        "p",
        "d",
        "q",
    ],
)
def test_arima_parameter_space_rejects_negative_order(
    name,
):

    kwargs = {
        "p": (0, 1),
        "d": (0, 1),
        "q": (0, 1),
    }

    kwargs[name] = (
        0,
        -1,
    )

    with pytest.raises(
        ValueError,
    ):
        ARIMAParameterSpace(
            **kwargs
        )


@pytest.mark.parametrize(
    "name",
    [
        "p",
        "d",
        "q",
    ],
)
def test_arima_parameter_space_rejects_non_integer_order(
    name,
):

    kwargs = {
        "p": (0, 1),
        "d": (0, 1),
        "q": (0, 1),
    }

    kwargs[name] = (
        0,
        1.5,
    )

    with pytest.raises(
        TypeError,
    ):
        ARIMAParameterSpace(
            **kwargs
        )


def test_arima_parameter_space_rejects_boolean_order():

    with pytest.raises(
        TypeError,
    ):
        ARIMAParameterSpace(
            p=(False, 1),
            d=(1,),
            q=(0,),
        )


def test_arima_parameter_space_repr():

    space = ARIMAParameterSpace(
        p=(0, 1),
        d=(1,),
        q=(0, 1),
    )

    representation = repr(
        space
    )

    assert "ARIMAParameterSpace" in representation
    assert "p=(0, 1)" in representation
    assert "d=(1,)" in representation
    assert "q=(0, 1)" in representation