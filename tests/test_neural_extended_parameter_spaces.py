import pytest
from zenerestimation.optimization.spaces import (
    LSTMParameterSpace, GRUParameterSpace, HybridNeuralParameterSpace,
)


@pytest.mark.parametrize("cls", [LSTMParameterSpace, GRUParameterSpace, HybridNeuralParameterSpace])
def test_legacy_candidates_unchanged(cls):
    space = cls(window=[4, 6], units=[8])
    assert space.candidate_count == 2
    assert space.candidates() == [
        {"window": 4, "units": 8},
        {"window": 6, "units": 8},
    ]


@pytest.mark.parametrize("cls", [LSTMParameterSpace, GRUParameterSpace, HybridNeuralParameterSpace])
def test_extended_candidates_deterministic(cls):
    space = cls(window=[4], units=[8, 16], epochs=[20, 40], batch_size=[8, 16])
    assert space.candidate_count == 8
    expected = [
        {"window": 4, "units": u, "epochs": e, "batch_size": b}
        for u in [8, 16] for e in [20, 40] for b in [8, 16]
    ]
    assert space.candidates() == expected
    assert space.candidates() == expected


@pytest.mark.parametrize("cls", [LSTMParameterSpace, GRUParameterSpace, HybridNeuralParameterSpace])
def test_single_optional_dimension(cls):
    space = cls(window=[6], units=[32], epochs=[50])
    assert space.candidates() == [{"window": 6, "units": 32, "epochs": 50}]


@pytest.mark.parametrize("cls", [LSTMParameterSpace, GRUParameterSpace, HybridNeuralParameterSpace])
@pytest.mark.parametrize("kwargs", [
    {"epochs": []}, {"batch_size": [0]}, {"epochs": [True]},
    {"batch_size": [1.5]}, {"epochs": "100"},
])
def test_invalid_optional_dimensions(cls, kwargs):
    with pytest.raises((TypeError, ValueError)):
        cls(window=[6], units=[32], **kwargs)
