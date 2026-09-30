import pytest

from zenerestimation.evaluation.benchmark import (
    BenchmarkModelSpec, OptimizedBenchmark,
)


class FakeModel:
    """Minimal model used to test fresh construction."""

    def __init__(
        self,
        value=1,
    ):
        self.value = value


def test_benchmark_model_spec_accepts_valid_configuration():
    spec = BenchmarkModelSpec(
        name="FakeModel",
        factory=FakeModel,
        params={
            "value": 5,
        },
    )

    assert spec.name == "FakeModel"

    assert spec.params == {
        "value": 5,
    }


def test_benchmark_model_spec_rejects_empty_name():
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        BenchmarkModelSpec(
            name="",
            factory=FakeModel,
            params={},
        )


def test_benchmark_model_spec_rejects_noncallable_factory():
    with pytest.raises(
        TypeError,
        match="callable",
    ):
        BenchmarkModelSpec(
            name="FakeModel",
            factory="invalid",
            params={},
        )


def test_benchmark_model_spec_rejects_non_dictionary_params():
    with pytest.raises(
        TypeError,
        match="dictionary",
    ):
        BenchmarkModelSpec(
            name="FakeModel",
            factory=FakeModel,
            params=[],
        )


def test_benchmark_model_spec_defensively_copies_params():
    params = {
        "value": 5,
    }

    spec = BenchmarkModelSpec(
        name="FakeModel",
        factory=FakeModel,
        params=params,
    )

    params["value"] = 999

    assert spec.params == {
        "value": 5,
    }


def test_benchmark_model_spec_creates_configured_model():
    spec = BenchmarkModelSpec(
        name="FakeModel",
        factory=FakeModel,
        params={
            "value": 5,
        },
    )

    model = spec.create_model()

    assert isinstance(
        model,
        FakeModel,
    )

    assert model.value == 5


def test_benchmark_model_spec_creates_fresh_models():
    spec = BenchmarkModelSpec(
        name="FakeModel",
        factory=FakeModel,
        params={
            "value": 5,
        },
    )

    first = spec.create_model()
    second = spec.create_model()

    assert first is not second


# ---------------------------------------------------------------------
# OptimizedBenchmark helpers
# ---------------------------------------------------------------------


class FakeEvaluator:
    """Minimal evaluator used to test benchmark orchestration."""

    def __init__(self):
        self.calls = []

    def evaluate(
        self,
        dataset,
        model,
    ):
        self.calls.append(
            {
                "dataset": dataset,
                "model": model,
            }
        )

        return {
            "value": model.value,
        }


# ---------------------------------------------------------------------
# OptimizedBenchmark — construction
# ---------------------------------------------------------------------


def test_optimized_benchmark_accepts_evaluator():
    evaluator = FakeEvaluator()

    benchmark = OptimizedBenchmark(
        evaluator=evaluator,
    )

    assert benchmark.evaluator is evaluator


def test_optimized_benchmark_rejects_none_evaluator():
    with pytest.raises(
        TypeError,
        match="evaluator",
    ):
        OptimizedBenchmark(
            evaluator=None,
        )


def test_optimized_benchmark_rejects_evaluator_without_evaluate():
    with pytest.raises(
        TypeError,
        match="evaluate",
    ):
        OptimizedBenchmark(
            evaluator=object(),
        )


# ---------------------------------------------------------------------
# OptimizedBenchmark — specification validation
# ---------------------------------------------------------------------


def test_optimized_benchmark_requires_nonempty_specs():
    benchmark = OptimizedBenchmark(
        evaluator=FakeEvaluator(),
    )

    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        benchmark.evaluate(
            dataset=object(),
            specs=(),
        )


def test_optimized_benchmark_requires_iterable_specs():
    benchmark = OptimizedBenchmark(
        evaluator=FakeEvaluator(),
    )

    with pytest.raises(
        TypeError,
        match="iterable",
    ):
        benchmark.evaluate(
            dataset=object(),
            specs=123,
        )


def test_optimized_benchmark_rejects_single_spec_object():
    benchmark = OptimizedBenchmark(
        evaluator=FakeEvaluator(),
    )

    spec = BenchmarkModelSpec(
        name="FakeModel",
        factory=FakeModel,
        params={
            "value": 1,
        },
    )

    with pytest.raises(
        TypeError,
        match="iterable",
    ):
        benchmark.evaluate(
            dataset=object(),
            specs=spec,
        )


def test_optimized_benchmark_rejects_invalid_spec_element():
    benchmark = OptimizedBenchmark(
        evaluator=FakeEvaluator(),
    )

    with pytest.raises(
        TypeError,
        match="BenchmarkModelSpec",
    ):
        benchmark.evaluate(
            dataset=object(),
            specs=(
                "invalid",
            ),
        )


def test_optimized_benchmark_rejects_duplicate_model_names():
    benchmark = OptimizedBenchmark(
        evaluator=FakeEvaluator(),
    )

    specs = (
        BenchmarkModelSpec(
            name="FakeModel",
            factory=FakeModel,
            params={
                "value": 1,
            },
        ),
        BenchmarkModelSpec(
            name="FakeModel",
            factory=FakeModel,
            params={
                "value": 2,
            },
        ),
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        benchmark.evaluate(
            dataset=object(),
            specs=specs,
        )


# ---------------------------------------------------------------------
# OptimizedBenchmark — orchestration
# ---------------------------------------------------------------------


def test_optimized_benchmark_evaluates_every_spec():
    evaluator = FakeEvaluator()

    benchmark = OptimizedBenchmark(
        evaluator=evaluator,
    )

    specs = (
        BenchmarkModelSpec(
            name="ModelA",
            factory=FakeModel,
            params={
                "value": 1,
            },
        ),
        BenchmarkModelSpec(
            name="ModelB",
            factory=FakeModel,
            params={
                "value": 2,
            },
        ),
    )

    results = benchmark.evaluate(
        dataset=object(),
        specs=specs,
    )

    assert list(
        results.keys()
    ) == [
        "ModelA",
        "ModelB",
    ]

    assert len(
        evaluator.calls
    ) == 2


def test_optimized_benchmark_preserves_specification_order():
    benchmark = OptimizedBenchmark(
        evaluator=FakeEvaluator(),
    )

    specs = (
        BenchmarkModelSpec(
            name="Third",
            factory=FakeModel,
            params={
                "value": 3,
            },
        ),
        BenchmarkModelSpec(
            name="First",
            factory=FakeModel,
            params={
                "value": 1,
            },
        ),
        BenchmarkModelSpec(
            name="Second",
            factory=FakeModel,
            params={
                "value": 2,
            },
        ),
    )

    results = benchmark.evaluate(
        dataset=object(),
        specs=specs,
    )

    assert list(
        results.keys()
    ) == [
        "Third",
        "First",
        "Second",
    ]


def test_optimized_benchmark_passes_same_dataset_to_every_evaluation():
    evaluator = FakeEvaluator()

    benchmark = OptimizedBenchmark(
        evaluator=evaluator,
    )

    dataset = object()

    specs = (
        BenchmarkModelSpec(
            name="ModelA",
            factory=FakeModel,
            params={
                "value": 1,
            },
        ),
        BenchmarkModelSpec(
            name="ModelB",
            factory=FakeModel,
            params={
                "value": 2,
            },
        ),
    )

    benchmark.evaluate(
        dataset=dataset,
        specs=specs,
    )

    assert (
        evaluator.calls[0]["dataset"]
        is dataset
    )

    assert (
        evaluator.calls[1]["dataset"]
        is dataset
    )


def test_optimized_benchmark_uses_frozen_parameters():
    evaluator = FakeEvaluator()

    benchmark = OptimizedBenchmark(
        evaluator=evaluator,
    )

    spec = BenchmarkModelSpec(
        name="FakeModel",
        factory=FakeModel,
        params={
            "value": 42,
        },
    )

    results = benchmark.evaluate(
        dataset=object(),
        specs=(
            spec,
        ),
    )

    assert (
        evaluator.calls[0]["model"].value
        == 42
    )

    assert results == {
        "FakeModel": {
            "value": 42,
        },
    }


def test_optimized_benchmark_creates_fresh_models():
    evaluator = FakeEvaluator()

    benchmark = OptimizedBenchmark(
        evaluator=evaluator,
    )

    spec = BenchmarkModelSpec(
        name="FakeModel",
        factory=FakeModel,
        params={
            "value": 5,
        },
    )

    benchmark.evaluate(
        dataset=object(),
        specs=(
            spec,
        ),
    )

    first_model = (
        evaluator.calls[0]["model"]
    )

    benchmark.evaluate(
        dataset=object(),
        specs=(
            spec,
        ),
    )

    second_model = (
        evaluator.calls[1]["model"]
    )

    assert (
        first_model
        is not second_model
    )


def test_optimized_benchmark_does_not_mutate_specs():
    evaluator = FakeEvaluator()

    benchmark = OptimizedBenchmark(
        evaluator=evaluator,
    )

    spec = BenchmarkModelSpec(
        name="FakeModel",
        factory=FakeModel,
        params={
            "value": 5,
        },
    )

    original_params = dict(
        spec.params
    )

    benchmark.evaluate(
        dataset=object(),
        specs=(
            spec,
        ),
    )

    assert (
        spec.params
        == original_params
    )


def test_optimized_benchmark_propagates_evaluation_failure():
    class FailingEvaluator:
        def evaluate(
            self,
            dataset,
            model,
        ):
            raise RuntimeError(
                "benchmark evaluation failed"
            )

    benchmark = OptimizedBenchmark(
        evaluator=FailingEvaluator(),
    )

    spec = BenchmarkModelSpec(
        name="FakeModel",
        factory=FakeModel,
        params={
            "value": 1,
        },
    )

    with pytest.raises(
        RuntimeError,
        match="benchmark evaluation failed",
    ):
        benchmark.evaluate(
            dataset=object(),
            specs=(
                spec,
            ),
        )