from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class BenchmarkModelSpec:
    """Frozen model configuration for final benchmark evaluation."""

    name: str
    factory: Callable[..., Any]
    params: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(
                "name must be a string"
            )

        if not self.name.strip():
            raise ValueError(
                "name must be a non-empty string"
            )

        if not callable(self.factory):
            raise TypeError(
                "factory must be callable"
            )

        if not isinstance(self.params, dict):
            raise TypeError(
                "params must be a dictionary"
            )

        object.__setattr__(
            self,
            "params",
            deepcopy(self.params),
        )

    def create_model(self):
        """Create a fresh model from the frozen configuration."""

        return self.factory(
            **deepcopy(self.params)
        )


class OptimizedBenchmark:
    """Evaluate frozen model configurations on one benchmark contract.

    OptimizedBenchmark is an orchestration layer. It does not optimize
    parameters, select seeds, preprocess data independently, calculate
    metrics independently, rank models, or select a winning model.

    Each BenchmarkModelSpec creates a fresh model instance. Scientific
    evaluation is delegated to the supplied ForecastEvaluator.
    """

    def __init__(
        self,
        evaluator,
    ) -> None:
        if evaluator is None:
            raise TypeError(
                "evaluator must not be None"
            )

        if not callable(
            getattr(
                evaluator,
                "evaluate",
                None,
            )
        ):
            raise TypeError(
                "evaluator must provide a callable "
                "evaluate method"
            )

        self._evaluator = evaluator

    @property
    def evaluator(self):
        """Evaluator used for all benchmark runs."""

        return self._evaluator

    def evaluate(
        self,
        dataset,
        specs,
    ) -> dict[str, Any]:
        """Evaluate frozen model specifications.

        Parameters
        ----------
        dataset:
            Dataset passed unchanged to the standardized evaluator.

        specs:
            Iterable of BenchmarkModelSpec objects.

        Returns
        -------
        dict[str, Any]
            Evaluation results keyed by model name, preserving
            specification order.

        Notes
        -----
        Each specification creates a fresh model instance. No result
        from one model evaluation influences any subsequent model.
        """

        specs = self._normalize_specs(
            specs
        )

        results: dict[str, Any] = {}

        for spec in specs:
            model = spec.create_model()

            result = self._evaluator.evaluate(
                dataset,
                model,
            )

            results[spec.name] = result

        return results

    @staticmethod
    def _normalize_specs(
        specs,
    ) -> tuple[BenchmarkModelSpec, ...]:
        """Validate and normalize benchmark specifications."""

        if isinstance(
            specs,
            BenchmarkModelSpec,
        ):
            raise TypeError(
                "specs must be an iterable of "
                "BenchmarkModelSpec objects"
            )

        try:
            normalized = tuple(
                specs
            )
        except TypeError as exc:
            raise TypeError(
                "specs must be an iterable of "
                "BenchmarkModelSpec objects"
            ) from exc

        if not normalized:
            raise ValueError(
                "at least one benchmark model "
                "specification is required"
            )

        if not all(
            isinstance(
                spec,
                BenchmarkModelSpec,
            )
            for spec in normalized
        ):
            raise TypeError(
                "all specs must be "
                "BenchmarkModelSpec instances"
            )

        names = [
            spec.name
            for spec in normalized
        ]

        if len(names) != len(
            set(names)
        ):
            raise ValueError(
                "benchmark model names must be unique"
            )

        return normalized