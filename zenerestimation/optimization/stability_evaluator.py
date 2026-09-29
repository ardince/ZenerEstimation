"""Repeated-seed stability evaluation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterable

from zenerestimation.data import BatteryDataset

from .evaluator import OptimizationEvaluator
from .stability import SeedResult, StabilityResult, FailedSeedResult


class StabilityEvaluator:
    """Evaluate a frozen model configuration across repeated seeds."""

    def __init__(
        self,
        evaluator: OptimizationEvaluator,
        *,
        model: str,
    ) -> None:
        if not isinstance(evaluator, OptimizationEvaluator):
            raise TypeError(
                "evaluator must be an OptimizationEvaluator"
            )

        if not isinstance(model, str):
            raise TypeError("model must be a string")

        if not model.strip():
            raise ValueError(
                "model must be a non-empty string"
            )

        self.evaluator = evaluator
        self.model = model

    def evaluate(
        self,
        dataset: BatteryDataset,
        model_factory: Callable[..., Any],
        params: dict[str, Any],
        seeds: Iterable[int],
    ) -> StabilityResult:
        """Evaluate one frozen configuration across multiple seeds."""

        if not isinstance(dataset, BatteryDataset):
            raise TypeError(
                "dataset must be a BatteryDataset"
            )

        if not callable(model_factory):
            raise TypeError(
                "model_factory must be callable"
            )

        if not isinstance(params, dict):
            raise TypeError(
                "params must be a dictionary"
            )

        if not params:
            raise ValueError(
                "params must not be empty"
            )

        if "seed" in params:
            raise ValueError(
                "params must not contain seed"
            )

        try:
            seed_values = tuple(seeds)
        except TypeError as exc:
            raise TypeError(
                "seeds must be an iterable of integers"
            ) from exc

        if not seed_values:
            raise ValueError(
                "seeds must not be empty"
            )

        for seed in seed_values:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError(
                    "all seeds must be integers"
                )

        if len(seed_values) != len(set(seed_values)):
            raise ValueError(
                "duplicate seeds are not allowed"
            )

        seed_results = []

        for seed in seed_values:
            candidate_params = deepcopy(params)
            candidate_params["seed"] = seed

            try:
                candidate = self.evaluator.evaluate(
                    dataset,
                    model_factory,
                    candidate_params,
                )

                seed_results.append(
                    SeedResult(
                        seed=seed,
                        score=candidate.score,
                        metrics=deepcopy(
                            candidate.metrics
                        ),
                        metadata={
                            "validation": deepcopy(
                                candidate.metadata
                            ),
                        },
                    )
                )

            except Exception as exc:
                seed_results.append(
                    FailedSeedResult(
                        seed=seed,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        metadata={
                            "params": deepcopy(
                                params
                            ),
                        },
                    )
                )

        if not any(
            isinstance(
                result,
                SeedResult,
            )
            for result in seed_results
        ):
            details = "; ".join(
                (
                    f"seed={result.seed}: "
                    f"{result.error_type}: "
                    f"{result.error_message}"
                )
                for result in seed_results
                if isinstance(
                    result,
                    FailedSeedResult,
                )
            )

            raise RuntimeError(
                "all repeated-seed evaluations failed: "
                f"{details}"
            )

        return StabilityResult(
            model=self.model,
            metric=self.evaluator.metric,
            params=deepcopy(params),
            seeds=tuple(seed_results),
            metadata={
                "analysis": "repeated_seed",
                "seed_count": len(seed_results),
                "selection": "none",
            },
        )