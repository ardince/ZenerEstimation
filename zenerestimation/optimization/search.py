"""
Generic forecasting-model optimization orchestration.

Evaluates explicit model parameter configurations using an
OptimizationEvaluator and selects the best candidate according
to the configured minimization metric.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from zenerestimation.data.dataset import BatteryDataset

from .candidate import CandidateResult
from .evaluator import ForecastAlignmentError, OptimizationEvaluator
from .result import OptimizationResult


class GenericOptimizer:
    """
    Evaluate and select forecasting-model parameter candidates.

    The optimizer itself does not fit forecasting models directly.
    Candidate evaluation is delegated to OptimizationEvaluator.

    Parameters
    ----------
    evaluator:
        Internal temporal optimization evaluator.

    model:
        Human-readable forecasting model name.
    """

    def __init__(
        self,
        evaluator: OptimizationEvaluator,
        *,
        model: str,
        max_candidates: int | None = None,
        max_epoch_folds: int | None = None,
    ) -> None:

        if not isinstance(
            evaluator,
            OptimizationEvaluator,
        ):
            raise TypeError(
                "evaluator must be an OptimizationEvaluator"
            )

        if (
            not isinstance(model, str)
            or not model
        ):
            raise ValueError(
                "model must be a non-empty string"
            )

        for name, value in (("max_candidates", max_candidates),
                            ("max_epoch_folds", max_epoch_folds)):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")

        self.max_candidates = max_candidates
        self.max_epoch_folds = max_epoch_folds
        self.evaluator = evaluator
        self.model = model

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def optimize(
        self,
        dataset: BatteryDataset,
        model_factory: Callable[..., Any],
        candidates: Iterable[dict[str, Any]],
    ) -> OptimizationResult:
        """
        Evaluate candidate configurations and select the best one.

        Parameters
        ----------
        dataset:
            Development dataset used only for internal optimization.

        model_factory:
            Callable constructing a fresh forecasting model.

        candidates:
            Iterable of explicit parameter dictionaries.

        Returns
        -------
        OptimizationResult
            Complete optimization result.
        """

        self._validate_dataset(
            dataset
        )

        if not callable(
            model_factory
        ):
            raise TypeError(
                "model_factory must be callable"
            )

        candidate_params = (
            self._materialize_candidates(
                candidates
            )
        )

        # Budgeting is opt-in. An epoch-fold is one requested epoch in
        # one temporal fold; this is a planned upper bound, not measured work.
        fold_count = None
        if self.max_epoch_folds is not None:
            fold_count = len(self.evaluator.splitter.split(dataset))
            if fold_count < 1:
                raise ValueError("the splitter must produce at least one fold")
            for params in candidate_params:
                epochs = params.get("epochs")
                if type(epochs) is not int or epochs < 1:
                    raise ValueError(
                        "max_epoch_folds requires a positive integer 'epochs' "
                        "in every candidate (including hybrid neural candidates)"
                    )

        results = []
        planned_epoch_folds = 0
        stop_reason = None

        for params in candidate_params:
            if self.max_candidates is not None and len(results) >= self.max_candidates:
                stop_reason = "max_candidates"
                break
            if self.max_epoch_folds is not None:
                cost = params["epochs"] * fold_count
                if planned_epoch_folds + cost > self.max_epoch_folds:
                    stop_reason = "max_epoch_folds"
                    break
                planned_epoch_folds += cost

            try:

                result = self.evaluator.evaluate(
                    dataset=dataset,
                    model_factory=model_factory,
                    params=params,
                )

            except Exception as exc:

                failure_metadata = {"validation": "internal_temporal"}
                if isinstance(exc, ForecastAlignmentError):
                    failure_metadata["alignment_failure"] = dict(exc.alignment_failure)

                result = CandidateResult.failed(
                    params=params,
                    error=exc,
                    metadata=failure_metadata,
                )

            results.append(
                result
            )


        if not results:
            raise ValueError("budget does not permit evaluation of the first candidate")

        best_candidate = self._select_best(
            results
        )

        return OptimizationResult(
            model=self.model,
            metric=self.evaluator.metric,
            best_params=best_candidate.params,
            best_score=best_candidate.score,
            candidates=tuple(results),
            metadata={
                "search": "explicit_candidates",
                "objective": "minimize",
                "candidate_count": len(results),
                "validation": "internal_temporal",
                **({
                    "budget": {
                        "max_candidates": self.max_candidates,
                        "max_epoch_folds": self.max_epoch_folds,
                        "planned_epoch_folds": planned_epoch_folds,
                        "fold_count": fold_count,
                        "evaluated_candidates": len(results),
                        "total_candidates": len(candidate_params),
                        "stop_reason": stop_reason,
                    }
                } if self.max_candidates is not None or self.max_epoch_folds is not None else {}),
            },
        )

    # ---------------------------------------------------------
    # Candidate handling
    # ---------------------------------------------------------

    @staticmethod
    def _materialize_candidates(
        candidates,
    ) -> tuple[dict[str, Any], ...]:
        """
        Validate and materialize candidate configurations.
        """

        if isinstance(
            candidates,
            (str, bytes),
        ):
            raise TypeError(
                "candidates must be an iterable "
                "of parameter dictionaries"
            )

        try:
            materialized = tuple(
                candidates
            )
        except TypeError as exc:
            raise TypeError(
                "candidates must be an iterable "
                "of parameter dictionaries"
            ) from exc

        if not materialized:
            raise ValueError(
                "candidates cannot be empty"
            )

        for params in materialized:

            if not isinstance(
                params,
                dict,
            ):
                raise TypeError(
                    "each candidate must be a dictionary"
                )

            if not params:
                raise ValueError(
                    "candidate parameter dictionaries "
                    "cannot be empty"
                )

        return materialized

    # ---------------------------------------------------------
    # Selection
    # ---------------------------------------------------------
    
    @staticmethod
    def _select_best(
        candidates,
    ):

        successful = tuple(
            candidate
            for candidate in candidates
            if candidate.status == "evaluated"
        )

        if not successful:
            raise RuntimeError(
                "optimization failed because "
                "no candidate completed evaluation"
            )

        return min(
            successful,
            key=lambda candidate: (
                candidate.score
            ),
        )
    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    @staticmethod
    def _validate_dataset(
        dataset,
    ) -> None:

        if not isinstance(
            dataset,
            BatteryDataset,
        ):
            raise TypeError(
                "dataset must be a BatteryDataset"
            )

    def __repr__(self) -> str:

        return (
            "GenericOptimizer("
            f"model={self.model!r}, "
            f"metric={self.evaluator.metric!r}"
            ")"
        )