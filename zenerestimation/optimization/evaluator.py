"""
Internal optimization evaluator.

Evaluates one forecasting-model parameter configuration using
leakage-safe temporal validation folds.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.evaluation.evaluator import ForecastEvaluator

from .candidate import CandidateResult
from .validation import ExpandingWindowSplitter


class OptimizationEvaluator:
    """
    Evaluate one model configuration using internal temporal validation.

    Parameters
    ----------
    splitter:
        Temporal validation splitter.

    metric:
        Metric used as the candidate selection score.

    preprocessor:
        Optional training-data preprocessor. A fresh copy is used
        independently inside every temporal fold.
    """

    SUPPORTED_METRICS = (
        "rmse",
        "mae",
        "mape",
    )

    def __init__(
        self,
        splitter: ExpandingWindowSplitter,
        *,
        metric: str = "rmse",
        preprocessor=None,
    ) -> None:

        if not isinstance(
            splitter,
            ExpandingWindowSplitter,
        ):
            raise TypeError(
                "splitter must be an ExpandingWindowSplitter"
            )

        if (
            not isinstance(metric, str)
            or metric not in self.SUPPORTED_METRICS
        ):
            raise ValueError(
                "metric must be one of: "
                + ", ".join(self.SUPPORTED_METRICS)
            )

        if (
            preprocessor is not None
            and not callable(
                getattr(
                    preprocessor,
                    "fit_transform",
                    None,
                )
            )
        ):
            raise TypeError(
                "preprocessor must provide "
                "a callable fit_transform method"
            )

        self.splitter = splitter
        self.metric = metric
        self.preprocessor = preprocessor

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def evaluate(
        self,
        dataset: BatteryDataset,
        model_factory: Callable[..., Any],
        params: dict[str, Any],
    ) -> CandidateResult:
        """
        Evaluate one parameter configuration.

        A fresh forecasting model is created for every temporal fold.

        Parameters
        ----------
        dataset:
            Development dataset. It must not include the external
            benchmark holdout.

        model_factory:
            Callable constructing a fresh forecasting model from
            candidate parameters.

        params:
            Candidate parameter configuration.

        Returns
        -------
        CandidateResult
            Aggregated internal-validation result.
        """

        self._validate_inputs(
            dataset,
            model_factory,
            params,
        )

        folds = self.splitter.split(
            dataset
        )

        fold_metrics = []

        for fold in folds:

            self._validate_validation_partition(
                fold.validation
            )

            train_dataset = fold.train

            if self.preprocessor is not None:

                fold_preprocessor = (
                    self._fresh_preprocessor()
                )

                train_dataset = (
                    fold_preprocessor
                    .fit_transform(
                        fold.train
                    )
                )

            model = model_factory(
                **params
            )

            self._validate_model(
                model
            )

            model.fit(
                train_dataset
            )

            forecast_result = model.predict(
                fold.validation_points
            )

            actual = np.asarray(
                fold.validation["microVolt"],
                dtype=float,
            )

            predicted = np.asarray(
                forecast_result.forecast,
                dtype=float,
            )

            if (
                len(predicted)
                != fold.validation_points
            ):
                raise ValueError(
                    "model prediction length must match "
                    "the temporal validation horizon"
                )

            metrics = {
                "rmse": ForecastEvaluator.rmse(
                    actual,
                    predicted,
                ),
                "mae": ForecastEvaluator.mae(
                    actual,
                    predicted,
                ),
                "mape": ForecastEvaluator.mape(
                    actual,
                    predicted,
                ),
            }

            fold_metrics.append(
                metrics
            )

        aggregated = (
            self._aggregate_metrics(
                fold_metrics
            )
        )

        return CandidateResult(
            params=params,
            score=aggregated[self.metric],
            metrics=aggregated,
            metadata={
                "validation": "expanding_window",
                "folds": len(folds),
                "validation_steps": (
                    self.splitter.validation_steps
                ),
                "selection_metric": self.metric,
                "fold_metrics": fold_metrics,
                "preprocessing": (
                    self._preprocessing_metadata()
                ),
            },
        )

    # ---------------------------------------------------------
    # Aggregation
    # ---------------------------------------------------------

    @staticmethod
    def _aggregate_metrics(
        fold_metrics: list[dict[str, float]],
    ) -> dict[str, float]:
        """
        Calculate mean validation metrics across folds.
        """

        if not fold_metrics:
            raise ValueError(
                "fold_metrics cannot be empty"
            )

        return {
            metric: float(
                np.mean(
                    [
                        fold[metric]
                        for fold in fold_metrics
                    ]
                )
            )
            for metric in (
                "rmse",
                "mae",
                "mape",
            )
        }

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        dataset,
        model_factory,
        params,
    ) -> None:

        if not isinstance(
            dataset,
            BatteryDataset,
        ):
            raise TypeError(
                "dataset must be a BatteryDataset"
            )

        if not callable(
            model_factory
        ):
            raise TypeError(
                "model_factory must be callable"
            )

        if not isinstance(
            params,
            dict,
        ):
            raise TypeError(
                "params must be a dictionary"
            )

        if not params:
            raise ValueError(
                "params cannot be empty"
            )

    @staticmethod
    def _validate_model(
        model,
    ) -> None:

        if not callable(
            getattr(
                model,
                "fit",
                None,
            )
        ):
            raise TypeError(
                "model_factory must return a model "
                "implementing fit(dataset)"
            )

        if not callable(
            getattr(
                model,
                "predict",
                None,
            )
        ):
            raise TypeError(
                "model_factory must return a model "
                "implementing predict(steps)"
            )

    @staticmethod
    def _validate_validation_partition(
        validation,
    ) -> None:
        """
        Require untouched observed validation targets.
        """

        if (
            "microVolt"
            not in validation.columns
        ):
            raise ValueError(
                "validation partition must contain "
                "a microVolt column"
            )

        missing_mask = (
            validation["microVolt"].isna()
        )

        if not missing_mask.any():
            return

        if "ds" in validation.columns:

            missing_dates = (
                validation.loc[
                    missing_mask,
                    "ds",
                ]
                .dt.strftime(
                    "%Y-%m-%d"
                )
                .tolist()
            )

            raise ValueError(
                "validation partition contains "
                "missing target values at: "
                f"{missing_dates}"
            )

        raise ValueError(
            "validation partition contains "
            "missing target values"
        )

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    def _fresh_preprocessor(self):
        """
        Create an independent preprocessor for one fold.
        """

        from copy import deepcopy

        return deepcopy(
            self.preprocessor
        )

    def _preprocessing_metadata(
        self,
    ) -> dict[str, Any]:

        if self.preprocessor is None:
            return {
                "enabled": False,
            }

        metadata = {
            "enabled": True,
            "name": type(
                self.preprocessor
            ).__name__,
        }

        method = getattr(
            self.preprocessor,
            "method",
            None,
        )

        if method is not None:
            metadata["method"] = method

        fill_edges = getattr(
            self.preprocessor,
            "fill_edges",
            None,
        )

        if fill_edges is not None:
            metadata[
                "fill_edges"
            ] = fill_edges

        return metadata

    def __repr__(self) -> str:

        return (
            "OptimizationEvaluator("
            f"metric={self.metric!r}, "
            f"splitter={self.splitter!r}"
            ")"
        )