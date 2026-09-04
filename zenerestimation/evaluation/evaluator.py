"""
Forecast evaluation engine.

This module provides reusable holdout evaluation for forecasting
models in ZenerEstimation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.evaluation.result import EvaluationResult


class ForecastEvaluator:
    """
    Evaluate a forecasting model using a deterministic holdout split.

    The final ``evaluation_steps`` observations are reserved for
    validation. The supplied forecasting model is fitted only on the
    preceding training observations.

    Parameters
    ----------
    evaluation_steps:
        Number of observations reserved for holdout evaluation.
    """

    def __init__(
        self,
        evaluation_steps: int = 6,
    ) -> None:

        if evaluation_steps <= 0:
            raise ValueError(
                "evaluation_steps must be greater than zero"
            )

        self.evaluation_steps = evaluation_steps

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def evaluate(
        self,
        dataset: BatteryDataset,
        model: Any,
    ) -> EvaluationResult:
        """
        Evaluate a forecasting model on the dataset holdout period.

        Parameters
        ----------
        dataset:
            Complete battery dataset.

        model:
            Forecasting model implementing::

                fit(dataset)
                predict(steps)

        Returns
        -------
        EvaluationResult
            Standardized holdout evaluation result.
        """

        self._validate_dataset(dataset)
        self._validate_model(model)

        train_dataset, validation_df = (
            self.split_dataset(dataset)
        )

        # ----------------------------------------------------
        # Fit only on training data
        # ----------------------------------------------------

        model.fit(train_dataset)

        # ----------------------------------------------------
        # Forecast holdout period
        # ----------------------------------------------------

        forecast_result = model.predict(
            self.evaluation_steps
        )

        actual = np.asarray(
            validation_df["microVolt"],
            dtype=float,
        )

        predicted = np.asarray(
            forecast_result.forecast,
            dtype=float,
        )

        if len(predicted) != self.evaluation_steps:
            raise ValueError(
                "model prediction length must match "
                "evaluation_steps"
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        rmse = self.rmse(
            actual,
            predicted,
        )

        mae = self.mae(
            actual,
            predicted,
        )

        mape = self.mape(
            actual,
            predicted,
        )

        # ----------------------------------------------------
        # Model name
        # ----------------------------------------------------

        model_name = self._model_name(
            model,
            forecast_result,
        )

        # ----------------------------------------------------
        # Standardized result
        # ----------------------------------------------------

        return EvaluationResult(
            model=model_name,
            evaluation_steps=self.evaluation_steps,
            rmse=rmse,
            mae=mae,
            mape=mape,
            actual=tuple(
                float(value)
                for value in actual
            ),
            predicted=tuple(
                float(value)
                for value in predicted
            ),
            dates=tuple(
                validation_df["ds"]
            ),
            metadata={
                "training_points": len(
                    train_dataset
                ),
                "validation_points": len(
                    validation_df
                ),
            },
        )

    # --------------------------------------------------------
    # Dataset splitting
    # --------------------------------------------------------

    def split_dataset(
        self,
        dataset: BatteryDataset,
    ) -> tuple[BatteryDataset, Any]:
        """
        Split a complete dataset into training and holdout portions.

        Returns
        -------
        tuple
            ``(training_dataset, validation_dataframe)``
        """

        self._validate_dataset(dataset)

        split_index = (
            len(dataset)
            - self.evaluation_steps
        )

        train_df = (
            dataset.data
            .iloc[:split_index]
            .copy()
        )

        validation_df = (
            dataset.data
            .iloc[split_index:]
            .copy()
        )

        train_dataset = BatteryDataset(
            train_df
        )

        return (
            train_dataset,
            validation_df,
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    @staticmethod
    def rmse(
        actual,
        predicted,
    ) -> float:
        """
        Root Mean Squared Error.
        """

        actual, predicted = (
            ForecastEvaluator._prepare_arrays(
                actual,
                predicted,
            )
        )

        return float(
            np.sqrt(
                np.mean(
                    (actual - predicted) ** 2
                )
            )
        )

    @staticmethod
    def mae(
        actual,
        predicted,
    ) -> float:
        """
        Mean Absolute Error.
        """

        actual, predicted = (
            ForecastEvaluator._prepare_arrays(
                actual,
                predicted,
            )
        )

        return float(
            np.mean(
                np.abs(
                    actual - predicted
                )
            )
        )

    @staticmethod
    def mape(
        actual,
        predicted,
    ) -> float:
        """
        Mean Absolute Percentage Error.

        Zero-valued actual observations are excluded from the
        percentage calculation because percentage error is undefined
        for those observations.
        """

        actual, predicted = (
            ForecastEvaluator._prepare_arrays(
                actual,
                predicted,
            )
        )

        non_zero = (
            actual != 0
        )

        if not np.any(non_zero):
            raise ValueError(
                "MAPE cannot be calculated when "
                "all actual values are zero"
            )

        percentage_error = np.abs(
            (
                actual[non_zero]
                - predicted[non_zero]
            )
            / actual[non_zero]
        )

        return float(
            np.mean(
                percentage_error
            )
            * 100.0
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def _validate_dataset(
        self,
        dataset: BatteryDataset,
    ) -> None:

        if not isinstance(
            dataset,
            BatteryDataset,
        ):
            raise TypeError(
                "dataset must be a BatteryDataset"
            )

        if len(dataset) <= self.evaluation_steps:
            raise ValueError(
                "evaluation_steps must be smaller "
                "than the dataset length"
            )

    @staticmethod
    def _validate_model(
        model: Any,
    ) -> None:

        if not callable(
            getattr(
                model,
                "fit",
                None,
            )
        ):
            raise TypeError(
                "model must implement fit(dataset)"
            )

        if not callable(
            getattr(
                model,
                "predict",
                None,
            )
        ):
            raise TypeError(
                "model must implement predict(steps)"
            )

    @staticmethod
    def _prepare_arrays(
        actual,
        predicted,
    ) -> tuple[np.ndarray, np.ndarray]:

        actual_array = np.asarray(
            actual,
            dtype=float,
        )

        predicted_array = np.asarray(
            predicted,
            dtype=float,
        )

        if actual_array.ndim != 1:
            raise ValueError(
                "actual must be one-dimensional"
            )

        if predicted_array.ndim != 1:
            raise ValueError(
                "predicted must be one-dimensional"
            )

        if len(actual_array) == 0:
            raise ValueError(
                "actual and predicted cannot be empty"
            )

        if len(actual_array) != len(
            predicted_array
        ):
            raise ValueError(
                "actual and predicted must have "
                "the same length"
            )

        if not np.all(
            np.isfinite(actual_array)
        ):
            raise ValueError(
                "actual must contain only finite values"
            )

        if not np.all(
            np.isfinite(predicted_array)
        ):
            raise ValueError(
                "predicted must contain only finite values"
            )

        return (
            actual_array,
            predicted_array,
        )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @staticmethod
    def _model_name(
        model: Any,
        forecast_result: Any,
    ) -> str:
        """
        Resolve a human-readable model name.
        """

        result_name = getattr(
            forecast_result,
            "model",
            None,
        )

        if result_name:
            return str(result_name)

        return model.__class__.__name__

    def __repr__(self) -> str:
        return (
            "ForecastEvaluator("
            f"evaluation_steps="
            f"{self.evaluation_steps}"
            ")"
        )