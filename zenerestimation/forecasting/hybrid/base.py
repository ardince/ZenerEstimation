"""
Base infrastructure for hybrid forecasting models.

Hybrid forecasters combine a trend model with a residual model
while exposing the same public API as every forecasting model
inside ZenerEstimation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from zenerestimation.forecasting import ForecastResult


class BaseHybridForecaster(ABC):
    """
    Base class for every hybrid forecasting model.
    """

    def __init__(
        self,
        trend_model,
        residual_model,
    ):

        self.trend_model = trend_model
        self.residual_model = residual_model

        self.dataset = None

        self.trend_result = None
        self.residual_result = None
        self.combined_result = None

        self.validate_components()

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validate_components(self):
        """
        Validate both forecasting components.
        """

        for model in (
            self.trend_model,
            self.residual_model,
        ):

            if not hasattr(model, "fit"):
                raise TypeError(
                    f"{model.__class__.__name__} "
                    "does not implement fit()."
                )

            if not hasattr(model, "predict"):
                raise TypeError(
                    f"{model.__class__.__name__} "
                    "does not implement predict()."
                )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def fit(
        self,
        dataset,
    ):
        """
        Train the hybrid model.
        """

        self.dataset = dataset

        self.trend_model.fit(dataset)

        residual_dataset = self.prepare_residuals(dataset)

        self.residual_model.fit(residual_dataset)

        return self

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict(
        self,
        steps,
    ):
        """
        Produce a hybrid forecast.
        """

        self.trend_result = self.trend_model.predict(steps)

        self.residual_result = self.residual_model.predict(steps)

        self.combined_result = self.combine_forecasts(
            self.trend_result,
            self.residual_result,
        )

        return self.combined_result

    # ---------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------

    def fit_predict(
        self,
        dataset,
        steps,
    ):

        self.fit(dataset)

        return self.predict(steps)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self):

        return {

            "family": "Hybrid",

            "trend_model":
                self.trend_model.__class__.__name__,

            "residual_model":
                self.residual_model.__class__.__name__,

            **self.summary_metadata(),
        }

    # ---------------------------------------------------------
    # Abstract methods
    # ---------------------------------------------------------

    @abstractmethod
    def prepare_residuals(
        self,
        dataset,
    ):
        """
        Build the residual dataset.
        """

    @abstractmethod
    def combine_forecasts(
        self,
        trend_result: ForecastResult,
        residual_result: ForecastResult,
    ):
        """
        Combine both forecasting results.
        """

    @abstractmethod
    def summary_metadata(self):
        """
        Return model-specific metadata.
        """