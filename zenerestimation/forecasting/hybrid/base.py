"""
Base infrastructure for hybrid forecasting models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from zenerestimation.forecasting import ForecastResult

from zenerestimation.diagnostics import HybridDiagnostics


class BaseHybridForecaster(ABC):
    """
    Base class for every hybrid forecasting model.
    """

    def __init__(
        self,
        residual_model,
    ):

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
        Validate the residual forecasting model.
        """

        if not hasattr(self.residual_model, "fit"):
            raise TypeError(
                f"{self.residual_model.__class__.__name__} "
                "does not implement fit()."
            )

        if not hasattr(self.residual_model, "predict"):
            raise TypeError(
                f"{self.residual_model.__class__.__name__} "
                "does not implement predict()."
            )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def fit(
        self,
        dataset,
    ):

        self.dataset = dataset

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

        self.trend_result = self.forecast_trend(steps)

        self.residual_result = self.residual_model.predict(
            steps
        )

        self.combined_result = self.combine_forecasts(
            self.trend_result,
            self.residual_result,
        )

        return self.combined_result

    # ---------------------------------------------------------

    def fit_predict(
        self,
        dataset,
        steps,
    ):

        self.fit(dataset)

        return self.predict(steps)

    # ---------------------------------------------------------

    def summary(self):

        return {

            "family": "Hybrid",

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
    def forecast_trend(
        self,
        steps,
    ) -> ForecastResult:
        """
        Forecast or extrapolate the trend component.
        """

    @abstractmethod
    def combine_forecasts(
        self,
        trend_result: ForecastResult,
        residual_result: ForecastResult,
    ):
        """
        Combine trend and residual forecasts.
        """

    @abstractmethod
    def summary_metadata(self):
        """
        Return model-specific metadata.
        """


    def diagnostics(
        self,
        dataset=None,
    ):
        """
        Run hybrid diagnostics.
        """

        if dataset is None:

            dataset = self.dataset

        return HybridDiagnostics(self).run(dataset)