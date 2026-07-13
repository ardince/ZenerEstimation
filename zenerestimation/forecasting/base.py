"""
Base forecasting model.

Every forecasting algorithm in ZenerEstimation should
inherit from BaseForecastModel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseForecastModel(ABC):
    """
    Abstract base class for forecasting models.
    """

    def __init__(self):

        self.dataset = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, dataset):
        """
        Train the forecasting model.
        """

    @abstractmethod
    def predict(self, steps=1):
        """
        Produce a forecast.
        """

    def fit_predict(self, dataset, steps=1):
        """
        Convenience method:
        fit() followed by predict().
        """

        self.fit(dataset)

        return self.predict(steps)