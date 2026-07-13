"""
Forecast model registry.

Forecasting models are registered here and can
be retrieved by name.
"""

from __future__ import annotations

from .base import BaseForecastModel


class ModelRegistry:
    """
    Registry for forecasting models.
    """

    def __init__(self):
        self._models = {}


    def register(self, name: str, model_cls):

        if not issubclass(model_cls, BaseForecastModel):
            raise TypeError(
                "Forecast model must inherit "
                "from BaseForecastModel."
            )

        self._models[name.lower()] = model_cls


    def get(self, name: str):

        key = name.lower()

        if key not in self._models:
            raise ValueError(
                f"Unknown forecasting model: '{name}'"
            )

        return self._models[key]

    def available_models(self):

        return sorted(self._models.keys())

    def __contains__(self, name):

        return name.lower() in self._models

    def __len__(self):

        return len(self._models)