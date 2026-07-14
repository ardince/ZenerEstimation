"""
Forecast pipeline.

Provides a common execution interface for all
forecasting models.
"""

# TODO (Sprint 5):
# Replace 'steps' with a ForecastConfig object once
# forecasting models (ARIMA, Kalman, LSTM, GRU) are implemented.

from __future__ import annotations

from .models import ModelRegistry


class ForecastPipeline:
    """
    Forecast execution pipeline.
    """

    def __init__(self, dataset):

        self.dataset = dataset

        self.registry = ModelRegistry()

    
    def register_model(self, name, model_cls):

        self.registry.register(name, model_cls)

    
    def available_models(self):

        return self.registry.available_models()

    
    def run(self, model_name, steps=1, *args, **kwargs):
        """
        Execute a forecasting model.
        """

        model_cls = self.registry.get(model_name)

        model = model_cls(*args, **kwargs)

        model.fit(self.dataset)

        return model.predict(steps)