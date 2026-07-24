"""
Base classes for neural forecasting models.

Provides the common infrastructure shared by
LSTM, GRU and future neural forecasters.
"""

from __future__ import annotations

from zenerestimation.forecasting import ForecastResult

from .windows import WindowGenerator
from .scaler import SequenceScaler


class BaseNeuralForecaster:
    """
    Base class for all neural forecasting models.
    """

    def __init__(
        self,
        window=6,
    ):

        self.window = window

        self.dataset = None

        self.generator = WindowGenerator(
            window=window,
        )

        self.scaler = SequenceScaler()

        self.model = None

        self.history = None

        self.fitted = None

    # ---------------------------------------------------------
    # Data preparation
    # ---------------------------------------------------------

    def prepare_data(
        self,
        dataset,
    ):

        dataset.prepare()

        self.dataset = dataset

        values = dataset.target.values

        scaled = self.scaler.fit_transform(values)

        X, y = self.generator.transform(
            scaled
        )

        return X, y

    # ---------------------------------------------------------
    # Model creation
    # ---------------------------------------------------------

    def build_model(self):
        """
        Create the neural network.

        Implemented by subclasses.
        """

        raise NotImplementedError

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def fit(
        self,
        dataset,
    ):

        raise NotImplementedError

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict(
        self,
        steps,
    ):

        raise NotImplementedError

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


    def summary(self):

        return {

            "window": self.window,

            "model": self.__class__.__name__,

            "dataset": self.dataset.name if self.dataset else None,}