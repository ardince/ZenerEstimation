"""
Baseline LSTM forecaster.

This implementation intentionally provides the
simplest possible LSTM architecture in order to
validate the neural forecasting infrastructure.

Future versions will extend this model with
dropout, configurable optimizers, Bayesian
hyperparameter search and hybrid architectures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

from zenerestimation.forecasting import ForecastResult

from .base import BaseNeuralForecaster
from .utils import set_seed


class LSTMForecaster(BaseNeuralForecaster):
    """
    Baseline Long Short-Term Memory forecaster.

    Notes
    -----
    This class intentionally implements the
    simplest possible LSTM architecture in order
    to validate the neural forecasting pipeline.

    Future versions will introduce:

    • Dropout
    • Early stopping
    • Learning-rate scheduling
    • Bayesian hyperparameter optimisation
    • Hybrid neural architectures
    """

    MODEL_NAME = "LSTM"

    # ---------------------------------------------------------
    # Construction
    # ---------------------------------------------------------

    def __init__(
        self,
        window=6,
        units=32,
        epochs=100,
        batch_size=8,
        seed=42,
    ):

        super().__init__(
            window=window,
        )

        self.units = int(units)

        self.epochs = int(epochs)

        self.batch_size = int(batch_size)

        self.seed = int(seed)

        self.model = None

        self.history = None

        self.fitted = None

    # ---------------------------------------------------------
    # Network architecture
    # ---------------------------------------------------------
    def build_model(self):
        """
        Build the baseline LSTM network.
        """

        model = Sequential(

            name="BaselineLSTM"

        )

        model.add(
            Input(
                shape=(self.window, 1)
            )
        )

        model.add(

            LSTM(

                units=self.units,

                name="lstm",

            )

        )

        model.add(

            Dense(

                units=1,

                name="forecast",

            )

        )

        return model

    # ---------------------------------------------------------
    # Model compilation
    # ---------------------------------------------------------

    def compile_model(self):
        """
        Compile the neural network.
        """

        self.model.compile(

            optimizer="adam",

            loss="mse",

        )


    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def fit(
        self,
        dataset,
    ):
        """
        Train the LSTM model.
        """

        # ---------------------------------------------
        # Reproducibility
        # ---------------------------------------------

        set_seed(self.seed)

        # ---------------------------------------------
        # Prepare data
        # ---------------------------------------------

        X, y = self.prepare_data(dataset)

        # Number of original observations

        n = len(dataset.target)

        # ---------------------------------------------
        # Build network
        # ---------------------------------------------

        self.model = self.build_model()

        self.compile_model()

        # ---------------------------------------------
        # Train
        # ---------------------------------------------

        self.history = self.model.fit(

            X,

            y,

            epochs=self.epochs,

            batch_size=self.batch_size,

            verbose=0,

        )

        # ---------------------------------------------
        # Fitted values
        # ---------------------------------------------

        prediction = self.model.predict(

            X,

            verbose=0,

        ).flatten()

        prediction = self.scaler.inverse_transform(
            prediction
        )

        # ---------------------------------------------
        # Align fitted values with dataset
        # ---------------------------------------------

        fitted = np.full(

            n,

            np.nan,

            dtype=float,

        )

        fitted[self.window:] = prediction

        self.fitted = pd.Series(

            fitted,

            index=self.dataset.data.index,

        )

        return self


    # ---------------------------------------------------------
    # Forecast
    # ---------------------------------------------------------

    def predict(
        self,
        steps=1,
    ):
        """
        Forecast future observations.
        """

        if self.model is None:

            raise RuntimeError(
                "Model has not been trained. "
                "Call fit() before predict()."
            )

        # ---------------------------------------------
        # Last observed window
        # ---------------------------------------------

        values = self.dataset.target.values

        scaled = self.scaler.transform(values)

        window = scaled[-self.window:].copy()

        forecast = []

        # ---------------------------------------------
        # Recursive forecasting
        # ---------------------------------------------

        for _ in range(steps):

            X = window.reshape(

                1,

                self.window,

                1,

            )

            yhat = self.model.predict(

                X,

                verbose=0,

            )[0, 0]

            forecast.append(yhat)

            window = np.concatenate(

                (

                    window[1:],

                    [yhat],

                )

            )

        # ---------------------------------------------
        # Back-transform
        # ---------------------------------------------

        forecast = self.scaler.inverse_transform(

            np.asarray(forecast).reshape(-1, 1)

        ).ravel()

        # ---------------------------------------------
        # Forecast dates
        # ---------------------------------------------

        dates = self.dataset.forecast_dates(steps)

        # ---------------------------------------------
        # Result
        # ---------------------------------------------

        return ForecastResult(

            model=self.MODEL_NAME,

            forecast=pd.Series(
                np.asarray(forecast).ravel(),
                index=dates,
            ),

            fitted=self.fitted,

            horizon=steps,

            dates=dates,

            metadata=self.summary(),

        )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    def summary(self):
        """
        Return model metadata.
        """

        summary = super().summary()

        summary.update(

            {

                "units": self.units,

                "epochs": self.epochs,

                "batch_size": self.batch_size,

                "seed": self.seed,

            }

        )

        return summary