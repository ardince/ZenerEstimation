from __future__ import annotations

import numpy as np
import pandas as pd

from zenerestimation.forecasting import (
    BaseForecastModel,
    ForecastResult,
)


class KalmanForecaster(BaseForecastModel):
    """
    Adaptive Kalman forecaster.

    State vector

        x = [value,
             drift]

    Drift is expressed in microVolt/year.
    """

    def __init__(
        self,
        dt=0.25,
        process_noise=1e-3,
        drift_noise=1e-4,
        adaptive=True,
        regime_factor=2.5,
        regime_multiplier=10.0,
    ):

        self.dt = dt

        self.process_noise = process_noise
        self.drift_noise = drift_noise

        self.adaptive = adaptive

        self.regime_factor = regime_factor
        self.regime_multiplier = regime_multiplier

        self.dataset = None

        self.states = None
        self.regimes = None

        self.bias = 0.0
        self.sigma = None

        self.last_state = None

    # ---------------------------------------------------------
    # Convenience properties
    # ---------------------------------------------------------

    @property
    def trend(self):

        if self.states is None:
            return None

        return self.states[:, 0]

    @property
    def drift(self):

        if self.states is None:
            return None

        return self.states[:, 1]

    # ---------------------------------------------------------
    # Fit
    # ---------------------------------------------------------

    def fit(self, dataset):

        dataset.prepare()

        self.dataset = dataset

        y = dataset.target.values

        n = len(y)

        dt = self.dt

        # ---------------------------------------------
        # Measurement noise
        # ---------------------------------------------

        dy = np.diff(y)

        R = np.array([[np.var(dy)]])

        self.measurement_noise = float(R[0, 0])

        # ---------------------------------------------
        # Process noise
        # ---------------------------------------------

        Q_base = np.array([
            [self.process_noise, 0.0],
            [0.0, self.drift_noise],
        ])

        # ----------------------------------------
        # Store filter configuration
        # ----------------------------------------

        #self.process_noise = float(Q_base[0, 0])

        #self.drift_noise = float(Q_base[1, 1])

        # ---------------------------------------------
        # Initial state
        # ---------------------------------------------

        init_drift = (y[5] - y[0]) / (5 * dt)

        x = np.array([
            y[0],
            init_drift,
        ])

        F = np.array([
            [1, dt],
            [0, 1],
        ])

        H = np.array([
            [1, 0],
        ])

        P = np.eye(2)

        threshold = self.regime_factor * np.std(dy)

        states = []

        regimes = []

        # ---------------------------------------------
        # Kalman loop
        # ---------------------------------------------

        for t in range(n):

            if (
                self.adaptive
                and t > 0
                and abs(y[t] - y[t - 1]) > threshold
            ):

                regime = "FAST"

                Q = (
                    Q_base
                    * self.regime_multiplier
                )

            else:

                regime = "SLOW"

                Q = Q_base

            regimes.append(regime)

            # Prediction

            x = F @ x

            P = F @ P @ F.T + Q

            # Update

            z = np.array([y[t]])

            residual = z - H @ x

            S = H @ P @ H.T + R

            K = P @ H.T @ np.linalg.inv(S)

            x = x + (K @ residual).flatten()

            P = (np.eye(2) - K @ H) @ P

            states.append(x.copy())

        self.states = np.array(states)

        self.bias = 0.0

        self.sigma = None

        self.regimes = regimes

        self.fast_regimes = self.regimes.count("FAST")

        self.slow_regimes = self.regimes.count("SLOW")

        self.last_state = self.states[-1].copy()

        # ---------------------------------------------
        # Bias correction
        # ---------------------------------------------

        self.bias = np.mean(

            y - self.trend

        )

        self.states[:, 0] += self.bias

        # ---------------------------------------------
        # Residual noise
        # ---------------------------------------------

        residuals = y - self.trend

        self.sigma = np.std(residuals)

        return self

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    def predict(self, steps=1):

        dt = self.dt

        F = np.array([
            [1, dt],
            [0, 1],
        ])

        x = self.last_state.copy()

        x[0] += self.bias

        forecast = []

        for _ in range(steps):

            x = F @ x

            forecast.append(x[0])

        last_date = self.dataset.data["ds"].iloc[-1]

        freq = pd.infer_freq(
            self.dataset.data["ds"]
        )

        if freq is None:
            freq = "QS-JAN"

        dates = pd.date_range(
            start=last_date,
            periods=steps + 1,
            freq=freq,
        )[1:]

        return ForecastResult(

            model="Kalman",

            forecast=pd.Series(forecast),

            fitted=pd.Series(self.trend),

            horizon=steps,

            dates=dates,

            metadata={

                "adaptive": self.adaptive,

                "dt": self.dt,

                "process_noise": self.process_noise,

                "drift_noise": self.drift_noise,

                "measurement_noise": round(self.measurement_noise, 6),

                "bias_correction": round(float(self.bias), 6),

                "residual_sigma": round(float(self.sigma), 6),

                "regime_factor": self.regime_factor,

                "fast_regimes": self.fast_regimes,

                "slow_regimes": self.slow_regimes,

            },

        )

    # ---------------------------------------------------------
    # Convenience method
    # ---------------------------------------------------------

    def fit_predict(self, dataset, steps=1):

        self.fit(dataset)

        return self.predict(steps)