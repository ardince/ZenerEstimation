"""
Adaptive Kalman filter.

Provides the numerical state estimation engine used by
Kalman-based forecasting models.

This class is intentionally independent from BatteryDataset
and ForecastResult.
"""

from __future__ import annotations

import numpy as np

from zenerestimation.visualization import forecast

#from zenerestimation.data import dataset


class AdaptiveKalmanFilter:
    """
    Adaptive two-state Kalman filter.

    State vector

        x = [value, drift]

    where

        value : estimated signal

        drift : first derivative

    The filter automatically estimates measurement noise
    from the observed signal.
    """

    # ---------------------------------------------------------
    # Construction
    # ---------------------------------------------------------

    def __init__(
        self,
        dt=0.25,
        process_noise=1e-4,
        measurement_noise=None,
    ):

        self.dt = float(dt)

        self.process_noise = process_noise

        self.measurement_noise = measurement_noise

        self.values = None

        self.states = None

        self.bias = 0.0

        self.residual_sigma = None

        self.P = None

        self.Q = None

        self.R = None

        self.F = None

        self.H = np.array([[1.0, 0.0]])

    # ---------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------

    def _initialize(
        self,
        values,
    ):
        """
        Initialise filter matrices.
        """

        values = np.asarray(
            values,
            dtype=float,
        )

        self.values = values

        self.n_samples = len(values)

        # ----------------------------------------
        # Initial drift estimate
        # ----------------------------------------

        if len(values) >= 6:

            drift = (

                values[5] - values[0]

            ) / (5 * self.dt)

        else:

            drift = 0.0

        # initial state

        self.x = np.array(

            [

                values[0],

                drift,

            ],

            dtype=float,

        )

        # covariance

        self.P = np.eye(2)

        # transition matrix

        self.F = np.array(

            [

                [1.0, self.dt],

                [0.0, 1.0],

            ]

        )

        # process noise

        self.Q = np.array(

            [

                [

                    self.process_noise,

                    0.0,

                ],

                [

                    0.0,

                    self.process_noise / 100.0,

                ],

            ]

        )

        # measurement noise

        if self.measurement_noise is None:

            dy = np.diff(values)

            variance = max(

                np.var(dy),

                1e-6,

            )

        else:

            variance = self.measurement_noise

        self.R = np.array(

            [

                [

                    variance,

                ]

            ]

        )

        # state history

        self.states = np.zeros(

            (

                self.n_samples,

                2,

            )

        )


    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def _predict(self):
        """
        Kalman prediction step.
        """

        self.x = self.F @ self.x

        self.P = (

            self.F
            @ self.P
            @ self.F.T

            + self.Q

        )

    # ---------------------------------------------------------
    # Update
    # ---------------------------------------------------------

    def _update(
        self,
        measurement,
    ):
        """
        Kalman measurement update.
        """

        innovation = (

            measurement

            - (self.H @ self.x)[0]

        )

        S = (

            self.H
            @ self.P
            @ self.H.T

            + self.R

        )

        K = (

            self.P
            @ self.H.T
            @ np.linalg.inv(S)

        )

        self.x = (

            self.x

            + (K.flatten() * innovation)

        )

        self.P = (

            np.eye(2)

            - K @ self.H

        ) @ self.P

        return innovation

    # ---------------------------------------------------------
    # Fit
    # ---------------------------------------------------------

    def fit(
        self,
        dataset,
    ):
        """
        Run the adaptive Kalman filter.
        """

        if hasattr(dataset, "target"):
            values = dataset.target.to_numpy(dtype=float)
        else:
            values = np.asarray(dataset, dtype=float)

        self._initialize(values)

        innovations = []

        for i, measurement in enumerate(self.values):

            self._predict()

            innovation = self._update(measurement)

            innovations.append(innovation)

            self.states[i] = self.x.copy()

        self.innovations = np.asarray(
            innovations,
            dtype=float,
        )

        # ---------------------------------------------------------
        # Store final Kalman state
        # ---------------------------------------------------------

        self._compute_bias()

        self._compute_sigma()

        self.state = self.x.copy()

        self.covariance = self.P.copy()

        return self


    # ---------------------------------------------------------
    # Bias correction
    # ---------------------------------------------------------

    def _compute_bias(self):
        """
        Estimate systematic bias between the
        measured signal and the Kalman trend.
        """

        trend = self.states[:, 0]

        self.bias = float(

            np.mean(

                self.values - trend

            )

        )

    # ---------------------------------------------------------
    # Residual statistics
    # ---------------------------------------------------------

    def _compute_sigma(self):
        """
        Estimate residual standard deviation.
        """

        corrected = self.states[:, 0] + self.bias

        residuals = (

            self.values

            - corrected

        )

        self.residual_sigma = float(

            np.std(residuals)

        )

    # ---------------------------------------------------------
    # Smoothed trend
    # ---------------------------------------------------------

    def smooth(self):
        """
        Return the bias-corrected trend.
        """

        if self.states is None:

            raise RuntimeError(

                "Filter has not been fitted."

            )

        if self.bias == 0.0:

            self._compute_bias()

        return self.states[:, 0] + self.bias

    # ---------------------------------------------------------
    # Residuals
    # ---------------------------------------------------------

    def residuals(self):
        """
        Return signal residuals.
        """

        trend = self.smooth()

        return self.values - trend

    # ---------------------------------------------------------
    # Residual sigma
    # ---------------------------------------------------------

    def sigma(self):
        """
        Return residual standard deviation.
        """

        if self.residual_sigma is None:

            self._compute_sigma()

        return self.residual_sigma

    # ---------------------------------------------------------
    # Median drift
    # ---------------------------------------------------------

    def drift(
        self,
        window=12,
    ):
        """
        Return median annualised drift.

        Parameters
        ----------
        window :
            Number of final samples used.
        """

        if self.states is None:

            raise RuntimeError(

                "Filter has not been fitted."

            )

        drift = self.states[:, 1] * 4.0

        if len(drift) < window:

            return float(

                np.median(drift)

            )

        return float(

            np.median(

                drift[-window:]

            )

        )


    def forecast(
        self,
        steps,
    ):
        """
        Forecast future trend using
        the last estimated Kalman state.
        """

        if not hasattr(self, "state"):
            raise RuntimeError(
            "AdaptiveKalmanFilter must be fitted before forecasting."
            )

        state = self.state.copy()

        F = np.array([
        [1.0, self.dt],
        [0.0, 1.0],
        ])

        forecast = []

        for _ in range(steps):

            state = F @ state

            forecast.append(state[0])

        return np.asarray(forecast)
    