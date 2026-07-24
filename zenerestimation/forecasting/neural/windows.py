"""
Sliding window generation utilities.

Converts a one-dimensional time series into
input/output sequences suitable for neural networks.
"""

from __future__ import annotations

import numpy as np


class WindowGenerator:
    """
    Generate sliding windows from a time series.
    """

    def __init__(
        self,
        window=6,
    ):

        self.window = int(window)

    def transform(
        self,
        values,
    ):
        """
        Convert a sequence into supervised learning windows.

        Returns
        -------
        X : ndarray
            Shape (samples, window, 1)

        y : ndarray
            Shape (samples,)
        """

        values = np.asarray(
            values,
            dtype=float,
        )

        X = []
        y = []

        for i in range(
            len(values) - self.window
        ):

            X.append(
                values[
                    i:i + self.window
                ]
            )

            y.append(
                values[
                    i + self.window
                ]
            )

        X = np.asarray(X)

        y = np.asarray(y)

        # TensorFlow expects
        # (samples, timesteps, features)

        X = X.reshape(
            X.shape[0],
            X.shape[1],
            1,
        )

        return X, y

    def summary(self):

        return {

            "window": self.window,

        }