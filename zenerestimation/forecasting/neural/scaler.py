"""
Sequence scaling utilities.

Provides a unified interface around scikit-learn scalers.
"""

from __future__ import annotations

import numpy as np

from sklearn.preprocessing import MinMaxScaler


class SequenceScaler:
    """
    Wrapper around MinMaxScaler.
    """

    def __init__(self):

        self.scaler = MinMaxScaler()

    def fit(
        self,
        values,
    ):

        values = np.asarray(values).reshape(
            -1,
            1,
        )

        self.scaler.fit(values)

        return self

    def transform(
        self,
        values,
    ):

        values = np.asarray(values).reshape(
            -1,
            1,
        )

        return self.scaler.transform(
            values
        ).flatten()

    def fit_transform(
        self,
        values,
    ):

        values = np.asarray(values).reshape(
            -1,
            1,
        )

        return self.scaler.fit_transform(
            values
        ).flatten()

    def inverse_transform(
        self,
        values,
    ):

        values = np.asarray(values).reshape(
            -1,
            1,
        )

        return self.scaler.inverse_transform(
            values
        ).flatten()