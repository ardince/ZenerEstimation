"""
Rolling window generator.

Every forecasting model should use this implementation.
"""

from __future__ import annotations

import numpy as np


class WindowGenerator:
    """
    Create rolling windows from a BatteryDataset.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def generate(self, window: int):

        self.dataset.prepare()

        values = self.dataset.data["microVolt"].to_numpy()

        X = []
        y = []

        for i in range(len(values) - window):

            X.append(values[i:i + window])

            y.append(values[i + window])

        return np.asarray(X), np.asarray(y)