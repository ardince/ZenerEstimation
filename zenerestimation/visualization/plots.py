"""
Visualization utilities for battery datasets.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


class DatasetPlotter:
    """
    Plot BatteryDataset objects.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def plot_series(self):

        self.dataset.prepare()

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            self.dataset.data["ds"],
            self.dataset.data["microVolt"],
            marker="o",
        )

        ax.set_title("Battery Voltage")

        ax.set_xlabel("Date")

        ax.set_ylabel("microVolt")

        ax.grid(True)

        return fig

    def plot_histogram(self):

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(self.dataset.data["microVolt"])

        ax.set_title("Voltage Distribution")

        return fig

    def plot_boxplot(self):

        fig, ax = plt.subplots(figsize=(4, 5))

        ax.boxplot(
            self.dataset.data["microVolt"]
        )

        ax.set_title("Voltage Boxplot")

        return fig