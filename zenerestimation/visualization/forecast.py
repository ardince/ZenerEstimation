"""
Forecast visualization utilities.

Provides a generic plotting interface for all forecasting models.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


class ForecastPlot:
    """
    Plot historical measurements together with forecast results.
    """

    def __init__(
    self,
    dataset,
    result,
    ):
        self.dataset = dataset
        self.result = result

        self._figure = None
        self._axes = None


    @property
    def figure(self):
        """
        Return the matplotlib Figure.
        """
        return self._figure
    

    @property
    def axes(self):
        """
        Return the matplotlib Axes.
        """
        return self._axes
    

    def plot(self):
        fig, ax = plt.subplots(
        figsize=(10, 5)
        )

        self._figure = fig
        self._axes = ax

        ax.plot(
            self.dataset.data["ds"],
            self.dataset.data["microVolt"],
            marker="o",
            label="History",
        )

        ax.plot(
            self.result.dates,
            self.result.forecast,
            marker="o",
            linestyle="--",
            label="Forecast",
        )

        ax.axvline(
            self.dataset.data["ds"].iloc[-1],
            linestyle=":",
            linewidth=1,
        )

        ax.set_title(
            "Battery Voltage Forecast"
        )

        ax.set_xlabel(
            "Date"
        )

        ax.set_ylabel(
            "Voltage (µV)"
        )

        ax.grid(True)

        ax.legend()

        return fig
    

    def save(
        self,
        filename,
        dpi=300,
    ):
        
        if self._figure is None:
            self.plot()

        self._figure.savefig(
            filename,
            dpi=dpi,
            bbox_inches="tight",
        )


    def show(self):
        if self._figure is None:
            self.plot()

        plt.show()