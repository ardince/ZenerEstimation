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
    experiment=None,
    evaluation=None,
    ):
        self.experiment = experiment
        self.evaluation = evaluation
        self.dataset = dataset
        self.result = result

        self._figure = None
        self._axes = None


    def _build_title(self):

        battery = getattr(
                self.dataset,
                "battery",
                "Unknown",
        )

        return (
            f"Battery {battery}\n"
            f"{self.result.model} Forecast\n"
        )


    def _build_experiment_info(self):
        """
        Build dataset/forecast/evaluation information
        displayed inside the forecast figure.
        """

        lines = []

        # --------------------------------------------------------
        # Missing periods
        # --------------------------------------------------------

        summary = self.dataset.summary()

        missing_period_count = getattr(
            self.dataset,
            "missing_period_count",
            None,
        )

        if missing_period_count is None:
            missing_period_count = summary[
                "missing_periods"
            ]

        lines.append(
            f"Missing Periods: {missing_period_count}"
        )

        # --------------------------------------------------------
        # Forecast horizon
        # --------------------------------------------------------

        horizon = getattr(
            self.result,
            "horizon",
            None,
        )

        if horizon is not None:

            lines.append(
                f"Horizon: {horizon} quarters"
            )

        # --------------------------------------------------------
        # Evaluation
        # --------------------------------------------------------

        if self.evaluation:

            rmse = self.evaluation.get(
                "rmse"
            )

            if rmse is not None:

                lines.append(
                    f"RMSE: {rmse:.2f} µV"
                )

        return "\n".join(lines)


    def _draw_experiment_box(self, ax):
        """
        Draw experiment information on the figure.
        """

        info = self._build_experiment_info()

        if not info:
            return

        ax.text(
            0.98,
            0.02,
            info,
            transform=ax.transAxes,

            ha="right",
            va="bottom",

            multialignment="left",   # Left-align multiline text

            fontsize=9,
            verticalalignment="bottom",
            family="monospace",

            bbox=dict(
                facecolor="white",
                edgecolor="gray",
                alpha=0.75,
                boxstyle="round,pad=0.4",
            ),
        )


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
    

    def plot(self, title="Battery Voltage Forecast",):

        fig, ax = plt.subplots(
        figsize=(10, 5)
        )

        self._figure = fig
        self._axes = ax

        ax.plot(
            self.dataset.data["ds"],
            self.dataset.data["microVolt"],
            marker="o",
            label="Measured",
        )

        # --------------------------------------------------------
        # Plot fitted values (if available)
        # --------------------------------------------------------

        fitted = self.result.fitted

        if fitted is not None:

            ax.plot(
                self.dataset.data["ds"].iloc[1:],
                fitted.iloc[1:],
                linewidth=2,
                alpha=0.8,
                label=f"{self.result.model} Fit",
            )

        
        ax.plot(

        self.result.dates,

        self.result.forecast,

        marker="s",

        linestyle="--",

        linewidth=2,

        color="tab:red",

        label=f"{self.result.model} Forecast",

        )

        ax.axvline(

        self.dataset.data["ds"].iloc[-1],

        color="gray",

        linestyle="--",

        linewidth=1.5,

        alpha=0.7,

        )

        if title is None:
            title = self._build_title()

        ax.set_title(title)

        self._draw_experiment_box(ax)

        ax.set_xlabel(
            "Date"
        )

        ax.set_ylabel(
            "Voltage (µV)"
        )

        ax.grid(True)

        ax.legend(loc="upper left")

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