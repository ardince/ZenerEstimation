"""
Remaining Useful Life Histogram

Visualizes Monte Carlo RUL simulations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


class RULHistogram:
    """
    Histogram for Monte Carlo RUL results.
    """

    def __init__(self, prognostic_result):

        self.result = prognostic_result

    def plot(self):

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.hist(
            self.result.samples,
            bins=25,
            alpha=0.75,
        )

        ax.set_title(
            f"{self.result.model} Remaining Useful Life"
        )

        ax.set_xlabel("Remaining Useful Life (years)")
        ax.set_ylabel("Frequency")

        text = (
            f"Mean      : {self.result.mean:.2f} years\n"
            f"Median    : {self.result.median:.2f}\n"
            f"P10       : {self.result.p10:.2f}\n"
            f"P90       : {self.result.p90:.2f}\n"
            f"\n"
            f"Threshold : {self.result.threshold:.3f} µV\n"
            f"Simulations : {self.result.simulations}"
        )

        ax.text(
            0.98,
            0.98,
            text,
            transform=ax.transAxes,
            va="top",
            ha="right",
            bbox=dict(
                facecolor="white",
                alpha=0.85,
            ),
        )

        plt.tight_layout()

        return fig

    def save(self, filename):

        fig = self.plot()

        fig.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)