"""
Experiment folder utilities.

Creates one dedicated directory for every experiment and
provides standard filenames for all generated artifacts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class ExperimentFolder:
    """
    Represents one experiment directory.

    Example
    -------
    results/
        20260717_153055_732B-5610410_arima/
            forecast.png
            forecast.json
            experiment.json
            report.txt
            info.txt
    """

    def __init__(
        self,
        battery: str,
        model: str,
        root: str = "results",
    ):

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.root = Path(root)

        self.path = (
            self.root
            / f"{timestamp}_{battery}_{model.lower()}"
        )

        self.path.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def forecast_png(self):

        return self.path / "forecast.png"

    @property
    def forecast_json(self):

        return self.path / "forecast.json"

    @property
    def experiment_json(self):

        return self.path / "experiment.json"

    @property
    def report_txt(self):

        return self.path / "report.txt"

    @property
    def info_txt(self):

        return self.path / "info.txt"

    @property
    def directory(self):

        return self.path

    def __str__(self):

        return str(self.path)

    def summary(self):

        """
        Return all artifact locations.
        """

        return {
            "directory": str(self.directory),
            "forecast_png": str(self.forecast_png),
            "forecast_json": str(self.forecast_json),
            "experiment_json": str(self.experiment_json),
            "report_txt": str(self.report_txt),
            "info_txt": str(self.info_txt),
        }