"""
Experiment result loader.

Discovers and loads previously generated experiment packages.

The loader is intentionally read-only. It never executes a forecasting
model and never modifies existing result files.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path


# ============================================================
# Result Package
# ============================================================


@dataclass(frozen=True)
class ResultPackage:
    """
    Structured representation of one stored experiment run.
    """

    battery: str
    model: str
    timestamp: str
    run_number: int

    directory: Path

    forecast: dict | None
    evaluation: dict | None
    experiment: dict | None

    report: Path | None
    figure: Path | None
    log: Path | None


# ============================================================
# Result Loader
# ============================================================


class ResultLoader:
    """
    Discover and load stored experiment results.

    Expected directory structure:

        results/
        └── <battery>/
            └── <model>/
                └── <timestamp>_<run_number>/
                    ├── forecast.json
                    ├── evaluation.json
                    ├── experiment.json
                    ├── forecast.png
                    ├── report.txt
                    └── experiment.log
    """

    def __init__(
        self,
        root="results",
    ):
        self.root = Path(root)

    # ========================================================
    # Discovery
    # ========================================================

    def discover(self):
        """
        Discover all experiment run directories.

        Returns
        -------
        list[Path]
            Experiment package directories.
        """

        if not self.root.exists():
            return []

        packages = []

        for battery_directory in self.root.iterdir():

            if not battery_directory.is_dir():
                continue

            for model_directory in battery_directory.iterdir():

                if not model_directory.is_dir():
                    continue

                for run_directory in model_directory.iterdir():

                    if not run_directory.is_dir():
                        continue

                    if self._is_run_directory(
                        run_directory
                    ):
                        packages.append(
                            run_directory
                        )

        return sorted(packages)

    # ========================================================
    # Run Directory Validation
    # ========================================================

    @staticmethod
    def _is_run_directory(
        directory,
    ):
        """
        Determine whether a directory follows the
        timestamp/run-number naming convention.
        """

        parts = directory.name.rsplit(
            "_",
            1,
        )

        if len(parts) != 2:
            return False

        timestamp = parts[0]
        run_number = parts[1]

        if len(timestamp) != 15:
            return False

        if timestamp[8] != "_":
            return False

        if not run_number.isdigit():
            return False

        return True

    # ========================================================
    # Load One Package
    # ========================================================

    def load(
        self,
        directory,
    ):
        """
        Load one experiment package.
        """

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Result directory does not exist: "
                f"{directory}"
            )

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Result path is not a directory: "
                f"{directory}"
            )

        if not self._is_run_directory(
            directory
        ):
            raise ValueError(
                f"Invalid experiment directory: "
                f"{directory}"
            )

        # ----------------------------------------------------
        # Extract identity from directory structure
        # ----------------------------------------------------

        model_directory = directory.parent
        battery_directory = model_directory.parent

        battery = battery_directory.name
        model = model_directory.name

        timestamp, run_number = (
            directory.name.rsplit(
                "_",
                1,
            )
        )

        run_number = int(run_number)

        # ----------------------------------------------------
        # Load JSON artifacts
        # ----------------------------------------------------

        forecast = self._load_json(
            directory / "forecast.json"
        )

        evaluation = self._load_json(
            directory / "evaluation.json"
        )

        experiment = self._load_json(
            directory / "experiment.json"
        )

        # ----------------------------------------------------
        # Optional presentation artifacts
        # ----------------------------------------------------

        figure = self._optional_path(
            directory / "forecast.png"
        )

        report = self._optional_path(
            directory / "report.txt"
        )

        log = self._optional_path(
            directory / "experiment.log"
        )

        return ResultPackage(

            battery=battery,

            model=model,

            timestamp=timestamp,

            run_number=run_number,

            directory=directory,

            forecast=forecast,

            evaluation=evaluation,

            experiment=experiment,

            report=report,

            figure=figure,

            log=log,

        )

    # ========================================================
    # Load Battery
    # ========================================================

    def load_battery(
        self,
        battery,
    ):
        """
        Load all experiment runs belonging to one battery.
        """

        battery = str(battery)

        results = []

        for directory in self.discover():

            battery_name = (
            directory.parent.parent.name
            )

            if battery_name != battery:
                continue

            results.append(
                self.load(directory)
            )

        return results

    # ========================================================
    # Load Model
    # ========================================================

    def load_model(
        self,
        battery,
        model,
    ):
        """
        Load all runs of one model for one battery.
        """

        battery = str(battery)
        model = str(model).lower()

        results = []

        for directory in self.discover():

            battery_name = (
            directory.parent.parent.name
            )

            model_name = (
            directory.parent.name.lower()
            )

            if battery_name != battery:
                continue

            if model_name != model:
                continue

            results.append(
                self.load(directory)
            )

        return results

    # ========================================================
    # Latest Run
    # ========================================================

    def latest(
        self,
        battery,
        model,
    ):
        """
        Return the latest run for a battery/model pair.

        Returns None if no run exists.
        """

        results = self.load_model(
            battery=battery,
            model=model,
        )

        if not results:
            return None

        return max(
            results,
            key=lambda result: (
                result.timestamp,
                result.run_number,
            ),
        )

    # ========================================================
    # JSON Loading
    # ========================================================

    @staticmethod
    def _load_json(
        filename,
    ):
        """
        Load a JSON artifact.

        Missing JSON files return None.
        """

        filename = Path(filename)

        if not filename.exists():
            return None

        with open(
            filename,
            "r",
            encoding="utf-8",
        ) as fp:

            return json.load(fp)

    # ========================================================
    # Optional Path
    # ========================================================

    @staticmethod
    def _optional_path(
        filename,
    ):
        """
        Return a Path if the artifact exists.
        """

        filename = Path(filename)

        if filename.exists():
            return filename

        return None