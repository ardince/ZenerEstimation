"""
Experiment result utilities.

Creates structured output directories for forecasting
and diagnostic experiments.

Directory structure:

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

from __future__ import annotations

import json

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# ============================================================
# Result Container
# ============================================================


@dataclass(frozen=True)
class ExperimentResult:
    """
    Output artifacts belonging to one experiment run.
    """

    figure: Path

    forecast: Path

    evaluation: Path

    experiment: Path

    report: Path

    log: Path

    directory: Path

    battery: str

    model: str

    timestamp: str

    run_number: int


# ============================================================
# Configuration
# ============================================================


RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Timestamp
# ============================================================


def timestamp():
    """
    Return a human-readable experiment timestamp.

    Format:

        YYYYMMDD_HHMMSS
    """

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


# ============================================================
# Run Number
# ============================================================

def _next_run_number(
    model_directory: Path,
):
    """
    Return the next sequential experiment number.

    The counter is stored independently from historical
    directory names so legacy microsecond-based directories
    cannot be mistaken for run numbers.
    """

    counter_file = (
        model_directory / ".run_counter"
    )

    if counter_file.exists():

        try:

            current = int(
                counter_file.read_text(
                    encoding="utf-8"
                ).strip()
            )

        except ValueError:

            current = 0

    else:

        current = 0

    next_number = current + 1

    counter_file.write_text(
        str(next_number),
        encoding="utf-8",
    )

    return next_number


# ============================================================
# Result Files
# ============================================================


def create_result_files(
    battery,
    model,
    suffix=None,
    root=None,
):
    """
    Create the directory structure for one experiment run.

    Parameters
    ----------
    battery : str
        Battery identifier.

    model : str
        Forecasting model identifier.

    suffix : str, optional
        Optional suffix retained for backwards compatibility.

    Returns
    -------
    ExperimentResult
        Paths belonging to the experiment package.
    """

    battery = str(battery)

    model = str(model).lower()

    results_root = (
        Path(root)
        if root is not None
        else RESULTS_DIR
    )

    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    battery_directory = (
        results_root / battery
    )

    model_directory = (
        battery_directory / model
    )

    battery_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_number = _next_run_number(
        model_directory
    )

    stamp = timestamp()

    run_name = (
        f"{stamp}_{run_number}"
    )

    if suffix:
        run_name += f"_{suffix}"

    directory = (
        model_directory / run_name
    )

    # --------------------------------------------------------
    # Protect against an extremely unlikely timestamp collision
    # --------------------------------------------------------

    while directory.exists():

        run_number += 1

        run_name = (
            f"{stamp}_{run_number}"
        )

        if suffix:
            run_name += f"_{suffix}"

        directory = (
            model_directory / run_name
        )

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # Standard artifact names
    # --------------------------------------------------------

    figure = (
        directory / "forecast.png"
    )

    forecast = (
        directory / "forecast.json"
    )

    evaluation = (
        directory / "evaluation.json"
    )

    experiment = (
        directory / "experiment.json"
    )

    report = (
        directory / "report.txt"
    )

    log = (
        directory / "experiment.log"
    )

    return ExperimentResult(

        figure=directory / "forecast.png",

        forecast=directory / "forecast.json",

        evaluation=directory / "evaluation.json",

        experiment=directory / "experiment.json",

        report=directory / "report.txt",

        log=directory / "experiment.log",

        directory=directory,

        battery=battery,

        model=model,

        timestamp=stamp,

        run_number=run_number,

    )


# ============================================================
# JSON Utilities
# ============================================================


def save_metadata(
    filename,
    metadata,
):
    """
    Save experiment metadata as JSON.
    """

    filename = Path(filename)

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as fp:

        json.dump(
            metadata,
            fp,
            indent=4,
            default=str,
        )


# ============================================================
# Report Utility
# ============================================================


def save_report(
    filename,
    text,
):
    """
    Save a human-readable report.
    """

    filename = Path(filename)

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as fp:

        fp.write(text)