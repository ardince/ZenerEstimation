"""
Experiment result utilities.

Creates timestamped output files for demonstrations
and forecasting experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass(frozen=True)
class ExperimentResult:
    """
    Output files belonging to one experiment.
    """

    figure: Path

    metadata: Path

    report: Path

    log: Path

    directory: Path


RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(
    exist_ok=True
)


def timestamp():

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def create_result_files(battery, model,suffix=None,):

    stamp = timestamp()

    prefix = f"{stamp}_{battery}_{model}"

    figure = RESULTS_DIR / f"{prefix}.png"

    metadata = RESULTS_DIR / f"{prefix}.json"

    report = RESULTS_DIR / f"{prefix}.txt"

    log = RESULTS_DIR / f"{prefix}.log"

    if suffix:
        prefix += f"_{suffix}"

    return ExperimentResult(
        figure=figure,
        metadata=metadata,
        report=report,
        log=log,
        directory=RESULTS_DIR,
    )


import json

def save_metadata(filename, metadata,):
    """
    Save experiment metadata as JSON.
    """

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


def save_report(filename, text,):

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as fp:

        fp.write(text)

