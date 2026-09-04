"""
Processed dataset persistence.

Writes canonical processed datasets and their provenance
metadata without modifying the original raw dataset.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .result import DatasetProcessingResult


@dataclass(frozen=True)
class ProcessedDatasetPaths:
    """
    Paths created for a persisted processed dataset.
    """

    data: Path
    metadata: Path

    @property
    def directory(self) -> Path:
        return self.data.parent


class ProcessedDatasetWriter:
    """
    Persist a DatasetProcessingResult.

    The writer creates:

    - <battery>.csv
    - <battery>.metadata.json

    Existing artifacts are protected by default.

    Parameters
    ----------
    output_directory:
        Directory used for processed datasets.
    """

    def __init__(
        self,
        output_directory: str | Path = "datasets/processed",
    ) -> None:

        self.output_directory = Path(
            output_directory
        )

    # ========================================================
    # Public API
    # ========================================================

    def save(
        self,
        result: DatasetProcessingResult,
        *,
        overwrite: bool = False,
    ) -> ProcessedDatasetPaths:
        """
        Persist a processed dataset and its metadata.

        Parameters
        ----------
        result:
            DatasetProcessingResult to persist.

        overwrite:
            Whether existing processed artifacts may be replaced.

        Returns
        -------
        ProcessedDatasetPaths
            Paths to the generated artifacts.
        """

        self._validate_result(
            result
        )

        paths = self.paths_for(
            result.battery
        )

        self._validate_overwrite(
            paths,
            overwrite=overwrite,
        )

        paths.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._write_csv_atomic(
            result,
            paths.data,
        )

        try:

            self._write_metadata_atomic(
                result,
                paths.metadata,
            )

        except Exception:

            # Avoid leaving a new CSV without matching
            # metadata if metadata persistence fails.
            if not overwrite:
                paths.data.unlink(
                    missing_ok=True
                )

            raise

        return paths

    def paths_for(
        self,
        battery: str,
    ) -> ProcessedDatasetPaths:
        """
        Return the canonical persistence paths for a battery.
        """

        self._validate_battery_filename(
            battery
        )

        return ProcessedDatasetPaths(
            data=(
                self.output_directory
                / f"{battery}.csv"
            ),
            metadata=(
                self.output_directory
                / f"{battery}.metadata.json"
            ),
        )

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _validate_result(
        result: DatasetProcessingResult,
    ) -> None:

        if not isinstance(
            result,
            DatasetProcessingResult,
        ):

            raise TypeError(
                "result must be a "
                "DatasetProcessingResult"
            )

        ProcessedDatasetWriter._validate_battery_filename(
            result.battery
        )

    @staticmethod
    def _validate_battery_filename(
        battery: str,
    ) -> None:

        if not isinstance(
            battery,
            str,
        ) or not battery.strip():

            raise ValueError(
                "battery must be a non-empty string"
            )

        candidate = Path(
            battery
        )

        if (
            candidate.name != battery
            or "/" in battery
            or "\\" in battery
        ):

            raise ValueError(
                "battery must be a safe filename identifier"
            )

    @staticmethod
    def _validate_overwrite(
        paths: ProcessedDatasetPaths,
        *,
        overwrite: bool,
    ) -> None:

        existing = [
            path
            for path in (
                paths.data,
                paths.metadata,
            )
            if path.exists()
        ]

        if existing and not overwrite:

            names = ", ".join(
                path.name
                for path in existing
            )

            raise FileExistsError(
                "processed dataset artifacts already exist: "
                f"{names}"
            )

    # ========================================================
    # CSV persistence
    # ========================================================

    @staticmethod
    def _write_csv_atomic(
        result: DatasetProcessingResult,
        destination: Path,
    ) -> None:

        temporary = destination.with_suffix(
            destination.suffix + ".tmp"
        )

        frame = result.dataframe()

        try:

            frame.to_csv(
                temporary,
                index=False,
                date_format="%Y-%m-%d",
                na_rep="",
            )

            os.replace(
                temporary,
                destination,
            )

        finally:

            temporary.unlink(
                missing_ok=True
            )

    # ========================================================
    # Metadata persistence
    # ========================================================

    @classmethod
    def _write_metadata_atomic(
        cls,
        result: DatasetProcessingResult,
        destination: Path,
    ) -> None:

        temporary = destination.with_suffix(
            destination.suffix + ".tmp"
        )

        metadata = cls._build_metadata(
            result
        )

        try:

            with open(
                temporary,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    metadata,
                    file,
                    indent=4,
                    ensure_ascii=False,
                    default=cls._json_default,
                )

                file.write(
                    "\n"
                )

            os.replace(
                temporary,
                destination,
            )

        finally:

            temporary.unlink(
                missing_ok=True
            )

    # ========================================================
    # Metadata
    # ========================================================

    @staticmethod
    def _build_metadata(
        result: DatasetProcessingResult,
    ) -> dict[str, Any]:

        return {
            "schema": (
                "zenerestimation.processed-dataset"
            ),
            "schema_version": "1.0",
            **result.metadata_dict(),
            "columns": {
                "ds": {
                    "role": "timestamp",
                    "format": "YYYY-MM-DD",
                },
                "microVolt": {
                    "role": "target",
                    "missing_representation": (
                        "empty CSV field"
                    ),
                },
                "is_observed": {
                    "role": "observation_flag",
                    "true": (
                        "original measured observation"
                    ),
                    "false": (
                        "inserted canonical period"
                    ),
                },
            },
        }

    # ========================================================
    # JSON conversion
    # ========================================================

    @staticmethod
    def _json_default(
        value: Any,
    ) -> Any:

        if isinstance(
            value,
            Path,
        ):
            return str(
                value
            )

        if isinstance(
            value,
            (
                datetime,
                date,
                pd.Timestamp,
            ),
        ):
            return value.isoformat()

        if isinstance(
            value,
            np.generic,
        ):
            return value.item()

        raise TypeError(
            "object is not JSON serializable: "
            f"{type(value).__name__}"
        )

    # ========================================================
    # Representation
    # ========================================================

    def __repr__(
        self,
    ) -> str:

        return (
            "ProcessedDatasetWriter("
            f"output_directory="
            f"{str(self.output_directory)!r}"
            ")"
        )