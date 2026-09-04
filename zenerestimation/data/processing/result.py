"""
Dataset processing result.

Defines the standardized result produced by the canonical
dataset-processing pipeline.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DatasetProcessingResult:
    """
    Immutable result of canonical dataset processing.

    Parameters
    ----------
    battery:
        Battery dataset identifier.

    data:
        Canonical processed dataframe.

    source_rows:
        Number of rows in the original raw dataset.

    processed_rows:
        Number of rows after timeline regularization.

    missing_periods:
        Number of missing periods inserted into the canonical
        timeline.

    frequency:
        Canonical dataset frequency.

    metadata:
        Additional processing metadata.
    """

    battery: str

    data: pd.DataFrame

    source_rows: int

    processed_rows: int

    missing_periods: int

    frequency: str

    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:

        if not isinstance(
            self.battery,
            str,
        ) or not self.battery.strip():

            raise ValueError(
                "battery must be a non-empty string"
            )

        if not isinstance(
            self.data,
            pd.DataFrame,
        ):

            raise TypeError(
                "data must be a pandas DataFrame"
            )

        required_columns = {
            "ds",
            "microVolt",
            "is_observed",
        }

        missing_columns = (
            required_columns
            - set(self.data.columns)
        )

        if missing_columns:

            raise ValueError(
                "processed data is missing "
                "required columns: "
                f"{sorted(missing_columns)}"
            )

        if self.source_rows <= 0:

            raise ValueError(
                "source_rows must be greater than zero"
            )

        
        if self.processed_rows != len(
            self.data
        ):

            raise ValueError(
                "processed_rows must match data length"
            )

        if self.missing_periods < 0:

            raise ValueError(
                "missing_periods cannot be negative"
            )

        observed_rows = int(
            self.data[
                "is_observed"
            ].sum()
        )

        expected_missing_periods = (
            self.processed_rows
            - observed_rows
        )

        if (
            self.missing_periods
            != expected_missing_periods
        ):

            raise ValueError(
                "missing_periods must equal "
                "processed_rows - observed_rows"
            )

        if not isinstance(
            self.frequency,
            str,
        ) or not self.frequency.strip():

            raise ValueError(
                "frequency must be a non-empty string"
            )

        # Defensive copies because frozen dataclasses do not
        # make mutable objects immutable automatically.

        object.__setattr__(
            self,
            "data",
            self.data.copy(
                deep=True
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            deepcopy(
                self.metadata
            )
            if self.metadata is not None
            else {},
        )

    @property
    def observed_rows(self) -> int:
        """
        Number of original observed timestamps
        retained in the canonical dataset.
        """

        return int(
            self.data[
                "is_observed"
            ].sum()
        )

    @property
    def inserted_rows(self) -> int:
        """
        Number of canonical rows inserted for missing periods.
        """

        return (
            self.processed_rows
            - self.source_rows
        )

    @property
    def duplicate_rows_resolved(
        self,
        ) -> int:
        """
        Number of excess raw rows removed while
        resolving duplicate timestamps.
        """

        return max(
            0,
            self.source_rows
            - self.observed_rows
        )

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Return a compact processing summary.
        """

        return {
            "battery": self.battery,
            "source_rows": self.source_rows,
            "processed_rows": self.processed_rows,
            "observed_rows": self.observed_rows,
            "missing_periods": self.missing_periods,
            "frequency": self.frequency,
        }

    def metadata_dict(
        self,
    ) -> dict[str, Any]:
        """
        Return serializable processing metadata.
        """

        return {
            "battery": self.battery,
            "source_rows": self.source_rows,
            "processed_rows": self.processed_rows,
            "observed_rows": self.observed_rows,
            "missing_periods": self.missing_periods,
            "frequency": self.frequency,
            "processing_version": "1.0",
            "target": "microVolt",
            **deepcopy(
                self.metadata
            ),
        }

    def dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Return a defensive copy of the processed dataframe.
        """

        return self.data.copy(
            deep=True
        )

    def __repr__(
        self,
    ) -> str:

        return (
            "DatasetProcessingResult("
            f"battery={self.battery!r}, "
            f"source_rows={self.source_rows}, "
            f"processed_rows={self.processed_rows}, "
            f"missing_periods={self.missing_periods}, "
            f"frequency={self.frequency!r}"
            ")"
        )