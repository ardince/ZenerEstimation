"""
Canonical battery dataset processor.

Transforms raw battery measurements into a deterministic,
regularized time-series representation without performing
model-specific preprocessing or target interpolation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .result import DatasetProcessingResult


class DatasetProcessor:
    """
    Canonicalize raw battery measurement data.

    The processor:

    - validates required columns,
    - converts dates and target values,
    - sorts observations chronologically,
    - rejects duplicate timestamps,
    - constructs a complete regular timeline,
    - inserts missing periods,
    - marks genuine observations with ``is_observed``.

    Missing target values are deliberately left as ``NaN``.

    Parameters
    ----------
    frequency:
        Canonical pandas frequency string.

        The ZenerEstimation quarterly default is ``"QS-MAR"``.
    """

    REQUIRED_COLUMNS = (
        "ds",
        "microVolt",
    )

    def __init__(
        self,
        frequency: str = "QS-MAR",
        *,
        dayfirst: bool = True,
        duplicate_policy: str = "error",
    ) -> None:

        if not isinstance(
            frequency,
            str,
        ) or not frequency.strip():

            raise ValueError(
                "frequency must be a non-empty string"
            )

        if not isinstance(
            dayfirst,
            bool,
        ):

            raise TypeError(
                "dayfirst must be a boolean"
            )

        allowed_policies = {
            "error",
            "first",
            "last",
            "mean",
        }

        if duplicate_policy not in allowed_policies:

            raise ValueError(
                "duplicate_policy must be one of: "
                "error, first, last, mean"
            )

        self.frequency = frequency
        self.dayfirst = dayfirst
        self.duplicate_policy = duplicate_policy

    # ========================================================
    # Public API
    # ========================================================

    def process(
        self,
        data: pd.DataFrame,
        battery: str,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetProcessingResult:
        """
        Convert raw measurements into the canonical representation.

        Parameters
        ----------
        data:
            Raw measurement dataframe containing ``ds`` and
            ``microVolt``.

        battery:
            Battery identifier.

        metadata:
            Optional provenance metadata.

        Returns
        -------
        DatasetProcessingResult
            Standardized processed dataset.
        """

        self._validate_input(
            data,
            battery,
        )

        source_rows = len(
            data
        )

        clean = self._prepare_dataframe(
            data
        )

        clean = self._handle_duplicates(
            clean
        )

        self._validate_alignment(
            clean
        )

        #timeline = self._build_timeline(
         #   clean
        #)

        processed = self._regularize(
            clean
        )

        missing_periods = int(
            (
                ~processed[
                    "is_observed"
                ]
            ).sum()
        )

        result_metadata = {
            "processor": self.__class__.__name__,
            "date_dayfirst": self.dayfirst,
            "duplicate_policy": self.duplicate_policy,
            **(
                metadata.copy()
                if metadata is not None
                else {}
            ),
        }

        return DatasetProcessingResult(
            battery=battery,
            data=processed,
            source_rows=source_rows,
            processed_rows=len(
                processed
            ),
            missing_periods=missing_periods,
            frequency=self.frequency,
            metadata=result_metadata,
        )

    # ========================================================
    # Input validation
    # ========================================================

    def _validate_input(
        self,
        data: pd.DataFrame,
        battery: str,
    ) -> None:

        if not isinstance(
            data,
            pd.DataFrame,
        ):

            raise TypeError(
                "data must be a pandas DataFrame"
            )

        if data.empty:

            raise ValueError(
                "data cannot be empty"
            )

        if not isinstance(
            battery,
            str,
        ) or not battery.strip():

            raise ValueError(
                "battery must be a non-empty string"
            )

        missing_columns = (
            set(
                self.REQUIRED_COLUMNS
            )
            - set(
                data.columns
            )
        )

        if missing_columns:

            raise ValueError(
                "raw data is missing required "
                f"columns: {sorted(missing_columns)}"
            )

    # ========================================================
    # Canonical preparation
    # ========================================================

    #@staticmethod
    def _prepare_dataframe(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        frame = data.loc[
            :,
            [
                "ds",
                "microVolt",
            ],
        ].copy()

        # ----------------------------------------------------
        # Dates
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Dates
        # ----------------------------------------------------

        date_values = frame["ds"]

        if pd.api.types.is_datetime64_any_dtype(
            date_values
        ):

            frame["ds"] = pd.to_datetime(
            date_values,
            errors="coerce",
        )

        else:

            text = (
                date_values
                .astype("string")
                .str.strip()
            )

            # ISO dates are always interpreted explicitly
            # as YYYY-MM-DD.

            iso_mask = text.str.match(
                r"^\d{4}-\d{2}-\d{2}$",
                na=False,
            )

            parsed = pd.Series(
                pd.NaT,
                index=frame.index,
                dtype="datetime64[ns]",
            )

            if iso_mask.any():

                parsed.loc[
                    iso_mask
                ] = pd.to_datetime(
                    text.loc[
                        iso_mask
                    ],
                    format="%Y-%m-%d",
                    errors="coerce",
                )

            non_iso_mask = ~iso_mask

            if non_iso_mask.any():

                parsed.loc[
                    non_iso_mask
                ] = pd.to_datetime(
                    text.loc[
                        non_iso_mask
                    ],
                    errors="coerce",
                    dayfirst=self.dayfirst,
                )

            frame["ds"] = parsed


        if frame[
            "ds"
            ].isna().any():

            raise ValueError(
                "ds contains invalid or missing dates"
            )
        
        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        frame["microVolt"] = pd.to_numeric(
            frame["microVolt"],
            errors="coerce",
        )

        if frame[
            "microVolt"
        ].isna().any():

            raise ValueError(
                "raw microVolt values must be "
                "numeric and non-missing"
            )

        values = frame[
            "microVolt"
        ].to_numpy(
            dtype=float
        )

        if not np.all(
            np.isfinite(values)
        ):

            raise ValueError(
                "raw microVolt values must be finite"
            )

        # ----------------------------------------------------
        # Ordering
        # ----------------------------------------------------

        frame = (
            frame
            .sort_values("ds")
            .reset_index(drop=True)
        )

        return frame

    # ========================================================
    # Duplicate validation
    # ========================================================

    @staticmethod
    def _validate_duplicates(
        data: pd.DataFrame,
    ) -> None:

        duplicated = data[
            "ds"
        ].duplicated(
            keep=False
        )

        if duplicated.any():

            duplicate_dates = (
                data.loc[
                    duplicated,
                    "ds",
                ]
                .dt.strftime(
                    "%Y-%m-%d"
                )
                .unique()
                .tolist()
            )

            raise ValueError(
                "duplicate timestamps are not allowed: "
                f"{duplicate_dates}"
            )

    # ========================================================
    # Frequency alignment
    # ========================================================

    def _validate_alignment(
        self,
        data: pd.DataFrame,
    ) -> None:
        """
        Ensure timestamps lie on the configured
        canonical frequency grid.
        """

        if data.empty:
            return

        try:

            offset = pd.tseries.frequencies.to_offset(
                self.frequency
            )

        except ValueError as exc:

            raise ValueError(
                "invalid frequency: "
                f"{self.frequency}"
            ) from exc

        off_grid: list[pd.Timestamp] = []

        for timestamp in data["ds"]:

            timestamp = pd.Timestamp(
                timestamp
            )

            if not offset.is_on_offset(
                timestamp
            ):

                off_grid.append(
                    timestamp
                )

        if off_grid:

            formatted = [
                timestamp.strftime(
                    "%Y-%m-%d"
                )
                for timestamp in off_grid
            ]

            raise ValueError(
                "timestamps are not aligned "
                f"with frequency {self.frequency}: "
                f"{formatted}"
            )

    # ========================================================
    # Timeline regularization
    # ========================================================

    def _regularize(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        start = data[
            "ds"
        ].iloc[0]

        end = data[
            "ds"
        ].iloc[-1]

        full_index = pd.date_range(
            start=start,
            end=end,
            freq=self.frequency,
        )

        original_dates = set(
            data[
                "ds"
            ]
        )

        regularized = (
            data
            .set_index("ds")
            .reindex(full_index)
        )

        regularized.index.name = "ds"

        regularized = (
            regularized
            .reset_index()
        )

        regularized[
            "is_observed"
        ] = regularized[
            "ds"
        ].isin(
            original_dates
        )

        regularized[
            "is_observed"
        ] = regularized[
            "is_observed"
        ].astype(bool)

        return regularized.loc[
            :,
            [
                "ds",
                "microVolt",
                "is_observed",
            ],
        ]


    def _handle_duplicates(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        duplicated = data[
            "ds"
        ].duplicated(
            keep=False
        )

        if not duplicated.any():
            return data

        if self.duplicate_policy == "error":

            duplicate_dates = (
                data.loc[
                    duplicated,
                    "ds",
                ]
                .dt.strftime(
                    "%Y-%m-%d"
                )
                .unique()
                .tolist()
            )

            raise ValueError(
                "duplicate timestamps are not allowed: "
                f"{duplicate_dates}"
            )

        if self.duplicate_policy == "first":

            return (
                data
                .drop_duplicates(
                    subset="ds",
                    keep="first",
                )
                .reset_index(drop=True)
            )

        if self.duplicate_policy == "last":

            return (
                data
                .drop_duplicates(
                    subset="ds",
                    keep="last",
                )
                .reset_index(drop=True)
            )

        if self.duplicate_policy == "mean":

            return (
                data
                .groupby(
                    "ds",
                    as_index=False,
                    sort=True,
                )
                .agg(
                    microVolt=(
                        "microVolt",
                        "mean",
                    )
                )
                .reset_index(
                    drop=True
                )
            )

        raise ValueError(
            "unsupported duplicate_policy: "
            f"{self.duplicate_policy}"
        )

    # ========================================================
    # Representation
    # ========================================================

    def __repr__(
        self,
    ) -> str:

        return (
            "DatasetProcessor("
            f"frequency={self.frequency!r}"
            f"dayfirst={self.dayfirst!r}"
            ")"
        )