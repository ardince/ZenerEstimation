from __future__ import annotations

import pandas as pd

from ..dataset import BatteryDataset

#from zenerestimation.data.temporal import (
 #   TemporalPreprocessor,
#)


class TemporalPreprocessor:
    """
    Leakage-safe temporal preparation for a
    training dataset.

    The preprocessor operates only on the
    dataset supplied to it. It does not perform
    train/validation splitting itself.

    Current responsibilities:
        - preserve the canonical timeline
        - interpolate internal missing target values
        - optionally fill leading/trailing gaps
        - preserve observation provenance

    Model-specific transformations such as
    scaling, differencing, and neural sequence
    generation are intentionally excluded.
    """

    SUPPORTED_METHODS = (
        "linear",
    )

    def __init__(
        self,
        *,
        method: str = "linear",
        fill_edges: bool = True,
    ) -> None:

        if method not in self.SUPPORTED_METHODS:

            raise ValueError(
                "unsupported interpolation method: "
                f"{method}"
            )

        if not isinstance(
            fill_edges,
            bool,
        ):

            raise TypeError(
                "fill_edges must be bool"
            )

        self.method = method
        self.fill_edges = fill_edges

    def transform(
        self,
        dataset: BatteryDataset,
    ) -> BatteryDataset:
        """
        Return a model-ready copy of the supplied
        temporal dataset.

        Only target missing values are filled.
        The canonical timestamps and
        is_observed flags are preserved.
        """

        self._validate_dataset(
            dataset
        )

        data = dataset.data.copy(
            deep=True
        )

        target = (
            pd.to_numeric(
                data["microVolt"],
                errors="coerce",
            )
            .astype(float)
        )

        target = target.interpolate(
            method=self.method,
            limit_area="inside",
        )

        if self.fill_edges:

            target = (
                target
                .ffill()
                .bfill()
            )

        if target.isna().any():

            missing_dates = (
                data.loc[
                    target.isna(),
                    "ds",
                ]
                .dt.strftime(
                    "%Y-%m-%d"
                )
                .tolist()
            )

            raise ValueError(
                "temporal preprocessing could "
                "not resolve missing target "
                f"values: {missing_dates}"
            )

        data["microVolt"] = target

        transformed = BatteryDataset(
            data
        )

        self._copy_dataset_metadata(
            source=dataset,
            target=transformed,
        )

        transformed._preprocessing = {
            "temporal": True,
            "method": self.method,
            "fill_edges": self.fill_edges,
        }

        return transformed

    def fit_transform(
        self,
        dataset: BatteryDataset,
    ) -> BatteryDataset:
        """
        Convenience alias for transform().

        Temporal interpolation currently has no
        learned parameters, but this method
        provides a stable preprocessing API for
        future extensions.
        """

        return self.transform(
            dataset
        )

    @staticmethod
    def _validate_dataset(
        dataset,
    ) -> None:

        if not isinstance(
            dataset,
            BatteryDataset,
        ):

            raise TypeError(
                "dataset must be "
                "a BatteryDataset"
            )

        required_columns = {
            "ds",
            "microVolt",
        }

        missing_columns = (
            required_columns
            - set(dataset.data.columns)
        )

        if missing_columns:

            raise ValueError(
                "dataset is missing required "
                f"columns: {sorted(missing_columns)}"
            )

        if dataset.data.empty:

            raise ValueError(
                "dataset cannot be empty"
            )

        if (
            dataset.data[
                "microVolt"
            ]
            .notna()
            .sum()
            == 0
        ):

            raise ValueError(
                "dataset contains no observed "
                "target values"
            )

    @staticmethod
    def _copy_dataset_metadata(
        *,
        source: BatteryDataset,
        target: BatteryDataset,
    ) -> None:
        """
        Preserve framework-level dataset identity
        and provenance without altering the
        original dataset.
        """

        attributes = (
            "_battery",
            "_source_type",
            "_source_path",
        )

        for attribute in attributes:

            if hasattr(
                source,
                attribute,
            ):

                setattr(
                    target,
                    attribute,
                    getattr(
                        source,
                        attribute,
                    ),
                )

    def __repr__(
        self,
    ) -> str:

        return (
            "TemporalPreprocessor("
            f"method={self.method!r}, "
            f"fill_edges={self.fill_edges!r}"
            ")"
        )