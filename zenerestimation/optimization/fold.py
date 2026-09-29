"""
Temporal optimization fold.

Defines an immutable expanding-window validation fold used
during forecasting-model optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from zenerestimation.data.dataset import BatteryDataset


@dataclass(frozen=True)
class TemporalFold:
    """
    One temporal training/validation fold.

    Parameters
    ----------
    index:
        Zero-based fold index.

    train:
        Training dataset containing observations strictly
        preceding the validation block.

    validation:
        Validation dataframe immediately following the
        training dataset.
    """

    index: int
    train: BatteryDataset
    validation: pd.DataFrame

    def __post_init__(self) -> None:
        """
        Validate temporal fold integrity.
        """

        if (
            not isinstance(self.index, int)
            or isinstance(self.index, bool)
        ):
            raise TypeError(
                "index must be an integer"
            )

        if self.index < 0:
            raise ValueError(
                "index must be greater than or equal to zero"
            )

        if not isinstance(
            self.train,
            BatteryDataset,
        ):
            raise TypeError(
                "train must be a BatteryDataset"
            )

        if not isinstance(
            self.validation,
            pd.DataFrame,
        ):
            raise TypeError(
                "validation must be a pandas DataFrame"
            )

        if len(self.train) == 0:
            raise ValueError(
                "train cannot be empty"
            )

        if self.validation.empty:
            raise ValueError(
                "validation cannot be empty"
            )

        for name in (
            "ds",
            "microVolt",
        ):
            if name not in self.train.data.columns:
                raise ValueError(
                    f"train must contain a {name} column"
                )

            if name not in self.validation.columns:
                raise ValueError(
                    f"validation must contain a {name} column"
                )

        train_end = pd.Timestamp(
            self.train.data["ds"].iloc[-1]
        )

        validation_start = pd.Timestamp(
            self.validation["ds"].iloc[0]
        )

        if train_end >= validation_start:
            raise ValueError(
                "training observations must strictly "
                "precede validation observations"
            )

    @property
    def training_points(self) -> int:
        """
        Number of training observations.
        """

        return len(self.train)

    @property
    def validation_points(self) -> int:
        """
        Number of validation observations.
        """

        return len(self.validation)