"""
Internal temporal validation utilities.

Provides leakage-safe expanding-window folds for
forecasting-model optimization.
"""

from __future__ import annotations

from zenerestimation.data.dataset import BatteryDataset

from .fold import TemporalFold


class ExpandingWindowSplitter:
    """
    Generate expanding-window temporal validation folds.

    Parameters
    ----------
    folds:
        Number of validation folds.

    validation_steps:
        Number of observations in each validation block.
    """

    def __init__(
        self,
        folds: int = 3,
        validation_steps: int = 4,
    ) -> None:

        self._validate_positive_integer(
            folds,
            "folds",
        )

        self._validate_positive_integer(
            validation_steps,
            "validation_steps",
        )

        self.folds = folds
        self.validation_steps = validation_steps

    def split(
        self,
        dataset: BatteryDataset,
    ) -> tuple[TemporalFold, ...]:
        """
        Generate expanding-window validation folds.
        """

        if not isinstance(
            dataset,
            BatteryDataset,
        ):
            raise TypeError(
                "dataset must be a BatteryDataset"
            )

        data = dataset.data.copy()

        required_validation = (
            self.folds
            * self.validation_steps
        )

        minimum_rows = (
            required_validation
            + 1
        )

        if len(data) < minimum_rows:
            raise ValueError(
                "dataset does not contain enough rows "
                "for the requested temporal validation"
            )

        initial_train_size = (
            len(data)
            - required_validation
        )

        results = []

        for fold_index in range(
            self.folds
        ):

            validation_start = (
                initial_train_size
                + (
                    fold_index
                    * self.validation_steps
                )
            )

            validation_end = (
                validation_start
                + self.validation_steps
            )

            train_df = (
                data
                .iloc[:validation_start]
                .copy()
            )

            validation_df = (
                data
                .iloc[
                    validation_start:
                    validation_end
                ]
                .copy()
            )

            train_dataset = BatteryDataset(
                train_df
            )

            results.append(
                TemporalFold(
                    index=fold_index,
                    train=train_dataset,
                    validation=validation_df,
                )
            )

        return tuple(results)

    @staticmethod
    def _validate_positive_integer(
        value,
        name: str,
    ) -> None:

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise TypeError(
                f"{name} must be an integer"
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than zero"
            )

    def __repr__(self) -> str:
        return (
            "ExpandingWindowSplitter("
            f"folds={self.folds}, "
            f"validation_steps={self.validation_steps}"
            ")"
        )