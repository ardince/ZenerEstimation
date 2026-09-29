"""
Optimization/benchmark boundary.

Separates the development region used for model optimization
from the untouched final benchmark holdout.
"""

from __future__ import annotations

import pandas as pd

from zenerestimation.data.dataset import BatteryDataset

from copy import deepcopy


class BenchmarkBoundary:
    """
    Separate development data from a final benchmark holdout.

    Parameters
    ----------
    evaluation_steps:
        Number of observations reserved for final benchmark
        evaluation.

    evaluation_end:
        Optional final date of the benchmark region. Data after
        this date is excluded from both optimization and benchmark
        evaluation.
    """

    def __init__(
        self,
        evaluation_steps: int,
        *,
        evaluation_end=None,
    ) -> None:

        if (
            not isinstance(evaluation_steps, int)
            or isinstance(evaluation_steps, bool)
        ):
            raise TypeError(
                "evaluation_steps must be an integer"
            )

        if evaluation_steps <= 0:
            raise ValueError(
                "evaluation_steps must be greater than zero"
            )

        self.evaluation_steps = evaluation_steps

        self.evaluation_end = (
            pd.Timestamp(evaluation_end)
            if evaluation_end is not None
            else None
        )

    def development_dataset(
        self,
        dataset: BatteryDataset,
    ) -> BatteryDataset:
        """
        Return only the development region available to optimization.
        """

        bounded = self._bounded_data(
            dataset
        )

        if len(bounded) <= self.evaluation_steps:
            raise ValueError(
                "dataset does not contain enough rows "
                "to create both development and benchmark regions"
            )

        development = (
            bounded
            .iloc[:-self.evaluation_steps]
            .copy()
            .reset_index(drop=True)
        )

        result = deepcopy(
            dataset
        )

        result.data = development

        return result
    

    def benchmark_data(
        self,
        dataset: BatteryDataset,
    ) -> pd.DataFrame:
        """
        Return the untouched final benchmark partition.
        """

        bounded = self._bounded_data(
            dataset
        )

        if len(bounded) <= self.evaluation_steps:
            raise ValueError(
                "dataset does not contain enough rows "
                "to create both development and benchmark regions"
            )

        return (
            bounded
            .iloc[-self.evaluation_steps:]
            .copy()
        )

    def _bounded_data(
        self,
        dataset: BatteryDataset,
    ) -> pd.DataFrame:
        """
        Apply the optional benchmark end-date boundary.
        """

        if not isinstance(
            dataset,
            BatteryDataset,
        ):
            raise TypeError(
                "dataset must be a BatteryDataset"
            )

        data = dataset.data.copy()

        if self.evaluation_end is not None:

            data = data.loc[
                data["ds"]
                <= self.evaluation_end
            ].copy()

            if data.empty:
                raise ValueError(
                    "evaluation_end precedes all dataset observations"
                )

        return (
            data
            .sort_values("ds")
            .reset_index(drop=True)
        )

    @property
    def metadata(
        self,
    ) -> dict:

        return {
            "evaluation_steps": self.evaluation_steps,
            "evaluation_end": (
                str(
                    self.evaluation_end.date()
                )
                if self.evaluation_end is not None
                else None
            ),
        }

    def __repr__(
        self,
    ) -> str:

        return (
            "BenchmarkBoundary("
            f"evaluation_steps={self.evaluation_steps}, "
            f"evaluation_end={self.evaluation_end!r}"
            ")"
        )