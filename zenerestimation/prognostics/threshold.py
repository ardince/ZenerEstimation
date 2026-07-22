"""
Threshold estimation utilities.

Provides reusable threshold strategies for
Remaining Useful Life estimation.
"""

from __future__ import annotations

import numpy as np


class ThresholdEstimator:
    """
    Utility class for estimating failure thresholds.
    """

    DEFAULT_PERCENTILE = 99
    DEFAULT_MARGIN = 5.0

    @staticmethod
    def fixed(value: float) -> float:
        """
        Return a user-defined threshold.
        """

        return float(value)

    @staticmethod
    def percentile(values, percentile=99):
        """
        Threshold from a percentile of historical data.
        """

        values = np.asarray(values)

        return float(
            np.percentile(
                values,
                percentile,
            )
        )

    @staticmethod
    def percentile_margin(
        values,
        percentile=99,
        margin=5.0,
    ):
        """
        Percentile threshold plus safety margin.
        """

        return (

            ThresholdEstimator.percentile(

                values,

                percentile,

            )

            + margin

        )

    @staticmethod
    def maximum_margin(
        values,
        margin=5.0,
    ):
        """
        Maximum observed value plus margin.
        """

        values = np.asarray(values)

        return float(

            np.max(values)

            + margin

        )
    

    @classmethod
    def default(cls, values):
        """
        Default threshold policy used by ZenerEstimation.
        """

        return cls.percentile_margin(
            values,
            percentile=cls.DEFAULT_PERCENTILE,
            margin=cls.DEFAULT_MARGIN,
        )